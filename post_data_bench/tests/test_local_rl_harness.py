"""Local RL harness: receipts on every exit, probe evaluation, output directories."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding.rl import local_training

QWEN3_4B = {
    "model_type": "qwen3",
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 9728,
}


def fake_checkpoint(path):
    """A tiny HF directory that passes usable_checkpoint and checkpoint_files."""
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(QWEN3_4B))
    (path / "tokenizer_config.json").write_text("{}")
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    (path / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"\0" * 4
    )
    return path


class ReceiptTests(unittest.TestCase):
    """execute_rl writes rl_submission.json even when Slime fails."""

    def test_crashed_run_keeps_receipts_for_saved_checkpoints(self):
        """hf_rollout9 saved before a crash gets completed_steps 10; partial skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            data, output = root / "rl_trial_1.data", root / "rl_trial_1"
            data.mkdir()

            def crash(command, env=None):
                fake_checkpoint(output / "hf_rollout9")
                (output / "hf_rollout19").mkdir()  # still being written
                raise subprocess.CalledProcessError(1, command)

            with patch.object(local_training, "run", side_effect=crash):
                with self.assertRaises(subprocess.CalledProcessError):
                    local_training.execute_rl(data, root / "base", output, None)
            receipt = json.loads(
                (output / "hf_rollout9/rl_submission.json").read_text()
            )
            self.assertEqual(receipt["completed_steps"], 10)
            self.assertEqual(receipt["data_directory"], str(data))
            self.assertEqual(receipt["backend"], "local")
            self.assertFalse((output / "hf_rollout19/rl_submission.json").exists())

    def test_receipts_appear_while_the_run_is_going(self):
        """A checkpoint saved mid-run has its receipt before Slime exits."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            data, output = root / "rl_trial_1.data", root / "rl_trial_1"
            data.mkdir()
            seen = []

            def train(command, env=None):
                fake_checkpoint(output / "hf_rollout9")
                receipt = output / "hf_rollout9/rl_submission.json"
                for _ in range(200):
                    if receipt.exists():
                        break
                    subprocess.run(["sleep", "0.01"])
                seen.append(receipt.exists())

            with patch.object(local_training, "run", side_effect=train):
                local_training.execute_rl(
                    data, root / "base", output, None, receipt_interval=0.01
                )
            self.assertEqual(seen, [True])


class ProbeEvaluationTests(unittest.TestCase):
    """Probes skip the holdout evaluation; full runs evaluate every 10 steps."""

    def dry_run(self, root, *extra):
        """The Slime script run_rl_training_and_evaluation.sh would execute."""
        binaries = root / "bin"
        binaries.mkdir(exist_ok=True)
        for name, body in {"nvidia-smi": "echo GPU0", "conda": "echo /conda"}.items():
            (binaries / name).write_text(f"#!/bin/sh\n{body}\n")
            (binaries / name).chmod(0o755)
        (root / "slime/bin").mkdir(parents=True, exist_ok=True)
        python = root / "slime/bin/python3"
        python.write_text("#!/bin/sh\nexit 1\n")
        python.chmod(0o755)
        (root / "deps/Megatron-LM").mkdir(parents=True, exist_ok=True)
        (root / "checkpoint").mkdir(exist_ok=True)
        (root / "data").mkdir(exist_ok=True)
        for name in ("train.jsonl", "val.jsonl"):
            (root / "data" / name).write_text("{}\n")
        command = [
            "bash",
            str(local_training.CODING / "rl/run_rl_training_and_evaluation.sh"),
            "--rl-data-dir",
            str(root / "data"),
            "--init-checkpoint",
            str(root / "checkpoint"),
            "--output-dir",
            str(root / "out"),
            "--slime-env",
            str(root / "slime"),
            "--slime-deps",
            str(root / "deps"),
            "--dry-run",
            *extra,
        ]
        env = {
            "PATH": f"{binaries}:/usr/bin:/bin",
            "HOME": str(root),
            "SLURM_SUBMIT_DIR": "",
            "TVM_FFI_CACHE_DIR": str(root / "tvm"),
            "FLASHINFER_WORKSPACE_BASE": str(root / "flash"),
            "LCB_RM_WORKERS": "4",
        }
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_probe_command_turns_evaluation_off(self):
        """The probe passes --eval-interval 0 and still saves every step."""
        probe = local_training.rl_command(
            "data", "ckpt", "out", "env", "deps", probe=True
        )
        last = len(probe) - 1 - probe[::-1].index("--eval-interval")
        self.assertEqual(probe[last + 1], "0")
        last = len(probe) - 1 - probe[::-1].index("--save-interval")
        self.assertEqual(probe[last + 1], "1")

    def test_eval_interval_zero_drops_the_holdout_from_slime(self):
        """No --eval-prompt-data at 0; the default run evaluates every 10 steps."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = self.dry_run(root, "--eval-interval", "0")
            full = self.dry_run(root)
        self.assertNotIn("--eval-prompt-data", probe)
        self.assertNotIn("--eval-interval", probe)
        self.assertIn("--eval-interval 10", full)
        self.assertIn("--eval-prompt-data generated_val", full)
        self.assertIn("--n-samples-per-eval-prompt 4", full)


if __name__ == "__main__":
    unittest.main()
