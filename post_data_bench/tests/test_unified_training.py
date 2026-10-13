"""Regression tests for unified training."""
import argparse
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from coding.rl import agent_training as training
from coding.rl import rubric_data as data
from coding.rl import rubric_rm as rm

ROOT = Path(__file__).resolve().parents[1]


def rows():
    """Create distinct shared-triple fixtures for submission tests."""
    return [
        {
            "prompt": f"Question {i}",
            "ground_truth": "<think>Reasoning</think>Answer",
            "rubric": [
                [
                    {
                        ("code"): (
                            "def grade(prompt, response, ground_truth):\n    return 1.0"
                        )
                    },
                    1,
                ]
            ],
        }
        for i in range(20)
    ]


def write_rows(path, values):
    """Write fixture rows as newline-delimited JSON."""
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in values))


def read(path):
    """Load newline-delimited JSON fixture records."""
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class DatasetTests(unittest.TestCase):
    """Regression checks for dataset."""

    def test_split_and_sft_rl_identity(self):
        """Verify split and sft rl identity."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_rows(root / "input", rows())
            data.compile_data(root / "input", root / "a")
            train, val, sft = [
                read(root / "a" / name)
                for name in ("train.jsonl", "val.jsonl", "sft.jsonl")
            ]
            self.assertEqual((len(train), len(val), len(sft)), (18, 2, 18))
            self.assertEqual(
                [r["prompt"] for r in train], [r["messages"][:-1] for r in sft]
            )
            self.assertFalse(
                {json.dumps(r["prompt"]) for r in train}
                & {json.dumps(r["prompt"]) for r in val}
            )
            self.assertNotIn("rubric", json.dumps([r["messages"] for r in sft]))
            write_rows(root / "input", list(reversed(rows())))
            data.compile_data(root / "input", root / "b")
            self.assertEqual(
                (root / "a/train.jsonl").read_bytes(),
                (root / "b/train.jsonl").read_bytes(),
            )

    def test_reject_duplicate_and_small(self):
        """Verify reject duplicate and small."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            for values in (rows()[:1], rows() + [rows()[0]]):
                write_rows(path, values)
                with self.assertRaises(ValueError):
                    data.compile_data(path, Path(tmp) / "out")

    def test_invalid_weights_and_union(self):
        """Verify invalid weights and union."""
        for rubric in (
            [],
            [[{"code": "x", "prompt": "x"}, 1]],
            [[{"prompt": "x"}, -1]],
            [[{"prompt": "x"}, float("nan")]],
            [[{"prompt": "x"}, 0]],
        ):
            with self.subTest(rubric=rubric), self.assertRaises(ValueError):
                data.validate_rubric(rubric)

    def test_thinking_and_overlength(self):
        """Verify thinking and overlength."""
        with self.assertRaises(ValueError):
            data.validate_row({**rows()[0], "ground_truth": "answer"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            write_rows(path, rows())
            tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [0] * 101)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                data.compile_data(path, Path(tmp) / "out", tokenizer, max_length=100)
            self.assertFalse((Path(tmp) / "out").exists())


class CodeTests(unittest.TestCase):
    """Regression checks for code."""

    def test_named_grade_and_helpers(self):
        """Verify named grade and helpers."""
        source = (
            "def grade(prompt, response, ground_truth):\n    return "
            "helper(response == ground_truth)\ndef helper(value):\n    "
            "return float(value)"
        )
        self.assertEqual(rm.code_score(source, [], "a", "a"), 1)
        self.assertEqual(rm.code_score(source, [], "a", "b"), 0)

    def test_forbidden_access_and_invalid_score(self):
        """Verify forbidden access and invalid score."""
        for source in (
            "import os\ndef grade(p,r,g):\n    return float(bool(os.environ))",
            "def grade(p,r,g):\n    return open('/etc/passwd').read()",
            "def grade(p,r,g):\n    return 12",
            "def grade(p,r,g):\n    return r.__class__",
        ):
            with self.subTest(source=source):
                self.assertEqual(rm.code_score(source, [], "a", "b"), 0)

    def test_timeout(self):
        """Verify timeout."""
        self.assertEqual(
            rm.code_score(
                "def grade(p,r,g):\n    while True:\n        pass", [], "a", "a"
            ),
            0,
        )


class RewardTests(unittest.IsolatedAsyncioTestCase):
    """Regression checks for reward."""

    def setUp(self):
        """Isolate persistent judge usage and environment settings."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.usage_file = Path(temporary.name) / "judge_usage.json"
        environment = patch.dict(
            os.environ,
            {
                "JUDGE_TOKEN_USAGE_FILE": str(self.usage_file),
                "JUDGE_TOKEN_UNIT_LIMIT": "75000000",
                "JUDGE_MODEL": rm.JUDGE_MODEL,
                "JUDGE_REASONING_EFFORT": "low",
                "PDB_TASK": "coding",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    async def test_mixed_weighted_reward_and_original_prompt(self):
        """Verify mixed weighted reward and original prompt."""
        sample = SimpleNamespace(
            response="<think>trace</think>answer",
            label="gold",
            prompt="templated",
            metadata={
                "original_prompt": [{"role": "user", "content": "original"}],
                "rubric": [
                    [{"code": "def grade(p, r, g):\n    return 0.0"}, 1],
                    [{"prompt": "criterion"}, 3],
                ],
            },
        )
        with patch.object(
            rm.asyncio, "to_thread", new=AsyncMock(return_value=0)
        ), patch.object(rm, "prompt_score", new=AsyncMock(return_value=1)) as judge:
            self.assertEqual(await rm.custom_rm(None, sample), 0.75)
            self.assertEqual(
                judge.call_args.args[1], sample.metadata["original_prompt"]
            )
            self.assertEqual(judge.call_args.args[2], sample.response)

    async def test_judge_model_and_separate_key(self):
        """Verify judge model and separate key."""
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"score": 0.75}'))
            ],
        )
        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "test-judge",
                "LITELLM_API_KEY": "test-generation",
                "LITELLM_BASE_URL": "https://example.test",
            },
        ), patch(
            "litellm.acompletion", new=AsyncMock(return_value=response)
        ) as completion:
            self.assertEqual(
                await rm.prompt_score("criterion", [], "response", "gold"), 0.75
            )
            self.assertEqual(completion.call_args.kwargs["api_key"], "test-judge")
            self.assertEqual(completion.call_args.kwargs["model"], rm.JUDGE_MODEL)
            self.assertEqual(
                completion.call_args.kwargs["reasoning"], {"effort": "low"}
            )

    async def test_consecutive_failure_circuit_and_no_secret_in_error(self):
        """Verify consecutive failure circuit and no secret in error."""
        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "test-judge",
                "LITELLM_BASE_URL": "https://example.test",
                "JUDGE_MAX_CONSECUTIVE_FAILURES": "3",
            },
        ), patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=RuntimeError("private provider body")),
        ) as completion, patch.object(
            rm.asyncio, "sleep", new=AsyncMock()
        ):
            with self.assertRaises(rm.JudgeUnavailable) as error:
                await rm.prompt_score("criterion", [], "response", "gold")
            self.assertEqual(completion.await_count, 9)
            self.assertNotIn("private", str(error.exception))
            with self.assertRaises(rm.JudgeUnavailable):
                await rm.prompt_score("criterion", [], "response", "gold")
            self.assertEqual(completion.await_count, 9)

    async def test_success_resets_failures(self):
        """Verify success resets failures."""
        success = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"score": 1}'))],
        )
        with patch.dict(
            os.environ,
            {
                "LITELLM_JUDGE_API_KEY": "test",
                "LITELLM_BASE_URL": "https://example.test",
                "JUDGE_MAX_CONSECUTIVE_FAILURES": "2",
            },
        ), patch(
            "litellm.acompletion",
            new=AsyncMock(
                side_effect=[RuntimeError(), success, RuntimeError(), success]
            ),
        ), patch.object(
            rm.asyncio, "sleep", new=AsyncMock()
        ):
            self.assertEqual(await rm.prompt_score("c", [], "r", "g"), 1)
            self.assertEqual(await rm.prompt_score("c", [], "r", "g"), 1)

    async def test_no_generation_key_fallback(self):
        """Verify no generation key fallback."""
        with patch.dict(
            os.environ, {"LITELLM_API_KEY": "generation", "LITELLM_JUDGE_API_KEY": ""}
        ):
            with self.assertRaises(rm.JudgeUnavailable):
                await rm.prompt_score("c", [], "r", "g")


class FinalizationTests(unittest.TestCase):
    """Regression checks for finalization."""

    def test_freeze_selected_checkpoint_and_detect_tampering(self):
        """Verify freeze selected checkpoint and detect tampering."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = root / "agent"
            checkpoint = agent / "selected"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text(
                json.dumps(
                    dict(
                        model_type="qwen3",
                        hidden_size=2560,
                        num_hidden_layers=36,
                        num_attention_heads=32,
                        num_key_value_heads=8,
                        intermediate_size=9728,
                    )
                )
            )
            (checkpoint / "tokenizer_config.json").write_text("{}")
            (checkpoint / "model.safetensors").write_bytes(b"test fixture weights")
            (checkpoint / "untrusted.py").write_text("raise Exception()")
            write_rows(agent / "generated_raw.jsonl", rows())
            data.compile_data(agent / "generated_raw.jsonl", root / "prepared")
            (checkpoint / "sft_submission.json").write_text(
                json.dumps({"sft_sha256": training.digest(root / "prepared/sft.jsonl")})
            )
            (agent / "submission.json").write_text(
                json.dumps({"checkpoint": str(checkpoint)})
            )
            args = argparse.Namespace(
                data=agent / "generated_raw.jsonl",
                submission=agent / "submission.json",
                agent_dir=agent,
                output_dir=root / "final",
                slime_env="/slime",
                slime_deps="/deps",
                prepare_only=True,
            )
            host_model = root / "host_model"
            host_model.mkdir()
            (host_model / "tokenizer_config.json").write_text('{"host": true}')
            with patch(
                "huggingface_hub.snapshot_download", return_value=str(host_model)
            ), patch.object(
                training,
                "prepare",
                side_effect=lambda src, dst, model: data.compile_data(src, dst),
            ):
                training.finalize(args)
            self.assertFalse((root / "final/initial_checkpoint/untrusted.py").exists())
            self.assertEqual(
                json.loads(
                    (
                        root / "final/initial_checkpoint/tokenizer_config.json"
                    ).read_text()
                ),
                {"host": True},
            )
            self.assertEqual(
                (root / "final/initial_checkpoint/model.safetensors").read_bytes(),
                b"test fixture weights",
            )
            with patch.dict(
                os.environ,
                {
                    "PDB_TASK": "HealthBench",
                    "JUDGE_MODEL": "custom/DeepSeek-V4-Flash-changed",
                    "JUDGE_REASONING_EFFORT": "high",
                    "JUDGE_TOKEN_UNIT_LIMIT": "230000000",
                    "JUDGE_TOKEN_USAGE_FILE": str(root / "different.json"),
                },
            ), patch.object(training, "run") as run:
                training.run_final(argparse.Namespace(bundle=root / "final"))
                command = run.call_args.args[0]
                self.assertIn("--rubric-rewards", command)
                self.assertNotIn("--no-thinking", command)
                self.assertEqual(command[command.index("--task") + 1], "coding")
                self.assertEqual(
                    command[command.index("--judge-token-unit-limit") + 1], "75000000.0"
                )
                self.assertEqual(
                    command[command.index("--judge-token-usage-file") + 1],
                    str(agent / "judge_token_usage.json"),
                )
                self.assertEqual(
                    command[command.index("--judge-reasoning-effort") + 1], "low"
                )
            (root / "final/data/train.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "changed"):
                training.run_final(argparse.Namespace(bundle=root / "final"))

    def test_probe_skips_secret_eval(self):
        """Verify probe skips secret eval."""
        command = training.rl_command(
            "data", "checkpoint", "out", "env", "deps", probe=True
        )
        self.assertIn("--skip-final-eval", command)
        self.assertIn("--rubric-rewards", command)


class RecipeTests(unittest.TestCase):
    """Regression checks for recipe."""

    def test_invalid_judge_config_fails_before_gpu_startup(self):
        """Validate the judge before checking GPU or training dependencies."""
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            data.mkdir()
            command = [
                "bash",
                str(ROOT / "coding/rl/run_rl_training_and_evaluation.sh"),
                "--rubric-rewards",
                "--rl-data-dir",
                str(data),
                "--judge-model",
                "azure_ai/FW-GLM-5.3",
                "--dry-run",
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("JUDGE_MODEL must contain DeepSeek-V4-Flash", result.stderr)
            self.assertNotIn("nvidia-smi", result.stderr)

    def test_dry_run_builds_fixed_reward_and_valid_shell_without_credentials(self):
        """Verify dry run builds fixed reward and valid shell without credentials."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binaries = root / "bin"
            binaries.mkdir()
            for name, body in {
                "conda": "echo /mock/conda",
                "nvidia-smi": "echo GPU0",
            }.items():
                target = binaries / name
                target.write_text("#!/bin/sh\n" + body + "\n")
                target.chmod(0o755)
            environment = root / "slime"
            (environment / "bin").mkdir(parents=True)
            python = environment / "bin/python"
            python.write_text("#!/bin/sh\nexit 1\n")
            python.chmod(0o755)
            (root / "deps/Megatron-LM").mkdir(parents=True)
            (root / "checkpoint").mkdir()
            (root / "data").mkdir()
            for name in ("train.jsonl", "val.jsonl"):
                (root / "data" / name).write_text("{}\n")
            env = {
                **os.environ,
                "PATH": str(binaries) + ":" + os.environ["PATH"],
                "TVM_FFI_CACHE_DIR": str(root / "tvm"),
                "FLASHINFER_WORKSPACE_BASE": str(root / "flash"),
                "LITELLM_JUDGE_API_KEY": "test-only-secret",
                "SLURM_SUBMIT_DIR": "",
            }
            command = training.rl_command(
                root / "data",
                root / "checkpoint",
                root / "out",
                str(environment),
                str(root / "deps"),
            ) + ["--dry-run"]
            result = subprocess.run(
                command, env=env, capture_output=True, text=True, check=True
            )
            self.assertNotIn("test-only-secret", result.stdout)
            self.assertIn("--custom-rm-path rl.rubric_rm.custom_rm", result.stdout)
            self.assertIn("enable_thinking", result.stdout)
            self.assertNotIn("--no-thinking", result.stdout)
            start = result.stdout.index("set -euo pipefail\ncd ")
            script = result.stdout[
                start : result.stdout.index("----- (--dry-run:", start)
            ]
            subprocess.run(["bash", "-n"], input=script, text=True, check=True)
            self.assertFalse((root / "out").exists())
            blocked = subprocess.run(
                command + ["--custom-rm-path", "evil.reward"],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(blocked.returncode, 0)


if __name__ == "__main__":
    unittest.main()
