"""The submitted checkpoint's SFT and RL runs: row cap, prompts from the final data."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from coding.rl import agent_training, lineage
from coding.rl import rubric_data as data

from tests.test_unified_training import rows, write_rows


class LineageTests(unittest.TestCase):
    """sft_trial -> rl_trial_1 -> rl_trial_2, each with its own compiled data."""

    def setUp(self):
        """A final dataset of 20 rows."""
        self.root = Path(tempfile.mkdtemp())
        self.final = rows()
        self.final_path = self.root / "generated_raw.jsonl"
        write_rows(self.final_path, self.final)

    def run_dir(self, name, values, stage, start=None):
        """A checkpoint with its receipt, its compiled data, and (RL) its start."""
        compiled = self.root / f"{name}.data"
        compiled.mkdir()
        write_rows(compiled / "submitted.jsonl", values)
        checkpoint = self.root / name
        checkpoint.mkdir()
        (checkpoint / f"{stage}_submission.json").write_text(
            json.dumps({"stage": stage, "data_directory": str(compiled)})
        )
        if start is not None:
            lineage.record_start(compiled, start)
        return checkpoint

    def chain(self, sft_rows=None, rl_rows=None):
        """SFT, then two RL runs; returns the last RL checkpoint."""
        sft = self.run_dir("sft_trial", sft_rows or self.final[:15], "sft")
        rl1 = self.run_dir("rl_trial_1", rl_rows or self.final, "rl", start=sft)
        return self.run_dir("rl_trial_2", self.final[5:], "rl", start=rl1)

    def test_stages_walk_back_to_the_sft_trial(self):
        """Newest first, ending at the SFT trial the first RL run started from."""
        runs = lineage.check(self.chain(), self.final_path)
        self.assertEqual(
            [(r["stage"], Path(r["checkpoint"]).name) for r in runs],
            [("rl", "rl_trial_2"), ("rl", "rl_trial_1"), ("sft", "sft_trial")],
        )

    def test_base_model_has_no_runs(self):
        """A checkpoint without receipts is the base model: nothing to check."""
        base = self.root / "Qwen3-4B"
        base.mkdir()
        self.assertEqual(lineage.check(base, self.final_path), [])

    def test_sft_prompts_outside_the_final_dataset_are_rejected(self):
        """A STaR or seed set with other prompts cannot be behind the submission."""
        other = rows()
        other[0]["prompt"] = "A prompt the final dataset does not have"
        with self.assertRaisesRegex(ValueError, "SFT run .* 1 prompts not in"):
            lineage.check(self.chain(sft_rows=other), self.final_path)

    def test_run_over_the_row_cap_is_rejected(self):
        """Each run in the lineage has the same cap as the final dataset."""
        with patch.object(lineage, "MAX_ROWS", 18):
            with self.assertRaisesRegex(
                ValueError, "RL run .* 20 rows; the limit is 18"
            ):
                lineage.check(self.chain(), self.final_path)

    def test_rl_run_without_its_start_is_rejected(self):
        """RL runs from before lineage.json cannot be traced, so they are refused."""
        rl = self.run_dir("rl_old", self.final, "rl")
        with self.assertRaisesRegex(
            ValueError, "did not record its starting checkpoint"
        ):
            lineage.check(rl, self.final_path)

    def test_dataset_over_the_row_cap_is_rejected(self):
        """validate_dataset refuses more than MAX_ROWS rows for any run."""
        path = self.root / "big.jsonl"
        write_rows(path, rows())
        with patch.object(data, "MAX_ROWS", 19):
            with self.assertRaisesRegex(ValueError, "at most 19"):
                data.validate_dataset(path)


class StepBudgetTests(unittest.TestCase):
    """Local checkpoints: exactly RL_STEP_BUDGET steps over the whole lineage."""

    def setUp(self):
        """A final dataset and a base model directory."""
        self.root = Path(tempfile.mkdtemp())
        self.final = rows()
        self.final_path = self.root / "generated_raw.jsonl"
        write_rows(self.final_path, self.final)
        self.base = self.root / "Qwen3-4B"
        self.base.mkdir()

    def local_run(self, name, steps, start):
        """A local RL checkpoint after `steps` steps, continuing from `start`."""
        compiled = self.root / f"{name}.data"
        compiled.mkdir()
        write_rows(compiled / "submitted.jsonl", self.final)
        lineage.record_start(compiled, start)
        checkpoint = self.root / name / f"hf_rollout{steps - 1}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "rl_submission.json").write_text(
            json.dumps(
                {
                    "stage": "rl",
                    "backend": "local",
                    "completed_steps": steps,
                    "data_directory": str(compiled),
                }
            )
        )
        return checkpoint

    def test_steps_add_up_over_continuations(self):
        """60 steps, then 20 more from that checkpoint: 80, submittable."""
        first = self.local_run("rl_trial_1", 60, self.base)
        second = self.local_run("rl_trial_2", 20, first)
        self.assertEqual(lineage.rl_steps(second), 80)
        self.assertEqual(len(lineage.check(second, self.final_path)), 2)

    def test_other_totals_are_rejected(self):
        """99 steps (80 + 19) and 70 steps both fail; the message gives the total."""
        first = self.local_run("rl_trial_1", 80, self.base)
        over = self.local_run("rl_trial_2", 19, first)
        with self.assertRaisesRegex(ValueError, "has 99 RL steps .* exactly 80"):
            lineage.check(over, self.final_path)
        under = self.local_run("rl_trial_3", 70, self.base)
        with self.assertRaisesRegex(ValueError, "has 70 RL steps"):
            lineage.check(under, self.final_path)

    def test_rl_refuses_a_run_past_the_budget(self):
        """From 60 steps, --num-rollout 30 is refused and 20 allowed."""
        first = self.local_run("rl_trial_1", 60, self.base)
        args = SimpleNamespace(
            num_rollout=30, rollout_batch_size=32, n_samples_per_prompt=8
        )
        with self.assertRaisesRegex(ValueError, "--num-rollout may be at most 20"):
            agent_training.check_rl_budget("local", first, args)
        args.num_rollout = 20
        agent_training.check_rl_budget("local", first, args)

    def test_rl_refuses_another_batch_shape(self):
        """More prompts per step would be more training in the same 80 steps."""
        args = SimpleNamespace(
            num_rollout=80, rollout_batch_size=64, n_samples_per_prompt=8
        )
        with self.assertRaisesRegex(ValueError, "--rollout-batch-size is fixed at 32"):
            agent_training.check_rl_budget("local", self.base, args)

    def test_tinker_has_no_budget(self):
        """Tinker runs keep any number of steps and batch shape."""
        args = SimpleNamespace(
            num_rollout=500, rollout_batch_size=64, n_samples_per_prompt=4
        )
        agent_training.check_rl_budget("tinker", self.base, args)


if __name__ == "__main__":
    unittest.main()
