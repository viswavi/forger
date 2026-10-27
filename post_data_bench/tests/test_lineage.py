"""The submitted checkpoint's SFT and RL runs: row cap, prompts from the final data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding.rl import lineage
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


if __name__ == "__main__":
    unittest.main()
