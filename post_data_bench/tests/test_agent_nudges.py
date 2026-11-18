"""The launcher's one nudge for an agent that finished with a small dataset."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_nudges import (  # noqa: E402
    count_rows,
    low_rows_message,
    new_prompt_count,
    prompt_hashes,
)


def _dataset(rows: int) -> Path:
    path = Path(tempfile.mkdtemp()) / "generated_raw.jsonl"
    path.write_text("".join('{"prompt": []}\n' for _ in range(rows)) + "\n\n")
    return path


class LowRowsMessageTests(unittest.TestCase):
    """When the launcher asks the agent to continue, and what it says."""

    def test_small_dataset_gets_the_numbers_and_how_to_decline(self):
        """321 rows: RL would see each prompt ~11 times; finishing again keeps it."""
        message = low_rows_message(_dataset(321), 1000, 2.5 * 3600)
        self.assertIn("has 321 rows", message)
        self.assertIn("seen about 11 times", message)
        self.assertIn("about 2.5 hours left", message)
        self.assertIn("keep the current pair intact", message)
        self.assertIn("simply finish again", message)

    def test_enough_rows_or_disabled_or_little_time_says_nothing(self):
        """At the minimum, with minimum=0, or under 30 minutes left: no nudge."""
        self.assertIsNone(low_rows_message(_dataset(1000), 1000, 3600))
        self.assertIsNone(low_rows_message(_dataset(10), 0, 3600))
        self.assertIsNone(low_rows_message(_dataset(10), 1000, 29 * 60))

    def test_starting_rows_do_not_count_toward_the_floor(self):
        """With a starting dataset, only rows outside it count.

        472 kept + 50 new is low.
        """
        import hashlib
        import json

        path = Path(tempfile.mkdtemp()) / "generated_raw.jsonl"
        rows = [{"prompt": [{"role": "user", "content": str(i)}]} for i in range(522)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        known = {
            hashlib.sha256(json.dumps(r["prompt"], sort_keys=True).encode()).hexdigest()
            for r in rows[:472]
        }
        message = low_rows_message(
            path, 100, 12 * 3600, target=200, starting_prompts=known
        )
        self.assertIn("has 522 rows, 50 of them new", message)
        self.assertIn("about 200 new rows is a good target", message)
        self.assertIsNone(
            low_rows_message(path, 50, 12 * 3600, target=200, starting_prompts=known)
        )

    def test_starting_hashes_read_rows_without_validating_rubrics(self):
        """A code criterion needs no sandbox package to hash its row's prompt.

        2 starting rows (one with code) + 1 new row counts 1 new.
        """
        import json

        folder = Path(tempfile.mkdtemp())
        rubric = [
            {"criterion": "x", "points": 1, "code": "def score(r, p, t): return 1"}
        ]
        start = [
            {"prompt": [{"role": "user", "content": str(i)}], "rubric": rubric}
            for i in range(2)
        ]
        (folder / "start.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in start)
        )
        new = start + [
            {"prompt": [{"role": "user", "content": "new"}], "rubric": rubric}
        ]
        (folder / "new.jsonl").write_text("".join(json.dumps(r) + "\n" for r in new))
        known = prompt_hashes(folder / "start.jsonl")
        self.assertEqual(len(known), 2)
        self.assertEqual(new_prompt_count(folder / "new.jsonl", known), 1)

    def test_missing_dataset_is_named(self):
        """No dataset file yet: the nudge says so, in minutes when under an hour."""
        missing = Path(tempfile.mkdtemp()) / "generated_raw.jsonl"
        message = low_rows_message(missing, 1000, 40 * 60)
        self.assertIn("There is no dataset at", message)
        self.assertIn("about 40 minutes left", message)
        self.assertIsNone(count_rows(missing))

    def test_blank_lines_do_not_count(self):
        """Trailing blank lines are not rows."""
        self.assertEqual(count_rows(_dataset(3)), 3)


if __name__ == "__main__":
    unittest.main()
