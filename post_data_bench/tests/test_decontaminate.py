"""Training data that overlaps the held-out test problems is removed before training."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coding.rl import decontaminate
from coding.rl.rubric_data import compile_data

from tests.test_unified_training import rows, write_rows

TEST_STATEMENT = (
    "Given a string s of lowercase letters, return the length of the longest "
    "substring in which every vowel appears an even number of times, counting "
    "only the vowels a e i o and u, and return zero when no such substring exists "
    "in the string at all after scanning it once from the left end to the right end."
)
OTHER_STATEMENT = (
    "You are given an integer array nums and an integer k. Split nums into exactly k "
    "non-empty contiguous parts so that the largest sum among the parts is as small "
    "as possible, and return that smallest possible largest sum for the array."
)


def prompt_row(text):
    """A row whose prompt is one user message."""
    return {"prompt": [{"role": "user", "content": text}]}


class DecontaminateTest(unittest.TestCase):
    """Matching, the standalone filter, and compile_data's use of it."""

    def setUp(self):
        """Fresh signature cache and a two-problem signature."""
        decontaminate.signatures.cache_clear()
        self.addCleanup(decontaminate.signatures.cache_clear)
        self.signature = decontaminate.build_signatures(
            [TEST_STATEMENT, OTHER_STATEMENT]
        )

    def test_copied_statement_overlaps(self):
        """A prompt that embeds a test statement is flagged; an unrelated one is not."""
        row = prompt_row("### Question:\n" + TEST_STATEMENT + "\n### Answer:")
        self.assertTrue(decontaminate.overlaps(row, self.signature))
        unrelated = prompt_row(
            "Count the islands in a grid of zeros and ones, where an island is a group "
            "of ones connected horizontally or vertically, and print the count."
        )
        self.assertFalse(decontaminate.overlaps(unrelated, self.signature))

    def test_signatures_hold_no_text(self):
        """Only hashes and sizes are stored: no statement words, ids or titles."""
        dumped = json.dumps(self.signature)
        self.assertNotIn("substring", dumped)
        self.assertEqual(
            set(self.signature), {"n", "min_matches", "fraction", "sizes", "windows"}
        )

    def test_boilerplate_windows_are_ignored(self):
        """Windows shared by COMMON statements (with background) do not count."""
        shared = (
            "a string a is lexicographically smaller than a string b "
            "if in the first position"
        )
        statements = [shared + " where they differ the letter is earlier", "x y z " * 5]
        background = [shared + " other text"] * decontaminate.COMMON
        signature = decontaminate.build_signatures(statements, background=background)
        self.assertFalse(decontaminate.overlaps(prompt_row(shared), signature))

    def test_filter_rows_counts_only(self):
        """filter_rows keeps the clean rows and returns how many it removed."""
        with mock.patch.object(
            decontaminate, "signatures", return_value=self.signature
        ):
            kept, removed = decontaminate.filter_rows(
                [prompt_row(TEST_STATEMENT), prompt_row("hello world")]
            )
        self.assertEqual((len(kept), removed), (1, 1))

    def test_no_signatures_keeps_everything(self):
        """Without a test set configured, nothing is removed and the count is None."""
        with mock.patch.object(decontaminate, "signatures", return_value=None):
            kept, removed = decontaminate.filter_rows([prompt_row(TEST_STATEMENT)])
        self.assertEqual((len(kept), removed), (1, None))

    def test_compile_data_drops_overlapping_rows(self):
        """compile_data writes only clean rows and says how many it removed."""
        data = rows()[:12]
        data[0]["prompt"] = [{"role": "user", "content": TEST_STATEMENT}]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            decontaminate, "signatures", return_value=self.signature
        ), mock.patch("builtins.print") as printed:
            source = Path(tmp) / "rows.jsonl"
            write_rows(source, data)
            out = compile_data(source, Path(tmp) / "out")
            submitted = (out / "submitted.jsonl").read_text().splitlines()
        self.assertEqual(len(submitted), 11)
        self.assertIn("removed 1 row(s)", printed.call_args.args[0])

    def test_compile_data_needs_ten_clean_rows(self):
        """Fewer than ten rows left after decontamination is an error."""
        data = rows()[:10]
        data[0]["prompt"] = [{"role": "user", "content": TEST_STATEMENT}]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            decontaminate, "signatures", return_value=self.signature
        ), mock.patch("builtins.print"):
            source = Path(tmp) / "rows.jsonl"
            write_rows(source, data)
            with self.assertRaisesRegex(ValueError, "after decontamination"):
                compile_data(source, Path(tmp) / "out")

    def test_short_situations_use_short_windows(self):
        """A benchmark of short situations is matched with 8-word windows."""
        situation = (
            "During a visit to her friend's house, Emily was offered a cup of tea "
            "and accepted it at once. Is what Emily did acceptable?"
        )
        other = (
            "At a dinner party, Tom passed the salt to his host with his left hand. "
            "Is what Tom did socially acceptable?"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".hidden").mkdir()
            write = lambda path, texts: path.write_text(  # noqa: E731
                "".join(json.dumps({"situation": t}) + "\n" for t in texts)
            )
            write(root / ".hidden" / "test_tasks.jsonl", [situation, other])
            write(root / "validation_tasks.jsonl", ["Ana shook hands firmly."])
            signature = decontaminate.benchmark_signatures(root)
        self.assertEqual(
            (signature["n"], signature["min_matches"]), decontaminate.SHORT
        )
        half = " ".join(situation.split()[:14])
        self.assertTrue(decontaminate.overlaps(prompt_row(half), signature))
        self.assertFalse(
            decontaminate.overlaps(
                prompt_row("Is what Emily did acceptable?"), signature
            )
        )


if __name__ == "__main__":
    unittest.main()
