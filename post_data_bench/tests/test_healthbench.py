"""Clinician-chat task: answer extraction, scoring, split and the agent's copy."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "healthbench"))

import checkers  # noqa: E402
import judge  # noqa: E402
import make_splits  # noqa: E402
from coding.rl.agent_validation import copy_validation_evaluator  # noqa: E402
from coding.split_contract import staged_build, write_manifest  # noqa: E402


def _row(i, specialty="cards", use_case="consult"):
    """Row."""
    return {
        "id": f"id{i:03d}",
        "messages": [{"role": "user", "content": f"Question {i} about a patient"}],
        "rubric_items": [
            {"criterion_text": "Recommends anticoagulation", "points": 5},
            {"criterion_text": "Suggests an unsafe dose", "points": -4},
        ],
        "physician_response": "Start apixaban.",
        "use_case": use_case,
        "type": "good_faith",
        "difficulty": "typical",
        "specialty": specialty,
    }


class AnswerTests(unittest.TestCase):
    """Answer checks."""

    def test_final_answer_drops_reasoning_and_stop_tokens(self):
        """Final answer drops reasoning and stop tokens."""
        self.assertEqual(
            checkers.final_answer("<think>plan</think>\nAnswer.<|im_end|>"), "Answer."
        )
        self.assertEqual(
            checkers.final_answer("<|channel>thought\nplan<channel|>Answer<turn|>"),
            "Answer",
        )

    def test_unclosed_reasoning_is_an_empty_answer(self):
        """Unclosed reasoning is an empty answer."""
        self.assertEqual(checkers.final_answer("<think>still thinking"), "")


class ScoreTests(unittest.TestCase):
    """Score checks."""

    items = [
        {"points": 5, "criterion_text": "a"},
        {"points": -4, "criterion_text": "b"},
    ]

    def test_negative_criteria_subtract_over_positive_total(self):
        """Negative criteria subtract over positive total."""
        self.assertEqual(judge.raw_score(self.items, [True, False]), 1.0)
        self.assertAlmostEqual(judge.raw_score(self.items, [True, True]), 0.2)
        self.assertAlmostEqual(judge.raw_score(self.items, [False, True]), -0.8)

    def test_ungraded_criteria_leave_the_score(self):
        """A criterion the content filter blocked counts neither way."""
        items = self.items + [{"points": 3, "criterion_text": "c"}]
        self.assertAlmostEqual(judge.raw_score(items, [True, None, False]), 5 / 8)
        self.assertAlmostEqual(judge.raw_score(items, [None, True, None]), -4 / 8)

    def test_length_adjustment_is_zero_at_2000_characters(self):
        """Length adjustment is zero at 2000 characters."""
        self.assertEqual(judge.length_adjusted(0.5, "x" * 2000), 0.5)
        self.assertAlmostEqual(judge.length_adjusted(0.5, "x" * 2500), 0.5 - 0.0147)

    def test_grader_prompt_shows_chat_answer_and_points(self):
        """Grader prompt shows chat answer and points."""
        prompt = judge.grader_prompt(
            [{"role": "user", "content": "Q?"}],
            "A.",
            {"points": -4, "criterion_text": "b"},
        )
        self.assertIn("user: Q?\n\nassistant: A.", prompt)
        self.assertIn("# Rubric item\n[-4] b", prompt)

    def test_parse_verdict_needs_a_boolean(self):
        """Parse verdict needs a boolean."""
        self.assertTrue(judge.parse_verdict('```json\n{"criteria_met": true}\n```'))
        self.assertIsNone(judge.parse_verdict('{"criteria_met": "yes"}'))
        self.assertIsNone(judge.parse_verdict("not json"))


class SplitTests(unittest.TestCase):
    """Split checks."""

    def test_every_specialty_keeps_a_test_row(self):
        """Every specialty keeps a row."""
        rows = [_row(i, "cards") for i in range(30)] + [
            _row(30 + i, "path") for i in range(2)
        ]
        counts = make_splits.specialty_counts(rows, 10)
        self.assertEqual(sum(counts.values()), 10)
        self.assertLessEqual(counts["path"], 1)


class AgentCopyTests(unittest.TestCase):
    """AgentCopy checks."""

    def test_copy_renames_ids_and_drops_labels(self):
        """Copy renames ids and drops labels."""
        validation, test = [_row(0), _row(1)], [_row(2)]
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / "build"
            with staged_build(build) as stage:
                checkers.write_jsonl(stage / "validation_tasks.jsonl", validation)
                checkers.write_jsonl(stage / ".hidden/test_tasks.jsonl", test)
                write_manifest(
                    stage,
                    [
                        stage / "validation_tasks.jsonl",
                        stage / ".hidden/test_tasks.jsonl",
                    ],
                    validation,
                    test,
                    seed=0,
                    inputs={},
                )
            copy = Path(tmp) / "copy"
            copy_validation_evaluator(build, copy, data_root=build, task="healthbench")
            rows = checkers.read_jsonl(copy / "validation_tasks.jsonl")
            self.assertEqual([r["id"] for r in rows], ["val_000", "val_001"])
            self.assertEqual(
                set(rows[0]), {"id", "messages", "rubric_items", "physician_response"}
            )
            self.assertTrue((copy / "judge.py").is_file())
            self.assertTrue((copy / "eval_launcher.sh").is_file())
            manifest = json.loads((copy / "validation_manifest.json").read_text())
            self.assertEqual(list(manifest["outputs"]), ["validation_tasks.jsonl"])


if __name__ == "__main__":
    unittest.main()
