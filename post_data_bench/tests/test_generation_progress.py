"""Interrupted paid generation must retain completed stage results (mocked LLM)."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GenerationProgressTests(unittest.TestCase):
    """Regression checks for generation progress."""

    def test_gold_stage_progress_is_emitted_before_later_failure(self):
        """Verify gold stage progress is emitted before later failure."""
        program = r"""
import sys
from types import SimpleNamespace
from unittest.mock import patch
import gold_generation as gold

rows = [{"question_id": "first"}, {"question_id": "second"}]
progress = []
count = 0


def sample(jobs, **kwargs):
    global count
    count += 1
    if count == 1:
        return [["pass"], ["fail"]]
    raise RuntimeError("interrupted next stage")


with patch.dict(
    sys.modules, {"llm_async": SimpleNamespace(sample_many=sample)}
), patch.object(gold, "_job_for", return_value={}), patch.object(
    gold.checkers, "row_to_problem", side_effect=lambda r: r
), patch.object(
    gold.checkers, "extract_code", side_effect=lambda r: r
), patch.object(
    gold.checkers,
    "check_lcb_solution",
    side_effect=lambda p, c, **kw: c == "pass",
):
    try:
        gold.run_gold_generation(
            rows, on_progress=lambda rows: progress.extend(rows)
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("fixture failed to interrupt")
assert progress == [
    {
        "question_id": "first",
        "gold_response": "pass",
        "verified_solvable": True,
    }
]
"""
        subprocess.run([sys.executable, "-c", program], cwd=ROOT / "coding", check=True)
