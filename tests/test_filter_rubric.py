"""Tests for filter.by_rubric_score: keep rows by a grade function's score."""

from __future__ import annotations

import logging

import pytest

from pyforger import _tracking, filter
from pyforger.inspect import rubric_scores
from pyforger.steps import step


@pytest.fixture(autouse=True)
def _quiet_and_fresh(monkeypatch):
    """Silence reports and forget earlier step records."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()
    yield
    _tracking.clear_step_records()


def _rows(n: int = 10) -> list[dict]:
    """Row i's response is "i"; the reference answer is 9."""
    return [{"id": i, "response": str(i), "answer": "9"} for i in range(n)]


class Counting:
    """A grade function that counts its calls; keyed by cache_key."""

    def __init__(self, fn, key="counting"):
        """Wrap ``fn``; ``key`` is the cache key."""
        self.fn, self.key, self.calls = fn, key, 0

    def cache_key(self):
        """The key Forger's op cache uses."""
        return ("Counting", self.key)

    def __call__(self, response, row):
        """Count the call and grade."""
        self.calls += 1
        return self.fn(response, row)


def closeness(response: str, row: dict) -> float:
    """Partial credit: 0.0 for "0" up to 1.0 for the answer, "9"."""
    return int(response) / int(row["answer"])


def test_keeps_every_row_in_range_not_a_sample():
    """All 300 rows are graded; bounds are inclusive and the score is stored."""
    rows = [{"id": i, "response": str(i % 10), "answer": "9"} for i in range(300)]
    kept = filter.by_rubric_score(
        rows, closeness, min_score=5 / 9, max_score=8 / 9, score_field="score"
    )
    assert len(kept) == 120  # responses 5..8 in each of 30 blocks of 10
    assert {r["response"] for r in kept} == {"5", "6", "7", "8"}
    assert kept[0] == {"id": 5, "response": "5", "answer": "9", "score": 5 / 9}
    assert "score" not in rows[5]  # input rows are not modified


def test_no_bounds_keeps_every_gradable_row_and_unchanged_rows():
    """Without bounds or score_field the rows come back as they were."""
    rows = _rows()
    kept = filter.by_rubric_score(rows, closeness)
    assert kept == rows and kept[3] is rows[3]
    assert filter.by_rubric_score([], closeness) == []


def test_ungradable_rows_are_dropped_unless_kept():
    """None, a None response, and a raised grade are ungradable."""
    rows = _rows(6)
    rows[1]["response"] = None

    def grade(response, row):
        """None for "2"; raises for "3"."""
        if response == "2":
            return None
        if response == "3":
            raise RuntimeError("sandbox did not start")
        return closeness(response, row)

    kept = filter.by_rubric_score(rows, grade, score_field="s")
    assert [r["id"] for r in kept] == [0, 4, 5]
    kept = filter.by_rubric_score(rows, grade, score_field="s", keep_ungradable=True)
    assert [r["s"] for r in kept] == [0.0, None, None, None, 4 / 9, 5 / 9]


def test_a_raised_grade_warns_and_is_not_cached(caplog):
    """The warning names the count and message; a rerun grades again."""

    def no_tests_for_3(response, row):
        """1.0, but raises for "3"."""
        if response == "3":
            raise KeyError("tests")
        return 1.0

    grade = Counting(no_tests_for_3)
    with caplog.at_level(logging.WARNING):
        kept = filter.by_rubric_score(_rows(), grade, min_score=1.0)
    assert len(kept) == 9
    assert "grade raised on 1 of 10 rows, e.g. KeyError: 'tests'" in caplog.text
    calls = grade.calls
    filter.by_rubric_score(_rows(), grade, min_score=1.0)
    assert grade.calls == calls * 2


def test_grades_are_cached_and_shared_with_rubric_scores():
    """A rerun with other bounds grades nothing; rubric_scores reuses the grades."""
    rows = _rows()
    grade = Counting(closeness)
    filter.by_rubric_score(rows, grade, min_score=0.5)
    assert grade.calls == 10
    filter.by_rubric_score(rows, grade, min_score=0.9, score_field="s")
    assert grade.calls == 10
    rubric_scores(rows, grade, n=None, control_columns=False)
    assert grade.calls == 10


def test_records_scores_and_ungradable_note_in_a_step():
    """The step report gets every row's score, the bounds, and a note."""
    rows = _rows()
    rows[0]["response"] = None

    @step
    def graded(rows):
        return filter.by_rubric_score(rows, closeness, min_score=0.5)

    assert len(graded(rows)) == 5
    op = _tracking.step_records()["graded"].ops[0]
    assert (op.op_name, op.category, op.n_in, op.n_out) == (
        "filter.by_rubric_score",
        "filter",
        10,
        5,
    )
    assert op.scores is not None and op.scores.label == "rubric score"
    assert op.scores.values[:3] == [None, 1 / 9, 2 / 9]
    assert (op.scores.keep, op.scores.lower, op.scores.upper) == ("between", 0.5, None)
    assert op.scores.note == "1 of 10 rows ungradable (1 no response): dropped"


def test_bad_arguments():
    """Bounds must be ordered and finite; the response field must exist."""
    with pytest.raises(ValueError, match="must not exceed"):
        filter.by_rubric_score(_rows(), closeness, min_score=0.9, max_score=0.1)
    with pytest.raises(ValueError, match="finite"):
        filter.by_rubric_score(_rows(), closeness, min_score=float("nan"))
    with pytest.raises(
        KeyError, match="filter.by_rubric_score: item is missing required field 'sol'"
    ):
        filter.by_rubric_score(_rows(), closeness, response_field="sol")
