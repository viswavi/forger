"""Tests for filter.by_predicate: a plain Python check as a reported filter."""

from __future__ import annotations

import re

import pytest

from pyforger import _tracking, filter
from pyforger.op import trace
from pyforger.steps import pipeline, render_step, step

ROWS = [{"id": i, "papers": ["p"] * i} for i in range(5)]


@pytest.fixture(autouse=True)
def _quiet_and_fresh(monkeypatch):
    """Silence reports and forget earlier step records."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()
    yield
    _tracking.clear_step_records()


def has_three_papers(row):
    """Need three papers."""
    return len(row["papers"]) >= 3


def test_keeps_rows_where_fn_is_truthy():
    """Truthy rows are kept, in order, as the same objects."""
    kept = filter.by_predicate(ROWS, has_three_papers)
    assert [r["id"] for r in kept] == [3, 4]
    assert kept[0] is ROWS[3]
    assert filter.by_predicate(ROWS, lambda r: r["papers"]) == ROWS[1:]
    assert filter.by_predicate([], has_three_papers) == []


def test_records_kept_scores_and_label_in_a_step():
    """The step lists the call with rows in/out and its per-row kept flags."""

    @step
    def enough(rows):
        rows = filter.by_predicate(rows, has_three_papers)
        rows = filter.by_predicate(rows, lambda r: r["id"] != 4)
        return filter.by_predicate(rows, lambda r: True, label="has ≥3 papers")

    assert [r["id"] for r in enough(ROWS)] == [3]
    rec = _tracking.step_records()["enough"]
    first, second, third = rec.ops
    assert (first.op_name, first.category) == ("filter.by_predicate", "filter")
    assert (first.n_in, first.n_out, first.yield_ratio) == (5, 2, 0.4)
    assert first.scores is not None and second.scores is not None
    assert first.scores.values == [False, False, False, True, True]
    assert first.scores.keep == "true"
    assert first.scores.label == "has_three_papers"  # fn.__name__
    assert second.scores.label == "predicate"  # a lambda
    assert third.scores is not None and third.scores.label == "has ≥3 papers"
    text = render_step(rec)
    # Chained checks: each gets its own row, and no (python) row between them.
    table = re.findall(r"filter\.by_predicate\s+(\d+)\s+(\d+)", text)
    assert table == [("5", "2"), ("2", "1"), ("1", "1")]
    assert "(python)" not in text
    line = next(ln for ln in text.splitlines() if "has_three_papers" in ln)
    assert "kept 2 of 5 (0.40) · dropped 3" in line
    assert line.rstrip().endswith("█" * 8 + "░" * 12)
    assert "has ≥3 papers · kept 1 of 1" in text


def test_is_not_cached_on_disk(tmp_path):
    """Each call runs fn again; nothing is written to the cache or the trace."""
    calls = []

    def check(row):
        calls.append(row["id"])
        return True

    @step(cache=False)
    def twice(rows):
        filter.by_predicate(rows, check)
        return filter.by_predicate(rows, check, _no_cache=True)

    twice(ROWS)
    twice(ROWS)
    assert len(calls) == 4 * len(ROWS)
    assert not (tmp_path / "filter.by_predicate").exists()
    assert not [r for r in trace() if r.op_name == "filter.by_predicate"]
    with pytest.raises(ValueError, match="never cached"):
        filter.by_predicate(ROWS, check, _cache_only=True)


def test_rows_by_keyword_and_yield_bounds():
    """Rows passed as items= count as input; a missed min_yield is a warning."""

    @step
    def strict_papers(rows):
        return filter.by_predicate(
            items=rows, fn=has_three_papers, min_yield=0.5, max_yield=0.9
        )

    strict_papers(ROWS)
    rec = _tracking.step_records()["strict_papers"]
    assert (rec.ops[0].n_in, rec.ops[0].n_out) == (5, 2)
    assert rec.warnings == ["filter.by_predicate: yield 0.40 below min_yield 0.50"]
    with pytest.raises(ValueError, match="min_yield must be in"):
        filter.by_predicate(ROWS, has_three_papers, min_yield=2)


def test_exception_names_the_row():
    """An error in fn propagates as the same type, with the row index."""
    rows = [{"papers": []}, {"title": "no papers"}]
    with pytest.raises(KeyError, match=r"has_three_papers\) failed on row 1") as info:
        filter.by_predicate(rows, has_three_papers)
    assert isinstance(info.value.__cause__, KeyError)

    class Picky(Exception):
        def __init__(self, code: int, detail: str) -> None:
            super().__init__(code, detail)

    def picky(row):
        raise Picky(3, "bad")

    with pytest.raises(Picky) as caught:
        filter.by_predicate(rows, picky, label="picky check")
    assert caught.value.__notes__ == [
        "filter.by_predicate (picky check) failed on row 0"
    ]


def test_does_not_count_as_a_quality_filter():
    """A predicate is a structural check; the pipeline still asks for a judge."""

    @step
    def first(rows):
        return filter.by_predicate(rows, has_three_papers)

    @step
    def second(rows):
        return filter.by_exact_match(rows, field="id")

    @pipeline
    def run(rows):
        return second(first(rows))

    run(ROWS)
    rec = _tracking.last_pipeline()
    assert rec is not None
    assert any("no quality filter ran" in s for s in rec.suggestions)
