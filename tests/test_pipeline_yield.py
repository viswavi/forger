"""The pipeline summary warns when few of the first step's rows reach the output."""

from __future__ import annotations

from pyforger import _tracking, filter, pipeline, step


@step
def make(n):
    """``n`` numbered rows."""
    return filter.by_predicate([{"i": i} for i in range(n)], lambda r: True)


@step
def keep(rows, k):
    """The first ``k`` rows."""
    return filter.by_predicate(rows, lambda r: r["i"] < k)


def _run(k, **flags):
    @pipeline(**flags)
    def gen(n):
        return keep(make(n), k)

    gen(100)
    rec = _tracking.last_pipeline()
    assert rec is not None
    return rec


def test_under_ten_percent_is_a_warning_naming_the_worst_step():
    """5 of 100 rows kept: a low_yield warning that names keep."""
    rec = _run(5)
    assert (
        "5 of 100 rows (5.0%) from step make reached the output; keep dropped the "
        "most (100 -> 5)"
    ) in rec.warnings
    assert rec.kinds[rec.warnings[-1]] == "low_yield"


def test_ten_to_thirty_percent_is_a_note():
    """20 of 100 rows kept: a note (notes never make strict=True raise)."""
    rec = _run(20)
    assert not any("reached the output" in w for w in rec.warnings)
    assert any("20 of 100 rows (20.0%)" in n for n in rec.notes)


def test_thirty_percent_or_more_says_nothing():
    """50 of 100 rows kept: no yield line."""
    rec = _run(50)
    assert not any("reached the output" in m for m in rec.warnings + rec.notes)


def test_low_yield_can_be_suppressed_with_a_reason():
    """suppress_low_yield_warning moves the warning out of the list."""
    rec = _run(5, suppress_low_yield_warning="only the hardest five percent are kept")
    assert not any("reached the output" in w for w in rec.warnings)


def test_limited_run_projects_rows_per_hundred():
    """A limited run says what every 100 rows from the first step give."""
    rec = _run(5, limit=50)
    assert any(
        "every 100 rows from make give about" in m for m in rec.warnings + rec.notes
    )
