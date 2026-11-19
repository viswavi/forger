"""Tests for (python) rows: rows that plain Python changed between ops and steps."""

from __future__ import annotations

import gc
import re
import weakref
from unittest.mock import patch

import pytest

from pyforger import _rowflow, _tracking, filter, op
from pyforger._tracking import PythonGap
from pyforger.inspect import inspect_step
from pyforger.steps import pipeline, render_pipeline, render_step, step


@pytest.fixture(autouse=True)
def _quiet_and_fresh(monkeypatch):
    """Silence reports and forget earlier step records."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()
    yield
    _tracking.clear_step_records()


def _rows(n: int) -> list[dict]:
    return [{"id": i, "text": f"row number {i}"} for i in range(n)]


@op(name="alter.test_tag", category="alter")
def _tag(rows: list[dict]) -> list[dict]:
    """New dicts, like an LLM op's output."""
    return [{**r, "tag": "t"} for r in rows]


@op(name="acquire.test_sample", category="acquire")
def _sample(prompt: str, *, n: int) -> list[dict]:
    return [{"id": i, "text": f"{prompt} {i}"} for i in range(n)]


@op(name="alter.test_kw_only", category="alter")
def _kw_only(*, rows: list[dict]) -> list[dict]:
    return list(rows)


def _keep_all(rows: list[dict]) -> list[dict]:
    return filter.by_length(rows, instruction_field="text", min_len=1)


def _gaps(name: str) -> list[tuple[str | None, str | None, int, int, bool]]:
    rec = _tracking.step_records()[name]
    return [(g.after, g.before, g.n_in, g.n_out, g.inferred) for g in rec.python_gaps]


# --------------------------------------------------------------------------- #
# Inside a step
# --------------------------------------------------------------------------- #


def test_identity_gap_between_ops_is_certain():
    """The next op's rows are some of the previous op's rows: a certain drop."""

    @step
    def tagged(rows):
        rows = _tag(rows)
        rows = [r for r in rows if r["id"] % 4]  # 8 → 6
        return _keep_all(rows)

    assert len(tagged(_rows(8))) == 6
    assert _gaps("tagged") == [("alter.test_tag", "filter.by_length", 8, 6, False)]
    text = render_step(_tracking.step_records()["tagged"])
    lines = [ln.split() for ln in text.splitlines() if ln.strip()]
    names = [ln[0] for ln in lines if ln[0] in ("alter.test_tag", "(python)")]
    assert names == ["alter.test_tag", "(python)"]
    assert ["(python)", "8", "6", "0.75"] in lines
    assert "inferred" not in text


def test_rebuilt_rows_fall_back_to_counts_and_are_marked_inferred():
    """Rows rebuilt with {**r} lose identity; counts still show the drop."""

    @step
    def rebuilt(rows):
        rows = _tag(rows)
        rows = [{**r, "n": 1} for r in rows if r["id"] % 4]
        return _keep_all(rows)

    rebuilt(_rows(8))
    assert _gaps("rebuilt") == [("alter.test_tag", "filter.by_length", 8, 6, True)]
    text = render_step(_tracking.step_records()["rebuilt"])
    assert re.search(r"\(python\)\s+8\s+6\s+0\.75\s+inferred", text)


def test_rows_added_in_python_show_too():
    """Python can add rows as well: rebuilt rows plus one new row."""

    @step
    def grown(rows):
        rows = _tag(rows)
        return _keep_all([{**r} for r in rows] + [{"id": 99, "text": "extra row"}])

    grown(_rows(3))
    assert _gaps("grown") == [("alter.test_tag", "filter.by_length", 3, 4, True)]


def test_step_input_to_first_op_and_last_op_to_return():
    """Drops before the first op and after the last op are (python) rows too."""

    @step
    def edges(rows):
        rows = _keep_all(rows[1:])  # 6 → 5 before the op
        return rows[:3]  # 5 → 3 after it

    edges(_rows(6))
    assert _gaps("edges") == [
        (None, "filter.by_length", 6, 5, False),
        ("filter.by_length", None, 5, 3, False),
    ]
    text = render_step(_tracking.step_records()["edges"])
    table = [ln.split()[0] for ln in text.splitlines() if ln.startswith("  ")]
    i = table.index("filter.by_length")
    assert table[i - 1] == "(python)" and table[i + 1] == "(python)"


def test_single_op_step_shows_its_table_when_python_dropped_rows():
    """A one-op step normally has no table; a (python) row brings it back."""

    @step
    def one(rows):
        return _tag(rows)[:2]

    one(_rows(4))
    text = render_step(_tracking.step_records()["one"])
    assert re.search(r"\(python\)\s+4\s+2\s+0\.50", text)


def test_chunked_contiguous_calls_have_no_false_gap():
    """Gen ×3 over chunks of the input, rows merged, one filter: no gap."""

    @step
    def chunked(rows):
        out = []
        for start in range(0, len(rows), 3):
            chunk = rows[start : start + 3]
            out += [{**r, **d} for r, d in zip(chunk, _tag(chunk))]
        return _keep_all(out)

    chunked(_rows(8))
    assert _gaps("chunked") == []

    @step
    def sampled(rows):
        out = []
        for n in (4, 64, 64, 64, 64):
            out += _sample("topic", n=n)
        return _keep_all(out)

    assert len(sampled(_rows(1))) == 260
    assert _gaps("sampled") == []


def test_chunked_calls_with_a_python_drop_show_one_gap():
    """Chunks of the input join into one stream; the drop after them shows."""

    @step
    def chunked(rows):
        out = []
        for start in range(0, len(rows), 3):
            out += _tag(rows[start : start + 3])
        return _keep_all([r for r in out if r["id"] != 0])

    chunked(_rows(8))
    assert _gaps("chunked") == [("alter.test_tag", "filter.by_length", 8, 7, False)]


def test_interleaved_chunk_calls_are_silent_on_counts():
    """Gen, filter, gen, filter...: rebuilt rows give no count-based gap."""

    @step
    def interleaved(rows):
        out = []
        for start in range(0, len(rows), 2):
            drafts = _tag(rows[start : start + 2])
            merged = [{**d, "n": 1} for d in drafts if d["id"] != 1]
            out += _keep_all(merged)
        return out

    interleaved(_rows(6))
    assert _gaps("interleaved") == []


def test_interleaved_chunk_calls_keep_identity_gaps():
    """Interleaved, but the filters take the generator's own rows: certain."""

    @step
    def interleaved(rows):
        out = []
        for start in range(0, len(rows), 2):
            drafts = _tag(rows[start : start + 2])
            out += _keep_all([d for d in drafts if d["id"] != 1])
        return out

    interleaved(_rows(6))
    assert _gaps("interleaved") == [("alter.test_tag", "filter.by_length", 6, 5, False)]


def test_chained_calls_of_one_op_have_no_false_gap():
    """by_predicate(by_predicate(rows)) is a chain, not two chunks."""

    @step
    def chained(rows):
        rows = filter.by_predicate(rows, lambda r: r["id"] > 0)
        return filter.by_predicate(rows, lambda r: r["id"] > 1)

    @step
    def chained_rebuilt(rows):
        rows = filter.by_predicate(rows, lambda r: r["id"] > 0)
        rows = [{**r} for r in rows]  # chunks or a chain? unknown: silent
        return filter.by_predicate(rows, lambda r: r["id"] > 1)

    chained(_rows(5))
    chained_rebuilt(_rows(5))
    assert _gaps("chained") == []
    assert _gaps("chained_rebuilt") == []


def test_merges_and_splits_are_silent():
    """Rows from two sources, or one op's rows split between two ops: no gap."""

    @step
    def merged(rows):
        new = _sample("new", n=4)
        return rows + _keep_all(new[:3])

    @step
    def split(rows):
        rows = _tag(rows)
        first = _keep_all(rows[:2])
        second = filter.by_predicate(rows[2:], lambda r: True)
        return first + second

    merged(_rows(5))
    split(_rows(5))
    # merged: sample → filter is a real 4 → 3 drop; the return mixes sources.
    assert _gaps("merged") == [("acquire.test_sample", "filter.by_length", 4, 3, False)]
    assert _gaps("split") == []


def test_several_list_arguments_and_non_list_inputs_are_silent():
    """A step with two lists has no single input; a keyword-only op has none."""

    @step
    def two_lists(rows, extra):
        return _keep_all(rows[:2])

    @step
    def keyword_rows(rows):
        rows = _kw_only(rows=rows[:2])
        return _keep_all(rows)

    two_lists(_rows(4), _rows(2))
    keyword_rows(_rows(4))
    assert _gaps("two_lists") == []
    assert _gaps("keyword_rows") == []


def test_string_rows_compare_only_by_count():
    """Strings share ids, so only counts compare (and only when rebuilt)."""

    @step
    def from_strings(topics):
        rows = [{"text": t} for t in topics if t != "skip"]
        return _keep_all(rows)

    from_strings(["alpha", "beta", "skip"])
    assert _gaps("from_strings") == [(None, "filter.by_length", 3, 2, True)]


def test_inspect_ops_are_skipped():
    """An inspect op returns a report, not rows; it does not break the stream."""

    @op(name="inspect.test_topics", category="inspect")
    def topics(rows):
        return ["topic"] * len(rows)

    @step
    def looked(rows):
        rows = _tag(rows)
        topics(rows)
        return _keep_all(rows[:2])

    looked(_rows(4))
    assert _gaps("looked") == [("alter.test_tag", "filter.by_length", 4, 2, False)]


def test_gaps_are_informational_even_under_strict():
    """(python) rows add no warnings and never raise."""

    @step(strict=True)
    def strict_step(rows):
        return _keep_all(rows[:2])

    strict_step(_rows(4))
    rec = _tracking.step_records()["strict_step"]
    assert rec.python_gaps and rec.warnings == []


def test_inspect_step_and_cached_steps_keep_gaps(capsys, monkeypatch):
    """A cache hit restores the gaps of the run that cached it."""

    @step
    def kept(rows):
        rows = _tag(rows)
        return _keep_all([{**r} for r in rows if r["id"]])

    kept(_rows(5))
    first = _tracking.step_records()["kept"].python_gaps
    kept(_rows(5))
    rec = _tracking.step_records()["kept"]
    assert rec.cache_hit and rec.python_gaps == first
    assert first == [
        PythonGap(
            after="alter.test_tag",
            before="filter.by_length",
            after_call=0,
            before_call=1,
            n_in=5,
            n_out=4,
            inferred=True,
        )
    ]
    monkeypatch.delenv("FORGER_STEP_REPORT")
    inspect_step("kept")
    out = capsys.readouterr().out
    assert re.search(r"\(python\)\s+5\s+4\s+0\.80\s+inferred", out)


def test_too_many_rows_turn_tracking_off():
    """Past the held-row cap a step stays silent rather than pin every row."""

    @step
    def big(rows):
        return _keep_all(_tag(rows)[:2])

    with patch.object(_rowflow, "MAX_HELD_ROWS", 10):
        big(_rows(8))
    assert _gaps("big") == []


def test_rows_are_not_held_after_the_step():
    """The step lets go of the rows its ops saw once it returns."""

    class Row:
        def cache_key(self) -> str:
            return "Row"

    @op(name="alter.test_objects", category="alter", cache=False)
    def objects(rows):
        return [Row() for _ in rows]

    @step(cache=False)
    def make(rows):
        out = objects(rows)
        return out[:1]

    produced = make([Row(), Row()])
    assert len(_tracking.step_records()["make"].python_gaps) == 1
    ref = weakref.ref(produced[0])
    del produced
    gc.collect()
    assert ref() is None


# --------------------------------------------------------------------------- #
# Between steps
# --------------------------------------------------------------------------- #


@step
def _solve(rows):
    """Tag rows."""
    return _tag(rows)


@step
def _check(rows):
    """Keep all rows."""
    return _keep_all(rows)


def test_pipeline_gap_by_identity():
    """Step B takes some of step A's output rows: a (python) row between them."""

    @pipeline
    def run(rows):
        solved = _solve(rows)
        return _check([r for r in solved if r["id"] % 3])

    run(_rows(9))
    rec = _tracking.last_pipeline()
    assert rec is not None
    assert [
        (g.after, g.before, g.position, g.n_in, g.n_out) for g in rec.python_gaps
    ] == [("_solve", "_check", 1, 9, 6)]
    assert rec.warnings == [w for w in rec.warnings if "python" not in w]
    lines = [ln.split() for ln in render_pipeline(rec).splitlines()]
    names = [ln[0] for ln in lines if ln and ln[0] in ("_solve", "(python)", "_check")]
    assert names == ["_solve", "(python)", "_check"]
    assert ["(python)", "9", "6", "0.67"] in lines


def test_no_pipeline_gap_for_unrelated_rebuilt_or_split_rows():
    """Counts never guess across steps; split and returned rows are not drops."""

    @pipeline
    def rebuilt(rows):
        solved = _solve(rows)
        return _check([{**r} for r in solved if r["id"] % 3])

    @pipeline
    def unrelated(rows):
        _solve(rows)
        return _check(_rows(3))

    @pipeline
    def split(rows):
        solved = _solve(rows)
        return _check(solved[:4]) + _check(solved[4:])

    @pipeline
    def returned(rows):
        solved = _solve(rows)
        _check(solved[:4])
        return solved

    for run in (rebuilt, unrelated, split, returned):
        run(_rows(9))
        rec = _tracking.last_pipeline()
        assert rec is not None and rec.python_gaps == [], run.__name__


def test_pipeline_gap_with_cached_steps():
    """Cached steps return new row objects; identity still holds within a run."""

    @pipeline
    def run(rows):
        solved = _solve(rows)
        return _check(solved[:5])

    run(_rows(9))
    run(_rows(9))
    rec = _tracking.last_pipeline()
    assert rec is not None and all(s.cache_hit for s in rec.steps)
    assert [(g.position, g.n_in, g.n_out) for g in rec.python_gaps] == [(1, 9, 5)]


def test_bare_ops_between_steps_take_part():
    """An op called in the pipeline outside a step is a point of the stream."""

    @pipeline
    def run(rows):
        solved = _solve(rows)
        kept = filter.by_predicate(solved[:6], lambda r: r["id"] % 2)
        return _check(kept)

    run(_rows(8))
    rec = _tracking.last_pipeline()
    assert rec is not None
    assert [s.name for s in rec.steps] == [
        "_solve",
        "(bare) filter.by_predicate",
        "_check",
    ]
    assert [(g.position, g.n_in, g.n_out) for g in rec.python_gaps] == [(1, 8, 6)]


# --------------------------------------------------------------------------- #
# Review regressions
# --------------------------------------------------------------------------- #


def test_one_op_before_and_after_another_has_no_false_gap():
    """A, B, A with no Python between them: three rows, no (python) row (#13)."""

    @step
    def aba(rows):
        rows = filter.by_length(rows, instruction_field="text", min_len=3)
        rows = filter.by_predicate(rows, lambda r: r["id"] != 9)
        return filter.by_length(rows, instruction_field="text", min_len=3, max_len=3)

    rows = _rows(10)
    for r in rows[:2]:
        r["text"] = "short"
    rows[2]["text"] = "a much longer text"
    assert len(aba(rows)) == 6  # 10 → 8 → 7 → 6
    assert _gaps("aba") == []
    text = render_step(_tracking.step_records()["aba"])
    names = [ln.split()[0] for ln in text.splitlines() if ln.startswith("  filter.")]
    assert names[:3] == ["filter.by_length", "filter.by_predicate", "filter.by_length"]
    assert "(python)" not in text


def test_gaps_around_a_repeated_op_sit_above_the_call_that_took_the_rows():
    """A, drop, B, drop, A: each (python) row sits above its own report row."""

    @step
    def aba(rows):
        rows = filter.by_predicate(rows, lambda r: True)
        rows = _keep_all(rows[1:])
        return filter.by_predicate(rows[1:], lambda r: True)

    aba(_rows(6))
    rec = _tracking.step_records()["aba"]
    gaps = [(g.after_call, g.before_call, g.n_in, g.n_out) for g in rec.python_gaps]
    assert gaps == [(0, 1, 6, 5), (1, 2, 5, 4)]
    lines = render_step(rec).splitlines()
    names = [ln.split()[0] for ln in lines if ln.startswith("  ") and ln.strip()]
    i = names.index("filter.by_length")
    assert names[i - 2 : i + 3] == [
        "filter.by_predicate",
        "(python)",
        "filter.by_length",
        "(python)",
        "filter.by_predicate",
    ]


def test_rebuilt_rows_split_between_two_ops_are_silent():
    """Easy rows to one op, hard to another, both rebuilt: no inferred row (#27)."""

    @step
    def split_rebuilt(rows):
        rows = _tag(rows)
        easy = [{**r} for r in rows if r["id"] < 4]
        hard = [{**r} for r in rows if r["id"] >= 4]
        return _keep_all(easy) + filter.by_predicate(hard, lambda r: True)

    @step
    def split_to_return(rows):
        rows = _tag(rows)
        easy = [{**r} for r in rows if r["id"] < 4]
        return _keep_all(easy) + [{**r} for r in rows if r["id"] >= 4]

    assert len(split_rebuilt(_rows(10))) == 10
    assert len(split_to_return(_rows(10))) == 10
    assert _gaps("split_rebuilt") == []
    assert _gaps("split_to_return") == []


def test_rebuilt_rows_passed_on_still_show_an_earlier_gap():
    """A later rebuild of the same stream does not hide an earlier drop."""

    @step
    def rebuilt_twice(rows):
        rows = _tag(rows)
        rows = _keep_all(rows[2:])  # 8 → 6, same dicts
        rows = [{**r, "n": 1} for r in rows]
        return filter.by_predicate(rows, lambda r: True)

    rebuilt_twice(_rows(8))
    assert _gaps("rebuilt_twice") == [
        ("alter.test_tag", "filter.by_length", 8, 6, False)
    ]


def test_pipeline_returning_a_list_of_row_lists_is_not_a_drop():
    """``return [kept, rest]``: the rest was returned, not dropped (#28)."""

    @pipeline
    def kept_and_rest(rows):
        made = _solve(rows)
        return [_check(made[:8]), made[8:]]

    @pipeline
    def nested(rows):
        made = _solve(rows)
        return {"splits": [_check(made[:8]), made[8:]]}

    for run in (kept_and_rest, nested):
        run(_rows(10))
        rec = _tracking.last_pipeline()
        assert rec is not None and rec.python_gaps == [], run.__name__
