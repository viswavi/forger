"""Tests for steps with several row inputs and dict outputs (named streams)."""

from __future__ import annotations

import re
from typing import Any, TypedDict

import pytest
from pydantic import BaseModel

from pyforger import _tracking, filter, op
from pyforger._render import _plain, pipeline_renderable
from pyforger.steps import (
    StepContractError,
    pipeline,
    render_pipeline,
    render_step,
    step,
)


@pytest.fixture(autouse=True)
def _quiet_and_fresh(monkeypatch):
    """Silence reports and forget earlier step records."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()
    yield
    _tracking.clear_step_records()


def _rows(n: int, start: int = 0) -> list[dict]:
    return [{"id": i, "text": f"row number {i}"} for i in range(start, start + n)]


@op(name="alter.test_tag", category="alter")
def _tag(rows: list[Any]) -> list[Any]:
    return [{**r, "tag": "t"} for r in rows]


def _keep_all(rows: list[Any]) -> Any:
    return filter.by_length(rows, instruction_field="text", min_len=1)


def _dicts(*rows: dict) -> Any:
    """Rows as plain dicts, also where a hint names a row model (rows stay dicts)."""
    return list(rows)


def _record(name: str) -> _tracking.StepRecord:
    return _tracking.step_records()[name]


class Train(BaseModel):
    """A training row."""

    id: int
    text: str


class Refusal(BaseModel):
    """A refusal row."""

    id: int
    reason: str


class Split(TypedDict):
    """Two streams, each with its own row model."""

    train: list[Train]
    refusals: list[Refusal]


class _Judge:
    """Not rows: an object argument such as a model."""

    def cache_key(self) -> str:
        return "judge"


# --------------------------------------------------------------------------- #
# Several inputs
# --------------------------------------------------------------------------- #


def test_several_row_inputs_are_counted_and_summed():
    """Every rows argument counts; the yield uses the sum."""

    @step
    def assemble(solvable, refusals, judge, limit, label):
        return _keep_all(solvable + refusals)

    out = assemble(_rows(4), _rows(3, start=10), _Judge(), 5, "x")
    assert len(out) == 7
    rec = _record("assemble")
    assert rec.n_in == 7 and rec.n_out == 7
    assert [(p.name, p.n_rows) for p in rec.inputs] == [
        ("solvable", 4),
        ("refusals", 3),
    ]
    text = render_step(rec)
    assert "solvable 4 + refusals 3 → 7 rows (yield 1.00)" in text


def test_which_arguments_hold_rows():
    """The first list always counts; later lists only as dicts or with a rows hint."""

    @step
    def mixed(topics, names, extra, empty_hinted: list[dict], empty, judges):
        return _keep_all([{"text": t} for t in topics])

    mixed(["a", "b"], ["n1", "n2", "n3"], _rows(2), [], [], [_Judge()])
    rec = _record("mixed")
    # topics (first) 2 + extra 2 + empty_hinted 0; names, empty, judges are not rows.
    assert [(p.name, p.n_rows) for p in rec.inputs] == [
        ("topics", 2),
        ("extra", 2),
        ("empty_hinted", 0),
    ]
    assert rec.n_in == 4

    @step
    def single(rows, topics):
        return _keep_all(rows)

    single(_rows(3), ["t1", "t2"])
    rec = _record("single")
    assert rec.n_in == 3 and rec.inputs == []
    assert "3 → 3 rows (yield 1.00)" in render_step(rec)


def test_keyword_and_star_row_arguments():
    """Rows passed by keyword or through *parts count under their names."""

    @step
    def by_keyword(rows, *, refusals):
        return _keep_all(rows + refusals)

    @step
    def star(*parts):
        return _keep_all([r for p in parts for r in p])

    by_keyword(_rows(2), refusals=_rows(1, start=5))
    star(_rows(2), _rows(3, start=5))
    assert [(p.name, p.n_rows) for p in _record("by_keyword").inputs] == [
        ("rows", 2),
        ("refusals", 1),
    ]
    assert [(p.name, p.n_rows) for p in _record("star").inputs] == [
        ("parts[0]", 2),
        ("parts[1]", 3),
    ]


def test_cache_controls_do_not_hide_the_inputs():
    """_no_cache=True is the step's own control, not an argument of the function."""

    @step
    def assemble(solvable, refusals):
        return _keep_all(solvable + refusals)

    assemble(_rows(2), _rows(1, start=5), _no_cache=True)
    assert _record("assemble").n_in == 3


def test_every_annotated_input_is_validated_by_name():
    """A list[X] hint on any rows parameter validates it; the error names it."""

    @step
    def assemble(solvable: list[Train], refusals: list[Refusal]):
        return _keep_all(solvable)

    with pytest.raises(StepContractError, match="input refusals row 1 does not match"):
        assemble(_rows(2), [{"id": 1, "reason": "no"}, {"id": 2}])
    assemble(_rows(2), [{"id": 1, "reason": "no"}])
    rec = _record("assemble")
    assert rec.input_type == "solvable list[Train] + refusals list[Refusal]"


def test_pipeline_table_shows_the_input_parts():
    """The in cell holds the sum; each input gets a dim row under the step."""

    @step
    def solve(rows):
        return _tag(rows)

    @step
    def assemble(solvable, refusals):
        return _keep_all(solvable + refusals)

    @pipeline
    def run(rows):
        return assemble(solve(rows), _rows(2, start=50))

    run(_rows(4))
    rec = _tracking.last_pipeline()
    assert rec is not None
    lines = [ln.split() for ln in render_pipeline(rec).splitlines()]
    assert ["assemble", "6", "6", "1.00"] == lines[
        [ln[:1] for ln in lines].index(["assemble"])
    ][:4]
    assert ["+", "solvable", "4"] in lines
    assert ["+", "refusals", "2"] in lines


# --------------------------------------------------------------------------- #
# Dict outputs
# --------------------------------------------------------------------------- #


@step(sample=2, show=["text"])
def _split(rows):
    """Split rows into train and refusals."""
    tagged = _tag(rows)
    return {
        "train": filter.by_predicate(tagged, lambda r: r["id"] % 3),
        "refusals": [
            {"id": r["id"], "reason": "no"} for r in tagged if not r["id"] % 3
        ],
    }


def test_dict_output_reports_each_stream():
    """One count, fields line, and sample block per stream; the yield uses the sum."""
    out = _split(_rows(8))
    assert sorted(out) == ["refusals", "train"]
    rec = _record("_split")
    assert rec.n_in == 8 and rec.n_out == 8
    assert rec.streams is not None
    assert [(s.name, s.n_rows) for s in rec.streams] == [("train", 5), ("refusals", 3)]
    assert rec.sample == [] and rec.fields is None
    text = render_step(rec)
    assert "8 → train 5 · refusals 3 (yield 1.00)" in text
    assert re.search(r"train fields: id int · text str · tag str\s+\(\+ tag\)", text)
    assert re.search(
        r"refusals fields: id int · reason str\s+\(\+ reason; − text\)", text
    )
    assert "train sample rows (2 of 5)" in text
    assert "refusals sample rows (2 of 3)" in text
    # show= prints the text field in full under each train row.
    assert re.search(r"^\s+text\n\s+row number 1$", text, re.M)
    assert rec.warnings == []


def test_show_warns_only_for_fields_no_stream_has():
    """show= checks the fields of every stream together."""

    @step(show=["reason", "nope"])
    def split(rows):
        rows = _tag(rows)
        return {"train": rows[:2], "refusals": [{"reason": "x"}]}

    split(_rows(3))
    assert _record("split").suggestions == [
        "show= names fields no output row has: nope"
    ]


def test_dict_of_all_one_model_validates_every_stream():
    """dict[str, list[X]] checks each stream; the error names stream and row."""

    @step
    def split(rows) -> dict[str, list[Train]]:
        rows = _tag(rows)
        return {"train": rows[:2], "held_out": rows[2:] + [{"id": "x"}]}

    with pytest.raises(
        StepContractError, match="output stream 'held_out' row 2 does not match Train"
    ):
        split(_rows(4))

    @step
    def fine(rows) -> dict[str, list[Train]]:
        rows = _tag(rows)
        return {"train": rows[:2], "held_out": rows[2:]}

    fine(_rows(4))
    assert _record("fine").output_type == "dict[str, list[Train]]"


def test_typed_dict_validates_each_stream_with_its_model():
    """A TypedDict names the streams and each stream's row model."""

    @step
    def split(rows) -> Split:
        rows = _tag(rows)
        return {
            "train": rows[:2],
            "refusals": _dicts({"id": 1, "reason": "no"}, {"id": 2}),
        }

    with pytest.raises(
        StepContractError, match="output stream 'refusals' row 1 does not match Refusal"
    ):
        split(_rows(3))

    @step
    def fine(rows) -> Split:
        rows = _tag(rows)
        return {"train": rows, "refusals": _dicts({"id": 1, "reason": "no"})}

    fine(_rows(3))
    rec = _record("fine")
    assert rec.output_type == "train list[Train] · refusals list[Refusal]"
    assert rec.streams is not None
    assert [s.row_type for s in rec.streams] == ["list[Train]", "list[Refusal]"]
    assert "rows validated: any → train list[Train] · refusals list[Refusal]" in (
        render_step(rec)
    )


def test_typed_dict_streams_must_match_their_names():
    """A missing or unknown stream, or a plain list, breaks a TypedDict hint."""

    @step
    def missing(rows) -> Split:
        return {"train": _tag(rows)}  # type: ignore[typeddict-item]

    @step
    def unknown(rows) -> Split:
        extra = {"train": _tag(rows), "refusals": [], "extra": []}
        return extra  # type: ignore[return-value]

    @step
    def as_list(rows) -> Split:
        return _tag(rows)

    @step
    def as_dict(rows) -> list[Train]:
        return {"train": _tag(rows)}  # type: ignore[return-value]

    with pytest.raises(StepContractError, match="no stream 'refusals'"):
        missing(_rows(2))
    with pytest.raises(StepContractError, match="stream 'extra' not in Split"):
        unknown(_rows(2))
    with pytest.raises(StepContractError, match="returned a list but its return"):
        as_list(_rows(2))
    with pytest.raises(StepContractError, match="returned dict but its return"):
        as_dict(_rows(2))


def test_dict_with_non_list_values_is_a_contract_error():
    """Only a dict of row lists is a valid result; the bad result is not cached."""

    @step
    def counts(rows):
        return {"train": _keep_all(rows), "n": 3}

    @step
    def int_keys(rows):
        return {1: _keep_all(rows)}

    for _ in range(2):  # not cached: the second call fails the same way
        with pytest.raises(
            StepContractError,
            match="value 'n' is int, not a list; a step returns rows or a dict of "
            "named row lists",
        ):
            counts(_rows(2))
    with pytest.raises(StepContractError, match="key 1 is not a string"):
        int_keys(_rows(2))


def test_stream_of_non_dicts_warns():
    """A stream of strings is allowed but has no fields or samples."""

    @step
    def split(rows):
        rows = _keep_all(rows)
        return {"train": rows, "ids": [str(r["id"]) for r in rows]}

    split(_rows(2))
    rec = _record("split")
    assert rec.n_out == 4
    assert rec.suggestions == [
        "rows of ids are not dicts; schema checks and samples skipped"
    ]


def test_cached_dict_step_restores_streams_and_one_liner(capsys, monkeypatch):
    """A dict result caches like a list; the cached line shows the streams."""
    first = _split(_rows(6))
    again = _split(_rows(6))
    assert again == first
    rec = _record("_split")
    assert rec.cache_hit
    assert rec.streams is not None
    assert [(s.name, s.n_rows) for s in rec.streams] == [("train", 4), ("refusals", 2)]
    monkeypatch.delenv("FORGER_STEP_REPORT")
    _split(_rows(6))
    out = capsys.readouterr().out
    assert "step _split · 6 → train 4 · refusals 2 · cached" in out


def test_cached_line_shows_input_parts(capsys, monkeypatch):
    """A cached multi-input step names its inputs in the one-liner."""

    @step
    def assemble(solvable, refusals):
        return _keep_all(solvable + refusals)

    assemble(_rows(2), _rows(1, start=5))
    monkeypatch.delenv("FORGER_STEP_REPORT")
    assemble(_rows(2), _rows(1, start=5))
    out = capsys.readouterr().out
    assert "step assemble · solvable 2 + refusals 1 → 3 rows · cached" in out


# --------------------------------------------------------------------------- #
# Streams downstream
# --------------------------------------------------------------------------- #


@step
def _check(rows):
    """Keep all rows."""
    return _keep_all(rows)


def _gaps() -> list[tuple[str | None, str | None, str | None, int | None, int, int]]:
    rec = _tracking.last_pipeline()
    assert rec is not None
    return [
        (g.after, g.stream, g.before, g.position, g.n_in, g.n_out)
        for g in rec.python_gaps
    ]


def test_downstream_step_matches_its_stream():
    """Rows taken from out["train"] compare with train, not with the total."""

    @pipeline
    def whole(rows):
        out = _split(rows)
        return {"train": _check(out["train"]), "refusals": out["refusals"]}

    @pipeline
    def dropped(rows):
        out = _split(rows)
        return _check(out["train"][:3])

    whole(_rows(8))
    assert _gaps() == []
    dropped(_rows(8))
    assert _gaps() == [("_split", "train", "_check", 1, 5, 3)]
    rec = _tracking.last_pipeline()
    assert rec is not None
    lines = [ln.split() for ln in render_pipeline(rec).splitlines()]
    assert ["(python)", "train", "5", "3", "0.60"] in lines
    # The streams of _split show under it, in the out column.
    assert ["→", "train", "5"] in lines and ["→", "refusals", "3"] in lines


def test_stream_rows_taken_later_are_not_a_gap():
    """Rows of a stream that a later step or the return value takes are not drops."""

    @pipeline
    def later(rows):
        out = _split(rows)
        first = _check(out["train"][:2])
        return first + _check(out["train"][2:])

    later(_rows(8))
    assert _gaps() == []


def test_several_inputs_from_one_stream_join():
    """Two inputs cut from one stream are one stream; a merge is skipped."""

    @step
    def assemble(a, b):
        return _keep_all(a + b)

    @pipeline
    def cut(rows):
        out = _split(rows)
        return assemble(out["train"][:2], out["train"][2:4])

    @pipeline
    def merged(rows):
        out = _split(rows)
        return assemble(out["train"][:2], out["train"][2:4] + _rows(1, start=99))

    cut(_rows(8))
    assert _gaps() == [("_split", "train", "assemble", 1, 5, 4)]
    merged(_rows(8))
    assert _gaps() == []


# --------------------------------------------------------------------------- #
# What a pipeline returns
# --------------------------------------------------------------------------- #


def _summary_line(rec) -> str:
    return render_pipeline(rec).splitlines()[1].strip()


def test_pipeline_may_return_a_tuple_dict_none_or_list():
    """The final row count is that of a list or a dict of lists; else left out."""

    @pipeline
    def as_tuple(rows):
        out = _split(rows)
        return out["train"], out["refusals"]

    @pipeline
    def as_none(rows):
        _check(_split(rows)["train"])

    @pipeline
    def as_dict(rows):
        out = _split(rows)
        return {"train": _check(out["train"]), "refusals": out["refusals"]}

    @pipeline
    def as_list(rows):
        rows = _check(rows)
        return rows[:2]  # the returned list counts, not the last step's output

    assert isinstance(as_tuple(_rows(8)), tuple)
    rec = _tracking.last_pipeline()
    assert rec is not None and rec.returned == "other" and rec.n_out is None
    assert _summary_line(rec).startswith("1 step · 8 rows in · ")

    assert as_none(_rows(8)) is None
    rec = _tracking.last_pipeline()
    assert rec is not None and rec.returned == "other"
    assert "→" not in _summary_line(rec)

    as_dict(_rows(8))
    rec = _tracking.last_pipeline()
    assert rec is not None and rec.returned == "streams" and rec.n_out == 8
    assert "2 steps · 8 → train 5 · refusals 3 (yield 1.00) · " in _summary_line(rec)

    as_list(_rows(8))
    rec = _tracking.last_pipeline()
    assert rec is not None and rec.returned == "rows" and rec.n_out == 2
    assert _summary_line(rec).startswith("1 step · 8 → 2 rows (yield 0.25) · ")


def test_narrow_pipeline_view_shows_stream_rows():
    """The one-line-per-row view prints a stream's count without a dangling arrow."""

    @pipeline
    def run(rows):
        out = _split(rows)
        return _check(out["train"])

    run(_rows(8))
    rec = _tracking.last_pipeline()
    assert rec is not None

    lines = [
        ln.strip() for ln in _plain(pipeline_renderable(rec, width=36)).splitlines()
    ]
    assert "→ train  5" in lines and "→ refusals  3" in lines
    assert not any(ln.endswith("→") for ln in lines)
