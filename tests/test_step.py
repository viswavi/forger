"""Tests for @step / @pipeline reports, yield checks, and filter score histograms."""

from __future__ import annotations

import contextvars
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from rich.console import Console

import pyforger
from pyforger import _tracking, acquire, alter, filter, op
from pyforger._render import _histogram, fmt_num, pipeline_renderable, step_renderable
from pyforger._tracking import PipelineRecord, ScoreRecord, StepRecord
from pyforger.acquire import Model
from pyforger.alter import dynamic
from pyforger.inspect import inspect_step
from pyforger.op import trace
from pyforger.steps import (
    PipelineContractError,
    StepContractError,
    pipeline,
    render_pipeline,
    render_scores,
    render_step,
    step,
)


@pytest.fixture(autouse=True)
def _quiet_and_fresh(monkeypatch):
    """Silence reports unless a test opts in, and forget earlier step records."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()
    yield
    _tracking.clear_step_records()


def _rows(*texts: str) -> list[dict]:
    return [{"text": t} for t in texts]


@op(name="alter.test_upper", category="alter")
def _upper(rows: list[dict]) -> list[dict]:
    return [{**r, "text": r["text"].upper()} for r in rows]


@op(name="alter.test_costly", category="alter")
def _costly(rows: list[dict]) -> list[dict]:
    _tracking.add_usage(100, 50, 0.25)
    return list(rows)


@op(name="alter.test_strings", category="alter")
def _strings(rows: list[dict]) -> list[str]:
    return [r["text"] for r in rows]


@op(name="inspect.test_report", category="inspect")
def _report(rows: list[dict]) -> list[str]:
    return ["topic" for _ in rows]


class _StableJudge:
    """Judge that scores by response length and has a stable cache key."""

    def __init__(self) -> None:
        self.call_count = 0

    def cache_key(self) -> tuple:
        return ("_StableJudge",)

    def generate(self, prompt, *, n=1, temperature=None, max_tokens=None):
        self.call_count += 1
        output = prompt.split("Output: ")[1].split("\n")[0]
        if output == "garbled":
            return ["not json"] * n
        return [json.dumps({"score": min(5, len(output))})] * n


def _length_filter(rows, **kwargs):
    return filter.by_length(rows, instruction_field="text", min_len=2, **kwargs)


TEXTS = ("one two", "three four five", "six", "seven eight", "nine")


# --------------------------------------------------------------------------- #
# @step records
# --------------------------------------------------------------------------- #


def test_step_records_ops_counts_and_categories():
    """A step records each top-level op with rows in/out, yield, and category."""

    @step
    def clean(rows):
        rows = _upper(rows)
        return _length_filter(rows)

    out = clean(_rows(*TEXTS))
    assert [r["text"] for r in out] == ["ONE TWO", "THREE FOUR FIVE", "SEVEN EIGHT"]
    rec = _tracking.step_records()["clean"]
    assert (rec.n_in, rec.n_out, rec.cache_hit) == (5, 3, False)
    assert [c.op_name for c in rec.ops] == ["alter.test_upper", "filter.by_length"]
    assert [c.category for c in rec.ops] == ["alter", "filter"]
    upper, length = rec.ops
    assert (upper.n_in, upper.n_out, upper.yield_ratio) == (5, 5, 1.0)
    assert (length.n_in, length.n_out) == (5, 3)
    assert length.yield_ratio == pytest.approx(0.6)
    assert length.scores is not None
    assert length.scores.values == [2, 3, 1, 2, 1]
    assert (length.scores.keep, length.scores.lower) == ("between", 2)
    assert rec.warnings == []
    assert rec.sample[0] == {"text": "ONE TWO"}


def test_usage_attributed_to_op_and_step():
    """Usage added inside an op lands on that op's call and on the step totals."""

    @step
    def costly(rows):
        return _length_filter(_costly(rows))

    costly(_rows(*TEXTS))
    rec = _tracking.step_records()["costly"]
    call = rec.ops[0]
    assert (call.prompt_tokens, call.completion_tokens) == (100, 50)
    assert call.cost_usd == pytest.approx(0.25)
    assert rec.ops[1].prompt_tokens == 0
    assert (rec.prompt_tokens, rec.completion_tokens) == (100, 50)
    assert rec.cost_usd == pytest.approx(0.25)


def test_nan_cost_counts_as_zero_and_is_never_printed(capsys, monkeypatch):
    """A NaN cost adds nothing and no report prints NaN."""
    monkeypatch.delenv("FORGER_STEP_REPORT")

    @op(name="alter.test_nan_cost", category="alter")
    def nan_cost(rows):
        _tracking.add_usage(None, float("nan"), float("nan"))
        return list(rows)

    @step
    def nan_step(rows):
        return _length_filter(nan_cost(rows))

    nan_step(_rows(*TEXTS))
    rec = _tracking.step_records()["nan_step"]
    assert rec.cost_usd == 0.0 and rec.prompt_tokens == 0
    assert not re.search(r"\bnan\b", capsys.readouterr().out, re.IGNORECASE)


def test_min_yield_violation_is_a_printed_warning(capsys, monkeypatch):
    """A filter below its min_yield adds a warning that the report prints."""
    monkeypatch.delenv("FORGER_STEP_REPORT")

    @step
    def picky(rows):
        return _length_filter(rows, min_yield=0.9)

    assert len(picky(_rows(*TEXTS))) == 3
    rec = _tracking.step_records()["picky"]
    expected = "filter.by_length: yield 0.60 below min_yield 0.90"
    assert rec.warnings == [expected]
    out = capsys.readouterr().out
    assert expected in out
    assert "LOW · min 0.90" in out


def test_min_yield_violation_raises_when_strict(capsys, monkeypatch):
    """strict=True raises after printing and storing the report, on every run."""
    monkeypatch.delenv("FORGER_STEP_REPORT")

    @step(strict=True)
    def picky(rows):
        return _length_filter(rows, min_yield=0.9)

    with pytest.raises(StepContractError, match="below min_yield 0.90"):
        picky(_rows(*TEXTS))
    out = capsys.readouterr().out
    assert "step picky" in out and "5 → 3 rows" in out
    assert inspect_step("picky").warnings == [
        "filter.by_length: yield 0.60 below min_yield 0.90"
    ]
    # The failing result was not cached, so the second run fails the same way.
    with pytest.raises(StepContractError, match="below min_yield 0.90"):
        picky(_rows(*TEXTS))
    assert not _tracking.step_records()["picky"].cache_hit


def test_strict_applies_to_cached_steps():
    """A result cached without strict still fails a strict step on a cache hit."""

    def picky(rows):
        return _length_filter(rows, min_yield=0.9)

    step(name="picky")(picky)(_rows(*TEXTS))
    strict_picky = step(name="picky", strict=True)(picky)
    with pytest.raises(StepContractError, match="below min_yield 0.90"):
        strict_picky(_rows(*TEXTS))
    assert _tracking.step_records()["picky"].cache_hit


def test_yield_bounds_on_acquire_ops_raise():
    """Acquire ops take no input rows, so yield bounds are rejected up front."""
    calls = []

    @op(name="acquire.test_fake_gen", category="acquire")
    def fake_gen(prompt, n):
        calls.append(1)
        return [{"text": f"{prompt} {i}"} for i in range(n)]

    with pytest.raises(ValueError, match="takes no input rows"):
        fake_gen("hello", 2, min_yield=0.5)
    assert calls == []


def test_yield_bounds_with_no_input_rows_warn():
    """An empty input misses min_yield; a non-list op cannot be checked."""

    @op(name="inspect.test_count", category="inspect")
    def count(rows):
        return {"n": len(rows)}

    @step
    def empty(rows):
        count(rows, max_yield=0.5)
        return _length_filter(rows, min_yield=0.5)

    empty([])
    assert _tracking.step_records()["empty"].warnings == [
        "inspect.test_count: yield unchecked; min_yield / max_yield need a list of "
        "rows in and a list out",
        "filter.by_length: no input rows, so yield is 0 and below min_yield 0.50",
    ]
    assert "LOW · min 0.50" in render_step(_tracking.step_records()["empty"])


def test_max_yield_violation():
    """A filter above its max_yield adds a warning."""

    @step
    def lenient(rows):
        return _length_filter(rows, max_yield=0.5)

    lenient(_rows(*TEXTS))
    rec = _tracking.step_records()["lenient"]
    assert rec.warnings == ["filter.by_length: yield 0.60 above max_yield 0.50"]
    assert rec.ops[0].max_yield == 0.5


def test_yield_bounds_outside_steps_log_a_warning(caplog):
    """Outside a step, a yield violation is logged."""
    with caplog.at_level("WARNING", logger="forger.op"):
        _length_filter(_rows(*TEXTS), min_yield=0.9)
    assert "below min_yield 0.90" in caplog.text


def test_yield_bounds_are_not_part_of_the_cache_key():
    """min_yield / max_yield do not change the cache key."""
    _length_filter(_rows(*TEXTS))
    _length_filter(_rows(*TEXTS), min_yield=0.1, max_yield=0.9)
    hits = [r.cache_hit for r in trace()]
    assert hits == [False, True]


@pytest.mark.parametrize(
    "bounds",
    [{"min_yield": 1.5}, {"max_yield": -0.1}, {"min_yield": 0.8, "max_yield": 0.2}],
)
def test_invalid_yield_bounds_raise(bounds):
    """Bounds outside [0, 1] or min above max raise ValueError."""
    with pytest.raises(ValueError, match="yield"):
        _length_filter(_rows(*TEXTS), **bounds)


@pytest.mark.parametrize("reserved", ["min_yield", "max_yield"])
def test_yield_names_are_reserved_at_decoration(reserved):
    """An op or step that declares min_yield / max_yield fails at decoration."""
    namespace: dict = {}
    exec(f"def fn(rows, {reserved}=None):\n    return rows", namespace)
    with pytest.raises(TypeError, match=f"reserved argument name '{reserved}'"):
        op(namespace["fn"])
    with pytest.raises(TypeError, match=f"reserved argument name '{reserved}'"):
        step(namespace["fn"])


# --------------------------------------------------------------------------- #
# @step contract
# --------------------------------------------------------------------------- #


def test_step_without_ops_raises():
    """A step that calls no Forger op is an error."""

    @step
    def plain(rows):
        return [r for r in rows if r["text"]]

    with pytest.raises(StepContractError, match="called no Forger operations"):
        plain(_rows(*TEXTS))
    # The failing result was not cached, so a second call fails the same way.
    with pytest.raises(StepContractError, match="called no Forger operations"):
        plain(_rows(*TEXTS))


def test_nested_step_raises():
    """Calling a step inside another step is an error."""

    @step
    def inner(rows):
        return _length_filter(rows)

    @step
    def outer(rows):
        return inner(_upper(rows))

    with pytest.raises(StepContractError, match="inside step outer"):
        outer(_rows(*TEXTS))


def test_step_returning_non_list_raises():
    """A step must return a list."""

    @step
    def counts(rows):
        return {"n": len(_length_filter(rows))}

    with pytest.raises(StepContractError, match="returned dict"):
        counts(_rows(*TEXTS))
    # The failing result was not cached, so a second call fails the same way.
    with pytest.raises(StepContractError, match="returned dict"):
        counts(_rows(*TEXTS))


class Row(BaseModel):
    """Input row model."""

    text: str


class Scored(BaseModel):
    """Output row model."""

    text: str
    score: int


def test_row_model_hints_validate_input_and_output():
    """list[Model] hints check every row and name the first bad one."""

    @step
    def score_rows(rows: list[Row]) -> list[Scored]:
        return _length_filter(_upper(rows))

    with pytest.raises(StepContractError, match=r"input row 1 does not match Row"):
        score_rows([{"text": "ok fine"}, {"txt": "missing"}])
    with pytest.raises(
        StepContractError, match=r"output row 0 does not match Scored: score"
    ):
        score_rows(_rows(*TEXTS))


def test_row_model_hints_pass_and_rows_stay_dicts():
    """Valid rows pass the schema check and are returned unchanged as dicts."""

    @step
    def keep(rows: list[Row]) -> list[Row]:
        return _length_filter(rows)

    out = keep(_rows(*TEXTS))
    assert all(type(r) is dict for r in out)
    rec = _tracking.step_records()["keep"]
    assert (rec.input_type, rec.output_type) == ("list[Row]", "list[Row]")


def test_non_dict_rows_warn_instead_of_failing():
    """A step may return non-dict rows; schema checks and samples are skipped."""

    @step
    def texts(rows):
        return _strings(_length_filter(rows))

    assert texts(_rows(*TEXTS)) == ["one two", "three four five", "seven eight"]
    rec = _tracking.step_records()["texts"]
    assert "rows are not dicts; schema checks and samples skipped" in rec.warnings
    assert rec.sample == []


def test_step_without_final_filter_warns():
    """A step whose last op is not a filter gets a warning."""

    @step
    def shout(rows):
        return _upper(rows)

    shout(_rows(*TEXTS))
    assert _tracking.step_records()["shout"].warnings == ["ends without a filter"]


def test_prod_and_env_silence_the_report(capsys, monkeypatch):
    """prod=True and FORGER_STEP_REPORT=0/false/no print nothing."""

    @step(prod=True)
    def quiet(rows):
        return _length_filter(rows)

    @step
    def loud(rows):
        return _length_filter(rows)

    monkeypatch.delenv("FORGER_STEP_REPORT")
    quiet(_rows(*TEXTS))
    assert capsys.readouterr().out == ""
    for value in ("0", "False", "no"):
        monkeypatch.setenv("FORGER_STEP_REPORT", value)
        loud(_rows(*TEXTS), _no_cache=True)
        assert capsys.readouterr().out == ""
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    loud(_rows(*TEXTS), _no_cache=True)
    out = capsys.readouterr().out
    assert "step loud" in out and "5 → 3 rows" in out


def test_cached_step_reports_its_ops_as_cached(capsys, monkeypatch):
    """A cache hit runs no ops but lists the ops of the run that cached it."""
    monkeypatch.delenv("FORGER_STEP_REPORT")
    calls = []

    @op(name="alter.test_counted", category="alter")
    def counted(rows):
        calls.append(1)
        return list(rows)

    @step
    def cached_step(rows):
        return _length_filter(counted(rows))

    cached_step(_rows(*TEXTS))
    capsys.readouterr()
    out_rows = cached_step(_rows(*TEXTS))
    assert len(out_rows) == 3 and calls == [1]
    rec = _tracking.step_records()["cached_step"]
    assert rec.cache_hit
    assert [(c.op_name, c.n_in, c.n_out, c.cache_hit) for c in rec.ops] == [
        ("alter.test_counted", 5, 5, True),
        ("filter.by_length", 5, 3, True),
    ]
    assert all(c.scores is None and c.duration_s == 0.0 for c in rec.ops)
    out = capsys.readouterr().out
    assert re.search(r"filter\.by_length\s+5\s+3\s+0\.60\s+-\s+cached", out)


def test_cached_step_replays_yield_warnings():
    """Warnings of the run that cached a step come back on every cache hit."""

    @step
    def picky(rows):
        return _length_filter(rows, min_yield=0.9)

    for _ in range(2):
        picky(_rows(*TEXTS))
        rec = _tracking.step_records()["picky"]
        assert rec.warnings == ["filter.by_length: yield 0.60 below min_yield 0.90"]
    assert rec.cache_hit


# --------------------------------------------------------------------------- #
# @pipeline
# --------------------------------------------------------------------------- #


@step
def _shout_step(rows):
    return _length_filter(_upper(rows))


@step
def _dedupe_step(rows):
    return filter.by_exact_match(rows, field="text")


def test_pipeline_without_steps_raises():
    """A pipeline that calls no step is an error, even with bare ops."""

    @pipeline
    def bare_only(rows):
        return _length_filter(rows)

    with pytest.raises(PipelineContractError, match="called no steps"):
        bare_only(_rows(*TEXTS))


def test_nested_pipeline_raises():
    """Pipelines cannot nest."""

    @pipeline
    def inner(rows):
        return _shout_step(rows)

    @pipeline
    def outer(rows):
        return inner(rows)

    with pytest.raises(PipelineContractError, match="inside pipeline outer"):
        outer(_rows(*TEXTS))


def test_pipeline_records_steps_and_warns(capsys, monkeypatch):
    """A pipeline records each step; one step and missing filters are warnings."""
    monkeypatch.delenv("FORGER_STEP_REPORT")

    @pipeline
    def one_step(rows):
        return _shout_step(rows)

    one_step(_rows(*TEXTS))
    rec = _tracking.last_pipeline()
    assert rec is not None and [s.name for s in rec.steps] == ["_shout_step"]
    joined = " | ".join(rec.warnings)
    assert "only one step ran" in joined
    assert "no quality filter ran" in joined
    assert "no diversity filter ran" in joined
    assert "no inspect op ran" in joined
    out = capsys.readouterr().out
    assert "pipeline one_step" in out and "1 step · 5 → 3 rows" in out
    assert "pipeline warnings" in out and "⚠ no inspect op ran" in out


def test_pipeline_bare_op_warns():
    """An op called in the pipeline outside a step is listed with a warning."""

    @pipeline
    def with_bare(rows):
        rows = _shout_step(rows)
        rows = filter.by_exact_match(rows, field="text")
        return _dedupe_step(rows)

    with_bare(_rows(*TEXTS))
    rec = _tracking.last_pipeline()
    assert rec is not None
    names = [s.name for s in rec.steps]
    assert names == ["_shout_step", "(bare) filter.by_exact_match", "_dedupe_step"]
    assert rec.steps[1].warnings == ["called outside a @step"]
    assert "1 op call outside a @step: filter.by_exact_match" in rec.warnings
    assert not any("only one step" in w for w in rec.warnings)


def test_dynamic_step_in_pipeline_counts_as_step(monkeypatch):
    """alter.dynamic_step called directly in a pipeline is a real step."""
    monkeypatch.setattr(dynamic, "_generate", lambda *args: "code")
    monkeypatch.setattr(dynamic, "_execute", lambda code, item, timeout: [item["text"]])

    @pipeline
    def with_dynamic(rows):
        outputs = alter.dynamic_step(rows, "goal", SimpleNamespace(_model_id="test"))
        return _shout_step([{"text": o[0]} for o in outputs])

    with_dynamic(_rows(*TEXTS))
    rec = _tracking.last_pipeline()
    assert rec is not None
    assert [(s.name, s.kind) for s in rec.steps] == [
        ("dynamic_step", "dynamic_step"),
        ("_shout_step", "step"),
    ]
    assert rec.steps[0].ops[0].op_name == "alter.dynamic_step"
    assert not any("outside a @step" in w for w in rec.warnings)
    assert not any("only one step" in w for w in rec.warnings)


def test_pipeline_with_quality_diversity_and_inspect_has_no_such_warnings():
    """Quality, diversity, and inspect ops anywhere silence those warnings."""

    @step
    def judge(rows):
        return filter.by_llm_score(
            rows, model=_StableJudge(), input_field="text", output_field="text"
        )

    @step
    def report_and_dedupe(rows):
        _report(rows)
        return filter.by_exact_match(rows, field="text")

    @pipeline
    def complete(rows):
        return report_and_dedupe(judge(rows))

    complete(_rows("abcdef", "abc", "ab"))
    rec = _tracking.last_pipeline()
    assert rec is not None and rec.warnings == []


def test_pipeline_strict_raises():
    """strict=True turns pipeline warnings into a PipelineContractError."""

    @pipeline(strict=True)
    def strict_one(rows):
        return _shout_step(rows)

    with pytest.raises(PipelineContractError, match="only one step ran"):
        strict_one(_rows(*TEXTS))


def test_pipeline_of_cached_steps_keeps_its_warnings():
    """Cached steps replay their op names, so a strict pipeline fails every run."""

    @pipeline(strict=True)
    def twice(rows):
        return _dedupe_step(_shout_step(rows))

    for _ in range(2):
        with pytest.raises(PipelineContractError, match="no inspect op ran"):
            twice(_rows(*TEXTS))
    rec = _tracking.last_pipeline()
    assert rec is not None and all(s.cache_hit for s in rec.steps)


def test_user_op_named_like_a_filter_is_not_a_filter():
    """A user op's name prefix gives it no category, so a no-op pipeline still warns."""

    @op(name="filter.keep_all")
    def keep_all(rows):
        return rows

    @op(name="inspect.nothing")
    def look(rows):
        return ["ok"] * len(rows)

    @step
    def make_rows(seeds):
        return keep_all([{"id": s, "text": f"row {s}"} for s in seeds])

    @step
    def finish(rows):
        look(rows)
        return filter.by_exact_match(rows, field="id", max_count=1)

    @pipeline(strict=True)
    def gamed(seeds):
        return finish(make_rows(seeds))

    with pytest.raises(PipelineContractError) as err:
        gamed(list(range(5)))
    assert "no quality filter ran" in str(err.value)
    assert "no inspect op ran" in str(err.value)
    assert "no diversity filter ran" not in str(err.value)
    rec = _tracking.last_pipeline()
    assert rec is not None
    assert [c.category for s in rec.steps for c in s.ops] == [
        "other",
        "other",
        "filter",
    ]
    assert "ends without a filter" in rec.steps[0].warnings


def test_user_op_with_declared_category_counts():
    """@op(category=...) is a deliberate declaration, so it satisfies the checks."""

    @op(name="my_judge", category="filter")
    def my_judge(rows):
        return rows[:-1]

    @op(category="inspect")
    def my_report(rows):
        return len(rows)

    @step
    def judge(rows):
        return my_judge(rows)

    @step
    def report_and_dedupe(rows):
        my_report(rows)
        return filter.by_exact_match(rows, field="text")

    @pipeline
    def declared(rows):
        return report_and_dedupe(judge(rows))

    declared(_rows("a", "b", "c"))
    rec = _tracking.last_pipeline()
    assert rec is not None and rec.warnings == []


def test_op_categories_come_from_the_registry():
    """Library ops get their package's category; unknown categories are rejected."""
    assert _tracking.category_of("filter.by_length") == "filter"
    assert _tracking.category_of("acquire.retrieve") == "acquire"
    assert _tracking.category_of("alter.generate_from_context") == "alter"
    assert _tracking.category_of("inspect.topic_based_clustering") == "inspect"
    assert _tracking.category_of("never.registered") == "other"
    with pytest.raises(ValueError, match="unknown op category 'filters'"):
        op(lambda rows: rows, name="typo", category="filters")


def test_step_cached_without_op_record_skips_which_ops_ran_checks(tmp_path):
    """A step cached before op records were stored skips the which-ops-ran checks."""

    @pipeline
    def twice(rows):
        return _dedupe_step(_shout_step(rows))

    twice(_rows(*TEXTS))
    first = _tracking.last_pipeline()
    assert first is not None and "no inspect op ran" in first.warnings
    for meta in tmp_path.rglob("*.step.json"):
        meta.unlink()
    twice(_rows(*TEXTS))
    second = _tracking.last_pipeline()
    assert second is not None and second.steps[0].cache_hit
    assert second.steps[0].ops == []
    assert second.warnings == []


def test_pyforger_exports_step_and_pipeline():
    """The package exports the decorators; the pipeline module still works."""
    from pyforger.pipeline import optimize, run

    assert pyforger.step is step and pyforger.pipeline is pipeline
    assert pyforger.run is run and pyforger.optimize is optimize


# --------------------------------------------------------------------------- #
# inspect_step and rendering
# --------------------------------------------------------------------------- #


def test_inspect_step_returns_latest_record_and_prints_full(capsys):
    """inspect_step prints the full report of a step's latest run."""

    @step
    def first(rows):
        return _length_filter(rows)

    first(_rows(*TEXTS))
    first(_rows("alpha beta", "gamma"))
    rec = inspect_step("first")
    assert isinstance(rec, StepRecord) and rec.n_in == 2
    assert inspect_step() is rec
    out = capsys.readouterr().out
    assert "step first" in out and "2 → 1 rows" in out
    assert re.search(r"\n\s+text\s+alpha beta\n", out)


def test_inspect_step_unknown_name_lists_known_steps():
    """An unknown name raises KeyError naming the steps that ran."""

    @step
    def known(rows):
        return _length_filter(rows)

    with pytest.raises(KeyError, match="no @step has run"):
        inspect_step("known")
    known(_rows(*TEXTS))
    with pytest.raises(KeyError, match="known steps: known"):
        inspect_step("missing")


def _bins(scores):
    """(label, count, cut, kept, share, current) per histogram bin."""
    return [
        (b.label, b.count, b.cut, b.kept, b.share, b.current)
        for b in _histogram(scores).bins
    ]


def _flat(text):
    """Report text with box characters removed and whitespace collapsed."""
    return " ".join(re.sub(r"[│║╭╮╰╯╔╗╚╝─═]", " ", text).split())


def test_render_int_scores_histogram():
    """Int scores get one bin per value; each bin says what a cut there keeps."""
    scores = ScoreRecord(
        label="score", values=[1, 2, 3, 3, 5, None], keep="ge", threshold=3
    )
    # Shares are of all rows; the unparsed row is never kept.
    assert _bins(scores) == [
        ("1", 1, "≥ 1", 5, pytest.approx(5 / 6), False),
        ("2", 1, "≥ 2", 4, pytest.approx(4 / 6), False),
        ("3", 2, "≥ 3", 3, pytest.approx(3 / 6), True),
        ("5", 1, "≥ 5", 1, pytest.approx(1 / 6), False),
    ]
    hist = _histogram(scores)
    assert hist.unparsed == 1
    assert hist.title == "score · keeps ≥ 3: 3 of 6 (0.50)"
    text = render_scores(scores, "filter.by_llm_score")
    assert "filter.by_llm_score · score · keeps ≥ 3: 3 of 6 (0.50)" in text
    assert re.search(r"▶\s+3\s+2\s+█+\s+≥ 3\s+3\s+0\.50\s+◀ current threshold", text)
    assert re.search(r"unparsed\s+1\s+never kept", text)


def test_render_judge_fractions_get_one_bin_each():
    """Three judges give scores 0, 1/3, 2/3, 1: one bin each, the cut marked."""
    values = [0.0] * 4 + [1 / 3] * 3 + [2 / 3] * 2 + [1.0] * 1
    scores = ScoreRecord(label="approval >=3", values=values, keep="ge", threshold=1.0)
    bins = _bins(scores)
    assert [(b[0], b[1]) for b in bins] == [("0", 4), ("1/3", 3), ("2/3", 2), ("1", 1)]
    assert [b[5] for b in bins] == [False, False, False, True]
    assert _histogram(scores).label == "judges ≥3"


def test_render_float_scores_histogram():
    """Many float scores in [0, 1] get ten bins of 0.1; < cuts sit at upper edges."""
    values = [round(i / 20 + 0.01, 2) for i in range(20)]  # two per bin
    bins = _bins(ScoreRecord(label="rouge_l", values=values, keep="lt", threshold=0.7))
    assert [b[0] for b in bins] == [f"{i / 10:g}–{(i + 1) / 10:g}" for i in range(10)]
    assert [b[1] for b in bins] == [2] * 10
    assert bins[6] == ("0.6–0.7", 2, "< 0.7", 14, pytest.approx(0.7), True)
    assert bins[4][2:5] == ("< 0.5", 10, pytest.approx(0.5))


def test_render_threshold_inside_a_bin_marks_that_bin():
    """A threshold in the middle of a bin marks it and shows the real cut."""
    values = [round(i / 20 + 0.01, 2) for i in range(20)]
    bins = _bins(ScoreRecord(label="rouge_l", values=values, keep="lt", threshold=0.65))
    marked = [b for b in bins if b[5]]
    # 0.01, 0.06, ..., 0.61 are below 0.65: 13 of 20 rows.
    assert marked == [("0.6–0.7", 2, "< 0.65", 13, pytest.approx(0.65), True)]


def test_render_float_scores_outside_unit_range_and_between():
    """Many distinct ints get round-width bins; between marks bins kept or not."""
    values = list(range(0, 200, 10))
    scores = ScoreRecord(
        label="words", values=values, keep="between", lower=20, upper=100
    )
    bins = _histogram(scores).bins
    assert [b.label for b in bins] == [f"{i}–{i + 19}" for i in range(0, 200, 20)]
    assert [b.count for b in bins] == [2] * 10
    assert [b.status for b in bins][:6] == ["no", "yes", "yes", "yes", "yes", "partly"]
    assert _histogram(scores).title == "words · keeps 20 to 100: 9 of 20 (0.45)"


def test_render_outlier_gets_an_overflow_row():
    """One huge value does not stretch the bins; it gets a ≥ row of its own."""
    values = list(range(20, 420)) + [21_000]
    bins = _histogram(
        ScoreRecord(label="words", values=values, keep="between", lower=50, upper=300)
    ).bins
    assert bins[-1].label.startswith("≥ ") and bins[-1].count == 1
    assert sum(b.count for b in bins) == len(values)
    assert len(bins) <= 12


def test_render_many_int_scores_uses_round_bins():
    """Ints with more than 12 values get round bins, and every value is counted."""
    values = list(range(3, 17))  # 14 distinct values
    bins = _bins(
        ScoreRecord(label="occurrence", values=values, keep="le", threshold=11)
    )
    assert [b[0] for b in bins] == ["2–3", "4–5", "6–7", "8–9", "10–11", "12–13"] + [
        "14–15",
        "16–17",
    ]
    assert [b[1] for b in bins] == [1, 2, 2, 2, 2, 2, 2, 1]
    # "<= 11" keeps the 10–11 bin and everything below it: 9 of 14 rows.
    assert bins[4] == ("10–11", 2, "≤ 11", 9, pytest.approx(9 / 14), True)


def test_render_empty_bins_are_merged():
    """Runs of empty bins fold into one row; an empty marked bin stays."""
    values = [0.01 + i * 0.001 for i in range(10)] + [
        0.95 + i * 0.001 for i in range(5)
    ]
    bins = _histogram(
        ScoreRecord(label="overlap", values=values, keep="lt", threshold=0.5)
    ).bins
    assert [(b.label, b.count, b.empty_run, b.current) for b in bins] == [
        ("0–0.1", 10, 0, False),
        ("0.1–0.4", 0, 3, False),
        ("0.4–0.5", 0, 0, True),
        ("0.5–0.9", 0, 4, False),
        ("0.9–1", 5, 0, False),
    ]


def test_render_pooled_scores_shows_only_the_current_threshold():
    """Pool-based scores (by_rouge_l) do not claim what other thresholds keep."""
    scores = ScoreRecord(
        label="rouge_l",
        values=[0.0, 0.55, 0.65, 0.9],
        keep="lt",
        threshold=0.7,
        pooled=True,
    )
    assert all(b[2] == "" and b[3] is None for b in _bins(scores))
    hist = _histogram(scores)
    assert hist.mode == "pooled"
    assert hist.title == "rouge_l · keeps < 0.7: 3 of 4 (0.75)"
    text = render_scores(scores, "filter.by_rouge_l")
    assert "if cut at" not in text
    assert "◀ current threshold" in text
    assert "Rerun to compare" in _flat(text)


def test_render_bool_scores():
    """Boolean scores show kept / dropped counts on one line."""
    text = render_scores(
        ScoreRecord(label="kept", values=[True, False, True], keep="true"),
        "filter.by_syntax",
    )
    assert text == "filter.by_syntax · kept 2 of 3 (0.67) · dropped 1"


def test_sample_values_keep_line_breaks():
    """Compact samples mark line breaks; the full view keeps them."""
    rec = StepRecord(
        name="code",
        n_in=1,
        n_out=1,
        sample=[{"solution": "def f(n):\n    return n"}],
    )
    assert "def f(n): ⏎ return n" in render_step(rec)
    assert re.search(r"def f\(n\):\n\s+    return n", render_step(rec, full=True))


def test_report_fits_a_narrow_terminal():
    """At 80 columns only the op name is cut; numbers and checks stay whole."""
    call = _tracking.OpCall(
        op_name="filter.a_really_long_custom_operation_name",
        category="filter",
        n_in=12345,
        n_out=6000,
        yield_ratio=0.49,
        min_yield=0.8,
        cache_hit=False,
        duration_s=12.0,
        prompt_tokens=1234567,
        completion_tokens=7654321,
        cost_usd=12.34,
    )
    rec = StepRecord(name="x", n_in=12345, n_out=6000, ops=[call, call])
    console = Console(file=io.StringIO(), width=80, color_system=None)
    console.print(step_renderable(rec, width=80))
    text = console.file.getvalue()
    assert all(len(line) <= 80 for line in text.splitlines())
    assert "LOW · min 0.80" in text and "12,345" in text and "$12.34" in text


def test_report_text_is_not_parsed_as_markup():
    """Sample rows and docstrings print literally, brackets and all."""
    rec = StepRecord(
        name="markup",
        doc="Doc with [bold]brackets[/bold].",
        n_in=1,
        n_out=1,
        sample=[{"text": "[red]not red[/red] :smile:"}],
    )
    text = render_step(rec)
    assert "[bold]brackets[/bold]" in text
    assert "[red]not red[/red] :smile:" in text


def test_render_step_lines_fit_and_cached_op_hides_scores():
    """Report lines stay within 120 columns; cached ops print (cached)."""
    scores = ScoreRecord(label="score", values=[1, 2, 3], keep="ge", threshold=2)
    call = _tracking.OpCall(
        op_name="filter.by_llm_score",
        category="filter",
        n_in=1234,
        n_out=617,
        yield_ratio=0.5,
        min_yield=0.3,
        cache_hit=True,
        duration_s=12.0,
        prompt_tokens=4120,
        completion_tokens=38900,
        cost_usd=0.0412,
        scores=scores,
    )
    rec = StepRecord(
        name="long",
        doc="Doc line.",
        n_in=1234,
        n_out=617,
        ops=[call],
        sample=[{"text": "x" * 500, "other": "y" * 500}],
        prompt_tokens=4120,
        completion_tokens=38900,
        cost_usd=0.0412,
    )
    for full in (False, True):
        text = render_step(rec, full=full)
        assert all(len(line) <= 120 for line in text.splitlines())
        assert re.search(r"4,120\s+38\.9K", text) and "$0.041" in text
        assert re.search(r"cached\s+ok · min 0\.30", text)
        assert "no score histogram for a cached result" in text
        assert "if cut at" not in text


# --------------------------------------------------------------------------- #
# Threads, dynamic_step, retrieval, usage, and per-filter scores
# --------------------------------------------------------------------------- #


def test_step_with_ops_in_worker_threads_warns_instead_of_failing():
    """Ops run in a thread pool are not itemized, but the step is not an error."""

    @step
    def threaded(rows):
        with ThreadPoolExecutor(2) as pool:
            parts = list(pool.map(lambda r: _length_filter([r]), rows))
        return [row for part in parts for row in part]

    assert len(threaded(_rows(*TEXTS))) == 3
    rec = _tracking.step_records()["threaded"]
    assert rec.ops == []
    assert rec.warnings[0].startswith("its ops ran in worker threads")


def test_step_ops_submitted_with_a_copied_context_are_itemized():
    """Submitting with contextvars.copy_context().run attributes worker ops."""

    @step
    def threaded(rows):
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(contextvars.copy_context().run, _length_filter, [row])
                for row in rows
            ]
            return [row for future in futures for row in future.result()]

    threaded(_rows(*TEXTS))
    rec = _tracking.step_records()["threaded"]
    assert [c.op_name for c in rec.ops] == ["filter.by_length"] * 5
    assert rec.warnings == []


def test_steps_called_from_worker_threads_join_the_pipeline():
    """Steps run in a thread pool still count as the pipeline's steps."""

    @pipeline
    def parallel(rows):
        with ThreadPoolExecutor(2) as pool:
            parts = list(pool.map(_shout_step, [rows[:2], rows[2:]]))
        return _dedupe_step([row for part in parts for row in part])

    parallel(_rows(*TEXTS))
    rec = _tracking.last_pipeline()
    assert rec is not None
    assert sorted(s.name for s in rec.steps) == [
        "_dedupe_step",
        "_shout_step",
        "_shout_step",
    ]


def test_dynamic_step_in_pipeline_is_stored_and_printed(capsys, monkeypatch):
    """dynamic_step called in a pipeline prints a report and works with inspect_step."""
    monkeypatch.delenv("FORGER_STEP_REPORT")
    monkeypatch.setattr(dynamic, "_generate", lambda *args: "code")
    monkeypatch.setattr(dynamic, "_execute", lambda code, item, timeout: [item["text"]])

    @pipeline
    def with_dynamic(rows):
        outputs = alter.dynamic_step(rows, "goal", SimpleNamespace(_model_id="test"))
        return _shout_step([{"text": o[0]} for o in outputs])

    with_dynamic(_rows(*TEXTS))
    out = capsys.readouterr().out
    assert "step dynamic_step" in out and "5 → 5 rows" in out
    assert out.index("step dynamic_step") < out.index("step _shout_step")
    assert inspect_step("dynamic_step").kind == "dynamic_step"


class _FakeStats:
    """Stands in for an OpenHands ConversationStats."""

    def get_combined_metrics(self):
        usage = SimpleNamespace(prompt_tokens=30, completion_tokens=7)
        return SimpleNamespace(accumulated_cost=0.5, accumulated_token_usage=usage)


class _FakeConversation:
    """Stands in for an OpenHands Conversation after it ran."""

    conversation_stats = _FakeStats()


@op(name="alter.test_agent", category="alter")
def _agent(rows):
    dynamic._record_agent_usage(_FakeConversation())
    dynamic._record_agent_usage(SimpleNamespace())  # no stats: ignored
    return list(rows)


def test_dynamic_step_agent_usage_is_counted():
    """The OpenHands conversation's tokens and cost reach the step totals."""
    agent = _agent

    @step
    def with_agent(rows):
        return _length_filter(agent(rows))

    with_agent(_rows(*TEXTS))
    rec = _tracking.step_records()["with_agent"]
    assert (rec.ops[0].prompt_tokens, rec.ops[0].completion_tokens) == (30, 7)
    assert rec.cost_usd == pytest.approx(0.5)


def test_model_generate_batch_usage_reaches_the_step_report():
    """Model.generate_batch adds each response's tokens and cost to the op."""
    message = SimpleNamespace(content="hello there", reasoning_content=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=20, completion_tokens_details=None
        ),
        _hidden_params={"response_cost": 0.001},
    )
    model = Model("openai/test-model")

    @step
    def write(rows):
        return _length_filter(alter.generate_from_context(rows, "Say {text}", model))

    with patch("pyforger.acquire.model.litellm") as mock_litellm:
        mock_litellm.completion.return_value = response
        write(_rows("one two"))
    rec = _tracking.step_records()["write"]
    generate = rec.ops[0]
    assert generate.op_name == "alter.generate_from_context"
    assert (generate.prompt_tokens, generate.completion_tokens) == (10, 20)
    assert generate.cost_usd == pytest.approx(0.001)
    assert (rec.prompt_tokens, rec.completion_tokens) == (10, 20)


def test_retrieval_step_records_the_retrieve_call():
    """acquire.retrieve is listed in the step report."""

    class _Source(acquire.RetrievalSource):
        def cache_key(self):
            return "test-source"

        def fetch(self, mode, *, query=None, n=10, seed=None, **kwargs):
            return [{"text": f"{query} result {i}"} for i in range(n)]

    @step
    def find(queries):
        return _length_filter(acquire.retrieve(_Source(), queries=queries, n=2))

    assert len(find(["a", "b"])) == 4
    rec = _tracking.step_records()["find"]
    assert [(c.op_name, c.n_in, c.n_out) for c in rec.ops] == [
        ("acquire.retrieve", None, 4),
        ("filter.by_length", 4, 4),
    ]


def test_row_model_hints_defined_in_a_function_resolve():
    """A row model defined in the enclosing function still validates rows."""

    class Local(BaseModel):
        text: str

    @step
    def local_typed(rows: list[Local]) -> list[Local]:
        return _length_filter(rows)

    with pytest.raises(StepContractError, match="input row 0 does not match Local"):
        local_typed([{"nope": 1}])
    local_typed(_rows(*TEXTS))
    assert _tracking.step_records()["local_typed"].input_type == "list[Local]"


def test_unresolvable_row_model_hints_warn():
    """Hints that cannot be resolved are a warning, not a silent skip."""

    @step
    def unknown(rows: list[Missing]) -> list[Missing]:  # type: ignore[name-defined]  # noqa: F821, E501
        return _length_filter(rows)

    unknown(_rows(*TEXTS))
    warning = _tracking.step_records()["unknown"].warnings[0]
    assert warning.startswith("type hints could not be resolved (NameError")
    assert warning.endswith("schema checks skipped")


def test_long_names_and_totals_fit_in_120_columns():
    """Headers, op tables, and the pipeline summary stay within 120 columns."""
    call = _tracking.OpCall(
        op_name="filter.a_really_long_custom_operation_name_for_testing_widths",
        category="filter",
        n_in=12345,
        n_out=6000,
        yield_ratio=0.49,
        min_yield=0.8,
        cache_hit=False,
        duration_s=12.0,
        prompt_tokens=1234567,
        completion_tokens=7654321,
        cost_usd=12.34,
    )
    warnings = ["filter.by_llm_score: yield 0.43 below min_yield 0.50", "ends w/o"]
    rec = StepRecord(
        name="generate_candidate_solutions_for_every_competition_problem_with_traces",
        n_in=12345,
        n_out=6000,
        duration_s=12345.6,
        ops=[call],
        warnings=warnings,
        prompt_tokens=1234567,
        completion_tokens=7654321,
        cost_usd=12.34,
    )
    step_text = render_step(rec)
    assert all(len(line) <= 120 for line in step_text.splitlines())
    assert "1,234,567 in / 7,654,321 out tokens · $12.34" in step_text
    assert "LOW · min 0.80" in step_text
    assert "⚠ filter.by_llm_score: yield 0.43 below min_yield 0.50" in step_text
    pipe = PipelineRecord(name="p" * 40, steps=[rec], duration_s=1.0, cost_usd=12.34)
    pipe_text = render_pipeline(pipe)
    assert all(len(line) <= 120 for line in pipe_text.splitlines())
    flat = _flat(pipe_text)
    name = rec.name
    assert f"⚠ {name}: filter.by_llm_score: yield 0.43 below min_yield 0.50" in flat
    assert f"⚠ {name}: ends w/o" in flat


_FILTER_SCORE_CASES = [
    (
        lambda rows: filter.by_rouge_l(rows, field="text", threshold=0.7),
        [{"text": "a b c"}, {"text": "a b c"}, {"text": "x y z"}],
        ("rouge_l", "lt", 0.7, [0.0, 1.0, 0.0], True),
    ),
    (
        lambda rows: filter.by_n_gram_overlap(rows, field="text", threshold=0.7),
        [{"text": "a b c"}, {"text": ""}, {"text": "a b c"}, {"text": "d e f"}],
        ("overlap", "lt", 0.7, [0.0, None, 1.0, 0.0], True),
    ),
    (
        lambda rows: filter.by_exact_match(rows, field="text", max_count=2),
        [{"text": "a"}, {"text": "a"}, {"text": "b"}, {"text": "a"}],
        ("occurrence", "le", 2, [1, 2, 1, 3], False),
    ),
    (
        lambda rows: filter.by_rouge_l_agreement(
            rows, field_a="a", field_b="b", threshold=0.5
        ),
        [{"a": "x y", "b": "x y"}, {"a": "x y", "b": "p q"}],
        ("rouge_l", "ge", 0.5, [1.0, 0.0], False),
    ),
    (
        lambda rows: filter.by_rouge_l_agreement(
            rows, field_a="a", field_b="b", threshold=0.5, disagree=True
        ),
        [{"a": "x y", "b": "x y"}, {"a": "x y", "b": "p q"}],
        ("rouge_l", "lt", 0.5, [1.0, 0.0], False),
    ),
    (
        lambda rows: filter.by_exact_agreement(rows, field_a="a", field_b="b"),
        [{"a": 1, "b": 1}, {"a": 1, "b": 2}],
        ("kept", "true", None, [True, False], False),
    ),
    (
        lambda rows: filter.by_group_agreement(
            rows, group_by_input_field="q", match_output_field="a"
        ),
        [
            {"q": 1, "a": "x"},
            {"q": 1, "a": "x"},
            {"q": 2, "a": "y"},
            {"q": 2, "a": "z"},
        ],
        ("kept", "true", None, [True, True, False, False], False),
    ),
    (
        lambda rows: filter.by_invalid_response(
            rows, instruction_field="i", response_field="r"
        ),
        [{"i": "what is it", "r": "a different answer"}, {"i": "q", "r": " "}],
        ("kept", "true", None, [True, False], False),
    ),
    (
        lambda rows: filter.by_keyword_blocklist(
            rows, response_field="r", blocklist=["bad"]
        ),
        [{"r": "good"}, {"r": "bad thing"}],
        ("kept", "true", None, [True, False], False),
    ),
    (
        lambda rows: filter.by_syntax(rows, lang="json", field="j"),
        [{"j": "{}"}, {"j": "{"}],
        ("kept", "true", None, [True, False], False),
    ),
]


@pytest.mark.parametrize("run, rows, expected", _FILTER_SCORE_CASES)
def test_filters_record_per_row_scores(run, rows, expected):
    """Each filter records one score per row and the rule it kept rows by."""

    @step(name="scored")
    def scored(items):
        return run(items)

    kept = scored(rows)
    call = _tracking.step_records()["scored"].ops[0]
    assert call.scores is not None
    label, keep, threshold, values, pooled = expected
    got = call.scores
    assert (got.label, got.keep, got.threshold, got.pooled) == (
        label,
        keep,
        threshold,
        pooled,
    )
    assert got.values == [
        pytest.approx(v) if isinstance(v, float) else v for v in values
    ]
    if keep == "true":
        assert sum(got.values) == len(kept)


def test_by_rouge_l_keeps_the_first_item_at_threshold_zero():
    """With threshold <= 0 the first item is still kept, as before."""
    out = filter.by_rouge_l(_rows("a b", "c d", "a b"), field="text", threshold=0.0)
    assert out == [{"text": "a b"}]


def test_token_units_are_attributed_and_shown():
    """Token units recorded by a model land on the op, the step, and the reports."""

    @op(name="alter.fake_generate", category="alter")
    def fake_generate(rows):
        _tracking.add_usage(100, 50, 0.0, 250.0)
        return rows

    @step
    def gen(rows):
        return _length_filter(fake_generate(rows))

    @pipeline
    def main():
        return gen([{"text": "one two three"}])

    main()
    rec = _tracking.step_records()["gen"]
    assert rec.ops[0].token_units == 250.0
    assert rec.token_units == 250.0
    assert _tracking.last_pipeline().token_units == 250.0
    text = render_step(rec)
    assert "250 units" in text
    assert re.search(r"\n\s+op\s+in\s+out.*units", text)
    assert "units" in render_pipeline(_tracking.last_pipeline())


def test_units_column_hidden_without_units():
    """Steps whose models report no token units have no units column."""

    @step
    def plain(rows):
        return _length_filter(rows)

    plain([{"text": "one two three"}])
    assert "units" not in render_step(_tracking.step_records()["plain"])


def _floats_0_to_1():
    """200 scores in [0, 1], ten in each 0.05 slice, none on a 0.1 edge."""
    return [round(i / 200 + 0.0025, 4) for i in range(200)]


@pytest.mark.parametrize(
    ("keep", "threshold", "marked_label", "kept"),
    [
        ("ge", 0.5, "0.5–0.6", 100),
        ("le", 0.5, "0.4–0.5", 100),
        ("lt", 0.5, "0.4–0.5", 100),
        ("ge", 0.55, "0.5–0.6", 90),
    ],
)
def test_marker_on_a_float_bin_edge(keep, threshold, marked_label, kept):
    """A threshold on a bin edge marks the bin at the edge of the kept rows."""
    bins = _bins(
        ScoreRecord(label="s", values=_floats_0_to_1(), keep=keep, threshold=threshold)
    )
    marked = [b for b in bins if b[5]]
    assert [(b[0], b[3]) for b in marked] == [(marked_label, kept)]
    # No other row repeats the current cut.
    cuts = [b[2] for b in bins]
    assert len(cuts) == len(set(cuts))


def test_number_labels_never_collide_or_dangle():
    """0.999 is not '1.', tiny values are not '0.', and huge values stay short."""
    assert fmt_num(0.999) == "1" and fmt_num(0.001) == "0.001"
    assert fmt_num(1e12) == "1e+12" and fmt_num(7.65e6) == "7.65M"
    assert fmt_num(float("nan")) == "nan"
    values = [0.5, 0.95, 0.999, 0.999, 1.0]
    bins = _bins(ScoreRecord(label="s", values=values, keep="ge", threshold=0.999))
    labels = [b[0] for b in bins]
    assert len(labels) == len(set(labels))
    assert [b[2] for b in bins if b[5]] == ["≥ 0.999"]


def _render_at(renderable, width):
    console = Console(file=io.StringIO(), width=width, color_system=None)
    console.print(renderable)
    return console.file.getvalue()


@pytest.mark.parametrize("width", [40, 60, 80, 100])
def test_narrow_widths_keep_numbers_and_checks(width):
    """Narrow consoles drop optional columns instead of cropping numbers."""
    call = _tracking.OpCall(
        op_name="filter.by_llm_score",
        category="filter",
        n_in=200,
        n_out=90,
        yield_ratio=0.45,
        min_yield=0.6,
        cache_hit=False,
        duration_s=61.0,
        prompt_tokens=412_500,
        completion_tokens=1_200_000,
        cost_usd=12.5,
        token_units=24e6,
        scores=ScoreRecord(
            label="score", values=_floats_0_to_1(), keep="ge", threshold=0.55
        ),
    )
    rec = StepRecord(name="judge", n_in=200, n_out=90, ops=[call, call])
    pipe = PipelineRecord(name="p", steps=[rec, rec], warnings=["no inspect op ran"])
    step_text = _render_at(step_renderable(rec, width=width), width)
    pipe_text = _render_at(pipeline_renderable(pipe, width=width), width)
    for text in (step_text, pipe_text):
        assert all(len(line) <= width for line in text.splitlines()), text
        # Row counts are never cut: a table row, or the one-line view.
        assert re.search(r"\b200\s+90\s+0\.45\b|200 → 90 · yield\s+0\.45", text), text
    assert "▶" in step_text and "LOW" in step_text


def test_control_characters_never_reach_the_report():
    """ANSI escapes and tabs in names, docs, warnings, and keys are neutralized."""
    bad = "\x1b[31mRED\x1b[0m\ttab\x07"
    call = _tracking.OpCall(
        op_name=f"filter.{bad}",
        category="filter",
        n_in=2,
        n_out=1,
        yield_ratio=0.5,
        cache_hit=False,
        duration_s=0.1,
        scores=ScoreRecord(label=bad, values=[1, 2], keep="ge", threshold=2),
    )
    rec = StepRecord(
        name=bad,
        doc=bad,
        n_in=2,
        n_out=1,
        ops=[call, call],
        warnings=[bad],
        sample=[{bad: bad}],
    )
    pipe = PipelineRecord(name=bad, steps=[rec], warnings=[bad])
    for text in (render_step(rec), render_step(rec, full=True), render_pipeline(pipe)):
        assert "\x1b" not in text and "\x07" not in text and "\t" not in text
        assert all(len(line) <= 120 for line in text.splitlines())


def test_long_sample_keys_leave_room_for_values():
    """A long field name is cut so its value still shows in the block view."""
    row = {"k" * 200: "visible value", "a": 1, "b": 2, "c": 3, "d": 4}
    rec = StepRecord(name="keys", n_in=1, n_out=1, sample=[row])
    assert "visible value" in render_step(rec)


def test_colliding_sample_keys_use_the_block_view():
    """Keys 1 and '1' both show their values."""
    row = {1: "int key", "1": "str key"}
    text = render_step(StepRecord(name="keys", n_in=1, n_out=1, sample=[row]))
    assert "int key" in text and "str key" in text


def test_nan_yield_is_not_reported_as_high():
    """A NaN yield or bound gives no verdict instead of a wrong one."""
    call = _tracking.OpCall(
        op_name="filter.x",
        category="filter",
        n_in=2,
        n_out=1,
        yield_ratio=float("nan"),
        max_yield=float("inf"),
        min_yield=float("nan"),
        cache_hit=False,
        duration_s=0.1,
    )
    text = render_step(StepRecord(name="n", ops=[call, call]))
    assert "HIGH" not in text and "LOW" not in text and "nan" not in text.lower()
