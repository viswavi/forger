"""Tests for @step / @pipeline reports, yield checks, and filter score histograms."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import pyforger
from pyforger import _tracking, alter, filter, op
from pyforger._tracking import ScoreRecord, StepRecord
from pyforger.alter import dynamic
from pyforger.inspect import inspect_step
from pyforger.op import trace
from pyforger.steps import (
    PipelineContractError,
    StepContractError,
    pipeline,
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


@op(name="alter.test_upper")
def _upper(rows: list[dict]) -> list[dict]:
    return [{**r, "text": r["text"].upper()} for r in rows]


@op(name="alter.test_costly")
def _costly(rows: list[dict]) -> list[dict]:
    _tracking.add_usage(100, 50, 0.25)
    return list(rows)


@op(name="alter.test_strings")
def _strings(rows: list[dict]) -> list[str]:
    return [r["text"] for r in rows]


@op(name="inspect.test_report")
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

    @op(name="alter.test_nan_cost")
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
    assert "min 0.90 LOW" in out


def test_min_yield_violation_raises_when_strict():
    """strict=True turns a yield warning into a StepContractError."""

    @step(strict=True)
    def picky(rows):
        return _length_filter(rows, min_yield=0.9)

    with pytest.raises(StepContractError, match="below min_yield 0.90"):
        picky(_rows(*TEXTS))


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
    assert "── step loud: 5 → 3 rows" in capsys.readouterr().out


def test_cached_step_reports_cached_without_ops(capsys, monkeypatch):
    """A cache hit runs no ops, is not an error, and prints (cached)."""
    monkeypatch.delenv("FORGER_STEP_REPORT")
    calls = []

    @op(name="alter.test_counted")
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
    assert rec.cache_hit and rec.ops == []
    assert "(cached)" in capsys.readouterr().out


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
    assert "══ pipeline one_step: 1 step · 5 → 3 rows" in out
    assert "pipeline warnings:" in out


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


def test_cached_step_skips_which_ops_ran_checks():
    """When a cached step hides its ops, the which-ops-ran warnings are skipped."""

    @pipeline
    def twice(rows):
        return _dedupe_step(_shout_step(rows))

    twice(_rows(*TEXTS))
    first = _tracking.last_pipeline()
    assert first is not None and "no inspect op ran" in first.warnings
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
    assert "── step first: 2 → 1 rows" in out
    assert '      text: "alpha beta"' in out


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


def test_render_int_scores_histogram():
    """Int scores get one column per value and a keep-at line."""
    lines = render_scores(
        ScoreRecord(
            label="score", values=[1, 2, 3, 3, 5, None], keep="ge", threshold=3
        ),
        indent=0,
    )
    assert lines[0].split() == ["score", "1", "2", "3", "5", "unparsed", "1"]
    assert lines[1].split() == ["rows", "1", "1", "2", "1"]
    # Fractions are of all rows; the unparsed row is never kept.
    assert lines[2].split() == ["keep", "at", ">=1", "0.83", ">=2", "0.67"] + [
        "[>=3",
        "0.50]",
        ">=5",
        "0.17",
    ]


def test_render_float_scores_histogram():
    """Float scores in [0, 1] get ten bins and keep-at lines at 0.5..0.9."""
    values = [0.05, 0.15, 0.55, 0.65, 0.72, 0.95]
    lines = render_scores(
        ScoreRecord(label="rouge_l", values=values, keep="lt", threshold=0.7),
        indent=0,
    )
    assert lines[0].split()[1:] == [f"{i / 10:.1f}" for i in range(10)]
    assert lines[1].split() == [
        "rows",
        "1",
        "1",
        "0",
        "0",
        "0",
        "1",
        "1",
        "1",
        "0",
        "1",
    ]
    keep = lines[2]
    assert "<0.5 0.33" in keep and "[<0.7 0.67]" in keep and "<0.9 0.83" in keep


def test_render_float_scores_outside_unit_range_and_between():
    """Wide-ranging scores bin over [min, max]; between shows the kept range."""
    values = list(range(0, 200, 10))
    lines = render_scores(
        ScoreRecord(label="words", values=values, keep="between", lower=20, upper=100),
        indent=0,
    )
    assert lines[0].split()[:3] == ["words", "0", "19"]
    assert lines[-1].split() == ["keep", "[20,", "100]:", "9", "of", "20", "(0.45)"]


def test_render_bool_scores():
    """Boolean scores show kept / dropped counts."""
    lines = render_scores(
        ScoreRecord(label="kept", values=[True, False, True], keep="true"), indent=2
    )
    assert lines == ["  kept 2  dropped 1"]


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
        assert "4,120/38,900" in text and "$0.0412" in text
        assert "min 0.30 ok  (cached)" in text
        assert "keep at" not in text
