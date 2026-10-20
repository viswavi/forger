"""Tests for the suppress_*_warning flags of @step / @pipeline and their guard rails."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pyforger import _tracking, filter, op
from pyforger.acquire.model import Model
from pyforger.alter import generate_from_context
from pyforger.steps import (
    PipelineContractError,
    StepContractError,
    pipeline,
    render_pipeline,
    render_step,
    step,
)

_REASON = "verify drops cut-off drafts"
_DOCS = Path(__file__).resolve().parents[1] / "docs" / "agents" / "pipelines.md"


@pytest.fixture(autouse=True)
def _quiet_and_fresh(monkeypatch):
    """Silence reports and forget earlier step records."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    monkeypatch.delenv("FORGER_SMOKE_ROWS", raising=False)
    _tracking.clear_step_records()
    yield
    _tracking.clear_step_records()


@pytest.fixture
def cut_off():
    """Fake litellm: prompts starting with "long" stop at max_tokens mid-JSON."""

    def completion(messages, n=1, **kwargs):
        long = messages[-1]["content"].startswith("long")
        content, reason = ('{"x": ', "length") if long else ('{"x": 1}', "stop")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, tool_calls=None),
                    finish_reason=reason,
                )
                for _ in range(n)
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=20, completion_tokens_details=None
            ),
            _hidden_params={"response_cost": 0.001},
        )

    with patch("pyforger.acquire.model.litellm") as mock:
        mock.get_supported_openai_params.return_value = None
        mock.completion.side_effect = completion
        yield


def _gen(rows: list[dict]) -> list[dict]:
    model = Model("gpt-4o", show_progress=False, max_tokens=100)
    return generate_from_context(
        rows, "{q}", model, output_schema={"x": "int"}, merge=True
    )


def _qs(*qs: str) -> list[dict]:
    return [{"q": q} for q in qs]


@op(name="alter.test_suppress_copy", category="alter")
def _copy(rows: list[dict]) -> list[dict]:
    return [dict(r) for r in rows]


class TestGuardRails:
    """Flags are checked when the step or pipeline is defined."""

    def test_typo_is_a_type_error_with_the_closest_flag(self):
        """A misspelled flag never suppresses silently."""
        with pytest.raises(
            TypeError, match="did you mean 'suppress_truncation_warning'"
        ):
            step(suppress_truncaton_warning=_REASON)

    def test_flag_of_the_other_decorator_says_where(self):
        """A pipeline flag on a step, and a step flag on a pipeline, raise."""
        with pytest.raises(TypeError, match="is a @pipeline flag"):
            step(suppress_no_inspect_warning=_REASON)
        with pytest.raises(TypeError, match="is a @step flag"):
            pipeline(suppress_truncation_warning=_REASON)

    def test_unsuppressible_kind_lists_the_flags(self):
        """Parse failures and yield misses have a fix, so there is no flag."""
        with pytest.raises(TypeError, match="only the problems below") as err:
            step(suppress_parse_failure_warning=_REASON)
        assert "suppress_truncation_warning, suppress_no_ops_warning" in str(err.value)

    @pytest.mark.parametrize("reason", ["", "  ", "ok", "known issue", 3])
    def test_short_reason_raises(self, reason):
        """A reason must say why the rows are still fine."""
        with pytest.raises(ValueError, match="needs a reason of at least 3 words"):
            step(suppress_truncation_warning=reason)

    def test_docs_reasons_are_rejected(self):
        """Every reason the docs show is a placeholder that cannot be copied."""
        reasons = re.findall(r'suppress_\w+_warning="([^"]*)"', _DOCS.read_text())
        assert reasons
        for reason in reasons + ["long proofs are filtered by length next"]:
            with pytest.raises(ValueError, match="example reason"):
                step(suppress_truncation_warning=reason)


class TestStepSuppression:
    """A suppressed problem leaves the warnings but keeps its size."""

    def test_truncation_is_suppressed_with_its_size(self, cut_off):
        """One dim line with the size and reason; strict=True passes."""

        @step(suppress_truncation_warning=_REASON, strict=True)
        def draft(rows: list[dict]) -> list[dict]:
            return _gen(rows)

        assert len(draft(_qs("long a", "b", "c", "long d"))) == 2
        rec = _tracking.step_records()["draft"]
        assert rec.warnings == []
        [hidden] = rec.suppressed
        assert (hidden.kind, hidden.flag, hidden.reason) == (
            "truncation",
            "suppress_truncation_warning",
            _REASON,
        )
        assert hidden.sizes == ["2 of 4 completions cut off, 2 rows dropped"]
        assert "stopped at max_tokens=100" in hidden.messages[0]
        text = render_step(rec)
        assert (
            "suppressed: truncation in draft (2 of 4 completions cut off, 2 rows "
            f"dropped) — {_REASON}"
        ) in text
        assert "⚠" not in text

    def test_cached_step_keeps_its_suppression(self, cut_off):
        """Kinds and sizes are stored with the cached ops."""

        @step(suppress_truncation_warning=_REASON)
        def draft(rows: list[dict]) -> list[dict]:
            return _gen(rows)

        draft(_qs("long a", "b"))
        draft(_qs("long a", "b"))
        rec = _tracking.step_records()["draft"]
        assert rec.cache_hit and rec.warnings == []
        assert rec.suppressed[0].sizes == ["1 of 2 completions cut off, 1 row dropped"]

    def test_other_problems_still_warn(self):
        """The flag hides truncation only; a yield miss still fails strict."""

        @step(suppress_truncation_warning=_REASON, strict=True)
        def keep_even(rows: list[dict]) -> list[dict]:
            return filter.by_predicate(rows, lambda r: r["i"] % 2 == 0, min_yield=0.9)

        with pytest.raises(StepContractError, match="below min_yield"):
            keep_even([{"i": i} for i in range(4)])

    def test_no_ops_suggestion(self):
        """suppress_no_ops_warning hides "called no Forger operations"."""

        @step(
            suppress_no_ops_warning="a pure-Python reshuffle of fields, no rows change"
        )
        def reshape(rows: list[dict]) -> list[dict]:
            return [{"text": r["q"]} for r in rows]

        reshape(_qs("a"))
        rec = _tracking.step_records()["reshape"]
        assert rec.suggestions == []
        assert [(x.kind, x.warning) for x in rec.suppressed] == [
            ("no_forger_ops", False)
        ]

    def test_stale_flag_outside_a_pipeline_is_a_warning(self):
        """A data-problem flag that hid nothing warns, so it cannot hide a later one."""

        @step(suppress_truncation_warning=_REASON)
        def copy(rows: list[dict]) -> list[dict]:
            return _copy(rows)

        copy(_qs("a"))
        rec = _tracking.step_records()["copy"]
        assert rec.warnings == [
            "suppress_truncation_warning is set on copy, but no truncation occurred; "
            "remove it so it cannot hide a later one"
        ]


class TestPipelineSuppression:
    """Pipeline flags, the summary's lists, and stale flags."""

    def test_summary_names_the_flags_of_open_problems(self, cut_off):
        """One dim line after the data problems names each open problem's flag."""

        @step
        def draft(rows: list[dict]) -> list[dict]:
            return _gen(rows)

        @step
        def keep(rows: list[dict]) -> list[dict]:
            return filter.by_exact_match(rows, field="q")

        @pipeline
        def run():
            return keep(draft(_qs("long a", "b")))

        run()
        rec = _tracking.last_pipeline()
        lines = [line.strip() for line in render_pipeline(rec).strip().splitlines()]
        start = next(i for i, line in enumerate(lines) if line.startswith("Fix these"))
        last = " ".join(
            lines[start:]
        )  # the hint wraps at 120 columns, and ends the summary
        assert "@step(suppress_truncation_warning=...) on draft" in last
        # A suggestion names its flag on its own line, not in the closing hint.
        assert "suppress_no_inspect_warning" not in last
        assert (
            "suggestion: no inspect op ran, so nothing looked at what the rows cover "
            "(e.g. inspect.topic_based_clustering) (does not apply? say why with "
            "@pipeline(suppress_no_inspect_warning=...))"
        ) in " ".join(lines[:start])

    def test_no_hint_without_flaggable_problems(self):
        """Only problems without a flag (a yield miss) are open: no hint line."""
        from pyforger._render import _suppress_hint
        from pyforger._tracking import PipelineRecord, StepRecord

        miss = "filter.by_predicate: yield 0.50 below min_yield 0.90"
        step_rec = StepRecord(
            name="keep", warnings=[miss], kinds={miss: "yield_bounds"}
        )
        assert _suppress_hint(PipelineRecord(name="run", steps=[step_rec])) is None

    def test_summary_lists_suppressions_last(self, cut_off):
        """The last block counts open and suppressed problems and lists both."""

        @step(suppress_truncation_warning=_REASON)
        def draft(rows: list[dict]) -> list[dict]:
            return _gen(rows)

        @step
        def keep(rows: list[dict]) -> list[dict]:
            return filter.by_predicate(rows, lambda r: r["q"] == "b", min_yield=0.9)

        @pipeline(
            suppress_no_diversity_filter_warning="one row per seed question, no repeats"
        )
        def run():
            return keep(draft(_qs("long a", "b", "c")))

        run()
        rec = _tracking.last_pipeline()
        assert not any("no diversity filter" in s for s in rec.suggestions)
        lines = [line.strip() for line in render_pipeline(rec).splitlines()]
        assert (
            "suppressed: no_diversity_filter in pipeline run — one row per seed "
            "question, no repeats"
        ) in lines
        end = lines.index("data problems: 1 open, 1 suppressed")
        assert lines[end + 1].startswith("⚠ keep: filter.by_predicate: yield 0.50")
        assert lines[end + 2] == (
            "suppressed: truncation in draft (1 of 3 completions cut off, 1 row "
            f"dropped) — {_REASON}"
        )

    def test_stale_flags(self):
        """A stale data-problem flag warns, a stale suggestion flag suggests."""

        @step(suppress_truncation_warning=_REASON)
        def copy(rows: list[dict]) -> list[dict]:
            return _copy(rows)

        @step
        def keep(rows: list[dict]) -> list[dict]:
            return filter.by_exact_match(rows, field="q")

        @pipeline(
            suppress_no_diversity_filter_warning="one row per seed, no repeats",
            strict=True,
        )
        def run():
            copy(_qs("a"))
            return keep(copy(_qs("b")))

        with pytest.raises(PipelineContractError, match="no truncation occurred"):
            run()
        rec = _tracking.last_pipeline()
        assert rec.warnings == [
            "suppress_truncation_warning is set on copy, but no truncation occurred; "
            "remove it so it cannot hide a later one"
        ]
        assert (
            "suppress_no_diversity_filter_warning is set on pipeline run, but a "
            "diversity filter ran; remove it so it cannot hide a later one"
        ) in rec.suggestions

    def test_one_run_of_a_step_is_enough(self, cut_off):
        """A step run several times is stale only when no run truncated."""

        @step(suppress_truncation_warning=_REASON)
        def draft(rows: list[dict]) -> list[dict]:
            return _gen(rows)

        @pipeline
        def run():
            draft(_qs("b"))
            return draft(_qs("long a", "c"))

        run()
        rec = _tracking.last_pipeline()
        assert not any("is set on" in m for m in rec.warnings + rec.suggestions)

    def test_smoke_run_checks_no_stale_flags(self):
        """3 rows rarely show a problem, so a smoke run never calls a flag stale."""

        @step(suppress_truncation_warning=_REASON)
        def copy(rows: list[dict]) -> list[dict]:
            return _copy(rows)

        @pipeline(smoke=2)
        def run():
            return copy(_qs("a", "b", "c"))

        run()
        rec = _tracking.last_pipeline()
        assert not any("is set on" in m for m in rec.warnings + rec.suggestions)


class TestKinds:
    """Every warning and suggestion carries its kind from where it was made."""

    def test_op_error_and_yield_kinds(self):
        """An op that raised inside a step, and a yield miss, have kinds."""

        @op(name="alter.test_suppress_boom", category="alter")
        def boom(rows: list[dict]) -> list[dict]:
            raise ValueError("bad row")

        @step
        def s(rows: list[dict]) -> list[dict]:
            try:
                boom(rows)
            except ValueError:
                pass
            return filter.by_predicate(rows, lambda r: r["i"] % 2 == 0, min_yield=0.9)

        s([{"i": i} for i in range(4)])
        rec = _tracking.step_records()["s"]
        assert sorted(rec.kinds[w] for w in rec.warnings) == [
            "op_error",
            "yield_bounds",
        ]

    def test_op_body_warning_kind(self):
        """add_op_warning passes its kind to the op's report."""

        @op(name="alter.test_suppress_warns", category="alter")
        def warns(rows: list[dict]) -> list[dict]:
            _tracking.add_op_warning(
                "2 of 4 judge calls gave no score", "judge_unscored"
            )
            return rows

        @step
        def s(rows: list[dict]) -> list[dict]:
            return warns(rows)

        s(_qs("a"))
        rec = _tracking.step_records()["s"]
        assert rec.kinds == {"2 of 4 judge calls gave no score": "judge_unscored"}
