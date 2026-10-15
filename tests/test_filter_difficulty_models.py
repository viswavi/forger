"""Tests for the model-based difficulty filters (comparative and pass@k)."""
from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import StrictBool, TypeAdapter

from pyforger import _tracking
from pyforger._render import render_scores
from pyforger.acquire.model import GenerateResult, Model
from pyforger.filter import by_comparative_difficulty, by_pass_at_k
from pyforger.filter.model_difficulty import _default_extract
from pyforger.steps import pipeline, step

# A copy of filter.by_success_rate's input contract, rate math, and selection
# rule, so these tests fail if the outcomes we write stop being re-selectable by
# it or our decisions drift from its decisions.
_OUTCOMES = TypeAdapter(dict[str, list[StrictBool | None]])


def _success_rate(grades: list[bool | None]) -> float | None:
    valid = sum(value is not None for value in grades)
    return sum(value is True for value in grades) / valid if valid else None


def _success_rate_keeps(
    outcomes: dict, min_rate: float, max_rate: float, min_valid: int | None
) -> bool:
    groups = _OUTCOMES.validate_python(outcomes)
    return bool(groups) and all(
        (rate := _success_rate(grades)) is not None
        and sum(g is not None for g in grades) >= (min_valid or len(grades))
        and min_rate <= rate <= max_rate
        for grades in groups.values()
    )


class ScriptedModel(Model):
    r"""Fake model answering each task with scripted completions, cycled to ``n``.

    The task is the text after ``Task:\n`` in the default prompt (or the whole
    prompt). Counts generate_batch calls and records each call's ``n`` and size.
    """

    def __init__(self, name: str, script: dict[str, list[str]]) -> None:
        """Store the script; ``name`` keeps two fakes' cache keys apart."""
        super().__init__(model_id=f"test/{name}")
        self.name = name
        self.script = script
        self.calls: list[tuple[int, int]] = []  # (n, prompts) per call
        self.fail = False  # raise from generate_batch, e.g. a provider outage

    def cache_key(self) -> tuple:
        """Exclude the call log from the @op cache key."""
        return ("ScriptedModel", self.name, tuple(sorted(self.script.items())))

    def generate_batch(  # type: ignore[override]
        self, prompts: list[Any], *, n: int = 1, **kwargs: Any
    ) -> list[GenerateResult]:
        """Return the scripted completions for each prompt."""
        self.calls.append((n, len(prompts)))
        if self.fail:
            raise RuntimeError("provider unavailable")
        out = []
        for prompt in prompts:
            completions = self.script[prompt.rsplit("Task:\n", 1)[-1]]
            out.append(
                GenerateResult(
                    [completions[i % len(completions)] for i in range(n)],
                    usage={},
                    reasoning=[None] * n,
                    model_id=self._model_id,
                    raw=None,
                )
            )
        return out


ROWS = [
    {"id": "hard", "task": "t1", "answer": "1"},
    {"id": "easy", "task": "t2", "answer": "2"},
    {"id": "too_hard", "task": "t3", "answer": "3"},
    {"id": "inverted", "task": "t4", "answer": "4"},
    {"id": "empty", "task": "t5", "answer": "5"},
]
WEAK = {
    "t1": ["ANSWER: 9"],
    "t2": ["ANSWER: 2"],
    "t3": ["ANSWER: 9"],
    "t4": ["ANSWER: 4"],
    "t5": ["ANSWER: 9"],
}
STRONG = {
    "t1": ["ANSWER: 1"],
    "t2": ["ANSWER: 2"],
    "t3": ["ANSWER: 9"],
    "t4": ["ANSWER: 9"],
    "t5": [""],
}


def _comparative(rows=ROWS, weak=None, strong=None, **kwargs):
    return by_comparative_difficulty(
        rows,
        weak_model=weak or ScriptedModel("weak", WEAK),
        strong_model=strong or ScriptedModel("strong", STRONG),
        input_field="task",
        ground_truth_field="answer",
        **kwargs,
    )


def _pass_at_k(rows, model, **kwargs):
    return by_pass_at_k(
        rows, model=model, input_field="task", ground_truth_field="answer", **kwargs
    )


class TestComparative:
    """by_comparative_difficulty keeps rows only the strong model solves."""

    def test_keeps_strong_right_weak_wrong(self):
        """Easy, too-hard, inverted, and ungraded rows are all dropped."""
        assert [r["id"] for r in _comparative()] == ["hard"]

    def test_one_prediction_per_model(self):
        """Each model gets one generate_batch call with n=1 for all rows."""
        weak, strong = ScriptedModel("weak", WEAK), ScriptedModel("strong", STRONG)
        _comparative(weak=weak, strong=strong)
        assert weak.calls == strong.calls == [(1, len(ROWS))]

    def test_scores_in_step(self, monkeypatch):
        """The step report gets strong - weak per row, None when ungraded."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")

        @step
        def compare(rows):
            return _comparative(rows)

        compare(ROWS)
        scores = _tracking.step_records()["compare"].ops[0].scores
        assert scores is not None
        assert (scores.keep, scores.lower, scores.upper) == ("between", 1, 1)
        assert scores.values == [1, 0, 0, -1, None]
        assert "keeps 1 to 1: 1 of 5" in render_scores(scores)

    def test_outcomes_field(self):
        """Kept rows carry weak/strong grades in by_success_rate's format."""
        assert _comparative() == [ROWS[0]]
        kept = _comparative(outcomes_field="evaluations")
        assert kept == [{**ROWS[0], "evaluations": {"weak": [False], "strong": [True]}}]
        _OUTCOMES.validate_python(kept[0]["evaluations"])
        assert "evaluations" not in ROWS[0]

    def test_outcomes_field_replaces_existing_value(self):
        """An existing value under outcomes_field is replaced, not merged."""
        rows = [{**ROWS[0], "evaluations": {"other": [True]}}]
        kept = _comparative(rows, outcomes_field="evaluations")
        assert kept[0]["evaluations"] == {"weak": [False], "strong": [True]}

    def test_weak_without_answer_is_not_hard(self):
        """A weak completion with no answer (cut off mid-thought) drops the row."""
        weak = ScriptedModel("weak", {"t": ["<think>Let me try 9, then"]})
        strong = ScriptedModel("strong", {"t": ["ANSWER: 1"]})
        rows = [{"task": "t", "answer": "1"}]
        assert _comparative(rows, weak, strong) == []
        weak = ScriptedModel("weak", {"t": ["<think>hm</think>ANSWER: 9"]})
        assert _comparative(rows, weak, strong) == rows

    def test_model_error_propagates_and_is_not_cached(self):
        """A failing model call raises; the next call samples again."""
        strong = ScriptedModel("strong", STRONG)
        weak = ScriptedModel("weak", WEAK)
        weak.fail = True
        with pytest.raises(RuntimeError):
            _comparative(weak=weak, strong=strong)
        weak.fail = False
        assert [r["id"] for r in _comparative(weak=weak, strong=strong)] == ["hard"]
        assert len(weak.calls) == 2

    def test_boxed_answer_line(self):
        r"""'ANSWER: \boxed{1}' matches the reference 1."""
        weak = ScriptedModel("weak", {"t": ["ANSWER: \\boxed{9}"]})
        strong = ScriptedModel("strong", {"t": ["ANSWER: $\\boxed{1}$."]})
        rows = [{"task": "t", "answer": "1"}]
        assert _comparative(rows, weak, strong) == rows

    def test_threshold_change_makes_no_model_calls(self):
        """Changing only rouge_l_threshold regrades cached predictions."""
        weak = ScriptedModel("weak", {"t": ["ANSWER: the cat sat"]})
        strong = ScriptedModel("strong", {"t": ["ANSWER: the cat sat on the mat"]})
        rows = [{"task": "t", "answer": "the cat sat on the mat"}]
        assert _comparative(rows, weak, strong, metric="rouge_l") == rows
        assert (
            _comparative(rows, weak, strong, metric="rouge_l", rouge_l_threshold=0.5)
            == []
        )
        assert len(weak.calls) == len(strong.calls) == 1


class TestPassAtK:
    """by_pass_at_k keeps rows whose pass rate over k samples is in bounds."""

    SCRIPT = {
        "t1": ["ANSWER: 1"],  # 4/4
        "t2": ["ANSWER: 2", "ANSWER: 0", "ANSWER: 0", "ANSWER: 0"],  # 1/4
        "t3": ["ANSWER: 0"],  # 0/4
        "t4": ["ANSWER: 4", "ANSWER: 4", "ANSWER: 4", "ANSWER: 0"],  # 3/4
    }
    ROWS = [{"task": f"t{i}", "answer": str(i)} for i in range(1, 5)]

    def test_default_bounds(self):
        """Rows never or always solved are dropped by the 0.1-0.9 default."""
        kept = _pass_at_k(self.ROWS, ScriptedModel("m", self.SCRIPT))
        assert [r["task"] for r in kept] == ["t2", "t4"]

    def test_bounds_are_inclusive(self):
        """Rates equal to min_rate or max_rate are kept."""
        model = ScriptedModel("m", self.SCRIPT)
        kept = _pass_at_k(self.ROWS, model, min_rate=0.25, max_rate=0.25)
        assert [r["task"] for r in kept] == ["t2"]
        kept = _pass_at_k(self.ROWS, model, min_rate=0.75, max_rate=1.0)
        assert [r["task"] for r in kept] == ["t1", "t4"]

    def test_one_call_with_n_equal_k(self):
        """All prompts go out in one generate_batch call with n=k."""
        model = ScriptedModel("m", self.SCRIPT)
        _pass_at_k(self.ROWS, model, k=4)
        assert model.calls == [(4, len(self.ROWS))]

    def test_batches_prompts(self):
        """batch_size splits prompts across calls, each still with n=k."""
        model = ScriptedModel("m", self.SCRIPT)
        _pass_at_k(self.ROWS, model, k=2, batch_size=3)
        assert model.calls == [(2, 3), (2, 1)]

    def test_rate_changes_make_no_model_calls(self):
        """New min_rate / max_rate / min_valid reuse the cached samples."""
        model = ScriptedModel("m", self.SCRIPT)
        _pass_at_k(self.ROWS, model)
        _pass_at_k(self.ROWS, model, min_rate=0.0, max_rate=1.0)
        _pass_at_k(self.ROWS, model, min_valid=2)
        assert len(model.calls) == 1

    def test_no_cache_samples_again(self):
        """_no_cache=True reaches the inner sampling op."""
        model = ScriptedModel("m", self.SCRIPT)
        _pass_at_k(self.ROWS, model)
        _pass_at_k(self.ROWS, model, _no_cache=True)
        assert len(model.calls) == 2

    def test_metric_change_does_not_regenerate(self):
        """Switching metric regrades the cached samples."""
        model = ScriptedModel("m", self.SCRIPT)
        _pass_at_k(self.ROWS, model)
        kept = _pass_at_k(self.ROWS, model, metric=lambda pred, row: pred == "0")
        assert [r["task"] for r in kept] == ["t2", "t4"]  # rates 3/4 and 1/4
        assert len(model.calls) == 1

    def test_none_grades_excluded_and_min_valid(self):
        """Empty samples are excluded; all k must be valid unless min_valid allows."""
        model = ScriptedModel("m", {"t": ["ANSWER: 1", "", "ANSWER: 0", "ANSWER: 0"]})
        rows = [{"task": "t", "answer": "1"}]
        assert _pass_at_k(rows, model) == []
        assert _pass_at_k(rows, model, min_valid=3) == rows  # rate 1/3
        assert _pass_at_k(rows, model, min_valid=3, max_rate=0.3) == []

    def test_no_valid_grades_never_pass(self):
        """A row whose samples are all empty fails even with [0, 1] bounds."""
        model = ScriptedModel("m", {"t": [""]})
        rows = [{"task": "t", "answer": "1"}]
        assert _pass_at_k(rows, model, min_rate=0.0, max_rate=1.0, min_valid=1) == []

    def test_unanswered_sample_is_excluded(self):
        """A sample with no extractable answer grades None, like an empty one."""
        model = ScriptedModel("m", {"t": ["ANSWER: 1", "I don't know"]})
        rows = [{"task": "t", "answer": "1"}]
        kept = _pass_at_k(rows, model, k=2, outcomes_field="evaluations")
        assert kept == []
        kept = _pass_at_k(
            rows, model, k=2, min_valid=1, max_rate=1.0, outcomes_field="evaluations"
        )
        assert kept[0]["evaluations"] == {"model": [True, None]}

    def test_whole_completion_for_verifier(self):
        """extract_answer=lambda t: t hands the verifier the raw completion."""
        code = "def f():\n    return 1"
        model = ScriptedModel("m", {"t": [code]})
        rows = [{"task": "t", "answer": "1"}]
        kept = _pass_at_k(
            rows,
            model,
            k=1,
            metric=lambda prediction, row: prediction == code,
            extract_answer=lambda text: text,
            min_rate=1.0,
            max_rate=1.0,
        )
        assert kept == rows

    def test_rouge_l_threshold(self):
        """rouge_l passes at F1 >= rouge_l_threshold."""
        model = ScriptedModel("m", {"t": ["ANSWER: the cat sat"]})
        rows = [{"task": "t", "answer": "the cat sat on the mat"}]  # F1 2/3
        kw = dict(k=1, metric="rouge_l", min_rate=1.0, max_rate=1.0)
        assert _pass_at_k(rows, model, **kw) == []
        assert _pass_at_k(rows, model, rouge_l_threshold=0.6, **kw) == rows
        assert len(model.calls) == 1

    def test_custom_verifier(self):
        """A verifier gets the extracted prediction and the row; None excludes."""
        seen: list[tuple[str, str]] = []

        def verify(prediction, row):
            seen.append((prediction, row["task"]))
            return None if prediction == "skip" else prediction == row["answer"]

        model = ScriptedModel("m", {"t": ["ANSWER: yes", "ANSWER: skip", "ANSWER: no"]})
        rows = [{"task": "t", "answer": "yes"}]
        kept = _pass_at_k(
            rows, model, k=3, metric=verify, min_valid=2, outcomes_field="evaluations"
        )
        assert kept[0]["evaluations"] == {"model": [True, None, False]}
        assert seen == [("yes", "t"), ("skip", "t"), ("no", "t")]

    def test_verifier_must_return_bool(self):
        """A non-boolean grade is rejected, as by_success_rate would."""
        model = ScriptedModel("m", {"t": ["ANSWER: 1"]})
        with pytest.raises(TypeError):
            _pass_at_k([{"task": "t", "answer": "1"}], model, metric=lambda p, r: 1)

    def test_outcomes_field_matches_success_rate(self):
        """Kept rows' outcomes give by_success_rate the same rate and decision."""
        model = ScriptedModel("m", self.SCRIPT)
        kept = _pass_at_k(self.ROWS, model, outcomes_field="evaluations")
        assert [r["evaluations"] for r in kept] == [
            {"model": [True, False, False, False]},
            {"model": [True, True, True, False]},
        ]
        for row in kept:
            outcomes = _OUTCOMES.validate_python(row["evaluations"])
            rate = _success_rate(outcomes["model"])
            assert rate is not None and 0.1 <= rate <= 0.9
        unchanged = _pass_at_k(self.ROWS, model)
        assert unchanged == [self.ROWS[1], self.ROWS[3]]

    @pytest.mark.parametrize(
        "min_rate,max_rate,min_valid",
        [(0.1, 0.9, None), (0.0, 1.0, None), (0.5, 1.0, 2), (0.0, 0.0, 3), (1, 1, 1)],
    )
    def test_same_decisions_as_success_rate(self, min_rate, max_rate, min_valid):
        """by_success_rate on the stored grades keeps exactly the same rows."""
        script = {
            **self.SCRIPT,
            "t5": ["ANSWER: 5", "", "ANSWER: 0", "ANSWER: 5"],  # 2/3, one excluded
            "t6": ["ANSWER: 0", "", "", "ANSWER: 0"],  # 0/2, two excluded
            "t7": [""],  # nothing valid
        }
        rows = [{"task": f"t{i}", "answer": str(i)} for i in range(1, 8)]
        model = ScriptedModel("m", script)
        bounds = dict(min_rate=min_rate, max_rate=max_rate, min_valid=min_valid)
        kept = _pass_at_k(rows, model, **bounds)
        # All grades, attached to every row whose grades are not all None.
        graded = _pass_at_k(
            rows, model, min_rate=0.0, max_rate=1.0, min_valid=1, outcomes_field="ev"
        )
        assert len(graded) == 6
        reselected = [
            {k: v for k, v in row.items() if k != "ev"}
            for row in graded
            if _success_rate_keeps(row["ev"], **bounds)
        ]
        assert kept == reselected
        assert len(model.calls) == 1

    def test_scores_in_step(self, monkeypatch):
        """The step report gets the pass rate per row and the inclusive bounds."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        model = ScriptedModel("m", self.SCRIPT)

        @step
        def hard(rows):
            return _pass_at_k(rows, model, min_rate=0.25, max_rate=0.75)

        hard(self.ROWS)
        scores = _tracking.step_records()["hard"].ops[0].scores
        assert scores is not None
        assert (scores.keep, scores.lower, scores.upper) == ("between", 0.25, 0.75)
        assert scores.values == [1.0, 0.25, 0.0, 0.75]
        assert "keeps 0.25 to 0.75: 2 of 4" in render_scores(scores)

    def test_counts_as_quality_filter(self, monkeypatch):
        """A pipeline running by_pass_at_k gets no 'no quality filter' warning."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        model = ScriptedModel("m", self.SCRIPT)

        @step
        def hard(rows):
            return _pass_at_k(rows, model)

        @pipeline
        def run(rows):
            return hard(rows)

        run(self.ROWS)
        warnings = _tracking.last_pipeline().warnings  # type: ignore[union-attr]
        assert not any("quality filter" in w for w in warnings)

    @pytest.mark.parametrize(
        "change",
        [
            {"k": 2},
            {"temperature": 0.5},
            {"max_tokens": 64},
            {"prompt_template": "Task:\n{task}"},
        ],
    )
    def test_sampling_changes_regenerate(self, change):
        """A new k, temperature, max_tokens, or prompt samples again."""
        model = ScriptedModel("m", self.SCRIPT)
        _pass_at_k(self.ROWS, model)
        _pass_at_k(self.ROWS, model, **change)
        assert len(model.calls) == 2

    def test_model_change_regenerates(self):
        """Another model is sampled, not served the first model's samples."""
        first, second = ScriptedModel("a", self.SCRIPT), ScriptedModel("b", self.SCRIPT)
        _pass_at_k(self.ROWS, first)
        _pass_at_k(self.ROWS, second)
        assert len(first.calls) == len(second.calls) == 1

    def test_empty_input(self):
        """No rows: no model call, no rows out, an empty score list."""
        model = ScriptedModel("m", {})
        assert _pass_at_k([], model) == []
        assert model.calls == []

    def test_order_duplicates_and_no_mutation(self):
        """Kept rows keep input order, duplicates are graded each, inputs unchanged."""
        rows = [self.ROWS[3], self.ROWS[1], self.ROWS[3], self.ROWS[0]]
        before = copy.deepcopy(rows)
        kept = _pass_at_k(rows, ScriptedModel("m", self.SCRIPT), outcomes_field="ev")
        assert [r["task"] for r in kept] == ["t4", "t2", "t4"]
        assert rows == before

    def test_fewer_completions_than_k(self, caplog):
        """Missing samples (e.g. a provider ignoring n) grade None, with a warning."""

        class OneChoice(ScriptedModel):
            def generate_batch(self, prompts, *, n=1, **kwargs):
                return super().generate_batch(prompts, n=1, **kwargs)

        model = OneChoice("one", {"t": ["ANSWER: 1"]})
        rows = [{"task": "t", "answer": "1"}]
        kept = _pass_at_k(
            rows, model, k=3, min_valid=1, max_rate=1.0, outcomes_field="ev"
        )
        assert kept[0]["ev"] == {"model": [True, None, None]}
        assert "1 of 1 prompts got fewer than n=3 completions" in caplog.text

    def test_custom_prompt_template(self):
        """prompt_template replaces the default prompt."""
        model = ScriptedModel("m", {"Q: t1": ["ANSWER: 1"]})
        rows = [{"task": "t1", "answer": "1"}]
        kept = _pass_at_k(
            rows, model, k=1, prompt_template="Q: {task}", min_rate=1.0, max_rate=1.0
        )
        assert kept == rows


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"min_rate": -0.1},
        {"max_rate": 1.5},
        {"min_rate": 0.8, "max_rate": 0.2},
        {"min_rate": float("nan")},
        {"min_valid": 0},
        {"min_valid": 5},
        {"temperature": 0.0},
        {"metric": "bleu"},
        {"rouge_l_threshold": 1.5},
        {"batch_size": 0},
    ],
)
def test_pass_at_k_validation(kwargs):
    """Bad bounds, k, temperature, or metric raise before any model call."""
    model = ScriptedModel("m", {})
    with pytest.raises(ValueError):
        _pass_at_k([{"task": "t", "answer": "1"}], model, **kwargs)
    assert model.calls == []


def test_greedy_single_sample_allowed():
    """temperature=0 is fine with k=1."""
    model = ScriptedModel("m", {"t": ["ANSWER: 1"]})
    rows = [{"task": "t", "answer": "1"}]
    assert _pass_at_k(rows, model, k=1, temperature=0.0, max_rate=1.0) == rows


@pytest.mark.parametrize(
    "kwargs", [{"metric": "bleu"}, {"rouge_l_threshold": -0.1}, {"batch_size": 0}]
)
def test_comparative_validation(kwargs):
    """Bad metric settings or batch size raise before any model call."""
    weak, strong = ScriptedModel("weak", WEAK), ScriptedModel("strong", STRONG)
    with pytest.raises(ValueError):
        _comparative(weak=weak, strong=strong, **kwargs)
    assert weak.calls == strong.calls == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Work... ANSWER: Paris.", "Paris"),
        ("ANSWER: $1,000$", "1,000"),
        ("so the result is \\boxed{42}", "42"),
        ("<think>ANSWER: 3</think>Final ANSWER: 4", "4"),
        ("no answer here", None),
        ("**ANSWER:** 42", "42"),
        ("ANSWER:\n42\n\nHope this helps.", "42"),
        ("<think>so ANSWER: 3, but wait", None),
        ("ANSWER: \\boxed{42}", "42"),
        ("ANSWER: $\\boxed{\\frac{1}{2}}$.", "\\frac{1}{2}"),
        ("ANSWER: \\boxed{1} or \\boxed{2}", "\\boxed{1} or \\boxed{2}"),
    ],
)
def test_default_extract(text, expected):
    """The default extractor reads the final ANSWER: line, else a number."""
    assert _default_extract(text) == expected
