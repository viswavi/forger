"""Tests for the model-based difficulty filters (comparative and pass@k)."""
from __future__ import annotations

import copy
import threading
import time
from typing import Any

import pytest
from pydantic import StrictBool, TypeAdapter

from pyforger import _tracking
from pyforger._render import render_scores, render_step
from pyforger.acquire.model import GenerateResult, Model
from pyforger.filter import by_comparative_difficulty, by_pass_at_k, program_verifier
from pyforger.filter.model_difficulty import answer_segment
from pyforger.op import op, trace
from pyforger.steps import pipeline, step

# An independent copy of the outcomes contract, rate math, and selection rule,
# so these tests fail if the outcomes we write stop being re-selectable by
# pass rate or our decisions drift from that rule.
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
    """Fake model answering each task with scripted completions, cycled to ``n``.

    The task is the prompt with any ``"Q: "`` prefix removed. Records each
    prompt, and each generate_batch call's ``n`` and size.
    """

    def __init__(self, name: str, script: dict[str, list[str]]) -> None:
        """Store the script; ``name`` keeps two fakes' cache keys apart."""
        super().__init__(model_id=f"test/{name}")
        self.name = name
        self.script = script
        self.calls: list[tuple[int, int]] = []  # (n, prompts) per call
        self.prompts: list[Any] = []
        self.fail = False  # raise from generate_batch, e.g. a provider outage

    def cache_key(self) -> tuple:
        """Exclude the call log from the @op cache key."""
        return ("ScriptedModel", self.name, tuple(sorted(self.script.items())))

    def completions(self, prompt: Any, n: int) -> list[str]:
        """Return ``n`` scripted completions for one prompt."""
        scripted = self.script[prompt.removeprefix("Q: ")]
        return [scripted[i % len(scripted)] for i in range(n)]

    def generate_batch(  # type: ignore[override]
        self, prompts: list[Any], *, n: int = 1, **kwargs: Any
    ) -> list[GenerateResult]:
        """Return the scripted completions for each prompt."""
        self.calls.append((n, len(prompts)))
        self.prompts.extend(prompts)
        if self.fail:
            raise RuntimeError("provider unavailable")
        out = []
        for prompt in prompts:
            completions = self.completions(prompt, n)
            out.append(
                GenerateResult(
                    completions,
                    usage={},
                    reasoning=[None] * len(completions),
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
WEAK = {"t1": ["9"], "t2": ["2"], "t3": ["9"], "t4": ["4"], "t5": ["9"]}
STRONG = {"t1": ["1"], "t2": ["2"], "t3": ["9"], "t4": ["9"], "t5": [""]}


def _comparative(rows=ROWS, weak=None, strong=None, **kwargs):
    return by_comparative_difficulty(
        rows,
        weak_model=weak or ScriptedModel("weak", WEAK),
        strong_model=strong or ScriptedModel("strong", STRONG),
        input_field="task",
        ground_truth_field="answer",
        **kwargs,
    )


def _answer_line(text):
    """Keep what follows the last ANSWER: after the thinking, as a caller might."""
    segment = answer_segment(text)
    if segment is None or "ANSWER:" not in segment:
        return None
    return segment.rsplit("ANSWER:", 1)[1].strip().replace(",", "")


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

    def test_default_prompt_is_the_task(self):
        """Without prompt_template, each model is sent the task text unchanged."""
        weak, strong = ScriptedModel("weak", WEAK), ScriptedModel("strong", STRONG)
        _comparative(weak=weak, strong=strong)
        assert weak.prompts == strong.prompts == [r["task"] for r in ROWS]

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
        """Kept rows carry weak/strong grades as {group: [grade]}."""
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
        """A weak completion cut off mid-thought grades None and drops the row."""
        weak = ScriptedModel("weak", {"t": ["<think>Let me try 9, then"]})
        strong = ScriptedModel("strong", {"t": ["1"]})
        rows = [{"task": "t", "answer": "1"}]
        assert _comparative(rows, weak, strong) == []
        weak = ScriptedModel("weak", {"t": ["<think>hm</think>9"]})
        assert _comparative(rows, weak, strong) == rows

    def test_truncated_fail_counts_weak_as_wrong(self):
        """truncated="fail" grades a weak completion cut off mid-thought False."""
        weak = ScriptedModel("weak", {"t": ["<think>Let me try 9, then"]})
        strong = ScriptedModel("strong", {"t": ["1"]})
        rows = [{"task": "t", "answer": "1"}]
        kept = _comparative(rows, weak, strong, truncated="fail", outcomes_field="ev")
        assert kept == [{**rows[0], "ev": {"weak": [False], "strong": [True]}}]

    def test_truncation_note_in_scores(self, monkeypatch):
        """The step report says how many samples were truncated, per model."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        weak = ScriptedModel("weak", {"t": ["<think>Let me try 9, then"]})
        strong = ScriptedModel("strong", {"t": ["1"]})

        @step
        def hard(rows):
            return _comparative(rows, weak, strong)

        hard([{"task": "t", "answer": "1"}])
        scores = _tracking.step_records()["hard"].ops[0].scores
        assert scores is not None
        assert scores.note is not None
        assert scores.note.startswith("weak: 1 of 1 samples truncated")
        assert "excluded" in scores.note
        assert "weak: 1 of 1 samples truncated" in render_scores(scores)

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

    def test_extract_answer_opt_in(self):
        """ANSWER: lines match only when extract_answer picks the answer out."""
        weak = ScriptedModel("weak", {"t": ["Work.\nANSWER: 9"]})
        strong = ScriptedModel("strong", {"t": ["<think>6+1</think>\nANSWER: 1,000"]})
        rows = [{"task": "t", "answer": "1000"}]
        assert _comparative(rows, weak, strong) == []  # whole segments compared
        kept = _comparative(rows, weak, strong, extract_answer=_answer_line)
        assert kept == rows
        assert len(weak.calls) == len(strong.calls) == 1

    def test_threshold_change_makes_no_model_calls(self):
        """Changing only rouge_l_threshold regrades cached predictions."""
        weak = ScriptedModel("weak", {"t": ["the cat sat"]})
        strong = ScriptedModel("strong", {"t": ["the cat sat on the mat"]})
        rows = [{"task": "t", "answer": "the cat sat on the mat"}]
        assert _comparative(rows, weak, strong, metric="rouge_l") == rows
        assert (
            _comparative(rows, weak, strong, metric="rouge_l", rouge_l_threshold=0.5)
            == []
        )
        assert len(weak.calls) == len(strong.calls) == 1

    def test_missing_result_raises(self):
        """A model returning fewer results than prompts raises, naming the gap."""

        class DropsLast(ScriptedModel):
            def generate_batch(self, prompts, *, n=1, **kwargs):
                return super().generate_batch(prompts, n=n, **kwargs)[:-1]

        weak = DropsLast("drops", WEAK)
        with pytest.raises(ValueError, match="returned 4 results for 5 prompts"):
            _comparative(weak=weak)


class TestPassAtK:
    """by_pass_at_k keeps rows whose pass rate over k samples is in bounds."""

    SCRIPT = {
        "t1": ["1"],  # 4/4
        "t2": ["2", "0", "0", "0"],  # 1/4
        "t3": ["0"],  # 0/4
        "t4": ["4", "4", "4", "0"],  # 3/4
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

    def test_default_prompt_is_the_task(self):
        """Without prompt_template, the prompt is the task text unchanged."""
        model = ScriptedModel("m", self.SCRIPT)
        _pass_at_k(self.ROWS, model)
        assert model.prompts == ["t1", "t2", "t3", "t4"]

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

    def test_exact_match_compares_whole_segment(self):
        """exact_match normalizes the answer segment but does not search it."""
        model = ScriptedModel(
            "m", {"t": ["<think>4?</think>\n 1,000 \n", "The answer is 1000."]}
        )
        rows = [{"task": "t", "answer": 1000}]
        kept = _pass_at_k(
            rows, model, k=2, min_rate=0.0, max_rate=1.0, outcomes_field="ev"
        )
        assert kept[0]["ev"] == {"model": [True, False]}

    def test_none_grades_excluded_and_min_valid(self):
        """Empty samples are excluded; all k must be valid unless min_valid allows."""
        model = ScriptedModel("m", {"t": ["1", "", "0", "0"]})
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
        """With truncated="exclude", unclosed thinking grades None, like ''."""
        model = ScriptedModel(
            "m", {"t": ["1", "<think>still going", "<think>1</think>"]}
        )
        rows = [{"task": "t", "answer": "1"}]
        kwargs = dict(k=3, outcomes_field="evaluations", truncated="exclude")
        kept = _pass_at_k(rows, model, **kwargs)
        assert kept == []
        kept = _pass_at_k(rows, model, min_valid=1, max_rate=1.0, **kwargs)
        assert kept[0]["evaluations"] == {"model": [True, None, None]}

    def test_truncated_sample_fails_by_default(self):
        """A <think> block that never closes grades False; nothing after it, None."""
        model = ScriptedModel(
            "m", {"t": ["1", "<think>still going", "<think>1</think>"]}
        )
        rows = [{"task": "t", "answer": "1"}]
        assert _pass_at_k(rows, model, k=3) == []  # the empty answer is still None
        kept = _pass_at_k(rows, model, k=3, min_valid=2, outcomes_field="ev")
        assert kept[0]["ev"] == {"model": [True, False, None]}  # rate 1/2

    def test_all_truncated_row_is_hard(self):
        """A row the model always thinks itself to death on has pass rate 0."""
        model = ScriptedModel("m", {"t": ["<think>hmm", "<think>so maybe"]})
        rows = [{"task": "t", "answer": "1"}]
        assert _pass_at_k(rows, model, k=2, min_rate=0.0, max_rate=0.0) == rows
        assert _pass_at_k(rows, model, k=2, truncated="exclude", min_rate=0.0) == []

    def test_truncation_note_in_scores(self, monkeypatch):
        """The step report counts truncated samples graded as failures."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        model = ScriptedModel("m", {"t": ["1", "<think>still", "0", "<think>so"]})

        @step
        def hard(rows):
            return _pass_at_k(rows, model, min_rate=0.0)

        hard([{"task": "t", "answer": "1"}])
        scores = _tracking.step_records()["hard"].ops[0].scores
        assert scores is not None
        assert scores.values == [0.25]
        assert scores.note is not None
        assert scores.note.startswith("2 of 4 samples truncated")
        assert "graded failed" in scores.note
        assert "2 of 4 samples truncated" in render_scores(scores)

    def test_chunks_keep_each_truncation_note(self, monkeypatch):
        """Two by_pass_at_k chunks pool into one histogram with both notes."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        model = ScriptedModel(
            "m", {"t1": ["1", "<think>still"], "t2": ["<think>a", "<think>b"]}
        )

        @step
        def hard(rows):
            return [
                row
                for chunk in ([rows[0]], [rows[1]])
                for row in _pass_at_k(chunk, model, k=2, min_rate=0.0)
            ]

        rows = [{"task": "t1", "answer": "1"}, {"task": "t2", "answer": "1"}]
        hard(rows)
        hard(rows)  # a cached step keeps its ops' notes too
        rec = _tracking.step_records()["hard"]
        assert rec.cache_hit
        text = render_step(rec, full=True)
        assert "by_pass_at_k ×2" in text
        notes = [line.strip() for line in text.splitlines() if "truncated" in line]
        assert notes[0].startswith("1 of 2 samples truncated")
        assert notes[1].startswith("2 of 2 samples truncated")

    def test_no_truncation_note_without_truncation(self, monkeypatch):
        """No note when every sample closed its thinking."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        model = ScriptedModel("m", {"t": ["1", "<think>a</think>0"]})

        @step
        def hard(rows):
            return _pass_at_k(rows, model, k=2, min_rate=0.0)

        hard([{"task": "t", "answer": "1"}])
        scores = _tracking.step_records()["hard"].ops[0].scores
        assert scores is not None and scores.note is None

    def test_answer_segment_for_verifier(self):
        """A verifier gets the stripped completion after its closed <think> block."""
        code = "def f():\n    return 1"
        seen: list[str] = []

        def verify(prediction, row):
            seen.append(prediction)
            return prediction == code

        model = ScriptedModel("m", {"t": [f"<think>plan</think>\n{code}\n"]})
        rows = [{"task": "t", "answer": "1"}]
        kept = _pass_at_k(rows, model, k=1, metric=verify, min_rate=1.0, max_rate=1.0)
        assert kept == rows
        assert seen == [code]

    @pytest.mark.parametrize("truncated,grade", [("fail", False), ("exclude", None)])
    def test_verifier_skips_unclosed_thinking(self, truncated, grade):
        """A verifier is not called on a <think> block that never closes."""
        calls: list[str] = []
        model = ScriptedModel("m", {"t": ["<think>still going", "42"]})
        rows = [{"task": "t", "answer": "42"}]
        kept = _pass_at_k(
            rows,
            model,
            k=2,
            metric=lambda prediction, row: calls.append(prediction) or True,
            min_valid=1,
            outcomes_field="evaluations",
            max_rate=1.0,
            min_rate=0.5,
            truncated=truncated,
        )
        assert kept[0]["evaluations"] == {"model": [grade, True]}
        assert calls == ["42"]

    def test_truncated_ignores_custom_extract_answer(self):
        """truncated="fail" grades a cut-off sample before extract_answer sees it."""
        seen: list[str] = []
        model = ScriptedModel("m", {"t": ["<think>ANSWER: 1", "ANSWER: 1"]})
        rows = [{"task": "t", "answer": "1"}]
        kept = _pass_at_k(
            rows,
            model,
            k=2,
            extract_answer=lambda t: seen.append(t)
            or t.rsplit("ANSWER:", 1)[1].strip(),
            outcomes_field="ev",
        )
        assert kept[0]["ev"] == {"model": [False, True]}
        assert seen == ["ANSWER: 1"]

    def test_rouge_l_threshold(self):
        """rouge_l scores the answer segment and passes at F1 >= the threshold."""
        model = ScriptedModel(
            "m", {"t": ["<think>the cat sat on the mat</think>the cat sat"]}
        )
        rows = [{"task": "t", "answer": "the cat sat on the mat"}]  # F1 2/3
        kw = dict(k=1, metric="rouge_l", min_rate=1.0, max_rate=1.0)
        assert _pass_at_k(rows, model, **kw) == []
        assert _pass_at_k(rows, model, rouge_l_threshold=0.6, **kw) == rows
        assert len(model.calls) == 1

    def test_extract_answer_replaces_the_default(self):
        """A custom extract_answer sees each raw completion, thinking included."""
        seen: list[str] = []

        def extract(text):
            seen.append(text)
            return _answer_line(text)

        model = ScriptedModel(
            "m", {"t": ["<think>ANSWER: 7</think>\nANSWER: 3\n", "<think>ANSWER: 3"]}
        )
        rows = [{"task": "t", "answer": "3"}]
        kept = _pass_at_k(
            rows,
            model,
            k=2,
            extract_answer=extract,
            min_valid=1,
            max_rate=1.0,
            outcomes_field="ev",
            truncated="exclude",  # else the cut-off sample is never extracted
        )
        assert kept[0]["ev"] == {"model": [True, None]}
        assert seen == ["<think>ANSWER: 7</think>\nANSWER: 3\n", "<think>ANSWER: 3"]

    def test_custom_verifier(self):
        """With extract_answer, a verifier gets its output and the row."""
        seen: list[tuple[str, str]] = []

        def verify(prediction, row):
            seen.append((prediction, row["task"]))
            return None if prediction == "skip" else prediction == row["answer"]

        model = ScriptedModel("m", {"t": ["ANSWER: yes", "ANSWER: skip", "ANSWER: no"]})
        rows = [{"task": "t", "answer": "yes"}]
        kept = _pass_at_k(
            rows,
            model,
            k=3,
            metric=verify,
            extract_answer=lambda text: text.removeprefix("ANSWER: "),
            min_valid=2,
            outcomes_field="evaluations",
        )
        assert kept[0]["evaluations"] == {"model": [True, None, False]}
        # Verifiers run concurrently, so only the set of calls is fixed.
        assert sorted(seen) == [("no", "t"), ("skip", "t"), ("yes", "t")]

    def test_verifier_must_return_bool(self):
        """A non-boolean grade is rejected."""
        model = ScriptedModel("m", {"t": ["1"]})
        with pytest.raises(TypeError):
            _pass_at_k([{"task": "t", "answer": "1"}], model, metric=lambda p, r: 1)

    def test_outcomes_field_matches_success_rate(self):
        """Kept rows' outcomes give the same rate and decision when re-selected."""
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
        """Re-selecting on the stored grades keeps exactly the same rows."""
        script = {
            **self.SCRIPT,
            "t5": ["5", "", "0", "5"],  # 2/3, one excluded
            "t6": ["0", "", "", "0"],  # 0/2, two excluded
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
            {"prompt_template": "Q: {task}"},
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

    def test_fewer_completions_than_k_raises(self):
        """A prompt short of k samples (e.g. n ignored) raises; nothing is padded."""

        class ShortOnT2(ScriptedModel):
            def completions(self, prompt, n):
                got = super().completions(prompt, n)
                return got[:1] if prompt == "t2" else got

        model = ShortOnT2("short", self.SCRIPT)
        with pytest.raises(ValueError, match=r"returned 1 completions for prompt 1, b"):
            _pass_at_k(self.ROWS, model, k=3)
        with pytest.raises(ValueError, match="n=3 were requested"):
            _pass_at_k(self.ROWS, model, k=3)
        assert len(model.calls) == 2  # the failure was not cached

    def test_missing_result_in_later_batch_raises(self):
        """A batch with fewer results than prompts raises, naming its prompts."""

        class DropsLast(ScriptedModel):
            def generate_batch(self, prompts, *, n=1, **kwargs):
                out = super().generate_batch(prompts, n=n, **kwargs)
                return out[:-1] if prompts[0] == "t3" else out

        model = DropsLast("drops", self.SCRIPT)
        with pytest.raises(ValueError, match=r"1 results for 2 prompts \(prompts 2"):
            _pass_at_k(self.ROWS, model, batch_size=2)

    def test_custom_prompt_template(self):
        """prompt_template replaces the default prompt."""
        model = ScriptedModel("m", {"t1": ["1"]})
        rows = [{"task": "t1", "answer": "1"}]
        kept = _pass_at_k(
            rows, model, k=1, prompt_template="Q: {task}", min_rate=1.0, max_rate=1.0
        )
        assert kept == rows
        assert model.prompts == ["Q: t1"]


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
        {"truncated": "skip"},
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
    model = ScriptedModel("m", {"t": ["1"]})
    rows = [{"task": "t", "answer": "1"}]
    assert _pass_at_k(rows, model, k=1, temperature=0.0, max_rate=1.0) == rows


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric": "bleu"},
        {"rouge_l_threshold": -0.1},
        {"batch_size": 0},
        {"truncated": "none"},
    ],
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
        ("42", "42"),
        ("  Paris.\n", "Paris."),
        ("ANSWER: 4", "ANSWER: 4"),  # no format is parsed
        ("<think>try 3</think>\n 4 \n", "4"),
        ("<think>a</think>b</think> c", "c"),  # after the last </think>
        ("<think>so 3, but wait", None),  # cut off mid-thought
        ("<think>done</think>  \n", None),  # nothing after the thinking
        ("", None),
    ],
)
def test_answer_segment(text, expected):
    """The default prediction is the stripped text after a closed <think> block."""
    assert answer_segment(text) == expected


class _ConcurrencyProbe:
    """Verifier recording the most calls running at once (exact match grades).

    With ``barrier_parties``, each call waits until that many are running.
    """

    def __init__(self, barrier_parties: int | None = None) -> None:
        self.barrier = (
            threading.Barrier(barrier_parties, timeout=10) if barrier_parties else None
        )
        self.lock = threading.Lock()
        self.active = self.peak = 0

    def cache_key(self) -> tuple:
        """A fresh key per probe, so every test grades anew."""
        return ("probe", id(self))

    def __call__(self, prediction: str, row: dict) -> bool:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        if self.barrier is not None:
            self.barrier.wait()
        else:
            time.sleep(0.01)
        with self.lock:
            self.active -= 1
        return prediction == row["answer"]


class _CountingVerifier:
    """Exact-match verifier recording each prediction it grades."""

    def __init__(self) -> None:
        self.graded: list[str] = []

    def cache_key(self) -> tuple:
        """Leave the call log out of the cache key."""
        return ("counting",)

    def __call__(self, prediction: str, row: dict) -> bool:
        self.graded.append(prediction)
        return prediction == row["answer"]


class TestCompletionsAndVerifiers:
    """completions_field, optional ground truth, and concurrent verifiers."""

    SCRIPT = TestPassAtK.SCRIPT
    ROWS = TestPassAtK.ROWS

    def test_pass_at_k_completions_align_with_outcomes(self):
        """Kept rows carry the k raw completions in the order of their grades."""
        model = ScriptedModel("m", {"t": ["<think>a</think>1", "", "0", "1"]})
        rows = [{"task": "t", "answer": "1"}]
        kept = _pass_at_k(
            rows,
            model,
            min_valid=3,
            outcomes_field="ev",
            completions_field="samples",
        )
        assert kept[0]["ev"] == {"model": [True, None, False, True]}
        assert kept[0]["samples"] == {"model": ["<think>a</think>1", None, "0", "1"]}
        assert "samples" not in rows[0]

    def test_truncated_completions_kept_with_failed_grades(self):
        """completions_field keeps a truncated sample's raw text next to its False."""
        model = ScriptedModel("m", {"t": ["1", "<think>going", "1", "<think>on"]})
        rows = [{"task": "t", "answer": "1"}]
        kept = _pass_at_k(rows, model, outcomes_field="ev", completions_field="samples")
        assert kept[0]["ev"] == {"model": [True, False, True, False]}
        assert kept[0]["samples"] == {"model": ["1", "<think>going", "1", "<think>on"]}
        excluded = _pass_at_k(
            rows,
            model,
            min_valid=2,
            max_rate=1.0,
            outcomes_field="ev",
            completions_field="samples",
            truncated="exclude",
        )
        assert excluded[0]["ev"] == {"model": [True, None, True, None]}
        assert excluded[0]["samples"] == kept[0]["samples"]

    def test_completions_field_reuses_cached_samples(self):
        """Asking for completions later makes no new model call."""
        model = ScriptedModel("m", self.SCRIPT)
        plain = _pass_at_k(self.ROWS, model)
        with_samples = _pass_at_k(self.ROWS, model, completions_field="samples")
        assert [r["task"] for r in with_samples] == [r["task"] for r in plain]
        assert with_samples[0]["samples"] == {"model": ["2", "0", "0", "0"]}
        assert len(model.calls) == 1

    def test_comparative_completions(self):
        """by_comparative_difficulty keeps each model's completion per group."""
        kept = _comparative(completions_field="answers")
        assert kept == [{**ROWS[0], "answers": {"weak": ["9"], "strong": ["1"]}}]

    def test_ground_truth_optional_with_verifier(self):
        """A verifier reads the row itself; no ground_truth_field is needed."""
        model = ScriptedModel("m", {"t": ["yes", "no"]})
        kept = by_pass_at_k(
            [{"task": "t"}],
            model=model,
            input_field="task",
            k=2,
            metric=lambda prediction, row: prediction == "yes",
            outcomes_field="ev",
        )
        assert kept[0]["ev"] == {"model": [True, False]}

    @pytest.mark.parametrize("metric", ["exact_match", "rouge_l"])
    def test_ground_truth_required_for_reference_metrics(self, metric):
        """exact_match and rouge_l raise without ground_truth_field, before sampling."""
        model = ScriptedModel("m", {"t": ["1"]})
        with pytest.raises(ValueError, match="ground_truth_field is required"):
            by_pass_at_k(
                [{"task": "t"}], model=model, input_field="task", metric=metric
            )
        weak, strong = ScriptedModel("weak", WEAK), ScriptedModel("strong", STRONG)
        with pytest.raises(ValueError, match="ground_truth_field is required"):
            by_comparative_difficulty(
                ROWS, weak_model=weak, strong_model=strong, input_field="task"
            )
        assert model.calls == weak.calls == strong.calls == []

    @pytest.mark.parametrize(
        "kwargs",
        [{"verifier_workers": 0}, {"outcomes_field": "x", "completions_field": "x"}],
    )
    def test_validation(self, kwargs):
        """Bad worker counts or clashing fields raise before any model call."""
        model = ScriptedModel("m", {})
        with pytest.raises(ValueError):
            _pass_at_k([{"task": "t", "answer": "1"}], model, **kwargs)
        assert model.calls == []

    def test_verifiers_run_concurrently(self):
        """Four verifier calls wait on each other, so they must run at once."""
        model = ScriptedModel("m", self.SCRIPT)
        verify = _ConcurrencyProbe(barrier_parties=4)
        kept = _pass_at_k(self.ROWS, model, k=1, metric=verify, max_rate=1.0)
        assert [r["task"] for r in kept] == ["t1", "t2", "t4"]
        assert verify.peak == 4

    def test_verifier_workers_one_is_serial(self):
        """verifier_workers=1 never runs two verifier calls at once."""
        model = ScriptedModel("m", self.SCRIPT)
        verify = _ConcurrencyProbe()
        _pass_at_k(self.ROWS, model, metric=verify, verifier_workers=1, max_rate=1.0)
        assert verify.peak == 1

    def test_verifier_workers_in_cache_key(self):
        """A new verifier_workers regrades (timeouts depend on load), not resamples."""
        verify = _CountingVerifier()
        model = ScriptedModel("m", self.SCRIPT)
        first = _pass_at_k(self.ROWS, model, metric=verify, verifier_workers=4)
        again = _pass_at_k(self.ROWS, model, metric=verify, verifier_workers=1)
        assert again == first
        assert len(verify.graded) == 32 and len(model.calls) == 1
        hits = [r.cache_hit for r in trace() if r.op_name == "filter.by_pass_at_k"]
        assert hits == [False, False]
        _pass_at_k(self.ROWS, model, metric=verify, verifier_workers=1)
        assert len(verify.graded) == 32
        weak, strong = ScriptedModel("weak", WEAK), ScriptedModel("strong", STRONG)
        for workers in (3, 1):
            _comparative(
                weak=weak, strong=strong, metric=verify, verifier_workers=workers
            )
        assert len(weak.calls) == len(strong.calls) == 1
        assert len(verify.graded) == 32 + 2 * 9  # 9 non-empty predictions per run

    def test_verifier_error_propagates(self):
        """An exception in one verifier call fails the filter."""

        def verify(prediction, row):
            if prediction == "0":
                raise RuntimeError("runner crashed")
            return True

        with pytest.raises(RuntimeError, match="runner crashed"):
            _pass_at_k(self.ROWS, ScriptedModel("m", self.SCRIPT), metric=verify)

    def test_verifier_runs_in_callers_context(self, monkeypatch):
        """An op a verifier calls folds into the filter's row of the step report."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        calls: list[str] = []

        @op(name="custom.check")
        def check(prediction: str) -> bool:
            calls.append(prediction)
            return prediction == "1"

        model = ScriptedModel("m", self.SCRIPT)

        @step
        def hard(rows):
            return _pass_at_k(rows, model, k=1, metric=lambda p, r: check(p))

        hard(self.ROWS)
        assert sorted(calls) == ["0", "1", "2", "4"]
        names = [o.op_name for o in _tracking.step_records()["hard"].ops]
        assert names == ["filter.by_pass_at_k"]

    def test_program_verifier_runs_programs(self):
        """program_verifier grades sampled programs by running the row's tests."""
        right = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
        wrong = "```python\nprint(0)\n```"
        model = ScriptedModel(
            "m", {"add": [f"<think>easy</think>{right}", wrong, "no code here"]}
        )
        rows = [
            {
                "task": "add",
                "tests": [
                    {"input": "1 2\n", "output": "3"},
                    {"input": "5 5\n", "output": "10\n"},
                ],
            }
        ]
        kept = by_pass_at_k(
            rows,
            model=model,
            input_field="task",
            k=3,
            metric=program_verifier("tests", timeout_s=20),
            outcomes_field="ev",
            completions_field="samples",
        )
        assert kept[0]["ev"] == {"model": [True, False, False]}
        assert kept[0]["samples"]["model"][0].endswith(right)
