"""Failed requests in the LLM filters, and dropped rows' notes in the report."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import litellm
import numpy as np
import pytest

from pyforger import PromptTooLongError, _tracking
from pyforger._render import render_scores, render_step
from pyforger.acquire.model import GenerateResult, Model
from pyforger.filter import (
    by_comparative_difficulty,
    by_judged_difficulty,
    by_llm_score,
    by_pass_at_k,
)
from pyforger.inspect import topic_based_clustering
from pyforger.op import trace
from pyforger.steps import step


class GatewayTimeout(Exception):
    """Stands in for litellm's 504 after its retries."""

    status_code = 504


_CANDIDATE_RE = re.compile(r'<candidate label="([A-Z]+)">\n(.*?)\n</candidate>', re.S)


def _judge_answer(prompt: str, scores: dict[str, int]) -> str:
    """A judge answer scoring each shown output by ``scores`` (default 5)."""
    shown = dict(_CANDIDATE_RE.findall(prompt))
    if shown:
        return json.dumps(
            {
                "candidates": [
                    {"label": k, "feedback": f"on {v}", "score": scores.get(v, 5)}
                    for k, v in shown.items()
                ]
            }
        )
    if "<output>" in prompt:
        text = prompt.split("<output>\n")[1].split("\n</output>")[0]
        return json.dumps({"feedback": f"on {text}", "score": scores.get(text, 5)})
    return json.dumps({"explanation": "rated", "score": 4})


class FlakyJudge(Model):
    """Answers like a judge; prompts containing a ``fail`` text 504 instead.

    ``cut`` texts get a ``<think>`` block that never closes (out of tokens),
    ``fatal`` texts a non-transient error. With ``row_error``, ``fail`` texts
    raise ``row_error(text)`` instead of a 504. Records each batch's
    ``return_exceptions``.
    """

    def __init__(
        self,
        scores: dict[str, int] | None = None,
        fail: tuple[str, ...] = (),
        cut: tuple[str, ...] = (),
        fatal: Exception | None = None,
        answer: Any = None,
        row_error: Any = None,
    ) -> None:
        """Store what to score, fail, and cut off."""
        super().__init__("test/flaky-judge")
        self.scores = scores or {}
        self.fail = fail
        self.cut = cut
        self.fatal = fatal
        self.answer = answer
        self.row_error = row_error
        self.caught: list[bool] = []
        self.sent: list[str] = []  # every prompt, as text

    def cache_key(self) -> tuple:
        """Settings only, so the op cache works across instances."""
        return ("FlakyJudge", sorted(self.scores.items()), self.fail, self.cut)

    def generate_batch(  # type: ignore[override]
        self,
        prompts,
        *,
        n=1,
        temperature=None,
        max_tokens=None,
        stop=None,
        show_progress=None,
        return_exceptions=False,
        cache_salt=None,
        cache_rows=None,
    ):
        """One result per prompt, or the exception for failing prompts."""
        self.caught.append(return_exceptions)
        out: list[Any] = []
        for prompt in prompts:
            text = prompt if isinstance(prompt, str) else json.dumps(prompt)
            self.sent.append(text)
            exc: Exception | None = None
            if self.fatal is not None:
                exc = self.fatal
            elif any(f in text for f in self.fail) and self.row_error is not None:
                exc = self.row_error(text)
            elif any(f in text for f in self.fail):
                exc = GatewayTimeout(f"504 Gateway Timeout for {text[:20]!r}")
            if exc is not None:
                if not return_exceptions:
                    raise exc
                out.append(exc)
                continue
            if any(c in text for c in self.cut):
                answer = "<think>let me weigh each candidate"
            elif self.answer is not None:
                answer = self.answer(text)
            else:
                answer = _judge_answer(text, self.scores)
            out.append(
                GenerateResult(
                    [answer] * n,
                    usage={},
                    reasoning=[None] * n,
                    model_id=self._model_id,
                    raw=None,
                )
            )
        return out


ROWS = [
    {"q": "q1", "a": "good"},
    {"q": "q2", "a": "TIMEOUT"},
    {"q": "q3", "a": "bad"},
]


def _score(rows, judge, **kwargs):
    return by_llm_score(rows, model=judge, input_field="q", output_field="a", **kwargs)


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


class TestByLlmScore:
    """A failed judge request is a missing vote, not a crash."""

    def test_failed_request_leaves_row_unscored(self, monkeypatch, caplog):
        """The 504 row is dropped and unscored; the others are judged."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        judge = FlakyJudge(scores={"bad": 2}, fail=("TIMEOUT",))

        @step
        def judged(rows):
            return _score(rows, judge)

        with caplog.at_level(logging.WARNING):
            kept = judged(ROWS)
        assert kept == [ROWS[0]]
        assert judge.caught == [True]
        call = _tracking.step_records()["judged"].ops[0]
        assert call.scores is not None
        assert call.scores.values == [5, None, 2]
        assert call.warnings == [
            "filter.by_llm_score: 1 of 3 judge requests failed after retries; their "
            "votes are missing: GatewayTimeout ×1; last error: GatewayTimeout: 504 "
            "Gateway Timeout for 'Score the output bel' (result not cached)",
            "filter.by_llm_score: 1 of 3 judge calls gave no score (1 failed) → "
            "1 row unscored",
        ]
        assert call.warnings[0] in _messages(caplog)

    def test_failed_result_is_not_cached(self):
        """A rerun asks the judges again and scores the row that failed."""
        _score(ROWS, FlakyJudge(fail=("TIMEOUT",)))
        healthy = FlakyJudge()
        assert _score(ROWS, healthy) == ROWS
        assert healthy.caught  # judged again, not replayed from the cache
        assert [r.cache_hit for r in trace() if r.op_name == "filter.by_llm_score"] == [
            False,
            False,
        ]

    def test_non_transient_error_raises(self):
        """A bad request (400) or a spending cap is never swallowed."""

        class BadRequest(Exception):
            status_code = 400

        with pytest.raises(BadRequest):
            _score(ROWS, FlakyJudge(fatal=BadRequest("context too long")))
        with pytest.raises(_tracking.BudgetExceededError):
            _score(ROWS, FlakyJudge(fatal=_tracking.BudgetExceededError("cap")))

    def test_multi_judge_failure_is_one_missing_vote(self):
        """With several judges, a failed request costs its row one vote."""

        class OneFails(FlakyJudge):
            def generate_batch(self, prompts, *, return_exceptions=False, **kwargs):
                out = super().generate_batch(
                    prompts, return_exceptions=return_exceptions, **kwargs
                )
                out[0] = GatewayTimeout("504")  # the first judge of the first row
                return out

        rows = [{"q": "q1", "a": "x"}, {"q": "q2", "a": "y"}]
        kept = _score(
            rows,
            OneFails(),
            num_judges=2,
            temperature=0.5,
            agreement_threshold=0.5,
            score_field="s",
        )
        assert [r["a"] for r in kept] == ["x", "y"]
        assert kept[0]["s"] == 5  # the mean of the one vote that arrived

    def test_group_by_warning_counts_rows(self, caplog):
        """With group_by, a cut-off call's warning says how many rows it cost."""
        rows = [
            {"q": "slow", "a": "a1"},
            {"q": "slow", "a": "a2"},
            {"q": "slow", "a": "a3"},
            {"q": "fast", "a": "b1"},
            {"q": "fast", "a": "b2"},
            {"q": "down", "a": "c1"},
            {"q": "down", "a": "c2"},
        ]
        judge = FlakyJudge(cut=("a1",), fail=("c1",))
        with caplog.at_level(logging.WARNING):
            kept = _score(rows, judge, group_by="q")
        assert [r["a"] for r in kept] == ["b1", "b2"]
        assert (
            "filter.by_llm_score: 2 of 3 judge calls gave no score (1 failed, "
            "1 stopped mid-thought (unclosed <think>: raise max_tokens)) → "
            "5 rows unscored" in _messages(caplog)
        )


class TestDroppedNotes:
    """The step report shows a few dropped rows with the judge's feedback."""

    def test_dropped_rows_carry_feedback(self, monkeypatch):
        """Only dropped rows, at most three, with their output and feedback."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        rows = [{"q": f"q{i}", "a": f"answer {i}"} for i in range(6)]
        judge = FlakyJudge(scores={f"answer {i}": 2 for i in (1, 2, 4, 5)})

        @step
        def judged(rows):
            return _score(rows, judge)

        judged(rows)
        rec = _tracking.step_records()["judged"]
        scores = rec.ops[0].scores
        assert scores is not None
        assert [(d.value, d.preview, d.note) for d in scores.dropped] == [
            (2, "answer 1", "on answer 1"),
            (2, "answer 2", "on answer 2"),
            (2, "answer 4", "on answer 4"),
        ]
        assert scores.note is None  # not the last row's feedback
        text = render_step(rec)
        assert "dropped, with notes:" in text
        assert text.count("on answer 5") == 0 and text.count("on answer 1") == 1
        assert "2 · answer 1" in text
        assert "↳ on answer 1" in text
        assert "answer 5" not in text.split("dropped, with notes:")[1]

    def test_cached_reruns_keep_dropped_notes(self, monkeypatch):
        """A cached step, or a cached filter op in a new step, still shows them."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        rows = [{"q": f"q{i}", "a": f"answer {i}"} for i in range(3)]
        judge = FlakyJudge(scores={"answer 1": 2})
        expected = [(2, "answer 1", "on answer 1")]

        def notes(step_name):
            scores = _tracking.step_records()[step_name].ops[0].scores
            assert scores is not None
            return [(d.value, d.preview, d.note) for d in scores.dropped]

        @step
        def judged(rows):
            return _score(rows, judge)

        judged(rows)
        judged(rows)  # the step cache: scores come from its .step.json
        rec = _tracking.step_records()["judged"]
        assert rec.cache_hit
        assert notes("judged") == expected
        assert "↳ on answer 1" in render_step(rec, full=True)  # as inspect_step

        @step
        def judged_again(rows):
            # A new step, so only by_llm_score's op cache (.scores.json) hits.
            return _score(rows, judge)

        judged_again(rows)
        rec = _tracking.step_records()["judged_again"]
        assert not rec.cache_hit and rec.ops[0].cache_hit
        assert notes("judged_again") == expected
        assert rec.ops[0].scores is not None and rec.ops[0].scores.note is None
        assert "↳ on answer 1" in render_step(rec)

    def test_nothing_dropped_no_notes(self, monkeypatch):
        """No section when every row is kept."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")

        @step
        def judged(rows):
            return _score(rows, FlakyJudge())

        judged([{"q": "q", "a": "fine"}])
        rec = _tracking.step_records()["judged"]
        assert rec.ops[0].scores is not None and rec.ops[0].scores.dropped == []
        assert "dropped, with notes" not in render_step(rec)

    def test_lowest_judge_feedback_and_failure_note(self, monkeypatch):
        """Several judges: the lowest score's feedback; a failed row says so."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        seen: dict[str, int] = {}

        def answer(text):
            k = seen.get(text, 0)
            seen[text] = k + 1
            score = 5 if k == 0 else 1
            return json.dumps({"feedback": f"judge {k}", "score": score})

        judge = FlakyJudge(fail=("TIMEOUT",), answer=answer)

        @step
        def judged(rows):
            return _score(rows, judge, num_judges=2, temperature=0.5)

        judged([{"q": "q1", "a": "x"}, {"q": "q2", "a": "TIMEOUT"}])
        scores = _tracking.step_records()["judged"].ops[0].scores
        assert scores is not None
        assert [(d.preview, d.note) for d in scores.dropped] == [
            ("x", "judge 1"),
            ("TIMEOUT", "no score: the judge request failed"),
        ]

    def test_preview_and_note_are_cut(self, monkeypatch):
        """Long outputs and notes are cut to one short line."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        long_feedback = json.dumps({"feedback": "why " * 200, "score": 1})

        @step
        def judged(rows):
            return _score(rows, FlakyJudge(answer=lambda text: long_feedback))

        judged([{"q": "q", "a": "word\n" * 200}])
        record = _tracking.step_records()["judged"].ops[0].scores
        assert record is not None
        dropped = record.dropped[0]
        assert len(dropped.preview) == _tracking.PREVIEW_CHARS
        assert dropped.preview.endswith("…") and "\n" not in dropped.preview
        assert len(dropped.note) == _tracking.NOTE_CHARS
        assert "↳ why why" in render_scores(record)


class TestDifficultyFilters:
    """Failed samples and ratings are missing, not fatal."""

    def test_judged_difficulty_failure_drops_row(self, monkeypatch):
        """A failed rating request leaves its row unrated and warns once."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        judge = FlakyJudge(fail=("TIMEOUT",))

        @step
        def hard(rows):
            return by_judged_difficulty(rows, model=judge, task_field="task")

        rows = [{"task": "Prove it."}, {"task": "TIMEOUT task"}]
        assert hard(rows) == [rows[0]]
        call = _tracking.step_records()["hard"].ops[0]
        assert call.scores is not None and call.scores.values == [4, None]
        assert len(call.warnings) == 1
        assert call.warnings[0].startswith(
            "filter.by_judged_difficulty: 1 of 2 judge requests failed after "
            "retries; those rows are unrated: GatewayTimeout ×1; last error:"
        )

    def test_judged_difficulty_fatal_error_raises(self):
        """A non-transient error still raises."""
        with pytest.raises(_tracking.BudgetExceededError):
            by_judged_difficulty(
                [{"task": "t"}],
                model=FlakyJudge(fatal=_tracking.BudgetExceededError("cap")),
                task_field="task",
            )

    def test_pass_at_k_failed_samples_grade_none(self, monkeypatch):
        """A failed prompt's samples are None: excluded, so the row is dropped."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        student = FlakyJudge(fail=("TIMEOUT",), answer=lambda t: "4")

        @step
        def hard(rows):
            return by_pass_at_k(
                rows,
                model=student,
                input_field="task",
                ground_truth_field="answer",
                k=2,
                min_rate=0.0,
                max_rate=1.0,
                outcomes_field="ev",
            )

        rows = [{"task": "2+2", "answer": "4"}, {"task": "TIMEOUT", "answer": "4"}]
        kept = hard(rows)
        assert kept == [{**rows[0], "ev": {"model": [True, True]}}]
        call = _tracking.step_records()["hard"].ops[0]
        assert call.scores is not None and call.scores.values == [1.0, None]
        # Missing samples have no <think>: excluded, never counted as truncated
        # (by_pass_at_k grades truncated samples as failures by default).
        assert "truncated" not in (call.scores.note or "")
        assert len(call.warnings) == 1
        assert call.warnings[0].startswith(
            "difficulty filter: 2 of 4 samples from Model("
        )
        assert "GatewayTimeout ×1; last error: GatewayTimeout" in call.warnings[0]
        assert call.warnings[0].endswith("(result not cached)")

    def test_pass_at_k_partial_samples_are_padded(self):
        """Completions that arrived are kept; missing ones grade None."""

        class Short(FlakyJudge):
            def generate_batch(self, prompts, *, return_exceptions=False, **kwargs):
                out = super().generate_batch(
                    prompts, return_exceptions=return_exceptions, **kwargs
                )
                return [
                    GenerateResult(
                        list(r)[:1], usage={}, reasoning=[None], model_id="m", raw=None
                    )
                    for r in out
                ]

        rows = [{"task": "2+2", "answer": "4"}]
        kept = by_pass_at_k(
            rows,
            model=Short(answer=lambda t: "4"),
            input_field="task",
            ground_truth_field="answer",
            k=3,
            min_valid=1,
            max_rate=1.0,
            outcomes_field="ev",
        )
        assert kept[0]["ev"] == {"model": [True, None, None]}

    def test_comparative_failed_strong_drops_row(self):
        """A failed strong prediction leaves the row ungraded, not a crash."""
        rows = [{"task": "TIMEOUT", "answer": "4"}, {"task": "2+2", "answer": "4"}]
        weak = FlakyJudge(answer=lambda t: "5")
        strong = FlakyJudge(fail=("TIMEOUT",), answer=lambda t: "4")
        kept = by_comparative_difficulty(
            rows,
            weak_model=weak,
            strong_model=strong,
            input_field="task",
            ground_truth_field="answer",
        )
        assert kept == [rows[1]]

    def test_fatal_sample_error_raises(self):
        """A non-transient error from the sampler raises."""
        with pytest.raises(_tracking.BudgetExceededError):
            by_pass_at_k(
                [{"task": "t", "answer": "1"}],
                model=FlakyJudge(fatal=_tracking.BudgetExceededError("cap")),
                input_field="task",
                ground_truth_field="answer",
            )


def test_topic_clustering_tolerates_failed_assignments(monkeypatch):
    """A failed label request is not asked again: it falls back; one warning."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    animals = ("cats", "dogs", "fish")
    docs = [f"doc about {a} {i}" for a in animals for i in range(3)]

    def answer(text):
        if "Record: " in text:  # assign one record to a topic
            record = text.split("Record: ")[1]
            return json.dumps({"topic": next(a for a in animals if a in record)})
        return json.dumps({a: f"about {a}" for a in animals})  # summary, codebook

    # Only the assign prompt has "Record: <doc>"; the summary has "Records: ...".
    judge = FlakyJudge(fail=("Record: doc about fish 2",), answer=answer)

    @step
    def topics(rows):
        out = topic_based_clustering(
            rows,
            field="text",
            context="pets",
            model=judge,
            num_clusters=3,
            embed_fn=lambda d: np.array([[float(i), 1.0] for i in range(len(d))]),
            max_assign_retries=1,
        )
        assert out[:8] == ["cats"] * 3 + ["dogs"] * 3 + ["fish"] * 2
        assert len(out) == 9  # the failed one gets the fallback topic
        return rows

    topics([{"text": d} for d in docs])
    rec = _tracking.step_records()["topics"]
    warnings = [w for w in rec.warnings if "requests failed" in w]
    assert len(warnings) == 1
    assert warnings[0].startswith(
        "inspect.topic_based_clustering: 1 requests failed after retries"
    )
    # Asked once: the model's own retries are the only retries.
    assert sum("Record: doc about fish 2" in text for text in judge.sent) == 1


def _too_long(text: str) -> PromptTooLongError:
    """What a backend that checks lengths before sending raises."""
    return PromptTooLongError("Prompt 17 has 69685 tokens; the context window is 32768")


def _blocked(text: str) -> litellm.BadRequestError:
    """Azure's content-filter 400."""
    return litellm.BadRequestError(
        message="AzureException - {'error': {'code': 'content_filter'}}",
        llm_provider="azure",
        model="m",
    )


class TestRowErrors:
    """A content-filter block or an over-long prompt costs one row, not the op."""

    def _pass_at_k(self, rows, student):
        return by_pass_at_k(
            rows,
            model=student,
            input_field="task",
            ground_truth_field="answer",
            k=2,
            min_rate=0.0,
            max_rate=1.0,
        )

    def test_pass_at_k_over_long_prompt_is_unscored(self, monkeypatch):
        """The over-long row grades None (unscored); the warning names why."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        student = FlakyJudge(fail=("LONG",), answer=lambda t: "4", row_error=_too_long)

        @step
        def hard(rows):
            return self._pass_at_k(rows, student)

        rows = [{"task": "2+2", "answer": "4"}, {"task": "LONG", "answer": "4"}]
        assert hard(rows) == [rows[0]]
        call = _tracking.step_records()["hard"].ops[0]
        assert call.scores is not None and call.scores.values == [1.0, None]
        assert len(call.warnings) == 1
        assert (
            ": 1 prompt longer than the model's context window; last error: "
            "PromptTooLongError: Prompt 17 has 69685 tokens"
        ) in call.warnings[0]

    def test_judged_difficulty_over_long_prompt_is_unrated(self):
        """by_judged_difficulty leaves the row unrated instead of raising."""
        judge = FlakyJudge(fail=("LONG",), row_error=_too_long)
        rows = [{"task": "Prove it."}, {"task": "LONG task"}]
        assert by_judged_difficulty(rows, model=judge, task_field="task") == [rows[0]]

    def test_llm_score_content_filter_is_a_missing_vote(self, caplog):
        """A blocked judge request is a missing vote, counted by reason."""
        judge = FlakyJudge(fail=("TIMEOUT",), row_error=_blocked)
        with caplog.at_level(logging.WARNING):
            assert _score(ROWS, judge) == [ROWS[0], ROWS[2]]
        assert any(
            "votes are missing: 1 prompt blocked by the provider's content filter;"
            in message
            for message in _messages(caplog)
        )

    def test_every_prompt_row_error_raises(self):
        """Every prompt over the context window points at the template."""
        student = FlakyJudge(fail=("",), answer=lambda t: "4", row_error=_too_long)
        rows = [{"task": "2+2", "answer": "4"}, {"task": "3+1", "answer": "4"}]
        with pytest.raises(PromptTooLongError):
            self._pass_at_k(rows, student)
        with pytest.raises(litellm.BadRequestError):
            _score(ROWS, FlakyJudge(fail=("",), row_error=_blocked))

    def test_topic_clustering_every_label_blocked_raises(self):
        """Every label request blocked raises instead of all-fallback topics."""
        animals = ("cats", "dogs", "fish")
        judge = FlakyJudge(
            fail=("Record: ",),
            answer=lambda t: json.dumps({a: f"about {a}" for a in animals}),
            row_error=_blocked,
        )
        with pytest.raises(litellm.BadRequestError):
            topic_based_clustering(
                [{"text": f"doc about {a} {i}"} for a in animals for i in range(3)],
                field="text",
                context="pets",
                model=judge,
                num_clusters=3,
                embed_fn=lambda d: np.array([[float(i), 1.0] for i in range(len(d))]),
            )
