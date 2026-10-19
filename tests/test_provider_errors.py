"""Provider errors in generate_from_context / sample_from_LLM, and skip_cache_write.

Model retries a transient error itself (num_retries); the ops never ask a
failed request again, so a request makes at most num_retries + 1 attempts.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest
from fixtures import MockModel

from pyforger import _tracking, op
from pyforger.acquire import PromptTooLongError, sample_from_LLM
from pyforger.acquire._model_utils import _is_row_error, _is_transient
from pyforger.acquire.model import GenerateResult, Model
from pyforger.alter import generate_from_context
from pyforger.op import skip_cache_write, trace
from pyforger.steps import pipeline, step


class InternalServerError(Exception):
    """Stands in for litellm's 5xx error (e.g. a gateway 504)."""

    status_code = 504


def _result(completions: list[str], model_id: str = "test/flaky") -> GenerateResult:
    return GenerateResult(
        completions,
        usage={},
        reasoning=[None] * len(completions),
        model_id=model_id,
        raw=None,
    )


class FlakyModel(Model):
    """Answers "<prompt>#<j>"; prompts in ``fail`` raise that many times first."""

    def __init__(self, fail: dict[str, int] | None = None, error=None) -> None:
        """Fail each prompt in ``fail`` the given number of times.

        ``error(prompt)`` builds the exception; default a 504.
        """
        super().__init__("test/flaky")
        self.fail = dict(fail or {})
        self.error = error or (lambda prompt: InternalServerError(f"504 for {prompt}"))
        self.calls: list[tuple[str, int, str | None]] = []  # (prompt, n, salt)

    def cache_key(self) -> tuple:
        """Constant, so a cached op result would be reused across instances."""
        return ("FlakyModel",)

    def generate_batch(
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
    ):
        """Return one result per prompt, or the exception for failing ones."""
        out: list = []
        for prompt in prompts:
            self.calls.append((prompt, n, cache_salt))
            if self.fail.get(prompt, 0) > 0:
                self.fail[prompt] -= 1
                exc = self.error(prompt)
                if not return_exceptions:
                    raise exc
                out.append(exc)
                continue
            out.append(_result([f"{prompt}#{j}" for j in range(n)]))
        return out


class OldStyleModel:
    """Duck-typed model whose generate_batch predates return_exceptions."""

    def __init__(self, fail: bool = False) -> None:
        """Raise on every call when ``fail``."""
        self.fail_calls = fail
        self.calls = 0

    def cache_key(self) -> tuple:
        """Stable op-cache key."""
        return ("OldStyleModel", self.fail_calls)

    def generate_batch(self, prompts, *, n=1, temperature=None, cache_salt=None):
        """Answer every prompt, or raise like a provider error."""
        self.calls += 1
        if self.fail_calls:
            raise InternalServerError("504")
        return [[f"{p}#{j}" for j in range(n)] for p in prompts]

    def generate(self, prompt, *, n=1, temperature=None, cache_salt=None):
        """Single-prompt form, used by sample_from_LLM."""
        return self.generate_batch([prompt], n=n, cache_salt=cache_salt)[0]


ITEMS = [{"q": "A"}, {"q": "B"}, {"q": "C"}]


class TestGenerateFromContext:
    """A provider error fails only its own prompt."""

    def test_failed_prompt_is_not_asked_again(self):
        """A prompt that 504s (after Model's retries) is None, not re-sent."""
        model = FlakyModel(fail={"B": 1})
        rows = generate_from_context(ITEMS, "{q}", model, max_retries=2)
        assert rows == [{"text": "A#0"}, None, {"text": "C#0"}]
        assert model.calls == [("A", 1, None), ("B", 1, None), ("C", 1, None)]
        assert trace()[-1].cache_hit is False
        # Not cached, so the next call asks again; a complete result is cached.
        rows = generate_from_context(ITEMS, "{q}", FlakyModel(), max_retries=2)
        assert rows == [{"text": "A#0"}, {"text": "B#0"}, {"text": "C#0"}]
        assert trace()[-1].cache_hit is False
        assert generate_from_context(ITEMS, "{q}", FlakyModel(), max_retries=2) == rows
        assert trace()[-1].cache_hit is True

    def test_parse_failure_is_retried_but_provider_failure_is_not(self):
        """A reply that does not parse is re-asked; a later 504 ends that slot."""

        class ParseThenFail(FlakyModel):
            def generate_batch(
                self,
                prompts,
                *,
                n=1,
                return_exceptions=False,
                cache_salt=None,
                cache_rows=None,
                **kwargs,
            ):
                if cache_salt is None:  # first attempt: an unparseable reply
                    self.calls.append((prompts[0], n, cache_salt))
                    return [_result(["not json"] * n) for _ in prompts]
                self.fail = dict.fromkeys(prompts, 1)  # the retry 504s
                return super().generate_batch(
                    prompts,
                    n=n,
                    return_exceptions=return_exceptions,
                    cache_salt=cache_salt,
                )

        model = ParseThenFail()
        rows = generate_from_context(
            [{"q": "A"}], "{q}", model, output_schema={"k": str}, max_retries=3
        )
        assert rows == [None]
        # One parse retry, which 504s; no third attempt.
        assert [salt for _, _, salt in model.calls] == [None, "retry 1, n=1"]
        assert trace()[-1].cache_hit is False

    def test_failure_gives_none_and_skips_cache(self, caplog):
        """A failed prompt's slots are None, one warning, and no cache."""
        model = FlakyModel(fail={"B": 3})
        with caplog.at_level(logging.WARNING):
            rows = generate_from_context(
                ITEMS, "{q}", model, samples_per_item=2, max_retries=2
            )
        assert rows == [
            {"text": "A#0"},
            {"text": "A#1"},
            None,
            None,
            {"text": "C#0"},
            {"text": "C#1"},
        ]
        assert [c for c in model.calls if c[0] == "B"] == [("B", 2, None)]
        warnings = [r.getMessage() for r in caplog.records if "failed after" in r.msg]
        assert warnings == [
            "alter.generate_from_context: 1 of 3 prompts failed after the model's "
            "retries; their slots are None: InternalServerError ×1 (result not cached)"
        ]
        # Not cached: the next call runs again and fills the slots.
        retry = FlakyModel()
        again = generate_from_context(
            ITEMS, "{q}", retry, samples_per_item=2, max_retries=2
        )
        assert None not in again
        assert retry.calls  # the op ran instead of replaying the Nones

    def test_parse_failures_are_still_cached(self):
        """None slots from unparseable output are cached as before."""
        # FlakyModel echoes the prompt, whose format example {"k": "<str>"} parses.
        model = MockModel(responses=["not json"])
        rows = generate_from_context(
            [{"q": "A"}], "{q}", model, output_schema={"k": str}, max_retries=1
        )
        assert rows == [None]
        generate_from_context(
            [{"q": "A"}],
            "{q}",
            MockModel(responses=["not json"]),
            output_schema={"k": str},
            max_retries=1,
        )
        assert trace()[-1].cache_hit is True

    def test_warning_in_step_report_and_step_not_cached(self, monkeypatch):
        """The warning lands on the op in the step report; the step is not cached."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        _tracking.clear_step_records()

        @step
        def answer(rows, model):
            return generate_from_context(rows, "{q}", model, max_retries=1)

        out = answer(ITEMS, FlakyModel(fail={"C": 1}))
        assert out[2] is None
        rec = _tracking.step_records()["answer"]
        message = (
            "alter.generate_from_context: 1 of 3 prompts failed after the model's "
            "retries; their slots are None: InternalServerError ×1 (result not cached)"
        )
        assert rec.ops[0].warnings == [message]
        assert message in rec.warnings
        assert rec.cache_hit is False

        healthy = FlakyModel()
        out = answer(ITEMS, healthy)
        assert out[2] == {"text": "C#0"}
        assert healthy.calls  # the step ran again
        assert _tracking.step_records()["answer"].cache_hit is False

        cached = FlakyModel()
        assert answer(ITEMS, cached) == out
        assert cached.calls == []
        assert _tracking.step_records()["answer"].cache_hit is True

    def test_model_without_return_exceptions_still_works(self):
        """A generate_batch without the keyword is called as before."""
        rows = generate_from_context(ITEMS, "{q}", OldStyleModel(), samples_per_item=2)
        assert rows[0] == {"text": "A#0"} and len(rows) == 6

    def test_model_without_return_exceptions_raises(self):
        """Without return_exceptions a provider error still raises (old behavior)."""
        with pytest.raises(InternalServerError):
            generate_from_context(ITEMS, "{q}", OldStyleModel(fail=True))

    def test_merge_drops_failed_slots(self):
        """With merge=True, rows of prompts that failed are left out."""
        rows = generate_from_context(
            ITEMS,
            "{q}",
            FlakyModel(fail={"B": 9}),
            max_retries=1,
            merge=True,
            output_field="answer",
        )
        assert rows == [
            {"q": "A", "answer": {"text": "A#0"}},
            {"q": "C", "answer": {"text": "C#0"}},
        ]


class FailingCompletions:
    """Fake litellm.completion: contents in ``fail`` raise that many times first."""

    def __init__(self, fail: dict[str, int], numbered: bool = False) -> None:
        """Fail each content in ``fail`` the given number of times.

        ``numbered`` answers "<content>!<k>" for the k-th answer to a content.
        """
        self.fail = dict(fail)
        self.numbered = numbered
        self.answered: dict[str, int] = {}
        self.lock = threading.Lock()
        self.calls: list[str] = []

    def __call__(self, messages, n=1, **kwargs):
        """Answer "<content>!" n times, or raise a 504."""
        text = messages[-1]["content"]
        with self.lock:
            self.calls.append(text)
            failing = self.fail.get(text, 0) > 0
            if failing:
                self.fail[text] -= 1
            first = self.answered.get(text, 0)
            if not failing:
                self.answered[text] = first + n
        if failing:
            raise InternalServerError(f"504 for {text}")
        suffixes = [str(first + j) if self.numbered else "" for j in range(n)]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"{text}!{suffix}"),
                    finish_reason="stop",
                )
                for suffix in suffixes
            ],
            usage=SimpleNamespace(
                prompt_tokens=1, completion_tokens=1, completion_tokens_details=None
            ),
            _hidden_params={"response_cost": 0.0},
        )


def test_rerun_resends_only_failed_prompts(monkeypatch):
    """After a provider failure, a rerun of the step re-sends only that prompt."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()
    completions = FailingCompletions(fail={"B": 1})
    # num_retries=0: one request per prompt, as the counts assume.
    model = Model("gpt-4o", show_progress=False, num_retries=0)

    @step
    def answer(rows, model):
        return generate_from_context(rows, "{q}", model, max_retries=2)

    with patch("pyforger.acquire.model.litellm") as litellm:
        litellm.get_supported_openai_params.return_value = None
        litellm.completion.side_effect = completions
        first = answer(ITEMS, model)
        assert first == [{"text": "A!"}, None, {"text": "C!"}]
        assert sorted(completions.calls) == ["A", "B", "C"]

        completions.calls.clear()
        second = answer(ITEMS, model)
        assert second == [{"text": "A!"}, {"text": "B!"}, {"text": "C!"}]
        assert completions.calls == ["B"]  # A and C come from the request cache
        assert _tracking.step_records()["answer"].cache_hit is False

        completions.calls.clear()
        assert answer(ITEMS, model) == second
        assert completions.calls == []
        assert _tracking.step_records()["answer"].cache_hit is True


def test_repeated_prompt_retry_gets_its_own_completion(monkeypatch):
    """A failed copy of a repeated prompt does not reuse the other copy's answer."""
    monkeypatch.setattr("pyforger.acquire.model._sleep", lambda seconds: None)
    completions = FailingCompletions(fail={"A": 1}, numbered=True)
    with patch("pyforger.acquire.model.litellm") as litellm:
        litellm.get_supported_openai_params.return_value = None
        litellm.completion.side_effect = completions
        rows = generate_from_context(
            [{"q": "A"}, {"q": "A"}],
            "{q}",
            Model("gpt-4o", show_progress=False, num_retries=1),
            max_retries=1,
        )
    assert sorted(row["text"] for row in rows) == ["A!0", "A!1"]
    assert completions.calls == ["A", "A", "A"]


class TestSampleFromLLM:
    """A failed sample_from_LLM request is not asked for again."""

    def test_failed_round_is_not_replaced(self):
        """A 504 on one round drops its samples; later rounds ask for the rest."""
        model = FlakyModel(fail={"Write.": 1})
        rows = sample_from_LLM("Write.", model, n=3, batch_size=1)
        assert rows == [{"text": "Write.#0"}, {"text": "Write.#0"}]
        assert model.calls == [
            ("Write.", 1, None),
            ("Write.", 1, "round 1"),
            ("Write.", 1, "round 2"),
        ]
        assert trace()[-1].cache_hit is False

    def test_missing_completions_are_not_replaced(self):
        """A round that returns 1 of 2 (a failed fan-out request) is not topped up."""

        class HalfModel(FlakyModel):
            def generate_batch(
                self,
                prompts,
                *,
                n=1,
                return_exceptions=False,
                cache_salt=None,
                **kwargs,
            ):
                out = super().generate_batch(
                    prompts,
                    n=n,
                    return_exceptions=return_exceptions,
                    cache_salt=cache_salt,
                )
                return [_result(list(result)[: n // 2]) for result in out]

        model = HalfModel()
        assert sample_from_LLM("Write.", model, n=2) == [{"text": "Write.#0"}]
        assert model.calls == [("Write.", 2, None)]
        assert trace()[-1].cache_hit is False

    def test_failures_return_short_and_skip_cache(self, caplog):
        """A failed request returns what arrived, warns, and is not cached."""
        model = FlakyModel(fail={"Write.": 5})
        with caplog.at_level(logging.WARNING):
            rows = sample_from_LLM("Write.", model, n=2, max_retries=2)
        assert rows == []
        assert len(model.calls) == 1
        assert (
            "acquire.sample_from_LLM: 0 of 2 samples; 1 request(s) failed: "
            "InternalServerError ×1 (result not cached)"
        ) in caplog.text
        healthy = FlakyModel()
        assert len(sample_from_LLM("Write.", healthy, n=2, max_retries=2)) == 2
        assert healthy.calls

    def test_duck_typed_model_uses_generate(self):
        """A model with only generate (no return_exceptions) works as before."""
        assert sample_from_LLM("Hi", OldStyleModel(), n=2) == [
            {"text": "Hi#0"},
            {"text": "Hi#1"},
        ]
        with pytest.raises(InternalServerError):
            sample_from_LLM("Hi", OldStyleModel(fail=True), n=2)


class TestSkipCacheWrite:
    """skip_cache_write keeps the running op and every enclosing op uncached."""

    def test_propagates_to_enclosing_ops_only(self):
        """The inner op, its caller, and the caller's caller rerun; siblings hit."""
        runs: list[str] = []

        @op(name="alter.test_inner_flaky", category="alter")
        def inner(x):
            runs.append("inner")
            skip_cache_write()
            return x

        @op(name="alter.test_sibling", category="alter")
        def sibling(x):
            runs.append("sibling")
            return x

        @op(name="alter.test_middle", category="alter")
        def middle(x):
            runs.append("middle")
            return inner(x) + sibling(x)

        @op(name="alter.test_outer", category="alter")
        def outer(x):
            runs.append("outer")
            return middle(x)

        assert outer(1) == 2
        assert outer(1) == 2
        # The second call reruns the chain above inner; sibling is cached.
        assert runs == ["outer", "middle", "inner", "sibling"] + [
            "outer",
            "middle",
            "inner",
        ]
        assert sibling(1) == 1
        assert trace()[-1].cache_hit is True

    def test_noop_outside_ops(self):
        """Called outside any op, it does nothing."""
        skip_cache_write()

    def test_nested_op_warning_reaches_top_level_op(self, monkeypatch):
        """add_op_warning in a nested op shows on the top-level op in the report."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        _tracking.clear_step_records()

        @op(name="alter.test_warn_inner", category="alter")
        def warn_inner(rows):
            _tracking.add_op_warning("inner trouble")
            return rows

        @op(name="alter.test_warn_outer", category="alter")
        def warn_outer(rows):
            return warn_inner(rows)

        @step
        def run(rows):
            return warn_outer(rows)

        run([{"x": 1}])
        rec = _tracking.step_records()["run"]
        assert rec.ops[0].warnings == ["inner trouble"]
        assert "inner trouble" in rec.warnings


def _litellm_error(cls):
    """Build ``cls`` (a litellm exception) for an openai model."""
    return lambda prompt: cls(message=f"{prompt}", llm_provider="openai", model="m")


class TestFatalErrors:
    """A bad key, unknown model, or bad request raises instead of becoming None."""

    @pytest.mark.parametrize(
        "cls",
        [
            litellm.Timeout,
            litellm.APIConnectionError,
            litellm.RateLimitError,
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
            litellm.BadGatewayError,
        ],
    )
    def test_transient(self, cls):
        """Timeouts, connection errors, 429, and 5xx are retried."""
        assert _is_transient(_litellm_error(cls)("p"))

    @pytest.mark.parametrize(
        "cls",
        [
            litellm.AuthenticationError,
            litellm.NotFoundError,
            litellm.BadRequestError,
            litellm.ContextWindowExceededError,
        ],
    )
    def test_fatal(self, cls):
        """401, 404, and 400 (including context length) are not."""
        assert not _is_transient(_litellm_error(cls)("p"))

    def test_builtin_and_unknown_errors(self):
        """Builtin timeouts are transient; an error with no status is not."""
        assert _is_transient(TimeoutError()) and _is_transient(ConnectionError())
        assert not _is_transient(RuntimeError("boom"))

    def test_generate_from_context_raises_on_bad_key(self):
        """An AuthenticationError on one prompt raises at once, without retries."""
        model = FlakyModel(
            fail={"B": 9}, error=_litellm_error(litellm.AuthenticationError)
        )
        with pytest.raises(litellm.AuthenticationError):
            generate_from_context(ITEMS, "{q}", model, max_retries=3, merge=True)
        assert [c[0] for c in model.calls] == ["A", "B", "C"]

    def test_generate_from_context_raises_when_out_of_quota(self):
        """A 429 with OpenAI's insufficient_quota raises: no retry fixes it."""

        def out_of_quota(prompt):
            return litellm.RateLimitError(
                message="Error code: 429 - {'error': {'code': 'insufficient_quota'}}",
                llm_provider="openai",
                model="m",
            )

        model = FlakyModel(fail={"B": 9}, error=out_of_quota)
        with pytest.raises(litellm.RateLimitError):
            generate_from_context(ITEMS, "{q}", model, max_retries=3)
        assert [c[0] for c in model.calls] == ["A", "B", "C"]

    def test_sample_from_llm_raises_on_unknown_model(self):
        """A NotFoundError (e.g. a typo in the model id) raises."""
        model = FlakyModel(fail={"Hi": 9}, error=_litellm_error(litellm.NotFoundError))
        with pytest.raises(litellm.NotFoundError):
            sample_from_LLM("Hi", model, n=2)
        assert len(model.calls) == 1

    def test_real_model_bad_request_raises(self):
        """Through Model and litellm, a 400 still raises from the op."""
        with patch("pyforger.acquire.model.litellm") as mock:
            mock.get_supported_openai_params.return_value = None
            mock.completion.side_effect = _litellm_error(litellm.BadRequestError)("Q")
            with pytest.raises(litellm.BadRequestError):
                generate_from_context(
                    ITEMS, "{q}", Model("gpt-4o", show_progress=False, num_retries=0)
                )


class TimingOut:
    """Fake litellm.completion: contents in ``hang`` always time out."""

    def __init__(self, hang: set[str], unparseable: set[str]) -> None:
        """Time out on ``hang``; answer ``unparseable`` once with non-JSON."""
        self.hang = hang
        self.unparseable = set(unparseable)
        self.lock = threading.Lock()
        self.calls: list[str] = []

    def __call__(self, messages, n=1, **kwargs):
        """Raise litellm.Timeout, or answer {"k": "<content>"}."""
        # The prompt's first line is the row's text; the schema hint follows.
        text = messages[-1]["content"].split("\n")[0]
        with self.lock:
            self.calls.append(text)
            if text in self.hang:
                raise litellm.Timeout(
                    f"{text} timed out", model="gpt-4o", llm_provider="openai"
                )
            content = "not json" if text in self.unparseable else f'{{"k": "{text}"}}'
            self.unparseable.discard(text)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason="stop"
                )
                for _ in range(n)
            ],
            usage=SimpleNamespace(
                prompt_tokens=1, completion_tokens=1, completion_tokens_details=None
            ),
            _hidden_params={"response_cost": 0.0},
        )


@pytest.mark.parametrize("num_retries", [0, 2])
def test_timed_out_request_makes_num_retries_plus_one_attempts(
    num_retries, monkeypatch
):
    """A request that always times out is sent num_retries + 1 times in all.

    Through Model and a fake litellm.completion, whatever the op's
    max_retries; a reply that does not parse is still asked again.
    """
    sleeps: list[float] = []
    monkeypatch.setattr("pyforger.acquire.model._sleep", sleeps.append)
    completions = TimingOut(hang={"B"}, unparseable={"C"})
    model = Model("gpt-4o", show_progress=False, num_retries=num_retries)
    with patch("pyforger.acquire.model.litellm") as mock:
        mock.get_supported_openai_params.return_value = None
        mock.completion.side_effect = completions
        rows = generate_from_context(
            ITEMS, "{q}", model, output_schema={"k": str}, max_retries=3
        )
        assert rows == [{"k": "A"}, None, {"k": "C"}]
        assert completions.calls.count("B") == num_retries + 1
        assert completions.calls.count("C") == 2  # one parse retry
        assert len(sleeps) == num_retries

        # sample_from_LLM: the same bound for its one request.
        completions.calls.clear()
        assert sample_from_LLM("B", model, n=2, max_retries=3) == []
        assert completions.calls == ["B"] * (num_retries + 1)


def test_plain_thread_pool_step_is_not_cached(monkeypatch):
    """Ops in plain pool.map workers still keep their step out of the cache."""
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()

    @step
    def answer(rows, model):
        chunks = [rows[:2], rows[2:]]
        with ThreadPoolExecutor(max_workers=2) as pool:  # no copy_context
            parts = list(
                pool.map(
                    lambda chunk: generate_from_context(
                        chunk, "{q}", model, max_retries=1
                    ),
                    chunks,
                )
            )
        return filter_none(parts[0] + parts[1])

    first = answer(ITEMS, FlakyModel(fail={"C": 2}))
    assert first == [{"text": "A#0"}, {"text": "B#0"}]
    rec = _tracking.step_records()["answer"]
    assert any("worker thread" in w for w in rec.warnings)

    healthy = FlakyModel()
    second = answer(ITEMS, healthy)
    assert second == [{"text": "A#0"}, {"text": "B#0"}, {"text": "C#0"}]
    assert healthy.calls  # the step ran instead of replaying the failed result
    assert _tracking.step_records()["answer"].cache_hit is False

    cached = FlakyModel()
    assert answer(ITEMS, cached) == second
    assert cached.calls == []


class _Event(threading.Event):
    def cache_key(self) -> str:
        return "event"


@pytest.mark.parametrize("where", ["step", "submitted step", "bare op"])
def test_failure_elsewhere_does_not_uncache_a_submitted_step(monkeypatch, where):
    """A provider failure outside a submitted step leaves that step cached."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()
    started, failed = _Event(), _Event()
    runs: list[int] = []

    @op(name="alter.test_wait_for_failure", category="alter")
    def wait_for_failure(rows):
        runs.append(1)
        started.set()
        failed.wait(10)  # still running when the other op skips its write
        return list(rows)

    @step
    def slow(rows):
        return filter_none(wait_for_failure(rows))

    @step
    def flaky(rows, model):
        return filter_none(generate_from_context(rows, "{q}", model))

    @pipeline
    def main(rows):
        future = slow.submit(rows)
        started.wait(10)
        model = FlakyModel(fail={"A": 1})
        if where == "step":
            flaky(rows, model)
        elif where == "submitted step":
            flaky.submit(rows, model).result()
        else:
            generate_from_context(rows, "{q}", model)
        failed.set()
        return future.result()

    main(ITEMS)
    rec = _tracking.step_records()["slow"]
    assert rec.warnings == [] and not rec.cache_hit
    main(ITEMS)  # both events stay set: slow is a cache hit, nothing waits
    assert _tracking.step_records()["slow"].cache_hit
    assert runs == [1]


@op(name="filter.test_drop_none", category="filter")
def filter_none(rows):
    """Drop None slots (a filter, so the step ends with one)."""
    return [row for row in rows if row is not None]


def _content_filter(prompt):
    """Azure's content-filter 400, as litellm raises it."""
    return litellm.BadRequestError(
        message=(
            "AzureException BadRequestError - {'error': {'code': 'content_filter', "
            f"'message': 'Response content blocked for {prompt}'}}}}"
        ),
        llm_provider="azure",
        model="m",
    )


def _context_window(prompt):
    """The error litellm raises for a prompt over the context window."""
    return litellm.ContextWindowExceededError(
        message=f"{prompt} is too long", llm_provider="openai", model="m"
    )


class TestRowErrors:
    """A content-filter block or an over-long prompt fails only its own row."""

    @pytest.mark.parametrize(
        "error",
        [
            _litellm_error(litellm.ContentPolicyViolationError)("p"),
            _content_filter("p"),
            litellm.BadRequestError(
                message="violates content_policy", llm_provider="openai", model="m"
            ),
            _context_window("p"),
            PromptTooLongError(
                "Prompt 17 has 69685 tokens; the context window is 32768"
            ),
        ],
    )
    def test_row_errors(self, error):
        """Each is a row error, and so never transient."""
        assert _is_row_error(error)
        assert not _is_transient(error)

    @pytest.mark.parametrize(
        "error",
        [
            _litellm_error(litellm.BadRequestError)("unsupported parameter"),
            _litellm_error(litellm.AuthenticationError)("p"),
            _litellm_error(litellm.InternalServerError)("p"),
            ValueError("content_filter"),  # not a provider error
        ],
    )
    def test_not_row_errors(self, error):
        """Other 400s, auth errors, 5xx, and plain errors are not."""
        assert not _is_row_error(error)

    @pytest.mark.parametrize(
        ("error", "reason"),
        [
            (_content_filter, "1 prompt blocked by the provider's content filter"),
            (_context_window, "1 prompt longer than the model's context window"),
        ],
    )
    def test_generate_from_context_leaves_row_none(self, error, reason, caplog):
        """The op returns, the row's slot is None, and the warning names why."""
        model = FlakyModel(fail={"B": 9}, error=error)
        with caplog.at_level(logging.WARNING):
            rows = generate_from_context(ITEMS, "{q}", model, max_retries=3)
        assert rows == [{"text": "A#0"}, None, {"text": "C#0"}]
        assert [c[0] for c in model.calls] == ["A", "B", "C"]  # B not asked again
        assert (
            "alter.generate_from_context: 1 of 3 prompts failed after the model's "
            f"retries; their slots are None: {reason} (result not cached)"
        ) in caplog.text
        assert trace()[-1].cache_hit is False

    def test_reasons_and_types_are_counted_together(self, monkeypatch):
        """The step report counts row errors by reason next to other failures."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        _tracking.clear_step_records()

        def error(prompt):
            if prompt in ("A", "B"):
                return _content_filter(prompt)
            return InternalServerError("504")

        @step
        def answer(rows, model):
            return generate_from_context(rows, "{q}", model)

        items = [*ITEMS, {"q": "D"}]
        answer(items, FlakyModel(fail={"A": 1, "B": 1, "C": 1}, error=error))
        assert _tracking.step_records()["answer"].ops[0].warnings == [
            "alter.generate_from_context: 3 of 4 prompts failed after the model's "
            "retries; their slots are None: 2 prompts blocked by the provider's "
            "content filter, InternalServerError ×1 (result not cached)"
        ]

    def test_every_prompt_row_error_raises(self):
        """All prompts blocked points at the template: the first error raises."""
        model = FlakyModel(fail={"A": 1, "B": 1, "C": 1}, error=_content_filter)
        with pytest.raises(litellm.BadRequestError, match="blocked for A"):
            generate_from_context(ITEMS, "{q}", model)

    def test_single_prompt_row_error_does_not_raise(self):
        """With one prompt, a row error is that row's problem."""
        model = FlakyModel(fail={"A": 1}, error=_context_window)
        assert generate_from_context([{"q": "A"}], "{q}", model) == [None]

    def test_bad_request_without_content_filter_raises(self):
        """A 400 that is not a content-filter block still fails the op."""

        def unsupported(prompt):
            return litellm.BadRequestError(
                message="Unsupported value: 'temperature'",
                llm_provider="openai",
                model="m",
            )

        model = FlakyModel(fail={"B": 1}, error=unsupported)
        with pytest.raises(litellm.BadRequestError):
            generate_from_context(ITEMS, "{q}", model)

    def test_model_does_not_retry_a_row_error(self, monkeypatch):
        """Through Model and litellm, a blocked request is sent once."""
        sleeps: list[float] = []
        monkeypatch.setattr("pyforger.acquire.model._sleep", sleeps.append)
        completions = FailingCompletions(fail={})

        def blocked(messages, n=1, **kwargs):
            if messages[-1]["content"] == "B":
                completions.calls.append("B")
                raise _litellm_error(litellm.ContentPolicyViolationError)("B")
            return completions(messages, n=n, **kwargs)

        model = Model("gpt-4o", show_progress=False, num_retries=2)
        with patch("pyforger.acquire.model.litellm") as mock:
            mock.get_supported_openai_params.return_value = None
            mock.completion.side_effect = blocked
            rows = generate_from_context(ITEMS, "{q}", model, max_retries=2)
        assert rows == [{"text": "A!"}, None, {"text": "C!"}]
        assert completions.calls.count("B") == 1
        assert sleeps == []

    def test_sample_from_llm_raises_when_its_prompt_is_blocked(self):
        """sample_from_LLM's one prompt is the template: no samples, so it raises."""
        model = FlakyModel(fail={"Write.": 9}, error=_content_filter)
        with pytest.raises(litellm.BadRequestError):
            sample_from_LLM("Write.", model, n=2)
        assert len(model.calls) == 1

    def test_sample_from_llm_keeps_other_rounds(self, caplog):
        """One blocked round drops its samples; the others are returned."""
        model = FlakyModel(fail={"Write.": 1}, error=_content_filter)
        with caplog.at_level(logging.WARNING):
            rows = sample_from_LLM("Write.", model, n=3, batch_size=1)
        assert len(rows) == 2
        assert (
            "acquire.sample_from_LLM: 2 of 3 samples; 1 request(s) failed: "
            "1 prompt blocked by the provider's content filter (result not cached)"
        ) in caplog.text
