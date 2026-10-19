"""Model's default request timeout, progress while waiting, straggler warnings."""

from __future__ import annotations

import logging
import pickle
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import litellm
import openai
import pytest

import pyforger.acquire.model as model_module
from pyforger import op
from pyforger._tracking import BudgetExceededError
from pyforger.acquire.model import Model, _duration, _is_transient
from pyforger.tinker_usage import TinkerBudgetExceeded
from pyforger.token_units import TokenUnitQuotaExceeded

_LOGGER = "pyforger.acquire.model"


def make_response(text):
    """A litellm-shaped response with one choice."""
    choice = MagicMock()
    choice.message.content = text
    choice.message.reasoning_content = None
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = 1
    response.usage.completion_tokens = 1
    response.usage.completion_tokens_details.reasoning_tokens = 0
    response._hidden_params = {"response_cost": 0.0}
    return response


class Delayed:
    """Fake completion: answers each prompt after its delay (default 0)."""

    def __init__(self, delays=None):
        """`delays` maps prompt text to seconds."""
        self.delays = delays or {}

    def __call__(self, messages, **kwargs):
        """Fake litellm.completion."""
        text = messages[0]["content"]
        time.sleep(self.delays.get(text, 0))
        return make_response(text)


@pytest.fixture
def mock_litellm():
    """Patch litellm in pyforger.acquire.model; registry knows no model."""
    with patch("pyforger.acquire.model.litellm") as mock:
        mock.get_supported_openai_params.return_value = None
        mock.completion.side_effect = Delayed()
        yield mock


@pytest.fixture
def fast_clock(monkeypatch):
    """Check every 20ms; a request 100ms (and 5x the median) in is a straggler."""
    monkeypatch.setattr(model_module, "_PROGRESS_LOG_INTERVAL_S", 0.02)
    monkeypatch.setattr(model_module, "_STRAGGLER_MIN_S", 0.1)


def sent_timeouts(mock_litellm):
    """The timeout of each litellm.completion call; None when not passed."""
    return [c.kwargs.get("timeout") for c in mock_litellm.completion.call_args_list]


class TestDefaultTimeout:
    """Model passes a per-request timeout to litellm unless told otherwise."""

    def test_default_is_ten_minutes(self, mock_litellm):
        """No timeout and no max_tokens: 600s per attempt."""
        Model("gpt-4o").generate_batch(["A", "B"])
        assert sent_timeouts(mock_litellm) == [600.0, 600.0]

    def test_default_grows_with_max_tokens(self, mock_litellm):
        """A long max_tokens gets max_tokens / 25 seconds, per call too."""
        model = Model("gpt-4o", max_tokens=32768)
        model.generate_batch(["A"])
        model.generate_batch(["A"], max_tokens=20000)
        model.generate_batch(["A"], max_tokens=1000)
        assert sent_timeouts(mock_litellm) == [1311.0, 800.0, 600.0]

    def test_explicit_timeout_wins(self, mock_litellm):
        """Model(timeout=90) sends 90, whatever max_tokens is."""
        Model("gpt-4o", timeout=90, max_tokens=50000).generate_batch(["A"])
        assert sent_timeouts(mock_litellm) == [90]

    def test_none_leaves_litellm_default(self, mock_litellm):
        """Model(timeout=None) sends no timeout, so LiteLLM's own default applies."""
        Model("gpt-4o", timeout=None).generate_batch(["A"])
        assert "timeout" not in mock_litellm.completion.call_args.kwargs

    def test_timeout_is_not_in_the_cache_key(self, mock_litellm):
        """Changing the timeout keeps @op results cached."""
        assert (
            Model("gpt-4o", timeout=5).cache_key()
            == Model("gpt-4o").cache_key()
            == Model("gpt-4o", timeout=None).cache_key()
        )
        calls = []

        @op(name="test.ask")
        def ask(model: Model) -> list:
            calls.append(model)
            return list(model.generate("A"))

        assert ask(Model("gpt-4o")) == ask(Model("gpt-4o", timeout=30)) == ["A"]
        assert len(calls) == 1

    def test_unpickled_old_model_gets_the_default(self, mock_litellm):
        """A Model pickled before timeouts existed sends the default."""
        model = Model("gpt-4o")
        del model._timeout
        pickle.loads(pickle.dumps(model)).generate_batch(["A"])
        assert sent_timeouts(mock_litellm) == [600.0]

    @pytest.mark.parametrize("num_retries", [0, 1, 2])
    def test_timed_out_attempt_is_retried(self, num_retries, monkeypatch):
        """A timed-out attempt is retried: num_retries + 1 attempts in all.

        Real litellm with mock_timeout: no request leaves the process. LiteLLM
        gets num_retries=0, so only Model retries.
        """
        monkeypatch.setattr(model_module, "_sleep", lambda seconds: None)
        attempts = []
        handle = litellm.main._handle_mock_timeout

        def count(*args, **kwargs):
            attempts.append(kwargs.get("timeout"))
            return handle(*args, **kwargs)

        model = Model(
            "openai/gpt-4o",
            api_key="sk-test",
            timeout=0.01,
            num_retries=num_retries,
            mock_timeout=True,
        )
        with patch.object(litellm.main, "_handle_mock_timeout", count):
            with pytest.raises(litellm.Timeout) as excinfo:
                model.generate_batch(["A"])
        assert isinstance(excinfo.value, openai.APITimeoutError)
        assert attempts == [0.01] * (num_retries + 1)


# Gemini's per-minute rate limit: a retry fixes it.
GEMINI_RPM = (
    'VertexAIException - {"error": {"code": 429, "message": "You exceeded your '
    "current quota, please check your plan and billing details. Please retry in "
    '10s.", "status": "RESOURCE_EXHAUSTED"}}'
)
# OpenAI's hard quota: no retry fixes it.
OPENAI_QUOTA = (
    "You exceeded your current quota, please check your plan and billing "
    "details. {'error': {'type': 'insufficient_quota', 'code': "
    "'insufficient_quota'}}"
)


def rate_limit(message="slow down", retry_after=None):
    """A litellm 429, optionally with a Retry-After header."""
    headers = {} if retry_after is None else {"retry-after": retry_after}
    response = httpx.Response(
        429, headers=headers, request=httpx.Request("POST", "http://test")
    )
    return litellm.RateLimitError(
        message, llm_provider="openai", model="gpt-4o", response=response
    )


class Failing:
    """Fake completion: raises the errors in order, then answers."""

    def __init__(self, *errors):
        """Raise `errors` on the first calls."""
        self.errors = list(errors)
        self.calls = 0

    def __call__(self, messages, **kwargs):
        """Fake litellm.completion."""
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return make_response(messages[0]["content"])


@pytest.fixture
def sleeps(monkeypatch):
    """The backoff delays Model sleeps, without sleeping."""
    delays: list[float] = []
    monkeypatch.setattr(model_module, "_sleep", delays.append)
    return delays


class TestRetries:
    """Model retries transient errors itself; LiteLLM is told not to."""

    def test_litellm_gets_num_retries_zero(self, mock_litellm):
        """Neither LiteLLM's wrapper nor the openai client retries."""
        Model("gpt-4o", num_retries=5).generate_batch(["A"])
        kwargs = mock_litellm.completion.call_args.kwargs
        assert kwargs["num_retries"] == 0
        assert "max_retries" not in kwargs

    def test_explicit_max_retries_reaches_the_client(self, mock_litellm):
        """A caller's max_retries is sent, without the num_retries that overrides it."""
        Model("gpt-4o", max_retries=4).generate_batch(["A"])
        kwargs = mock_litellm.completion.call_args.kwargs
        assert kwargs["max_retries"] == 4
        assert "num_retries" not in kwargs

    def test_429_retried_with_backoff(self, mock_litellm, sleeps):
        """Two 429s, then an answer: 3 attempts, waits of about 1s then 2s."""
        fake = Failing(rate_limit(), rate_limit())
        mock_litellm.completion.side_effect = fake
        assert Model("gpt-4o").generate_batch(["A"]) == [["A"]]
        assert fake.calls == 3
        assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 2.0

    def test_retry_after_is_honoured(self, mock_litellm, sleeps):
        """A 429 with Retry-After: 7 waits 7s; a huge one is capped at 60s."""
        fake = Failing(rate_limit(retry_after="7"), rate_limit(retry_after="3600"))
        mock_litellm.completion.side_effect = fake
        Model("gpt-4o").generate_batch(["A"])
        assert sleeps == [7.0, 60.0]

    def test_gives_up_after_num_retries(self, mock_litellm, sleeps):
        """num_retries=2: the third 503 is raised, with no fourth attempt."""
        errors = [
            litellm.ServiceUnavailableError("busy", llm_provider="openai", model="m")
            for _ in range(4)
        ]
        fake = Failing(*errors)
        mock_litellm.completion.side_effect = fake
        with pytest.raises(litellm.ServiceUnavailableError):
            Model("gpt-4o", num_retries=2).generate_batch(["A"])
        assert fake.calls == 3
        assert len(sleeps) == 2

    @pytest.mark.parametrize(
        "error",
        [
            litellm.BadRequestError("too long", model="m", llm_provider="openai"),
            litellm.AuthenticationError("bad key", llm_provider="openai", model="m"),
            litellm.NotFoundError("no model", model="m", llm_provider="openai"),
            litellm.PermissionDeniedError(
                "denied",
                llm_provider="openai",
                model="m",
                response=httpx.Response(403, request=httpx.Request("POST", "http://t")),
            ),
            rate_limit(OPENAI_QUOTA),
            litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0),
        ],
        ids=["400", "401", "404", "403", "quota-429", "budget"],
    )
    def test_permanent_errors_are_not_retried(self, mock_litellm, sleeps, error):
        """A bad request, key, model, permission, or exhausted quota: one attempt."""
        fake = Failing(error)
        mock_litellm.completion.side_effect = fake
        with pytest.raises(type(error)):
            Model("gpt-4o").generate_batch(["A"])
        assert fake.calls == 1
        assert sleeps == []

    def test_gemini_rate_limit_is_retried(self, mock_litellm, sleeps):
        """Gemini's per-minute 429 says "exceeded your current quota" but is retried."""
        fake = Failing(rate_limit(GEMINI_RPM))
        mock_litellm.completion.side_effect = fake
        assert Model("gemini/gemini-2.5-pro").generate_batch(["A"]) == [["A"]]
        assert fake.calls == 2
        assert len(sleeps) == 1

    def test_retried_inside_a_batch(self, mock_litellm, sleeps):
        """In a batch, only the failing request is asked again."""
        calls: list[str] = []
        failed: set[str] = set()

        def fake(messages, **kwargs):
            text = messages[0]["content"]
            calls.append(text)
            if text == "B" and text not in failed:
                failed.add(text)
                raise litellm.InternalServerError("504", llm_provider="o", model="m")
            return make_response(text)

        mock_litellm.completion.side_effect = fake
        model = Model("gpt-4o", show_progress=False)
        assert model.generate_batch(["A", "B", "C"]) == [["A"], ["B"], ["C"]]
        assert sorted(calls) == ["A", "B", "B", "C"]
        assert len(sleeps) == 1

    def test_retried_response_is_charged_once(self, mock_litellm, sleeps):
        """A 429, then an answer: the totals count the one response that arrived."""
        mock_litellm.completion.side_effect = Failing(rate_limit())
        model = Model("gpt-4o")
        model.generate_batch(["A"])
        assert model.total_usage["prompt_tokens"] == 1
        assert len(model.history) == 1

    @pytest.mark.parametrize(
        "cap",
        [TokenUnitQuotaExceeded("quota reached"), BudgetExceededError("cap reached")],
        ids=["token-unit-quota", "spending-cap"],
    )
    def test_caps_are_checked_before_each_retry(
        self, mock_litellm, sleeps, monkeypatch, cap
    ):
        """A 504, then a cap reached before the retry: the cap raises, no retry."""
        fake = Failing(litellm.InternalServerError("504", llm_provider="o", model="m"))
        mock_litellm.completion.side_effect = fake
        model = Model("gpt-4o")
        checks: list[int] = []

        def before_request():
            checks.append(1)
            if len(checks) > 1:
                raise cap

        monkeypatch.setattr(model, "_before_request", before_request)
        with pytest.raises(type(cap)):
            model.generate_batch(["A"])
        assert fake.calls == 1
        assert len(checks) == 2  # the first attempt and the refused retry
        assert len(sleeps) == 1


@pytest.mark.parametrize(
    "error, transient",
    [
        (litellm.Timeout("t", model="m", llm_provider="openai"), True),
        (litellm.APIConnectionError("reset", llm_provider="openai", model="m"), True),
        (litellm.InternalServerError("ise", llm_provider="openai", model="m"), True),
        (TimeoutError(), True),
        (ConnectionResetError(), True),
        (rate_limit(), True),
        (rate_limit(GEMINI_RPM), True),
        (rate_limit(OPENAI_QUOTA), False),
        (rate_limit("You have hit billing_hard_limit_reached"), False),
        (rate_limit("Your credit balance is too low"), False),
        (litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0), False),
        (type("TokenUnitQuotaExceeded", (RuntimeError,), {})("230M units"), False),
        (TokenUnitQuotaExceeded("230M units"), False),
        (BudgetExceededError("max_cost_usd=20 reached"), False),
        (TinkerBudgetExceeded("$150 reached"), False),
        (RuntimeError("no status"), False),
    ],
)
def test_is_transient(error, transient):
    """Timeouts, lost connections, 429s and 5xx are transient; quota is not."""
    assert _is_transient(error) is transient


class TestHeartbeat:
    """The log line keeps coming while requests are in flight."""

    def test_line_without_completions(self, mock_litellm, fast_clock, caplog):
        """Nothing finished for a while: "0/2 done" lines, then "2/2 done"."""
        mock_litellm.completion.side_effect = Delayed({"A": 0.15, "B": 0.15})
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            Model("gpt-4o", show_progress=True).generate_batch(["A", "B"])
        lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        waiting = [line for line in lines if " 0/2 done" in line]
        assert len(waiting) >= 2
        assert lines[-1].startswith("LLM calls (gpt-4o): 2/2 done, ")

    def test_fast_batch_logs_nothing(self, mock_litellm, caplog):
        """A batch that finishes within the interval logs no line."""
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            Model("gpt-4o", show_progress=True).generate_batch(["A", "B"])
        assert not [r for r in caplog.records if "LLM calls" in r.getMessage()]


class TestStragglers:
    """One warning per request in flight far longer than the finished ones."""

    def stragglers(self, caplog):
        """The warning messages logged."""
        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    def test_one_warning_per_straggler(self, mock_litellm, fast_clock, caplog):
        """A request still running long after the others is warned about once."""
        mock_litellm.completion.side_effect = Delayed({"slow": 0.4})
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            Model("gpt-4o", show_progress=True).generate_batch(["a", "b", "c", "slow"])
        assert self.stragglers(caplog) == [
            "LLM calls (gpt-4o): 1 request in flight for 0s (median of completed "
            "requests 0s); each attempt times out after 10m (Model(timeout=...)) "
            "and is retried"
        ]

    def test_two_stragglers_one_warning(self, mock_litellm, fast_clock, caplog):
        """Requests that turn late at the same check share one warning."""
        mock_litellm.completion.side_effect = Delayed({"x": 0.3, "y": 0.3})
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            Model("gpt-4o", timeout=None, show_progress=False).generate_batch(
                ["a", "b", "x", "y"]
            )
        (warning,) = self.stragglers(caplog)
        assert ": 2 requests in flight for " in warning
        assert warning.endswith("Model(timeout=None) leaves LiteLLM's default timeout")

    def test_no_warning_without_a_finished_request(
        self, mock_litellm, fast_clock, caplog
    ):
        """Requests that all run long are not stragglers: nothing to compare with."""
        mock_litellm.completion.side_effect = Delayed({"a": 0.25, "b": 0.25})
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            Model("gpt-4o", show_progress=True).generate_batch(["a", "b"])
        assert self.stragglers(caplog) == []

    def test_threshold_follows_the_median(self, mock_litellm, fast_clock, caplog):
        """5x the median of finished requests: 0.1s calls make 0.3s not late."""
        delays = {"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.3}
        mock_litellm.completion.side_effect = Delayed(delays)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            Model("gpt-4o", show_progress=True).generate_batch(list(delays))
        assert self.stragglers(caplog) == []

    def test_reported_with_a_bar(self, mock_litellm, fast_clock, monkeypatch, caplog):
        """In a terminal (tqdm bar) the straggler is still warned about."""
        import tqdm

        monkeypatch.setattr(
            sys, "stderr", SimpleNamespace(isatty=lambda: True, write=lambda s: None)
        )
        monkeypatch.setattr(tqdm, "tqdm", lambda iterable, **kwargs: iterable)
        mock_litellm.completion.side_effect = Delayed({"slow": 0.3})
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            Model("gpt-4o").generate_batch(["a", "slow"])
        assert len(self.stragglers(caplog)) == 1
        # The bar replaces the log line.
        assert not [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "LLM calls" in r.getMessage()
        ]

    def test_waits_without_spinning(self, mock_litellm, monkeypatch):
        """Between checks the caller sleeps in wait(); it does not poll."""
        monkeypatch.setattr(model_module, "_PROGRESS_LOG_INTERVAL_S", 0.1)
        calls = []
        real_wait = model_module.wait

        def counting_wait(*args, **kwargs):
            calls.append(kwargs.get("timeout"))
            return real_wait(*args, **kwargs)

        monkeypatch.setattr(model_module, "wait", counting_wait)
        mock_litellm.completion.side_effect = Delayed({"a": 0.35, "b": 0.35})
        Model("gpt-4o", show_progress=False).generate_batch(["a", "b"])
        assert len(calls) <= 6  # ~4 checks at 0.1s and the completions


def test_duration():
    """Durations read like a clock: seconds, then minutes, then hours."""
    assert [_duration(s) for s in (41, 552, 600, 3900)] == [
        "41s",
        "9m 12s",
        "10m",
        "1h 5m",
    ]
