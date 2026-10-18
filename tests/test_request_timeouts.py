"""Model's default request timeout, progress while waiting, straggler warnings."""

from __future__ import annotations

import logging
import pickle
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import litellm
import openai
import pytest

import pyforger.acquire.model as model_module
from pyforger import op
from pyforger.acquire.model import Model, _duration

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

    def test_timed_out_attempt_is_retried(self):
        """A timed-out attempt raises litellm.Timeout (transient) and is retried.

        Real litellm with mock_timeout: no request leaves the process. One
        attempt plus num_retries=2 retries.
        """
        attempts = []
        handle = litellm.main._handle_mock_timeout

        def count(*args, **kwargs):
            attempts.append(kwargs.get("timeout"))
            return handle(*args, **kwargs)

        model = Model(
            "openai/gpt-4o", api_key="sk-test", timeout=0.01, mock_timeout=True
        )
        with patch.object(litellm.main, "_handle_mock_timeout", count):
            with pytest.raises(litellm.Timeout) as excinfo:
                model.generate_batch(["A"])
        assert isinstance(excinfo.value, openai.APITimeoutError)
        assert attempts == [0.01, 0.01, 0.01]


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
        assert caplog.records == []


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
        assert [r for r in caplog.records if r.levelno == logging.INFO] == []

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
