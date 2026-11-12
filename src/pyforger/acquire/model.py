"""LLM model wrapper for Forger.

Provides a simplified interface to LiteLLM for generating text
completions from any supported LLM provider.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import math
import os
import random
import re
import statistics
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Literal, Union, overload

import litellm

# Bound by name: tests patch this module's `litellm`, and isinstance needs classes.
from litellm.exceptions import BadRequestError as LitellmBadRequestError
from litellm.exceptions import ContentPolicyViolationError, ContextWindowExceededError

from pyforger import _tracking
from pyforger._progress import Progress
from pyforger._progress import done_line as _done_line
from pyforger._progress import duration as _duration
from pyforger._progress import progress_mode as _progress_mode
from pyforger._progress import vllm_tqdm
from pyforger.acquire import _local_dp, _request_cache
from pyforger.op import _NO_CACHE_SCOPE, _REQUEST_SCOPE, _hash
from pyforger.token_units import (
    TokenUnitBudget,
    TokenUnitQuotaExceeded,
    held_units,
    response_cost_usd,
    response_token_counts,
    token_units,
)
from pyforger.utils import get_formatted_logger

# LiteLLM prints "Give Feedback / Get Help: <url>" and a debug hint to stdout before
# raising each provider error; the raised error already says what went wrong.
# Set it back to False to see them.
litellm.suppress_debug_info = True

# Type aliases
Message = dict[str, str]
Prompt = Union[str, list[Message]]

# Threads per generate_batch call when the Model sets no max_concurrency.
_DEFAULT_MAX_WORKERS = 100
# Most completions asked of one request. A larger n is split into concurrent
# requests, e.g. n=40 on gpt-4o sends n=14, 13 and 13.
_MAX_N_PER_REQUEST = 16
# A batch projects its spend once this share of its completions (and at least
# _PROJECT_MIN_DONE requests) has finished, e.g. after 160 of 3,200; batches
# under _PROJECT_MIN_TOTAL completions are not projected.
_PROJECT_MIN_SHARE = 0.05
_PROJECT_MIN_DONE = 10
_PROJECT_MIN_TOTAL = 40
# Without a terminal (stderr redirected to a log file), progress is a plain log
# line this often instead of a redrawn bar, also while nothing finishes.
_PROGRESS_LOG_INTERVAL_S = 30.0
# A request in flight longer than both of these (seconds, and times the median
# duration of the batch's finished requests) gets one warning, e.g. at 9
# minutes when the others took 41s.
_STRAGGLER_MIN_S = 120.0
_STRAGGLER_FACTOR = 5.0
# Model(timeout="auto"): each request attempt may take this long, or longer
# when max_tokens at this slow a rate needs it (max_tokens=32768 gets 1311s).
# LiteLLM's own default is 6000s, so a stuck request blocks for 100 minutes.
_DEFAULT_TIMEOUT_S = 600.0
_SLOW_TOKENS_PER_S = 25.0
# Model(num_retries=) waits before retry k (0, 1, ...) about this many seconds
# times 2**k, at most _RETRY_MAX_S: 1s, 2s, 4s, ... A 429's Retry-After header
# is honoured up to _RETRY_AFTER_MAX_S.
_RETRY_BASE_S = 1.0
_RETRY_MAX_S = 30.0
_RETRY_AFTER_MAX_S = 60.0
# A rate-limit 429 (one that does not say the quota or budget is spent) gets at
# least this many retries, about 5s, 10s, 20s, 40s, 60s, 60s apart: a
# per-minute limit needs a minute to reset, and 3 attempts within 3 seconds
# only add to the load. num_retries=0 still means no retries at all.
_RATE_LIMIT_RETRIES = 6
_RATE_LIMIT_BASE_S = 5.0
_RATE_LIMIT_MAX_S = 60.0
# After a 429, requests to that model (from any Model instance in the process)
# are held to half as many in flight as when it hit, never fewer than
# _MIN_IN_FLIGHT; each success after that allows _IN_FLIGHT_GROWTH more.
_MIN_IN_FLIGHT = 4
_IN_FLIGHT_GROWTH = 0.25
# Status codes worth asking again: request timeout, conflict, too early, rate
# limit. Every 5xx is too.
_TRANSIENT_STATUS = frozenset({408, 409, 425, 429})
# A 429 (or other error) that means the account is out of quota or budget,
# which no retry fixes: OpenAI's "insufficient_quota" or "billing_hard_limit_
# reached" code, Anthropic's "credit balance is too low", LiteLLM's
# BudgetExceededError, Forger's spending caps (TokenUnitQuotaExceeded). Not
# "You exceeded your current quota ... billing details" alone: Gemini says
# that for a per-minute rate limit, which a retry does fix.
_QUOTA_MARKERS = (
    "insufficient_quota",
    "billing_",
    "credit balance",
    "budget",
    "quotaexceeded",
)

_logger = get_formatted_logger(__name__)


def _request_timeout(timeout: Any, max_tokens: int | None) -> Any:
    """The `timeout` litellm gets per request attempt; "auto" picks one."""
    if timeout != "auto":
        return timeout
    if max_tokens is None:
        return _DEFAULT_TIMEOUT_S
    return max(_DEFAULT_TIMEOUT_S, float(round(max_tokens / _SLOW_TOKENS_PER_S)))


class PromptTooLongError(ValueError):
    """A prompt longer than the model's context window, found before sending it.

    A row error (see `_is_row_error`): the op leaves that prompt's slots None
    instead of failing, e.g. ``PromptTooLongError("Prompt 17 has 69685 tokens;
    the context window is 32768")``.
    """


class ReplayedRowError(Exception):
    """A row error replayed from the request cache instead of asking again.

    A row error (see `_is_row_error`) is stored in the slots it failed, so a
    rerun gives the prompt the same error, and so the same rows, instead of
    asking again and maybe getting an answer this time. ``reason`` is the
    original's, e.g. "content filter". Pass ``_no_cache=True`` to the op to
    ask again.
    """

    def __init__(self, reason: str, error: str) -> None:
        """``error`` is the original, e.g. "BadRequestError: ... content_filter"."""
        super().__init__(
            f"{error} (replayed from the request cache; pass _no_cache=True to "
            "the op to ask again)"
        )
        self.reason = reason


# Words in a 400's message or body that mean the provider's content filter
# blocked the prompt or the reply, e.g. Azure's '"code":"content_filter"'.
_CONTENT_FILTER_MARKERS = ("content_filter", "content_policy")


def _row_error_reason(exc: BaseException) -> str | None:
    """Why ``exc`` failed only its prompt: "content filter", "context window", or None.

    These errors come from one prompt (or its reply), not from the key, the
    model, or the account, so neither Model nor the ops retry them, and an op
    gives that prompt's slots None instead of raising.
    """
    if isinstance(exc, ReplayedRowError):
        return exc.reason
    if isinstance(exc, (PromptTooLongError, ContextWindowExceededError)):
        return "context window"
    if isinstance(exc, ContentPolicyViolationError):
        return "content filter"
    if isinstance(exc, LitellmBadRequestError):
        text = f"{exc} {getattr(exc, 'body', '')}".lower()
        if any(marker in text for marker in _CONTENT_FILTER_MARKERS):
            return "content filter"
    return None


def _is_row_error(exc: BaseException) -> bool:
    """True for an error that fails one prompt: a content-filter block, a long prompt.

    That is `PromptTooLongError`, litellm's ``ContextWindowExceededError`` and
    ``ContentPolicyViolationError``, and a litellm ``BadRequestError`` whose
    message or body says content_filter or content_policy (Azure's filter 400).
    """
    return _row_error_reason(exc) is not None


def _quota_spent(exc: BaseException) -> bool:
    """True for a request Forger did not send because the token-unit quota is used up.

    Nothing was sent or charged, so an op can drop the prompt and keep the rest
    of its batch, e.g. completions the request cache replays. A provider's own
    out-of-quota 429 is not this: it raises (see `_is_transient`).
    """
    return isinstance(exc, TokenUnitQuotaExceeded)


def _is_transient(exc: BaseException) -> bool:
    """True for provider errors a retry can fix: timeouts, 429s, 5xx, lost connections.

    litellm's and openai's ``Timeout`` / ``APIConnectionError`` and builtin
    ``TimeoutError`` / ``ConnectionError`` are transient; other errors are judged
    by their ``status_code``. A bad key (401), unknown model (404), or bad
    request (400, e.g. context length) is not, nor is an error with no status,
    nor an exhausted quota or budget (a 429 with "insufficient_quota"), nor a
    row error (`_is_row_error`).
    """
    if _is_row_error(exc):
        return False
    text = f"{type(exc).__name__} {getattr(exc, 'code', '')} {exc}".lower()
    if any(marker in text for marker in _QUOTA_MARKERS):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    try:
        import openai
    except ImportError:  # pragma: no cover - openai ships with litellm
        pass
    else:
        if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
            return True
    # Tinker's SDK has its own copies of these (TinkerModel raises them once its
    # retries run out); its rate-limit and server errors carry a status_code.
    if type(exc).__module__.split(".")[0] == "tinker" and type(exc).__name__ in (
        "APITimeoutError",
        "APIConnectionError",
    ):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS or status >= 500
    return False


def _is_rate_limit(exc: BaseException) -> bool:
    """True for a 429 that a later retry can fix (not a spent quota or budget)."""
    status = getattr(exc, "status_code", None)
    named = type(exc).__name__ == "RateLimitError"
    return (status == 429 or named) and _is_transient(exc)


def _retry_after(exc: BaseException) -> float | None:
    """Seconds from the error's Retry-After header, or None without one.

    openai errors carry the httpx response; litellm's also copy its headers
    to ``litellm_response_headers``. An HTTP-date value is ignored.
    """
    response = getattr(exc, "response", None)
    for headers in (
        getattr(response, "headers", None),
        getattr(exc, "litellm_response_headers", None),
    ):
        if headers is None:
            continue
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
            return max(0.0, float(value))
        except (AttributeError, TypeError, ValueError):
            continue
    return None


def _retry_delay(retry: int, exc: BaseException) -> float:
    """Seconds to wait before retry `retry` (0, 1, ...) after `exc`."""
    after = _retry_after(exc)
    if after is not None:
        return min(after, _RETRY_AFTER_MAX_S)
    if _is_rate_limit(exc):
        backoff = min(_RATE_LIMIT_BASE_S * 2**retry, _RATE_LIMIT_MAX_S)
    else:
        backoff = min(_RETRY_BASE_S * 2**retry, _RETRY_MAX_S)
    # Half to full backoff, so 100 requests that hit one 429 do not return
    # together.
    return backoff * (0.5 + random.random() / 2)


def _retry_limit(retries: int, exc: BaseException) -> int:
    """How many retries a request gets when `exc` is its latest error."""
    if retries > 0 and _is_rate_limit(exc):
        return max(retries, _RATE_LIMIT_RETRIES)
    return retries


class _InFlightLimit:
    """How many requests to one model may be in flight, lowered by 429s.

    Unlimited until the first 429. A 429 halves the limit (from the number in
    flight when it hit, at most once per burst: only a request sent after the
    last cut can cut again), never below ``_MIN_IN_FLIGHT``; each success
    raises it by ``_IN_FLIGHT_GROWTH``. Shared by every Model instance for the
    same model and endpoint in the process, since a step often builds its own.
    """

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.limit = math.inf
        self.in_flight = 0
        self.last_cut = -math.inf

    @contextlib.contextmanager
    def slot(self) -> Any:
        """Hold one in-flight slot; yields the time the request was sent."""
        with self.cond:
            while self.in_flight >= self.limit:
                self.cond.wait()
            self.in_flight += 1
        try:
            yield time.monotonic()
        finally:
            with self.cond:
                self.in_flight -= 1
                self.cond.notify()

    def rate_limited(self, sent: float, model: str) -> None:
        """A request sent at `sent` got a 429: halve the limit once per burst."""
        with self.cond:
            if sent <= self.last_cut:
                return
            self.limit = max(_MIN_IN_FLIGHT, min(self.limit, self.in_flight + 1) // 2)
            self.last_cut = time.monotonic()
            limit = self.limit
        _logger.info(
            "%s is rate-limiting requests: at most %d in flight until requests "
            "succeed again",
            model,
            limit,
        )

    def succeeded(self) -> None:
        """A request got through: allow a little more."""
        with self.cond:
            if self.limit != math.inf:
                self.limit += _IN_FLIGHT_GROWTH
                self.cond.notify()


_IN_FLIGHT: dict[tuple[str, str], _InFlightLimit] = {}
_IN_FLIGHT_LOCK = threading.Lock()


def _in_flight_limit(call_kwargs: dict[str, Any]) -> _InFlightLimit:
    """The process-wide limit for the model and endpoint of `call_kwargs`."""
    key = (str(call_kwargs.get("model", "?")), str(call_kwargs.get("api_base") or ""))
    with _IN_FLIGHT_LOCK:
        if key not in _IN_FLIGHT:
            _IN_FLIGHT[key] = _InFlightLimit()
        return _IN_FLIGHT[key]


# Patched in tests, which also sleep in their fake completions.
_sleep = time.sleep


class _RequestClock:
    """When each request of one batch started and ended; warns about stragglers."""

    def __init__(self, desc: str, timeout: Any) -> None:
        """`timeout` is what litellm gets per attempt, named in the warning."""
        self.desc = desc
        self.timeout = timeout
        self.lock = threading.Lock()
        self.started: dict[int, float] = {}
        self.ended: dict[int, float] = {}
        self.warned: set[int] = set()

    def run(
        self,
        i: int,
        messages: list[Message],
        n: int,
        call_kwargs: dict[str, Any],
        semaphore: threading.BoundedSemaphore | None,
        retries: int = 0,
        on_response: Callable[[int, Any], None] | None = None,
        model: Model | None = None,
    ) -> Any:
        """`_complete_and_report` for request `i`, timed from its first slot."""

        def start() -> None:
            with self.lock:
                self.started.setdefault(i, time.monotonic())

        try:
            return _complete_and_report(
                i,
                messages,
                n,
                call_kwargs,
                semaphore,
                on_response,
                model,
                on_start=start,
                retries=retries,
            )
        finally:
            with self.lock:
                self.ended[i] = time.monotonic()

    def warn_stragglers(self, now: float) -> None:
        """Warn once about requests in flight far longer than the finished ones.

        Needs a finished request to compare with: a batch of long calls that
        all run about equally long is not warned about.
        """
        with self.lock:
            durations = [
                end - self.started[i]
                for i, end in self.ended.items()
                if i in self.started
            ]
            in_flight = [
                (i, now - t)
                for i, t in self.started.items()
                if i not in self.ended and i not in self.warned
            ]
        if not durations:
            return
        median = statistics.median(durations)
        limit = max(_STRAGGLER_MIN_S, _STRAGGLER_FACTOR * median)
        late = [(i, age) for i, age in in_flight if age > limit]
        if not late:
            return
        self.warned.update(i for i, _ in late)
        if isinstance(self.timeout, (int, float)):
            # A timed-out attempt is retried (num_retries), so a request can
            # stay in flight past its timeout.
            limit_text = (
                f"each attempt times out after {_duration(self.timeout)} "
                "(Model(timeout=...)) and is retried"
            )
        elif self.timeout is None:
            limit_text = "Model(timeout=None) leaves LiteLLM's default timeout"
        else:  # e.g. an httpx.Timeout
            limit_text = f"it times out per Model(timeout={self.timeout!r})"
        _logger.warning(
            "%s: %s in flight for %s (median of completed requests %s); %s",
            self.desc,
            "1 request" if len(late) == 1 else f"{len(late)} requests",
            _duration(max(age for _, age in late)),
            _duration(median),
            limit_text,
        )


def _watch(
    futures: Iterable[Future], total: int, desc: str, log: bool, clock: _RequestClock
) -> Iterator[Future]:
    """Yield `futures` as they finish, checking every 30s while any are in flight.

    A check warns about stragglers and, with `log`, logs "<desc>: 120/500
    done, 31s", also when nothing finished since the last line. A batch that
    finishes within the interval logs nothing.
    """
    start = time.monotonic()
    next_check = start + _PROGRESS_LOG_INTERVAL_S
    pending = set(futures)
    done = 0
    logged = False
    while pending:
        # Sleep until a request finishes or the next check is due (at least
        # 10ms, so a zero interval does not spin).
        finished, pending = wait(
            pending,
            timeout=max(next_check - time.monotonic(), 0.01),
            return_when=FIRST_COMPLETED,
        )
        done += len(finished)
        yield from finished
        now = time.monotonic()
        if pending and now >= next_check:
            if log:
                _logger.info("%s: %s", desc, _done_line(done, total, now - start))
                logged = True
            clock.warn_stragglers(now)
            next_check = now + _PROGRESS_LOG_INTERVAL_S
    if logged:
        _logger.info("%s: %s", desc, _done_line(done, total, time.monotonic() - start))


def _fmt_spend(amount: float, unit: str) -> str:
    return f"${amount:,.2f}" if unit == "usd" else f"{amount:,.0f} token units"


class _SpendProjection:
    """Stop a batch early when its finished requests show the rest won't fit a cap.

    Each completion weighs one, e.g. 400 prompts with n=8 weigh 3,200. Once
    ``_PROJECT_MIN_SHARE`` of them (and ``_PROJECT_MIN_DONE`` requests) have
    finished, each new request first checks that the rest fits what is left
    under every spending cap and the shared token-unit quota, even at the 95%
    lower bound of the cost per completion so far. If not, it raises
    ``SpendProjectionError``, and so does every request after it; requests in
    flight finish and are recorded.

    The lower bound keeps a few expensive early rows from stopping a batch
    that would fit, e.g. 3 long answers in the first 20 of 400. The batch is
    also sent in a fixed shuffled order (``_send_order``), so rows sorted by
    difficulty do not all come first.
    """

    def __init__(self, model: Model, total: int, shuffle: bool = True) -> None:
        self.model = model
        self.total = total
        # Send in _send_order; pointless for the parts of a single prompt.
        self.shuffle = shuffle
        self.done = 0
        self.requests = 0
        self.cost = 0.0
        self.units = 0.0
        # Sums of completions × (cost per completion)², for the spread.
        self.cost_sq = 0.0
        self.units_sq = 0.0
        self.lock = threading.Lock()
        self.error: _tracking.SpendProjectionError | None = None
        self.where = _tracking.spend_where()

    def add(self, completions: int, cost: float, units: float) -> None:
        """Count one finished request."""
        units = units if math.isfinite(units) else 0.0
        with self.lock:
            self.done += completions
            self.requests += 1
            self.cost += cost
            self.units += units
            if completions:
                self.cost_sq += cost * cost / completions
                self.units_sq += units * units / completions

    def _low(self, spent: float, spent_sq: float, done: int, requests: int) -> float:
        """95% lower bound of the cost per completion (requests as the samples)."""
        mean = spent / done
        var = max(spent_sq / done - mean * mean, 0.0)
        return max(mean - 1.96 * math.sqrt(var / max(requests, 1)), 0.0)

    def check(self) -> None:
        """Raise ``SpendProjectionError`` if the rest of the batch would pass a cap."""
        with self.lock:
            if self.error is not None:
                raise self.error
            done, requests, cost, units = (
                self.done,
                self.requests,
                self.cost,
                self.units,
            )
            sums_sq = {"usd": self.cost_sq, "units": self.units_sq}
        if requests < _PROJECT_MIN_DONE or done < _PROJECT_MIN_SHARE * self.total:
            return
        rest = self.total - done
        if rest <= 0:
            return
        headroom = _tracking.cap_headroom()
        budget = self.model._token_budget
        if budget is not None:
            left = budget.snapshot()["remaining_token_units"]
            headroom.append(
                ("the token-unit quota (FORGER_TOKEN_UNIT_LIMIT)", left, "units")
            )
        for label, left, unit in headroom:
            spent = cost if unit == "usd" else units
            need = rest * spent / done
            low = rest * self._low(spent, sums_sq[unit], done, requests)
            if low <= max(left, 0.0):
                continue
            left_text = _fmt_spend(max(left, 0.0), unit)
            at_least = (
                f" (at least {_fmt_spend(low, unit)})"
                if _fmt_spend(low, unit) != _fmt_spend(need, unit)
                else ""
            )
            error = _tracking.SpendProjectionError(
                f"{self.where}: the first {done:,} of {self.total:,} completions "
                f"cost {_fmt_spend(spent, unit)}; the other {rest:,} would cost "
                f"about {_fmt_spend(need, unit)}{at_least}, but {label} has "
                f"{left_text} left. Stopped before sending them; inside an op, "
                "the finished requests are in the request cache, so a rerun "
                "does not pay for them again. Make this op cheaper (fewer "
                "samples or judges, a cheaper model, shorter outputs) or give it "
                "fewer rows. "
                "FORGER_PROJECT_SPEND=0 turns this check off."
            )
            with self.lock:
                self.error = self.error or error
                raise self.error
        return


def _projecting() -> bool:
    return os.environ.get("FORGER_PROJECT_SPEND", "").strip().lower() not in (
        "0",
        "false",
        "no",
    )


_PROJECTION: contextvars.ContextVar[_SpendProjection | None] = contextvars.ContextVar(
    "forger_spend_projection", default=None
)

# A batch stops once this many completions finished and at least this share of
# them hit max_tokens: the rest would mostly be paid-for, cut-off text.
_TRUNCATION_MIN_FINISHED = 20
_TRUNCATION_STOP_SHARE = 0.8


class TruncationStopError(RuntimeError):
    """Most of a batch's finished completions hit max_tokens; the rest were not sent."""


class _TruncationStop:
    """Counts a batch's finished completions and stops it when most are cut off.

    Off with ``FORGER_ALLOW_TRUNCATION=1``.
    """

    def __init__(self, model_id: str, max_tokens: int | None) -> None:
        self.model_id, self.max_tokens = model_id, max_tokens
        self.finished = self.truncated = 0
        self.units = 0.0
        self._lock = threading.Lock()

    def add(self, response: Any, units: float) -> None:
        """Count a response's completions and the units of the cut-off ones."""
        choices = getattr(response, "choices", None) or []
        cut = sum(_finish_reason(choice) == "length" for choice in choices)
        with self._lock:
            self.finished += len(choices)
            self.truncated += cut
            if cut and math.isfinite(units):
                self.units += units * cut / len(choices)

    def check(self) -> None:
        """Raise before a request once most finished completions were cut off."""
        with self._lock:
            finished, truncated, units = self.finished, self.truncated, self.units
        if (
            finished < _TRUNCATION_MIN_FINISHED
            or truncated < _TRUNCATION_STOP_SHARE * finished
        ):
            return
        raise TruncationStopError(
            f"{truncated} of {finished} finished completions from {self.model_id} hit "
            f"max_tokens={self.max_tokens} ({units:,.0f} token units on cut-off text), "
            "so the rest of the batch was not sent. Raise max_tokens or lower "
            "reasoning_effort, and try a few prompts first; FORGER_ALLOW_TRUNCATION=1 "
            "turns this stop off."
        )


def _truncation_stopping() -> bool:
    return os.environ.get("FORGER_ALLOW_TRUNCATION", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


_TRUNCATION: contextvars.ContextVar[_TruncationStop | None] = contextvars.ContextVar(
    "forger_truncation_stop", default=None
)


def _make_semaphore(max_concurrency: int | None) -> threading.BoundedSemaphore | None:
    """The per-Model request cap, or None for no cap."""
    if max_concurrency is None:
        return None
    return threading.BoundedSemaphore(max_concurrency)


def _request_sizes(n: int, fan_out: bool) -> list[int]:
    """Per-request n values for one prompt, e.g. 40 -> [14, 13, 13].

    Fan-out models get n ones; others get near-equal parts of at most
    _MAX_N_PER_REQUEST.
    """
    if fan_out:
        return [1] * n
    parts = -(-n // _MAX_N_PER_REQUEST)
    return [n // parts + (1 if i < n % parts else 0) for i in range(parts)]


def _add_response_usage(response: Any, model_id: str) -> None:
    """Add one response's tokens and cost to the step report as it arrives."""
    usage = _extract_usage(response)
    _tracking.add_usage(
        usage["prompt_tokens"],
        usage["completion_tokens"],
        _extract_cost(response, model_id),
        _units_or_nan(model_id, usage["prompt_tokens"], usage["completion_tokens"]),
        model_id=model_id,
    )


def _zero_usage() -> dict[str, int]:
    """Empty token counts, as `Model.total_usage` reports them."""
    return {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}


def _units_or_nan(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Token units for this usage; NaN for a model without fixed weights."""
    try:
        return token_units(model_id, prompt_tokens, completion_tokens)
    except ValueError:
        return math.nan


def _complete(
    messages: list[Message],
    n: int,
    call_kwargs: dict[str, Any],
    semaphore: threading.BoundedSemaphore | None,
    model: Model | None = None,
    on_start: Callable[[], None] | None = None,
    retries: int = 0,
) -> Any:
    """One litellm.completion call, holding a `max_concurrency` slot if any.

    A transient error (`_is_transient`: a timeout, 429, 5xx, lost connection)
    is retried up to `retries` times after a short backoff, without holding
    the slot; any other error, or the last one, is raised. So a request makes
    at most `retries + 1` attempts (LiteLLM itself is told not to retry). A
    rate-limit 429 gets at least ``_RATE_LIMIT_RETRIES`` retries with a longer
    backoff (`_retry_limit`), and lowers how many requests to this model may
    be in flight at once (`_InFlightLimit`).

    The spending caps are checked each time the slot is free, right before
    an attempt is sent, so a queued request (or a retry) never starts past a
    cap; a cap or the token-unit quota is never retried. The response's usage
    is recorded here, in the worker, before the caller sees it: a batch that
    later raises or is interrupted keeps every finished request in the step
    report, the instance totals, and the token-unit ledger. Each attempt holds
    its worst case in that ledger while in flight (`Model._hold`), so a timed-out
    or killed attempt the gateway still bills stays charged. `on_start` runs
    each time the request holds its slot, just before it is sent.
    """
    limit = _in_flight_limit(call_kwargs)
    model_id = str(call_kwargs.get("model", "?"))
    retry = 0
    while True:
        sent = -math.inf
        hold = None
        try:
            with contextlib.ExitStack() as slot:
                if semaphore is not None:
                    slot.enter_context(semaphore)
                if model is not None:
                    model._before_request()  # caps, quota, spend projection
                sent = slot.enter_context(limit.slot())
                if on_start is not None:
                    on_start()
                if model is not None:
                    # Each attempt holds its worst case in the shared ledger.
                    hold = slot.enter_context(model._hold(messages, n, call_kwargs))
                response = litellm.completion(messages=messages, n=n, **call_kwargs)
            limit.succeeded()
            break
        except Exception as exc:
            if _is_rate_limit(exc):
                limit.rate_limited(sent, model_id)
            allowed = _retry_limit(retries, exc)
            if retry >= allowed or not _is_transient(exc):
                raise
            delay = _retry_delay(retry, exc)
            retry += 1
            _logger.info(
                "LLM request (%s) failed with %s; retry %d of %d in %.1fs",
                model_id,
                type(exc).__name__,
                retry,
                allowed,
                delay,
            )
            _sleep(delay)
    if model is not None:
        model._account_response(response, hold)
    else:
        _add_response_usage(response, call_kwargs.get("model", "?"))
    return response


def _complete_and_report(
    i: int,
    messages: list[Message],
    n: int,
    call_kwargs: dict[str, Any],
    semaphore: threading.BoundedSemaphore | None,
    on_response: Callable[[int, Any], None] | None,
    model: Model | None = None,
    on_start: Callable[[], None] | None = None,
    retries: int = 0,
) -> Any:
    """`_complete`, then `on_response(i, response)` in the same worker thread."""
    response = _complete(
        messages,
        n,
        call_kwargs,
        semaphore,
        model,
        on_start=on_start,
        retries=retries,
    )
    if on_response is not None:
        on_response(i, response)
    return response


def _send_order(total: int) -> list[int]:
    """A fixed shuffle of ``range(total)``: the same batch is sent the same way."""
    order = list(range(total))
    random.Random(total).shuffle(order)
    return order


def _batch_completion_with_progress(
    requests: list[tuple[list[Message], int]],
    call_kwargs: dict[str, Any],
    *,
    semaphore: threading.BoundedSemaphore | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    progress: str = "bar",
    on_response: Callable[[int, Any], None] | None = None,
    model: Model | None = None,
    retries: int = 0,
    shuffle: bool = False,
) -> list[Any]:
    """Send (messages, n) requests concurrently; return responses or errors.

    Results are in request order. Each request retries transient errors up
    to `retries` times (see `_complete`). `on_response(i, response)` runs in
    the worker as each request succeeds (the request cache writes there, so a
    run killed mid-batch keeps every finished request).

    A replacement for `litellm.batch_completion`, which retrieves
    `future.result()` in submission order, so the whole call looks frozen until
    the *slowest* early-submitted item finishes. This reports progress as each
    call actually finishes: a tqdm bar (`progress="bar"`), a log line every 30s
    (`"log"`), or nothing (`"off"`). Each request holds a slot of `semaphore`
    (the Model's `max_concurrency`) while in flight. In every mode, a request
    in flight far longer than the finished ones gets one warning. With
    `shuffle`, requests start in `_send_order` (for the spend projection).
    """
    results: list[Any] = [None] * len(requests)
    total = len(requests)
    desc = f"LLM calls ({call_kwargs.get('model', '?')})"
    clock = _RequestClock(desc, call_kwargs.get("timeout"))
    executor = ThreadPoolExecutor(max_workers=min(max_workers, total))
    try:
        future_to_idx: dict[Future, int] = {}
        for i in _send_order(total) if shuffle else range(total):
            msgs, n = requests[i]
            # One context copy per request, so add_usage in the worker sees the
            # caller's op / step / pipeline contextvars.
            run: Callable[..., Any] = contextvars.copy_context().run
            future = executor.submit(
                run,
                clock.run,
                i,
                msgs,
                n,
                call_kwargs,
                semaphore,
                retries,
                on_response,
                model,
            )
            future_to_idx[future] = i
        finished: Iterable[Future] = _watch(
            future_to_idx, total, desc, progress == "log", clock
        )
        if progress == "bar":
            from tqdm import tqdm

            finished = tqdm(finished, total=total, desc=desc, leave=False)
        for future in finished:
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = exc
    except BaseException:
        # Ctrl-C (or any error here): send no queued request; requests already
        # in flight finish in the background and still record their usage.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    return results


class GenerateResult(list):
    """list[str] with `usage` / `reasoning` / `finish_reasons` / `model_id` / `raw`.

    The completions are the list items: `result[0]`, `for text in result`, or
    `result.texts` for a plain list.

    `raw` is the LiteLLM response. When `Model.generate_batch` fans `n` out into
    separate requests, it is the list of those responses, in completions order.
    It holds only the requests actually sent: None when every completion came
    from the request cache.

    `finish_reasons` holds each completion's stop reason, e.g. "stop", or
    "length" when it hit max_tokens (None when the backend gave none).
    """

    usage: dict[str, Any]
    reasoning: list[str | None]
    model_id: str
    raw: Any
    finish_reasons: list[str | None]

    def __init__(
        self,
        completions: list[str],
        *,
        usage: dict[str, Any],
        reasoning: list[str | None],
        model_id: str,
        raw: Any,
        finish_reasons: list[str | None] | None = None,
    ) -> None:
        """Wrap completions with per-call metadata."""
        super().__init__(completions)
        self.usage = usage
        self.reasoning = reasoning
        self.model_id = model_id
        self.raw = raw
        self.finish_reasons = (
            list(finish_reasons) if finish_reasons is not None else [None] * len(self)
        )

    @property
    def truncated(self) -> int:
        """How many completions stopped at max_tokens (finish reason "length")."""
        return sum(1 for reason in self.finish_reasons if reason == "length")

    @property
    def texts(self) -> list[str]:
        """The completion strings as a plain list (same as `list(result)`)."""
        return list(self)

    def __getattr__(self, name: str) -> Any:
        """Raise an AttributeError that names the real attributes.

        Only runs for missing attributes, e.g. `result.completions`. Dunder and
        private names fail plainly so pickle, copy and hasattr probes behave.
        """
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        raise AttributeError(
            f"GenerateResult has no attribute {name!r}; it is a list of completion "
            "strings (use the result itself, e.g. result[0], or .texts for a plain "
            "list); metadata: .usage, .reasoning, .finish_reasons, .truncated, "
            ".model_id, .raw"
        )

    def __reduce__(self):
        """Pickle support."""
        return (
            _rebuild_generate_result,
            (
                list(self),
                self.usage,
                self.reasoning,
                self.model_id,
                self.raw,
                self.finish_reasons,
            ),
        )


def _rebuild_generate_result(
    completions: list[str],
    usage: dict[str, Any],
    reasoning: list[str | None],
    model_id: str,
    raw: Any,
    finish_reasons: list[str | None] | None = None,
) -> GenerateResult:
    """Pickle helper for GenerateResult (pickles from before finish_reasons too)."""
    return GenerateResult(
        completions,
        usage=usage,
        reasoning=reasoning,
        model_id=model_id,
        raw=raw,
        finish_reasons=finish_reasons,
    )


def _note_truncation(
    results: list[Any],
    model_id: str,
    max_tokens: int | None,
    advice: str | None = None,
) -> None:
    """Count completions that hit max_tokens toward the running op's report.

    Outside any op the count is logged here, since no report will show it.
    ``advice`` is the warning's tail (see ``_tracking.truncation_message``).
    The warning also says how many stopped mid-thought
    (``"<think>still thinking"``); inside ``_tracking.reporting_unclosed_think``
    those are left out, since the op warns about them itself.
    """
    done = [r for r in results if isinstance(r, GenerateResult)]
    cut = [
        isinstance(text, str) and "<think>" in text and "</think>" not in text
        for r in done
        for text, reason in zip(r, r.finish_reasons)
        if reason == "length"
    ]
    mid_thought = sum(cut)
    truncated = len(cut)
    if _tracking.reports_unclosed_think():
        truncated, mid_thought = truncated - mid_thought, 0
    total = sum(len(r) for r in done)
    noted = _tracking.add_truncation(
        model_id, max_tokens, truncated, total, advice, mid_thought
    )
    if truncated and not noted:
        _logger.warning(
            _tracking.truncation_message(
                model_id, max_tokens, truncated, total, advice, mid_thought=mid_thought
            )
        )


_VALUE_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "o1": {"temperature": 1.0},
    "o3": {"temperature": 1.0},
    "o4": {"temperature": 1.0},
    "gpt-5": {"temperature": 1.0},
    "gemini-3": {"temperature": 1.0},
}

_PROVIDER_PREFIXES = ("openai/", "anthropic/", "gemini/", "xai/", "deepseek/")

# Models that reject or ignore n > 1 (directly or through a LiteLLM proxy).
# generate_batch sends them n separate n=1 requests and merges the results.
# Brands match anywhere in the model name ("us.anthropic.claude-sonnet-4-5"),
# version prefixes only at the start ("gpt-5.6-terra", "o4-mini").
_NO_N_BRANDS = ("claude", "grok", "gemini")
_NO_N_PREFIXES = ("o1", "o3", "o4", "gpt-5")
_NO_N_PROVIDERS = ("anthropic", "xai", "gemini")
# Model ids that returned fewer than n completions for one n request in this
# process (e.g. a gateway alias that ignores n). They fan out from then on.
_LEARNED_NO_N: set[str] = set()
_LEARNED_NO_N_LOCK = threading.Lock()


def _learn_no_n(model_id: str, n: int, got: int | str) -> None:
    """Fan out `model_id` from now on; warn the first time.

    ``got`` is the number of completions a request returned, or the provider's
    error when it rejected ``n`` outright.
    """
    with _LEARNED_NO_N_LOCK:
        if model_id in _LEARNED_NO_N:
            return
        _LEARNED_NO_N.add(model_id)
    what = (
        f"returned {got} completions for n={n}"
        if isinstance(got, int)
        else f"rejected n={n} ({got})"
    )
    _logger.warning(
        "%s %s; sending the missing ones as n=1 requests, and fanning out n for "
        "this model from now on.",
        model_id,
        what,
    )


# A 400 saying the provider takes only n=1, e.g. Fireworks serverless through a
# LiteLLM gateway: "n must be 1 on serverless. To generate multiple samples,
# send separate requests."
_REJECTS_N = re.compile(
    r"\bn must be 1\b|\bonly n ?= ?1\b|unsupported (?:parameter|value):? ?'?n'?\b"
)


def _rejects_n(exc: BaseException) -> bool:
    """True for a bad request (400) whose message says only n=1 is accepted."""
    if getattr(exc, "status_code", None) not in (400, None):
        return False
    return bool(_REJECTS_N.search(str(exc)))


def _family(model_id: str) -> str:
    """Strip `litellm_proxy/` / `openrouter/` gateway + provider prefix."""
    s = model_id
    if s.startswith("litellm_proxy/"):
        s = s[len("litellm_proxy/") :]
    if s.startswith("openrouter/"):
        s = s[len("openrouter/") :]
    for prefix in _PROVIDER_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    return s


def _fans_out_n(model_id: str) -> bool:
    """True when n > 1 must be sent as n separate requests for this model.

    E.g. "litellm_proxy/claude-sonnet-4-5", "azure_ai/gpt-5.6-terra",
    "litellm_proxy/xai/grok-4", or any model whose LiteLLM supported-params
    list is known and lacks "n".
    """
    if model_id in _LEARNED_NO_N:
        return True
    *providers, name = _family(model_id).lower().split("/")
    if any(p in _NO_N_PROVIDERS for p in providers):
        return True
    if any(brand in name for brand in _NO_N_BRANDS):
        return True
    for prefix in _NO_N_PREFIXES:
        if name == prefix or (name.startswith(prefix) and name[len(prefix)] in "-."):
            return True
    supported = set(litellm.get_supported_openai_params(model=model_id) or [])
    return bool(supported) and "n" not in supported


def _merge_responses(
    slots: range,
    hits: dict[int, dict[str, Any]],
    fresh: list[tuple[list[int], Any]],
    model_id: str,
) -> GenerateResult:
    """One prompt's GenerateResult: cached and fresh completions in slot order.

    `hits` maps slot -> cached entry; `fresh` pairs each successful response
    with the slots it was asked for, e.g. ([2, 3], response with 2 choices).
    Usage, cost, and token units sum the fresh responses only: a cache hit
    costs nothing. Token units are NaN for a model without fixed weights, or
    when a response lacks usage. A short response fills the first of its slots
    (the rest are refilled or stay missing); choices beyond the slots asked
    come after the slotted ones, so the completion-count check still sees the
    mismatch.
    """
    by_slot: dict[int, tuple[str, str | None, str | None]] = {
        slot: (entry["text"], entry.get("reasoning"), entry.get("finish_reason"))
        for slot, entry in hits.items()
    }
    unslotted: list[tuple[str, str | None, str | None]] = []
    usage: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "token_units": _units_or_nan(model_id, 0, 0),
    }
    responses = [response for _, response in fresh]
    for asked, response in fresh:
        texts = [choice.message.content or "" for choice in response.choices]
        reasons = [_finish_reason(choice) for choice in response.choices]
        pairs = list(zip(texts, _extract_reasoning(response, model_id), reasons))
        by_slot.update(zip(asked, pairs))
        unslotted.extend(pairs[len(asked) :])
        for key, value in _extract_usage(response).items():
            usage[key] += value
        usage["cost_usd"] += _extract_cost(response, model_id)
        try:
            usage["token_units"] += _units_or_nan(
                model_id, *response_token_counts(response)
            )
        except ValueError:  # missing or invalid usage: units unknown
            usage["token_units"] = math.nan
    if hits:
        usage["cached_completions"] = len(hits)
    ordered = [by_slot[slot] for slot in slots if slot in by_slot] + unslotted
    return GenerateResult(
        [text for text, _, _ in ordered],
        usage=usage,
        reasoning=[reasoning for _, reasoning, _ in ordered],
        model_id=model_id,
        raw=(
            None
            if not responses
            else responses[0]
            if len(responses) == 1
            else responses
        ),
        finish_reasons=[reason for _, _, reason in ordered],
    )


def _finish_reason(choice: Any) -> str | None:
    """A choice's finish reason as a string, e.g. "length", or None."""
    reason = getattr(choice, "finish_reason", None)
    return reason if isinstance(reason, str) else None


def _cache_entry(choice: Any, reasoning: Any) -> dict[str, Any] | None:
    """What the request cache stores for one choice, or None to not cache it.

    Only plain text is cached. Empty or missing text (often a transient glitch)
    and choices carrying tool calls, a function call, or a refusal are not: a
    replay could not rebuild them, so they are asked again. Empty text that
    stopped at max_tokens is cached: a reasoning model that spent the whole
    budget thinking does the same again at that budget. So is a reply the
    content filter cut off, e.g. "" with finish_reason "content_filter": asked
    again it might pass, and a rerun must give the rows the first run got.
    """
    message = choice.message
    text = message.content
    finish_reason = _finish_reason(choice)
    stopped = finish_reason in ("length", "content_filter")
    if text is None and stopped:
        text = ""
    if not isinstance(text, str) or (not text and not stopped):
        return None
    for extra in ("tool_calls", "function_call", "refusal"):
        if getattr(message, extra, None):
            return None
    return {
        "text": text,
        "reasoning": reasoning if isinstance(reasoning, str) else None,
        "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
    }


def _cache_kwargs(method: Callable[..., Any], **values: Any) -> dict[str, Any]:
    """The request-cache keywords (`cache_rows`, `cache_salt`, ...) `method` takes.

    Values that are None are dropped, and so are keywords a subclass's own
    generate_batch does not accept (test fakes, backends written before these keywords),
    e.g. ``model.generate_batch(prompts, **_cache_kwargs(model.generate_batch,
    cache_rows=items))``.
    """
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return {}
    takes_any = any(p.kind is p.VAR_KEYWORD for p in params.values())
    return {
        name: value
        for name, value in values.items()
        if value is not None and (takes_any or name in params)
    }


def _check_completion_count(
    model_id: str,
    n: int,
    results: list[Any],
    fanned_out: bool,
    skip: set[int] = frozenset(),  # type: ignore[assignment]
) -> None:
    """Raise `ValueError` when a prompt got other than `n` completions.

    Exceptions (from `return_exceptions=True`) and prompts in `skip` (partial
    fan-out results) are not checked.
    """
    for i, result in enumerate(results):
        if isinstance(result, BaseException) or i in skip or len(result) == n:
            continue
        if fanned_out:
            hint = f"{n} separate n=1 requests did not return one completion each."
        else:
            hint = (
                "The model or provider may not support n > 1; add it to "
                "_NO_N_BRANDS / _NO_N_PREFIXES in pyforger/acquire/model.py "
                "to send n separate requests instead."
            )
        raise ValueError(
            f"{model_id} returned {len(result)} completions for prompt {i}, "
            f"but n={n} were requested. {hint}"
        )


# How each known model family reads a reasoning effort, matched by substring of
# the model id. The gateway serves DeepSeek through Azure AI, which ignores
# `reasoning_effort` (thinking stays off, or leaks into the answer text); GLM is
# served by Fireworks, which reads `reasoning_effort` ("low", "medium", "high").
_REASONING_FORMS: dict[str, Callable[[str], dict[str, Any]]] = {
    "deepseek-v4": lambda effort: {"reasoning": {"effort": effort}},
    "glm-5.3": lambda effort: {"reasoning_effort": effort},
}


def _reasoning_kwargs(
    model_id: str, effort: str | None, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Return completion kwargs with `effort` in the form the model reads.

    An explicit `effort` overrides any `reasoning`/`reasoning_effort` in
    `kwargs` or in `kwargs["extra_body"]`. Known families use the form in
    `_REASONING_FORMS`; other models get LiteLLM's standard `reasoning_effort`.
    With no `effort`, `kwargs` are returned unchanged.
    """
    if effort is None:
        return dict(kwargs)
    kwargs = {
        k: v for k, v in kwargs.items() if k not in ("reasoning", "reasoning_effort")
    }
    if isinstance(kwargs.get("extra_body"), dict):
        extra_body = {
            k: v
            for k, v in kwargs["extra_body"].items()
            if k not in ("reasoning", "reasoning_effort")
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        else:
            del kwargs["extra_body"]
    lowered = model_id.lower()
    form = next(
        (f for key, f in _REASONING_FORMS.items() if key in lowered),
        lambda effort: {"reasoning_effort": effort},
    )
    return {**kwargs, **form(effort)}


def _validate_params(model_id: str, **params: Any) -> None:
    """Raise `ValueError` on unsupported params or constraint violations."""
    supported_set = set(litellm.get_supported_openai_params(model=model_id) or [])

    for name, value in params.items():
        if value is None:
            continue
        if supported_set and name not in supported_set:
            raise ValueError(
                f"{model_id} does not support parameter '{name}'. "
                f"Supported parameters: {sorted(supported_set)}"
            )

    family = _family(model_id)
    required: dict[str, Any] = {}
    for key in sorted(_VALUE_CONSTRAINTS, key=len, reverse=True):
        if family == key or (family.startswith(key) and family[len(key)] in "-."):
            required = dict(_VALUE_CONSTRAINTS[key])
            break
    thinking = params.get("thinking")
    if (
        isinstance(thinking, dict)
        and thinking.get("type") == "enabled"
        and (family.lower().startswith("claude-") or "-claude-" in family.lower())
    ):
        required.setdefault("temperature", 1.0)

    for name, required_value in required.items():
        got = params.get(name)
        if got is not None and got != required_value:
            raise ValueError(
                f"{model_id} requires {name}={required_value} (got {got}). "
                f"Pass {name}={required_value} or omit the parameter."
            )


def _extract_reasoning(response: Any, model_id: str) -> list[str | None]:
    """Extract reasoning per choice; `None` when absent.

    Skips `litellm.supports_reasoning` (false negatives on proxy aliases).
    OpenRouter uses `message.reasoning` instead of `reasoning_content`.
    """
    s = (
        model_id[len("litellm_proxy/") :]
        if model_id.startswith("litellm_proxy/")
        else model_id
    )
    is_openrouter = s.startswith("openrouter/")
    out: list[str | None] = []
    for choice in response.choices:
        primary = getattr(choice.message, "reasoning_content", None)
        if primary:
            out.append(primary)
            continue
        if is_openrouter:
            fallback = getattr(choice.message, "reasoning", None)
            out.append(fallback if fallback else None)
        else:
            out.append(None)
    return out


_COST_WARNING_SEEN: set[str] = set()


@overload
def _extract_cost(response: Any, model_id: str) -> float:
    ...


@overload
def _extract_cost(response: Any, model_id: str, *, allow_missing: bool) -> float | None:
    ...


def _extract_cost(
    response: Any, model_id: str, *, allow_missing: bool = False
) -> float | None:
    """Cost in USD; `0.0` when unavailable (None with `allow_missing`).

    Warns once per `model_id` on a miss.
    """
    cost = response_cost_usd(response, model_id)
    if cost is None:
        if model_id not in _COST_WARNING_SEEN:
            _COST_WARNING_SEEN.add(model_id)
            warnings.warn(
                f"Cost unavailable for {model_id} (model may be absent from "
                f"LiteLLM's pricing registry). total_cost may undercount.",
                UserWarning,
                stacklevel=2,
            )
        return None if allow_missing else 0.0
    return cost


def _extract_usage(response: Any) -> dict[str, int]:
    """Dict with `prompt_tokens` / `completion_tokens` / `reasoning_tokens`."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}

    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = (
        getattr(details, "reasoning_tokens", 0) if details is not None else 0
    ) or 0

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "reasoning_tokens": reasoning_tokens,
    }


def _to_messages(prompt: Prompt) -> list[Message]:
    """Convert a prompt to LiteLLM message format.

    Args:
        prompt: A string (wrapped as a user message) or a list of
            chat messages in OpenAI format.

    Returns:
        List of message dicts ready for litellm.completion().
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


class Model:
    """Thin LiteLLM wrapper for generating LLM completions.

    Handles API key configuration, prompt formatting, and batching.
    String prompts are automatically wrapped as user messages.

    Example:
        >>> model = Model("gpt-4o", api_key="sk-...")
        >>> model.generate("What is 2+2?")
        ["4"]
    """

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        num_retries: int = 2,
        show_progress: bool | None = None,
        max_concurrency: int | None = None,
        request_cache: bool = True,
        reasoning_effort: str | None = None,
        timeout: float | Literal["auto"] | None = "auto",
        **kwargs: Any,
    ) -> None:
        """Create a Model instance.

        Args:
            model_id: LiteLLM model string (e.g. "gpt-4o",
                "anthropic/claude-sonnet-4-20250514").
            api_key: API key. Falls back to provider-specific env vars
                (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) when None; for
                ``litellm_proxy/`` models, to LITELLM_API_KEY, and `api_base`
                to LITELLM_BASE_URL.
            temperature: Default sampling temperature.
            max_tokens: Default max tokens. None lets the provider decide.
            num_retries: Retries per request on transient failures (a
                timeout, 429, 5xx, or lost connection), after a 1s, 2s, ...
                backoff or a 429's Retry-After. A request makes at most
                num_retries + 1 attempts: LiteLLM is sent num_retries=0, so
                neither it nor the openai client retries on top. A rate-limit
                429 gets at least 6 retries, about 5s, 10s, 20s, 40s, 60s, 60s
                apart, and lowers how many requests to the model are in
                flight (0 still retries nothing). A 400, 401, 403, 404, or an
                exhausted quota is not retried.
            show_progress: Progress for a generate_batch() call that sends
                more than one request (several prompts, or n > 1 on a model
                that does not support n): a tqdm bar when stderr is a
                terminal, otherwise a log line every 30 seconds. None (default)
                shows it unless the environment sets FORGER_PROGRESS=0.
            max_concurrency: Most requests this instance has in flight at once,
                across all generate_batch() calls and threads (each n fan-out
                copy counts). None (default) caps each call at 100 requests.
                Not part of cache_key().
            request_cache: Keep each completion on disk under
                $FORGER_CACHE/requests and reuse it when the same op call asks
                the same request again (see generate_batch). False, or the
                environment variable FORGER_REQUEST_CACHE=0, turns it off. Not
                part of cache_key().
            reasoning_effort: Reasoning effort (e.g. "none", "low", "high"),
                sent in the form the model reads: DeepSeek-V4 gets
                ``reasoning={"effort": ...}`` and GLM-5.3 ``reasoning_effort``
                (see `_REASONING_FORMS`). Overrides any reasoning setting in
                `kwargs` or `extra_body`. DeepSeek-V4 on the gateway does not
                think at all without it.
            timeout: Seconds one request attempt may take before it fails
                with a timeout, which is retried like other transient errors
                (num_retries). "auto" (default) is 600, or
                max_tokens / 25 when that is longer (32768 tokens get 1311).
                None leaves LiteLLM's default (6000). Not part of cache_key().
            **kwargs: Passed through to litellm.completion()
                (e.g. api_base, organization).
        """
        if max_concurrency is not None and max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        # Gateway models read the LiteLLM environment, e.g. LITELLM_BASE_URL;
        # without a base URL, litellm would send the request to OpenAI.
        if model_id.startswith("litellm_proxy/"):
            api_key = api_key or os.environ.get("LITELLM_API_KEY")
            if kwargs.get("api_base") is None and os.environ.get("LITELLM_BASE_URL"):
                kwargs = {**kwargs, "api_base": os.environ["LITELLM_BASE_URL"]}
        self._request_cache = request_cache
        self._model_id = model_id
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_retries = num_retries
        self._show_progress = show_progress
        self._max_concurrency = max_concurrency
        self._semaphore = _make_semaphore(max_concurrency)
        self._timeout = timeout
        self._kwargs = _reasoning_kwargs(model_id, reasoning_effort, kwargs)
        self.history: list[GenerateResult] = []
        # Instance totals, charged per response as it arrives (see
        # _account_response), so a batch that raises keeps its finished spend
        # even though `history` only gets the results of calls that return.
        self._usage_lock = threading.RLock()
        self._usage_totals = _zero_usage()
        self._total_cost = 0.0
        self._total_token_units = self._calculate_token_units(0, 0)
        self._usage_log_progress = 0.0
        self._next_usage_log = 0
        self._token_budget = self._make_token_budget()
        self._check_params(
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=kwargs.get("thinking"),
        )

    def __repr__(self) -> str:
        """Show model id for debugging."""
        return f"Model({self._model_id!r})"

    def __getstate__(self) -> dict[str, Any]:
        """Pickle/copy without the semaphore and lock (they cannot be pickled)."""
        with self._usage_lock:
            state = self.__dict__.copy()
            state["_usage_totals"] = dict(self._usage_totals)
        state.pop("_semaphore", None)
        state.pop("_usage_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore, with a fresh semaphore and an independent accounting lock."""
        self.__dict__.update(state)
        self._semaphore = _make_semaphore(state.get("_max_concurrency"))
        self._usage_lock = threading.RLock()
        # A Model pickled before Model(timeout=) existed.
        self.__dict__.setdefault("_timeout", "auto")

    def _make_token_budget(self) -> TokenUnitBudget | None:
        """The shared token-unit quota named by the environment, if any."""
        path = os.environ.get("FORGER_TOKEN_USAGE_FILE")
        limit = os.environ.get("FORGER_TOKEN_UNIT_LIMIT")
        if path is None and limit is None:
            return None
        if not path or not limit:
            raise ValueError(
                "Set both FORGER_TOKEN_USAGE_FILE and FORGER_TOKEN_UNIT_LIMIT "
                "to enable the shared token-unit budget."
            )
        return TokenUnitBudget(path, float(limit))

    def _calculate_token_units(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Token units for this model's usage; NaN without fixed weights."""
        return _units_or_nan(self._model_id, prompt_tokens, completion_tokens)

    @property
    def reasoning_effort(self) -> str | None:
        """Effort sent to the provider, whichever form it takes; None if unset."""
        effort = self._kwargs.get("reasoning_effort")
        reasoning = self._kwargs.get("reasoning")
        if effort is None and isinstance(reasoning, dict):
            effort = reasoning.get("effort")
        return effort

    @property
    def total_token_units(self) -> float:
        """Cumulative units since reset; NaN if any usage cannot be weighted."""
        with self._usage_lock:
            return self._total_token_units

    def _check_params(self, **params: Any) -> None:
        """Validate params against LiteLLM's registry for this model id."""
        _validate_params(self._model_id, **params)

    def cache_key(self) -> tuple:
        """Stable hash key for @op caching using all functional attributes."""
        kwargs_items = tuple(sorted(self._kwargs.items(), key=lambda kv: str(kv[0])))
        return (
            type(self).__name__,
            self._model_id,
            self._temperature,
            self._max_tokens,
            self._num_retries,
            kwargs_items,
        )

    def request_scope_key(self) -> tuple:
        """This Model as an op argument in request-cache scopes.

        Only what can change a completion: the model id and output-changing
        params. Unlike cache_key(), retries and timeouts are left out, so
        changing them reuses the request cache.
        """
        params = _request_cache.keyed_params(
            dict(temperature=self._temperature, max_tokens=self._max_tokens)
            | self._kwargs
        )
        items = tuple(sorted(params.items(), key=lambda kv: str(kv[0])))
        return (type(self).__name__, self._model_id, items)

    @property
    def total_cost(self) -> float:
        """USD cost of every response this instance received since reset.

        Counted as each response arrives, so it includes requests of a
        generate_batch call that later raised (which add nothing to `history`).
        Request-cache hits cost nothing.
        """
        with self._usage_lock:
            return self._total_cost

    @property
    def total_usage(self) -> dict[str, int]:
        """Token counts of every response this instance received since reset."""
        with self._usage_lock:
            return dict(self._usage_totals)

    def reset_history(self) -> None:
        """Reset history and instance totals, but not the shared token-unit ledger."""
        with self._usage_lock:
            self.history = []
            self._usage_totals = _zero_usage()
            self._total_cost = 0.0
            self._total_token_units = self._calculate_token_units(0, 0)
            self._usage_log_progress = 0.0
            self._next_usage_log = 0

    def _before_request(self) -> None:
        """Raise if a spending cap or the token-unit quota is reached.

        Called right before each request is sent (after it gets a
        `max_concurrency` slot), so requests queued behind a cap never start.
        """
        _tracking.check_budget()
        if self._token_budget is not None:
            token_units(self._model_id, 0, 0)  # unweighted models fail closed
            self._token_budget.check()
        projection = _PROJECTION.get()
        if projection is not None:
            projection.check()
        truncation = _TRUNCATION.get()
        if truncation is not None:
            truncation.check()

    def _add_totals(self, usage: dict[str, Any], cost_usd: float, units: float) -> None:
        """Add one response's usage to the instance totals; log every 1M units."""
        with self._usage_lock:
            for key in self._usage_totals:
                self._usage_totals[key] += usage.get(key, 0) or 0
            self._total_cost += cost_usd
            self._total_token_units += units
            # Unknown weights use raw tokens only to set the logging interval.
            self._usage_log_progress += (
                units
                if math.isfinite(units)
                else sum(
                    value
                    for key in ("prompt_tokens", "completion_tokens")
                    if isinstance(value := usage.get(key), int) and value > 0
                )
            )
            if self._usage_log_progress >= self._next_usage_log:
                _logger.info(
                    "%s cumulative usage: %s token units, $%.6f USD recorded cost",
                    self._model_id,
                    (
                        "NaN"
                        if math.isnan(self._total_token_units)
                        else f"{self._total_token_units:.1f}"
                    ),
                    self._total_cost,
                )
                self._next_usage_log = (
                    int(self._usage_log_progress // 1_000_000) + 1
                ) * 1_000_000

    def _hold(
        self, messages: list[Message], n: int, call_kwargs: dict[str, Any]
    ) -> contextlib.AbstractContextManager[str | None]:
        """Hold one request's worst case in the shared ledger while it is in flight.

        A no-op without a token-unit quota. See `TokenUnitBudget.hold`.
        """
        if self._token_budget is None:
            return contextlib.nullcontext()
        units = held_units(self._model_id, messages, n, call_kwargs.get("max_tokens"))
        return self._token_budget.hold(self._model_id, units)

    def _account_response(self, response: Any, hold: str | None = None) -> None:
        """Charge one LiteLLM response as it arrives, before it is parsed.

        It goes to the instance totals, the step report and spend ledger
        (`_tracking.add_usage`, with its token units), and the shared
        token-unit quota when one is configured. Charges are kept even when
        pricing or parsing fails. With a quota, a response without valid
        usage marks the ledger unmetered and raises ValueError: the quota
        cannot be enforced past it. The request's ``hold`` (see `_hold`) is
        replaced by its actual usage, or charged when usage is missing.
        """
        usage: dict[str, Any] = dict(_extract_usage(response))
        try:
            prompt_tokens, completion_tokens = response_token_counts(response)
        except ValueError:
            complete_usage = False
            units = math.nan
        else:
            complete_usage = True
            usage["prompt_tokens"] = prompt_tokens
            usage["completion_tokens"] = completion_tokens
            units = self._calculate_token_units(prompt_tokens, completion_tokens)
        cost = None
        try:
            cost = _extract_cost(response, self._model_id, allow_missing=True)
        finally:
            cost_usd = cost if cost is not None else 0.0
            self._add_totals(usage, cost_usd, units)
            projection = _PROJECTION.get()
            if projection is not None:
                projection.add(
                    len(getattr(response, "choices", None) or [1]), cost_usd, units
                )
            truncation = _TRUNCATION.get()
            if truncation is not None:
                truncation.add(response, units)
            _tracking.add_usage(
                usage["prompt_tokens"],
                usage["completion_tokens"],
                cost_usd,
                units,
                model_id=self._model_id,
            )
            if self._token_budget is not None:
                if not complete_usage:
                    self._token_budget.mark_unmetered(hold)
                    raise ValueError(
                        f"Missing or invalid token usage for {self._model_id}; "
                        "cannot account for the shared budget."
                    )
                self._token_budget.record(
                    self._model_id,
                    prompt_tokens,
                    completion_tokens,
                    cost_usd=cost,
                    hold=hold,
                )

    def generate(
        self,
        prompt: Prompt,
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        cache_salt: str | None = None,
        cache_row: Any = None,
    ) -> GenerateResult:
        """Generate n responses for a single prompt.

        Convenience wrapper around generate_batch for the single-prompt
        case.

        Args:
            prompt: A string (wrapped as a user message) or a list of
                chat messages in OpenAI format.
            n: Number of completions to generate.
            temperature: Sampling temperature. Overrides the instance
                default.
            max_tokens: Max tokens. Overrides the instance default.
            stop: Stop sequences. Generation stops when any of these
                strings is produced.
            cache_salt: Request-cache salt (see generate_batch).
            cache_row: The row the prompt was built from (see generate_batch's
                cache_rows).

        Returns:
            List of n response strings.
        """
        return self.generate_batch(
            [prompt],
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **_cache_kwargs(
                self.generate_batch,
                cache_salt=cache_salt,
                cache_rows=None if cache_row is None else [cache_row],
            ),
        )[0]

    def generate_batch(
        self,
        prompts: list[Prompt],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        show_progress: bool | None = None,
        return_exceptions: bool = False,
        cache_salt: str | None = None,
        cache_rows: list[Any] | None = None,
    ) -> list[GenerateResult]:
        """Generate n responses for each prompt in a batch.

        Models that do not support n > 1 (Claude, Grok, Gemini, o1/o3/o4,
        gpt-5, or any whose LiteLLM supported params lack "n") get n separate
        concurrent n=1 requests per prompt, merged into one GenerateResult
        with summed usage and cost. Other models get one request with n, or
        several when n > 16 (n=40 sends n=14, 13, 13), merged the same way.
        All requests are sent at once, up to the Model's max_concurrency.

        Request cache (unless off, see __init__): each completion is stored
        under a key of the enclosing @op call (op, body, and every argument
        but its rows), the prompt's row (`cache_rows`, else all of the op's
        rows), the messages, output-changing params, and `cache_salt`, plus a
        slot index. Slots are numbered per op call: n=4 fills slots 0..3, and
        the same request asked again in that op call (a duplicate row, a
        retry, a later chunk) fills 4..7. A rerun of the op asks the same
        requests in the same order, so it reuses cached slots and requests only
        missing ones. Results are written as each request finishes; errors,
        empty text, and tool calls are never cached. A reused completion adds
        no usage or cost, and `usage["cached_completions"]` counts them.
        `_no_cache=True` on an enclosing op skips reading (fresh results are
        still written). Outside any @op / @step (or from a thread that did not
        copy the context) the request cache is off.

        Args:
            prompts: List of prompts. Each prompt is either a string or
                a list of chat messages in OpenAI format.
            n: Number of completions per prompt.
            temperature: Sampling temperature. Overrides the instance default.
            max_tokens: Max tokens. Overrides the instance default.
            stop: Stop sequences.
            show_progress: Per-call override of the instance's show_progress
                (see __init__). None (default) uses the instance setting.
            return_exceptions: When True, a request that still fails after
                retries (e.g. a gateway 504) does not discard the others: a
                prompt whose requests all failed gets the exception in its
                slot, and a prompt with only some failed requests (fan-out
                copies or n > 16 parts) gets the completions that arrived
                (fewer than n, not checked) when those failures are transient
                (timeouts, 429, 5xx). A non-transient one (a spending cap, the
                token-unit quota, a bad request) puts that exception in the
                prompt's slot even so; the completions that arrived are
                charged and request-cached. A row error (a content-filter
                block, an over-long prompt) is request-cached too: a rerun
                gets a `ReplayedRowError` with the same reason instead of
                asking again.
                When False (default), the first failure is raised.
            cache_salt: Part of the request-cache key. Pass a new value to get
                fresh completions for a prompt asked before on purpose, e.g.
                "retry 1" when re-asking prompts whose output failed to parse.
            cache_rows: For an op that builds one prompt per row, the row
                each prompt came from (aligned with `prompts`; None entries
                allowed). Changing one row then re-sends only its requests.
                Without it, a prompt is keyed on all of the op's rows.

        Returns:
            List with one entry per prompt. Each GenerateResult is a list of
            n completion strings carrying usage / reasoning / model_id / raw
            as attributes; with return_exceptions=True an entry can instead be
            the exception its requests raised.

        Raises:
            BudgetExceededError: Spend reached a spending cap (the running
                pipeline's ``max_cost_usd`` / ``max_token_units``,
                ``FORGER_MAX_COST_USD``, or ``FORGER_MAX_TOKEN_UNITS``). Checked
                before the batch and again before each request is sent, so
                requests queued behind the cap are not sent; requests in
                flight finish and are charged.
            TokenUnitQuotaExceeded: The shared token-unit quota
                (``FORGER_TOKEN_USAGE_FILE`` / ``FORGER_TOKEN_UNIT_LIMIT``) is
                used up. Checked before each request, like the caps.
            ValueError: If a prompt whose requests all succeeded gets other
                than n completions. Usage is recorded in history before
                raising. With the token-unit quota, also a model without fixed
                token-unit weights, or a response without valid usage.
        """
        if not prompts:
            return []
        if cache_rows is not None and len(cache_rows) != len(prompts):
            raise ValueError(
                f"cache_rows has {len(cache_rows)} entries for {len(prompts)} prompts"
            )
        _tracking.check_budget()  # raises BudgetExceededError at a spending cap
        _tracking.note_prompt_texts(prompts)  # did retrieved text reach a prompt?

        temperature = temperature if temperature is not None else self._temperature
        max_tokens = max_tokens if max_tokens is not None else self._max_tokens

        self._check_params(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            thinking=self._kwargs.get("thinking"),
        )

        conversations = [_to_messages(prompt) for prompt in prompts]
        call_kwargs = dict(
            model=self._model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            api_key=self._api_key,
            **self._kwargs,
        )
        # Model retries in _complete, so LiteLLM must not: num_retries=0 turns
        # off its retries and sets the openai client's max_retries to 0. A
        # max_retries the caller passed would be overwritten by it, so then
        # only the openai client retries, as asked.
        if "max_retries" not in self._kwargs:
            call_kwargs["num_retries"] = 0
        timeout = _request_timeout(self._timeout, max_tokens)
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        # Key each prompt on the op call it comes from and its row, e.g. in
        # generate_from_context(rows, "Solve {q}", model) the row {"q": "2+2"}.
        # Slots are handed out per op call, so the same request asked twice in
        # it (two identical rows, three judge copies) gets new slots each time.
        # Outside any op (a direct call, or a thread that did not copy the
        # context) the request cache is off: a key of prompt text and params
        # alone would share samples between unrelated call sites.
        scope = _REQUEST_SCOPE.get()
        caching = (
            scope is not None
            and _request_cache.enabled(getattr(self, "_request_cache", True))
            and _request_cache.cacheable(call_kwargs)
        )
        keys: list[str | None] = [None] * len(prompts)
        slot_ranges: list[range] = [range(n)] * len(prompts)
        hits: list[dict[int, dict[str, Any]]] = [{} for _ in prompts]
        # Prompts whose cached slots hold a row error, e.g. a content-filter
        # block: replayed as that error, and not asked again.
        refused: dict[int, dict[str, Any]] = {}
        # Each row's "answered before" entry, keyed on the op and the enclosing
        # step but not the prompt (see _request_cache.row_id). `run` tells this
        # run of the step from earlier ones: the step's cache key changes when
        # its code, a global prompt, or its arguments do, and two ops of one
        # step run (the same rows judged twice) share it.
        row_ids: list[str | None] = [None] * len(prompts)
        step = _tracking.active_step()
        run = None
        if caching:
            assert scope is not None  # mypy hint
            run = step.cache_key if step is not None and step.cache_key else scope.key
            read = not _NO_CACHE_SCOPE.get()  # an enclosing op got _no_cache=True
            earlier: list[dict[str, Any]] = []
            for i, messages in enumerate(conversations):
                row = None if cache_rows is None else cache_rows[i]
                row_key = _hash(row)[0] if row is not None else scope.rows_key
                key = _request_cache.request_key(
                    scope.key, row_key, messages, call_kwargs, cache_salt
                )
                first = scope.take_slots(key, n)
                keys[i], slot_ranges[i] = key, range(first, first + n)
                if row is not None:
                    row_ids[i] = _request_cache.row_id(
                        scope.op_name, step.name if step is not None else None, row_key
                    )
                if read:
                    for slot in slot_ranges[i]:
                        entry = _request_cache.read_slot(key, slot)
                        if entry is not None and _request_cache.is_row_error(entry):
                            refused.setdefault(i, entry)
                        elif entry is not None:
                            hits[i][slot] = entry
                    row_id = row_ids[i]
                    before = (
                        _request_cache.read_row(row_id)
                        if row_id is not None and len(hits[i]) < n and i not in refused
                        else None
                    )
                    if before and before["key"] != key and before.get("run") != run:
                        earlier.append(before)
            # Warn before sending: e.g. a template edited for every row misses
            # the cache for every row, and the batch costs as much as the first.
            message = _request_cache.resent_warning(
                scope.op_name, earlier, len(prompts)
            )
            if message and not scope.resent_warned:
                scope.resent_warned = True
                _logger.warning(message)
                _tracking.add_op_warning(message, "request_cache_resent")

        # Request only the missing slots. Models that ignore or reject n > 1
        # get one n=1 request per slot, e.g. prompts [A, B] with n=2 send
        # [A, A, B, B]. Others get requests of at most _MAX_N_PER_REQUEST, e.g.
        # n=40 sends n=14, 13, 13.
        fan_out = n != 1 and _fans_out_n(self._model_id)
        requests: list[tuple[list[Message], int]] = []
        owners: list[tuple[int, list[int]]] = []  # (prompt, slots asked)
        for i, messages in enumerate(conversations):
            missing = [slot for slot in slot_ranges[i] if slot not in hits[i]]
            if not missing or i in refused:
                continue
            sizes = _request_sizes(len(missing), fan_out)
            start = 0
            for size in sizes:
                copy = [dict(m) for m in messages] if len(sizes) > 1 else messages
                requests.append((copy, size))
                owners.append((i, missing[start : start + size]))
                start += size

        if caching:
            # The op logs one INFO line per call; each batch only at DEBUG.
            reused = sum(n if i in refused else len(h) for i, h in enumerate(hits))
            if scope is not None:
                scope.count(reused, n * len(prompts))
            _logger.debug(
                "request cache: %d of %d completions reused (%s)",
                reused,
                n * len(prompts),
                self._model_id,
            )

        # What each row's fresh requests cost so far, for its row entry:
        # [token units, USD], None once unknown.
        spent: dict[int, list[float | None]] = {}
        spent_lock = threading.Lock()

        def store(j: int, response: Any) -> None:
            """Cache request j's completions as soon as it succeeds."""
            i, asked = owners[j]
            key = keys[i]
            if key is None:
                return
            try:
                if len(response.choices) > len(asked):
                    return  # the count check reports it; slots would be guesses
                reasoning = _extract_reasoning(response, self._model_id)
                stored = False
                for slot, choice, thinking in zip(asked, response.choices, reasoning):
                    entry = _cache_entry(choice, thinking)
                    if entry is not None:
                        _request_cache.write_slot(key, slot, entry)
                        stored = True
                row_id = row_ids[i]
                if stored and row_id is not None:
                    remember(i, row_id, key, response)
            except Exception as exc:  # a full disk must not fail the request
                _logger.warning("request cache write failed: %s", exc)

        def remember(i: int, row_id: str, key: str, response: Any) -> None:
            """Point row i's entry at `key`, with what its requests cost so far."""
            try:
                units: float | None = self._calculate_token_units(
                    *response_token_counts(response)
                )
            except ValueError:  # missing usage
                units = None
            cost = response_cost_usd(response, self._model_id)
            # Under the lock, so two requests of one row (n=1 fan-out) leave
            # their sum, not whichever wrote last.
            with spent_lock:
                total = spent.setdefault(i, [0.0, 0.0])
                for k, value in enumerate((units, cost)):
                    old = total[k]
                    if old is None or value is None or not math.isfinite(value):
                        total[k] = None
                    else:
                        total[k] = old + value
                _request_cache.write_row(
                    row_id,
                    {"key": key, "run": run, "units": total[0], "cost_usd": total[1]},
                )

        progress = _progress_mode(
            self._show_progress if show_progress is None else show_progress
        )
        # Workers copy this context, so their requests see the projection.
        weight = sum(size for _, size in requests)
        projection = (
            _SpendProjection(self, weight, shuffle=len(prompts) > 1)
            if _projecting() and weight >= _PROJECT_MIN_TOTAL
            else None
        )
        projection_token = _PROJECTION.set(projection)
        truncation_token = _TRUNCATION.set(
            _TruncationStop(self._model_id, call_kwargs.get("max_tokens"))
            if _truncation_stopping()
            else None
        )
        # A provider may reject n > 1 outright (a 400); those requests come back
        # as exceptions so their slots can be resent as n=1 requests below.
        # Row errors are request-cached before any failure is raised, so every
        # request is sent with its exception returned (see the end of this).
        tolerant = n != 1 and not fan_out
        try:
            responses = self._send(requests, call_kwargs, progress, True, store)
        finally:
            _TRUNCATION.reset(truncation_token)
            _PROJECTION.reset(projection_token)

        # A model that should support n but returned fewer completions (e.g. a
        # gateway alias that ignores n): send each missing slot as an n=1
        # request. A short response keeps the first of its slots.
        refilled = False
        if tolerant:
            rejected = {
                j
                for j, r in enumerate(responses)
                if isinstance(r, Exception) and _rejects_n(r)
            }
            failed = {
                i
                for j, ((i, _), r) in enumerate(zip(owners, responses))
                if isinstance(r, Exception) and j not in rejected
            }
            refill_owners = [
                (i, [slot])
                for j, ((i, asked), r) in enumerate(zip(owners, responses))
                if i not in failed
                for slot in (asked if j in rejected else asked[len(r.choices) :])
            ]
            if refill_owners:
                if rejected:
                    first = min(rejected)
                    _learn_no_n(
                        self._model_id,
                        len(owners[first][1]),
                        str(responses[first]).splitlines()[0][:200],
                    )
                    for j in rejected:
                        if owners[j][0] not in failed:
                            responses[j] = None  # resent below, slot by slot
                else:
                    first_short = next(
                        (len(asked), len(r.choices))
                        for (_, asked), r in zip(owners, responses)
                        if not isinstance(r, Exception) and len(r.choices) < len(asked)
                    )
                    _learn_no_n(self._model_id, *first_short)
                offset = len(owners)
                owners.extend(refill_owners)
                refill = [
                    ([dict(m) for m in conversations[i]], 1) for i, _ in refill_owners
                ]
                responses += self._send(
                    refill,
                    call_kwargs,
                    progress,
                    True,
                    lambda j, response: store(offset + j, response),
                )
                refilled = True
        if caching:
            self._cache_row_errors(owners, responses, keys)
        if not return_exceptions:  # as _send would have raised
            for response in responses:
                if isinstance(response, Exception):
                    raise response

        fresh: list[list[tuple[list[int], Any]]] = [[] for _ in prompts]
        errors: list[list[Exception]] = [[] for _ in prompts]
        for (i, asked), response in zip(owners, responses):
            if response is None:  # rejected n, resent as n=1 requests
                continue
            if isinstance(response, Exception):
                errors[i].append(response)
            else:
                fresh[i].append((asked, response))

        results: list[Any] = []
        partial: set[int] = set()
        for i in range(len(prompts)):
            if i in refused:
                entry = refused[i]
                replayed = ReplayedRowError(entry["row_error"], entry.get("error", ""))
                if not return_exceptions:
                    raise replayed
                results.append(replayed)
                continue
            # A cap, quota, or bad request is not a short result to retry: the
            # prompt gets the exception even if some of its requests finished
            # (they are charged and request-cached, so a rerun replays them).
            fatal = next((e for e in errors[i] if not _is_transient(e)), None)
            if fatal is not None or (errors[i] and not fresh[i] and not hits[i]):
                results.append(fatal if fatal is not None else errors[i][0])
                continue
            if errors[i]:
                partial.add(i)
            results.append(
                _merge_responses(slot_ranges[i], hits[i], fresh[i], self._model_id)
            )
        # Usage already reached the step report per response (see _complete),
        # and a cache hit adds none.
        self._record(
            [r for r in results if isinstance(r, GenerateResult)], add_usage=False
        )
        _note_truncation(results, self._model_id, max_tokens)
        if n != 1:
            _check_completion_count(
                self._model_id,
                n,
                results,
                fanned_out=fan_out or refilled,
                skip=partial,
            )
        return results

    @staticmethod
    def _cache_row_errors(
        owners: list[tuple[int, list[int]]], responses: list[Any], keys: list[Any]
    ) -> None:
        """Store each row error in the slots its request asked for.

        E.g. a content-filter 400 for prompt 3's slots [0, 1]: a rerun gives
        prompt 3 the same error instead of asking again, so it gets the same
        rows (the filter does not block a prompt every time).
        """
        for (i, asked), response in zip(owners, responses):
            reason = (
                _row_error_reason(response) if isinstance(response, Exception) else None
            )
            if reason is None or keys[i] is None:
                continue
            entry = _request_cache.row_error_entry(reason, response)
            try:
                for slot in asked:
                    _request_cache.write_slot(keys[i], slot, entry)
            except Exception as exc:  # a full disk must not fail the batch
                _logger.warning("request cache write failed: %s", exc)

    def _send(
        self,
        requests: list[tuple[list[Message], int]],
        call_kwargs: dict[str, Any],
        progress: str,
        return_exceptions: bool,
        on_response: Callable[[int, Any], None] | None = None,
    ) -> list[Any]:
        """Send (messages, n) requests; raise the first failure unless asked not to.

        `on_response(i, response)` runs as each request succeeds.
        """
        responses: list[Any] = []
        if len(requests) == 1:
            messages, size = requests[0]
            try:
                responses = [
                    _complete_and_report(
                        0,
                        messages,
                        size,
                        call_kwargs,
                        self._semaphore,
                        on_response,
                        self,
                        retries=self._num_retries,
                    )
                ]
            except Exception as exc:
                if not return_exceptions:
                    raise
                responses = [exc]
        elif requests:
            responses = _batch_completion_with_progress(
                requests,
                call_kwargs,
                semaphore=self._semaphore,
                max_workers=self._max_concurrency or _DEFAULT_MAX_WORKERS,
                progress=progress,
                on_response=on_response,
                model=self,
                retries=self._num_retries,
                shuffle=getattr(_PROJECTION.get(), "shuffle", False),
            )
        if not return_exceptions:
            for response in responses:
                if isinstance(response, Exception):
                    raise response
        return responses

    def _record(self, results: list[GenerateResult], *, add_usage: bool = True) -> None:
        """Add each result to history; with add_usage, also charge its usage.

        Charging adds it to the instance totals and the step report. Model
        charges each response as it arrives instead (add_usage=False), so
        results that were never returned still count.
        """
        with self._usage_lock:
            self.history.extend(results)
        if not add_usage:
            return
        for result in results:
            units = result.usage.get("token_units", math.nan)
            cost = result.usage.get("cost_usd", 0.0) or 0.0
            self._add_totals(result.usage, cost, units)
            _tracking.add_usage(
                result.usage.get("prompt_tokens", 0) or 0,
                result.usage.get("completion_tokens", 0) or 0,
                cost,
                units,
                model_id=self._model_id,
            )


# Default cap on generated tokens for a local reasoning model. Reasoning traces
# can be long (tens of thousands of tokens), so the LiteLLM default of "let the
# provider decide" is replaced here by an explicit, generous ceiling.
_DEFAULT_LOCAL_MAX_TOKENS = 32768
_LOCAL_STEP_TOKENS = 1024  # default max_num_batched_tokens; see LocalModel._engine


@dataclass(frozen=True)
class TargetLogprobs:
    """How likely a model finds one given response: see `LocalModel.target_logprobs`.

    `logprobs` holds the natural-log probability of each response token given
    the prompt and the tokens before it. `reasoning_tokens` counts the leading
    tokens through `</think>` (0 when the response has none). `truncated` means
    the response did not fit in the context and its end was not scored.
    """

    logprobs: list[float]
    reasoning_tokens: int
    truncated: bool


def _import_vllm() -> Any:
    """Import vllm, with an install hint when it is not installed."""
    try:
        import vllm
    except ModuleNotFoundError as e:
        if e.name != "vllm":  # vllm is installed but one of its imports failed
            raise
        raise ImportError(
            "vllm is required for LocalModel generation. "
            'Install it with: pip install "pyforger[vllm]"'
        ) from e
    return vllm


class LocalModel(Model):
    """On-GPU model served locally via vLLM, with the same interface as `Model`.

    Unlike the LiteLLM-backed :class:`Model`, this loads model weights into GPU
    memory once (lazily, on the first `generate`/`generate_batch` call) and
    runs batched inference in-process. This is the efficient path for sampling
    many completions per prompt from an open-weights student model (e.g. for
    on-policy distillation).

    The engine is created lazily so that importing `pyforger` -- and even
    constructing a `LocalModel` -- never requires `vllm` or a GPU. `vllm`
    is imported only when generation actually happens; it is an optional
    dependency, installed with `pip install "pyforger[vllm]"`.

    Chat prompts are rendered with the model's own chat template. For reasoning
    models (e.g. Qwen3) the generated text contains a `<think>...</think>`
    block followed by the answer; callers that need the two separated can use
    :func:`split_reasoning`.

    Example:
        >>> model = LocalModel("Qwen/Qwen3-4B", max_model_len=40960)
        >>> model.generate("What is 2+2?", n=2)  # doctest: +SKIP
        ['<think>...</think>4', '<think>...</think>4']
    """

    def __init__(
        self,
        model_id: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        data_parallel_size: int | str = "auto",
        dtype: str = "auto",
        enable_thinking: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
        show_progress: bool | None = None,
        **engine_kwargs: Any,
    ) -> None:
        """Create a vLLM-backed local model (engine is loaded lazily).

        Args:
            model_id: HuggingFace model id or local path (e.g. "Qwen/Qwen3-4B").
            temperature: Default sampling temperature.
            max_tokens: Default cap on generated tokens per completion. Defaults
                to a generous ceiling suitable for long reasoning traces.
            max_model_len: Maximum context length (prompt + generation) for the
                engine. None lets vLLM use the model's configured maximum.
            gpu_memory_utilization: Fraction of GPU memory vLLM may use.
            tensor_parallel_size: Number of GPUs to shard each engine across.
                Raise it only for a model too large for one GPU.
            data_parallel_size: Number of engines, each a full copy of the
                model on its own `tensor_parallel_size` GPUs in its own process,
                sampling an equal share of the prompts. "auto" (the default)
                uses every visible GPU (`CUDA_VISIBLE_DEVICES`, else
                `nvidia-smi`): 8 GPUs give 8 one-GPU engines. With fewer than
                two engines the model runs in this process, as `vllm.LLM`.
                The engines hold their GPUs until `close()` or exit, so set
                `CUDA_VISIBLE_DEVICES` to leave GPUs free for other jobs.
                Each engine gets its own Triton and Inductor cache
                directories unless TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR
                are set, and runs its engine core without a forked child
                (VLLM_ENABLE_V1_MULTIPROCESSING=0 unless set).
            dtype: Engine dtype ("auto", "bfloat16", "float16", ...).
            enable_thinking: For reasoning models, whether the chat template
                should enable the thinking block. Forwarded via
                `chat_template_kwargs={"enable_thinking": ...}` unless
                `chat_template_kwargs` is given explicitly.
            chat_template_kwargs: Extra kwargs for the chat template. When set,
                it is used verbatim and `enable_thinking` is ignored.
            seed: Sampling seed for reproducible generation.
            show_progress: Progress of generation and scoring, across every
                engine: vLLM's bar (with tokens per second) in a terminal,
                otherwise "N/total done, elapsed, ~left" every 30 s. None
                (default) shows it unless FORGER_PROGRESS=0.
            **engine_kwargs: Extra keyword arguments forwarded to `vllm.LLM`
                (e.g. `max_num_seqs` to cap concurrent sequences and avoid KV
                cache over-admission on long-generation workloads).
                `max_num_batched_tokens` defaults to 1024 tokens per engine
                step (at least `max_num_seqs`), so `target_logprobs` fits in
                memory beside the KV cache.
        """
        super().__init__(
            model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **engine_kwargs,
        )
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._tensor_parallel_size = tensor_parallel_size
        self._data_parallel_size = data_parallel_size
        self._dtype = dtype
        self._enable_thinking = enable_thinking
        self._chat_template_kwargs = chat_template_kwargs
        self._seed = seed
        self._show_progress = show_progress
        self._llm: Any | None = None

    def cache_key(self) -> tuple:
        """Stable hash key for @op caching from the functional configuration.

        Includes the sampling settings, the chat-template kwargs actually sent,
        `dtype` and every extra `vllm.LLM` kwarg (e.g. `revision`,
        `quantization`), since any of those can change outputs; two instances
        that differ in them must not share cached completions.
        `gpu_memory_utilization` and `tensor_parallel_size` only affect speed
        and memory, so they are left out -- except with `max_model_len=-1`,
        where vLLM (0.14+) sizes the context to fit memory and they cap
        generation length. `data_parallel_size` and `show_progress` are always
        left out.
        """
        engine_kwargs = tuple(sorted(self._kwargs.items(), key=lambda kv: str(kv[0])))
        memory = (
            (self._gpu_memory_utilization, self._tensor_parallel_size)
            if self._max_model_len == -1
            else ()
        )
        return (
            type(self).__name__,
            self._model_id,
            self._temperature,
            self._max_tokens,
            self._max_model_len,
            self._dtype,
            self._template_kwargs(),
            self._seed,
            engine_kwargs,
            memory,
        )

    def _check_params(self, **params: Any) -> None:
        """No-op: vLLM model ids are not in LiteLLM's parameter registry."""

    def _make_token_budget(self) -> None:
        """Local inference is exempt from the token-unit quota."""
        return None

    def _calculate_token_units(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """NaN: local inference is free, even for a model family with weights."""
        return math.nan

    def _engine(self) -> Any:
        """Lazily construct and cache the vLLM engine (one per GPU group).

        With two or more GPU groups, e.g. 8 visible GPUs and the defaults, this
        is a `DataParallelEngine` of worker processes with `vllm.LLM`'s `chat`.
        """
        if self._llm is None:
            vllm = _import_vllm()  # fails here, with the install hint, without vllm
            engine_kwargs = dict(
                model=self._model_id,
                gpu_memory_utilization=self._gpu_memory_utilization,
                tensor_parallel_size=self._tensor_parallel_size,
                dtype=self._dtype,
                seed=self._seed,
                max_model_len=self._max_model_len,
                **self._kwargs,
            )
            # Scoring (`target_logprobs`) takes a full-vocabulary log-softmax of
            # every prompt token in an engine step, outside the memory vLLM sets
            # aside: for Qwen3 a 16K-token step needs ~10 GB and a 4K one ~2.5 GB,
            # which ran out of memory beside the KV cache on H100s and L40S.
            # Fewer tokens per step only changes speed.
            engine_kwargs.setdefault(
                "max_num_batched_tokens",
                max(_LOCAL_STEP_TOKENS, engine_kwargs.get("max_num_seqs") or 0),
            )
            groups = _local_dp.device_groups(
                _local_dp.visible_gpus(),
                self._tensor_parallel_size,
                self._data_parallel_size,
            )
            if groups:
                _logger.info(
                    "LocalModel %s: %d engines, one per GPU group %s",
                    self._model_id,
                    len(groups),
                    "; ".join(",".join(g) for g in groups),
                )
                self._llm = _local_dp.DataParallelEngine.start(groups, engine_kwargs)
            else:
                _local_dp.prepare_in_process_engine()
                self._llm = vllm.LLM(**engine_kwargs)
        return self._llm

    def close(self) -> None:
        """Free the GPUs held by data-parallel engines; the next call restarts them."""
        if isinstance(self._llm, _local_dp.DataParallelEngine):
            self._llm.close()
            self._llm = None

    def _template_kwargs(self) -> dict[str, Any]:
        """Resolve chat-template kwargs from explicit override or enable_thinking."""
        if self._chat_template_kwargs is not None:
            return self._chat_template_kwargs
        return {"enable_thinking": self._enable_thinking}

    def generate_batch(
        self,
        prompts: list[Prompt],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        show_progress: bool | None = None,
        return_exceptions: bool = False,
        cache_salt: str | None = None,
        cache_rows: list[Any] | None = None,
    ) -> list[GenerateResult]:
        """Generate `n` completions per prompt with the local vLLM engine.

        Not request-cached: local inference is free. `cache_salt` and `cache_rows`
        are accepted for interface compatibility and ignored.

        Args:
            prompts: List of prompts (string or OpenAI-format message list).
            n: Number of completions per prompt.
            temperature: Sampling temperature. Overrides the instance default.
            max_tokens: Max generated tokens per completion. Overrides the
                instance default (which itself defaults to a generous ceiling).
            stop: Optional stop strings.
            show_progress: Per-call override of the instance's show_progress.
                None (default) uses the instance setting.
            return_exceptions: vLLM runs the batch as one engine call, so when
                True an engine error fills every prompt's slot instead of
                being raised.

        Returns:
            List of GenerateResult, one per prompt, in input order. Each is a
            list of `n` completion strings holding the raw generated text
            (including any `<think>...</think>` block for reasoning models;
            `reasoning` is therefore `None` per choice -- use
            :func:`split_reasoning` to separate it).
        """
        if not prompts:
            return []
        _tracking.check_budget()  # raises BudgetExceededError at a spending cap
        _tracking.note_prompt_texts(prompts)  # did retrieved text reach a prompt?

        vllm = _import_vllm()

        temperature = temperature if temperature is not None else self._temperature
        if max_tokens is None:
            max_tokens = (
                self._max_tokens
                if self._max_tokens is not None
                else _DEFAULT_LOCAL_MAX_TOKENS
            )

        conversations = [_to_messages(prompt) for prompt in prompts]
        sampling_params = vllm.SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=self._seed,
        )
        try:
            outputs = self._run(
                "chat",
                conversations,
                sampling_params,
                total=len(prompts) * n,
                unit="completions",
                show_progress=show_progress,
                chat_template_kwargs=self._template_kwargs(),
            )
        except Exception as exc:
            if not return_exceptions:
                raise
            return [exc] * len(prompts)  # type: ignore[list-item]

        results: list[GenerateResult] = []
        for output in outputs:
            completions = [completion.text for completion in output.outputs]
            # Local inference has no per-call monetary cost; cost is fixed (GPU
            # time). Thinking tokens are inline in the text, so not counted apart.
            usage: dict[str, Any] = {
                "prompt_tokens": len(output.prompt_token_ids or []),
                "completion_tokens": sum(
                    len(completion.token_ids) for completion in output.outputs
                ),
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
            }
            usage["token_units"] = self._calculate_token_units(
                usage["prompt_tokens"], usage["completion_tokens"]
            )
            # raw is dropped: vLLM's RequestOutput holds per-token id lists that
            # would pin gigabytes in self.history over a long sampling run.
            result = GenerateResult(
                completions,
                usage=usage,
                reasoning=[None] * len(completions),
                model_id=self._model_id,
                raw=None,
                finish_reasons=[
                    getattr(completion, "finish_reason", None)
                    for completion in output.outputs
                ],
            )
            results.append(result)

        self._record(results)
        _note_truncation(
            results, self._model_id, max_tokens, "; raise the LocalModel's max_tokens"
        )
        if n != 1:
            _check_completion_count(self._model_id, n, results, fanned_out=False)
        return results

    def target_logprobs(
        self,
        prompts: list[Prompt],
        targets: list[str],
        *,
        show_progress: bool | None = None,
    ) -> list[TargetLogprobs]:
        """Log-probability of each given response's tokens under this model.

        Each prompt is rendered with the chat template and generation prompt, as
        for sampling, and its target follows as the assistant's response: the
        tokens SFT trains on, without the end-of-turn token. Nothing is sampled;
        every visible GPU scores a share of the targets.

        Args:
            prompts: Prompts (string or OpenAI-format message list).
            targets: One response text per prompt, e.g. an SFT target
                `"<think>...</think>answer"`.
            show_progress: Per-call override of the instance's show_progress.

        Returns:
            One `TargetLogprobs` per prompt, in input order.
        """
        if len(prompts) != len(targets):
            raise ValueError(
                f"{len(prompts)} prompts but {len(targets)} targets; pass one each"
            )
        if not prompts:
            return []
        pairs = [(_to_messages(p), t) for p, t in zip(prompts, targets)]
        scores = self._run(
            "target_logprobs",
            pairs,
            total=len(pairs),
            unit="targets scored",
            show_progress=show_progress,
            chat_template_kwargs=self._template_kwargs(),
        )
        return [TargetLogprobs(**score) for score in scores]

    def _run(
        self,
        method: str,
        items: list,
        *args: Any,
        total: int,
        unit: str,
        show_progress: bool | None,
        **kwargs: Any,
    ) -> list:
        """``chat`` or ``target_logprobs`` on the engine(s), showing progress."""
        mode = _progress_mode(
            self._show_progress if show_progress is None else show_progress
        )
        engine = self._engine()
        desc = f"{self._model_id}: {unit}"
        if isinstance(engine, _local_dp.DataParallelEngine):
            with Progress(desc, total, mode) as progress:
                live = None if mode == "off" else progress
                return getattr(engine, method)(
                    items, *args, use_tqdm=False, progress=live, **kwargs
                )

        def call(use_tqdm: Any) -> list:
            if method == "chat":
                return engine.chat(items, *args, use_tqdm=use_tqdm, **kwargs)
            return _local_dp.target_logprobs(engine, items, use_tqdm=use_tqdm, **kwargs)

        if mode != "log":
            return call(mode == "bar")  # vLLM's own bar, with tokens per second
        with Progress(desc, total, mode) as progress:
            return call(vllm_tqdm(progress.update))


# Marks the boundary between a reasoning model's thinking block and its answer.
_THINK_CLOSE = "</think>"
_THINK_OPEN = "<think>"


def split_reasoning(text: str) -> tuple[str, str]:
    """Split a reasoning-model completion into (thinking, answer).

    Reasoning models such as Qwen3 emit `<think>...</think>` followed by the
    final answer. This splits on the closing `</think>` tag, stripping an
    optional leading `<think>`. When no closing tag is present (e.g. the model
    ran out of tokens mid-thought, or thinking was disabled), the whole text is
    returned as the answer with an empty thinking string.

    Args:
        text: Raw completion text.

    Returns:
        `(thinking, answer)` with surrounding whitespace stripped.
    """
    if _THINK_CLOSE not in text:
        return "", text.strip()
    thinking, _, answer = text.partition(_THINK_CLOSE)
    return thinking.strip().removeprefix(_THINK_OPEN).strip(), answer.strip()
