"""LLM model wrapper for Forger.

Provides a simplified interface to LiteLLM for generating text
completions from any supported LLM provider.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import random
import statistics
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Literal, Union

import litellm

from pyforger import _tracking
from pyforger.utils import get_formatted_logger

# Type aliases
Message = dict[str, str]
Prompt = Union[str, list[Message]]

# Threads per generate_batch call when the Model sets no max_concurrency.
_DEFAULT_MAX_WORKERS = 100
# Most completions asked of one request. A larger n is split into concurrent
# requests, e.g. n=40 on gpt-4o sends n=14, 13 and 13.
_MAX_N_PER_REQUEST = 16
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


def _progress_mode(show_progress: bool | None) -> str:
    """Return "bar" in a terminal, "log" if stderr is redirected, or "off".

    `show_progress=None` reads `FORGER_PROGRESS` (on unless "0"/"false"/"no").
    """
    if show_progress is None:
        flag = os.environ.get("FORGER_PROGRESS", "").strip().lower()
        show_progress = flag not in {"0", "false", "no"}
    if not show_progress:
        return "off"
    isatty = getattr(sys.stderr, "isatty", None)
    return "bar" if isatty is not None and isatty() else "log"


def _request_timeout(timeout: Any, max_tokens: int | None) -> Any:
    """The `timeout` litellm gets per request attempt; "auto" picks one."""
    if timeout != "auto":
        return timeout
    if max_tokens is None:
        return _DEFAULT_TIMEOUT_S
    return max(_DEFAULT_TIMEOUT_S, float(round(max_tokens / _SLOW_TOKENS_PER_S)))


def _is_transient(exc: BaseException) -> bool:
    """True for provider errors a retry can fix: timeouts, 429s, 5xx, lost connections.

    litellm's and openai's ``Timeout`` / ``APIConnectionError`` and builtin
    ``TimeoutError`` / ``ConnectionError`` are transient; other errors are judged
    by their ``status_code``. A bad key (401), unknown model (404), or bad
    request (400, e.g. context length) is not, nor is an error with no status,
    nor an exhausted quota or budget (a 429 with "insufficient_quota").
    """
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
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS or status >= 500
    return False


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
    # Half to full backoff, so 100 requests that hit one 429 do not return
    # together.
    backoff = min(_RETRY_BASE_S * 2**retry, _RETRY_MAX_S)
    return backoff * (0.5 + random.random() / 2)


# Patched in tests, which also sleep in their fake completions.
_sleep = time.sleep


def _duration(seconds: float) -> str:
    """Short duration, e.g. "41s", "9m 12s", "10m", "1h 5m"."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s" if s % 60 else f"{s // 60}m"
    return f"{s // 3600}h {s % 3600 // 60}m"


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
    ) -> Any:
        """`_complete` for request `i`, timed from when it first holds a slot."""

        def start() -> None:
            with self.lock:
                self.started.setdefault(i, time.monotonic())

        try:
            return _complete(
                messages, n, call_kwargs, semaphore, on_start=start, retries=retries
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
                _logger.info("%s: %d/%d done, %.0fs", desc, done, total, now - start)
                logged = True
            clock.warn_stragglers(now)
            next_check = now + _PROGRESS_LOG_INTERVAL_S
    if logged:
        _logger.info(
            "%s: %d/%d done, %.0fs", desc, done, total, time.monotonic() - start
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
    )


def _complete(
    messages: list[Message],
    n: int,
    call_kwargs: dict[str, Any],
    semaphore: threading.BoundedSemaphore | None,
    on_start: Callable[[], None] | None = None,
    retries: int = 0,
) -> Any:
    """One litellm.completion call, holding a `max_concurrency` slot if any.

    A transient error (`_is_transient`: a timeout, 429, 5xx, lost connection)
    is retried up to `retries` times after a short backoff, without holding
    the slot; any other error, or the last one, is raised. So a request makes
    at most `retries + 1` attempts (LiteLLM itself is told not to retry).

    The response's usage is recorded here, in the worker, before the caller
    sees it: a batch that later raises or is interrupted keeps every
    finished request in the step report. `on_start` runs each time the
    request holds its slot, just before it is sent.
    """
    retry = 0
    while True:
        try:
            with contextlib.ExitStack() as slot:
                if semaphore is not None:
                    slot.enter_context(semaphore)
                if on_start is not None:
                    on_start()
                response = litellm.completion(messages=messages, n=n, **call_kwargs)
            break
        except Exception as exc:
            if retry >= retries or not _is_transient(exc):
                raise
            delay = _retry_delay(retry, exc)
            retry += 1
            _logger.info(
                "LLM request (%s) failed with %s; retry %d of %d in %.1fs",
                call_kwargs.get("model", "?"),
                type(exc).__name__,
                retry,
                retries,
                delay,
            )
            _sleep(delay)
    _add_response_usage(response, call_kwargs.get("model", "?"))
    return response


def _batch_completion_with_progress(
    requests: list[tuple[list[Message], int]],
    call_kwargs: dict[str, Any],
    *,
    semaphore: threading.BoundedSemaphore | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    progress: str = "bar",
    retries: int = 0,
) -> list[Any]:
    """Send (messages, n) requests concurrently; return responses or errors.

    Results are in request order. Each request retries transient errors up
    to `retries` times (see `_complete`).

    A replacement for `litellm.batch_completion`, which retrieves
    `future.result()` in submission order, so the whole call looks frozen until
    the *slowest* early-submitted item finishes. This reports progress as each
    call actually finishes: a tqdm bar (`progress="bar"`), a log line every 30s
    (`"log"`), or nothing (`"off"`). Each request holds a slot of `semaphore`
    (the Model's `max_concurrency`) while in flight. In every mode, a request
    in flight far longer than the finished ones gets one warning.
    """
    results: list[Any] = [None] * len(requests)
    total = len(requests)
    desc = f"LLM calls ({call_kwargs.get('model', '?')})"
    clock = _RequestClock(desc, call_kwargs.get("timeout"))
    executor = ThreadPoolExecutor(max_workers=min(max_workers, total))
    try:
        future_to_idx: dict[Future, int] = {}
        for i, (msgs, n) in enumerate(requests):
            # One context copy per request, so add_usage in the worker sees the
            # caller's op / step / pipeline contextvars.
            run: Callable[..., Any] = contextvars.copy_context().run
            future = executor.submit(
                run, clock.run, i, msgs, n, call_kwargs, semaphore, retries
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
    """list[str] with `usage` / `reasoning` / `model_id` / `raw` attributes.

    `raw` is the LiteLLM response. When `Model.generate_batch` fans `n` out into
    separate requests, it is the list of those responses, in completions order.
    """

    usage: dict[str, Any]
    reasoning: list[str | None]
    model_id: str
    raw: Any

    def __init__(
        self,
        completions: list[str],
        *,
        usage: dict[str, Any],
        reasoning: list[str | None],
        model_id: str,
        raw: Any,
    ) -> None:
        """Wrap completions with per-call metadata."""
        super().__init__(completions)
        self.usage = usage
        self.reasoning = reasoning
        self.model_id = model_id
        self.raw = raw

    def __reduce__(self):
        """Pickle support."""
        return (
            _rebuild_generate_result,
            (list(self), self.usage, self.reasoning, self.model_id, self.raw),
        )


def _rebuild_generate_result(
    completions: list[str],
    usage: dict[str, Any],
    reasoning: list[str | None],
    model_id: str,
    raw: Any,
) -> GenerateResult:
    """Pickle helper for GenerateResult."""
    return GenerateResult(
        completions, usage=usage, reasoning=reasoning, model_id=model_id, raw=raw
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


def _learn_no_n(model_id: str, n: int, got: int) -> None:
    """Fan out `model_id` from now on; warn the first time."""
    with _LEARNED_NO_N_LOCK:
        if model_id in _LEARNED_NO_N:
            return
        _LEARNED_NO_N.add(model_id)
    _logger.warning(
        "%s returned %d completions for n=%d; sending the missing ones as n=1 "
        "requests, and fanning out n for this model from now on.",
        model_id,
        got,
        n,
    )


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


def _merge_responses(responses: list[Any], model_id: str) -> GenerateResult:
    """One GenerateResult from one prompt's responses, summing their usage."""
    completions: list[str] = []
    reasoning: list[str | None] = []
    usage: dict[str, Any] = {}
    for response in responses:
        completions.extend(choice.message.content or "" for choice in response.choices)
        reasoning.extend(_extract_reasoning(response, model_id))
        for key, value in _extract_usage(response).items():
            usage[key] = usage.get(key, 0) + value
        usage["cost_usd"] = usage.get("cost_usd", 0.0) + _extract_cost(
            response, model_id
        )
    return GenerateResult(
        completions,
        usage=usage,
        reasoning=reasoning,
        model_id=model_id,
        raw=responses[0] if len(responses) == 1 else list(responses),
    )


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


def _extract_cost(response: Any, model_id: str) -> float:
    """Cost in USD; `0.0` when unavailable. Warns once per `model_id` on miss."""
    hidden = getattr(response, "_hidden_params", None) or {}
    cost = hidden.get("response_cost")
    if cost is None:
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = None
    if cost is None:
        if model_id not in _COST_WARNING_SEEN:
            _COST_WARNING_SEEN.add(model_id)
            warnings.warn(
                f"Cost unavailable for {model_id} (model may be absent from "
                f"LiteLLM's pricing registry). total_cost may undercount.",
                UserWarning,
                stacklevel=2,
            )
        return 0.0
    return float(cost)


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
        temperature: float = 0.7,
        max_tokens: int | None = None,
        num_retries: int = 2,
        show_progress: bool | None = None,
        max_concurrency: int | None = None,
        timeout: float | Literal["auto"] | None = "auto",
        **kwargs: Any,
    ) -> None:
        """Create a Model instance.

        Args:
            model_id: LiteLLM model string (e.g. "gpt-4o",
                "anthropic/claude-sonnet-4-20250514").
            api_key: API key. Falls back to provider-specific env vars
                (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) when None.
            temperature: Default sampling temperature.
            max_tokens: Default max tokens. None lets the provider decide.
            num_retries: Retries per request on transient failures (a
                timeout, 429, 5xx, or lost connection), after a 1s, 2s, ...
                backoff or a 429's Retry-After. A request makes at most
                num_retries + 1 attempts: LiteLLM is sent num_retries=0, so
                neither it nor the openai client retries on top. A 400, 401,
                403, 404, or an exhausted quota is not retried.
            show_progress: Progress for a generate_batch() call that sends
                more than one request (several prompts, or n > 1 on a model
                that does not support n): a tqdm bar when stderr is a
                terminal, otherwise a log line every 30 seconds. None (default)
                shows it unless the environment sets FORGER_PROGRESS=0.
            max_concurrency: Most requests this instance has in flight at once,
                across all generate_batch() calls and threads (each n fan-out
                copy counts). None (default) caps each call at 100 requests.
                Not part of cache_key().
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
        self._model_id = model_id
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_retries = num_retries
        self._show_progress = show_progress
        self._max_concurrency = max_concurrency
        self._semaphore = _make_semaphore(max_concurrency)
        self._timeout = timeout
        self._kwargs = kwargs
        self.history: list[GenerateResult] = []
        self._check_params(
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=kwargs.get("thinking"),
        )

    def __repr__(self) -> str:
        """Show model id for debugging."""
        return f"Model({self._model_id!r})"

    def __getstate__(self) -> dict[str, Any]:
        """Pickle/copy without the semaphore (locks cannot be pickled)."""
        state = self.__dict__.copy()
        state.pop("_semaphore", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore, with a fresh semaphore of the same max_concurrency."""
        self.__dict__.update(state)
        self._semaphore = _make_semaphore(state.get("_max_concurrency"))
        # A Model pickled before Model(timeout=) existed.
        self.__dict__.setdefault("_timeout", "auto")

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

    @property
    def total_cost(self) -> float:
        """Sum of cost_usd across every call in this Model's history."""
        return sum(r.usage.get("cost_usd", 0.0) for r in self.history)

    @property
    def total_usage(self) -> dict[str, int]:
        """Sum of token counts across every call in this Model's history."""
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        for r in self.history:
            for key in totals:
                totals[key] += r.usage.get(key, 0)
        return totals

    def reset_history(self) -> None:
        """Drop all recorded calls; total_cost and total_usage reset to zero."""
        self.history = []

    def generate(
        self,
        prompt: Prompt,
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
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

        Returns:
            List of n response strings.
        """
        return self.generate_batch(
            [prompt],
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
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
    ) -> list[GenerateResult]:
        """Generate n responses for each prompt in a batch.

        Models that do not support n > 1 (Claude, Grok, Gemini, o1/o3/o4,
        gpt-5, or any whose LiteLLM supported params lack "n") get n separate
        concurrent n=1 requests per prompt, merged into one GenerateResult
        with summed usage and cost. Other models get one request with n, or
        several when n > 16 (n=40 sends n=14, 13, 13), merged the same way.
        All requests are sent at once, up to the Model's max_concurrency.

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
                (fewer than n, not checked).
                When False (default), the first failure is raised.

        Returns:
            List with one entry per prompt. Each GenerateResult is a list of
            n completion strings carrying usage / reasoning / model_id / raw
            as attributes; with return_exceptions=True an entry can instead be
            the exception its requests raised.

        Raises:
            ValueError: If a prompt whose requests all succeeded gets other
                than n completions. Usage is recorded in history before
                raising.
        """
        if not prompts:
            return []

        temperature = temperature if temperature is not None else self._temperature
        max_tokens = max_tokens if max_tokens is not None else self._max_tokens

        self._check_params(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            thinking=self._kwargs.get("thinking"),
        )

        # Models that ignore or reject n > 1 get n concurrent n=1 requests per
        # prompt, e.g. prompts [A, B] with n=2 send [A, A, B, B]. Others get
        # requests of at most _MAX_N_PER_REQUEST, e.g. n=40 sends n=14, 13, 13.
        fan_out = n != 1 and _fans_out_n(self._model_id)
        sizes = _request_sizes(n, fan_out)
        copies = len(sizes)
        prompt_messages = [_to_messages(prompt) for prompt in prompts]
        requests = [
            ([dict(m) for m in messages] if copies > 1 else messages, size)
            for messages in prompt_messages
            for size in sizes
        ]
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
        progress = _progress_mode(
            self._show_progress if show_progress is None else show_progress
        )
        responses = self._send(requests, call_kwargs, progress, return_exceptions)
        groups = [responses[i : i + copies] for i in range(0, len(responses), copies)]

        # A model that should support n but returned fewer completions (e.g. a
        # gateway alias that ignores n): send the missing ones as n=1 requests.
        refilled = False
        if n != 1 and not fan_out:
            short = {
                i: n - got
                for i, group in enumerate(groups)
                if not any(isinstance(r, Exception) for r in group)
                and (got := sum(len(r.choices) for r in group)) < n
            }
            if short:
                first = next(iter(short))
                _learn_no_n(self._model_id, n, n - short[first])
                refill = [
                    ([dict(m) for m in prompt_messages[i]], 1)
                    for i, missing in short.items()
                    for _ in range(missing)
                ]
                extra = iter(
                    self._send(refill, call_kwargs, progress, return_exceptions)
                )
                for i, missing in short.items():
                    groups[i] = groups[i] + [next(extra) for _ in range(missing)]
                refilled = True

        results: list[Any] = []
        partial: set[int] = set()
        for group in groups:
            finished = [r for r in group if not isinstance(r, Exception)]
            if not finished:
                results.append(group[0])
                continue
            if len(finished) < len(group):
                partial.add(len(results))
            results.append(_merge_responses(finished, self._model_id))
        # Usage already reached the step report per response (see _complete).
        self._record(
            [r for r in results if isinstance(r, GenerateResult)], add_usage=False
        )
        if n != 1:
            _check_completion_count(
                self._model_id,
                n,
                results,
                fanned_out=fan_out or refilled,
                skip=partial,
            )
        return results

    def _send(
        self,
        requests: list[tuple[list[Message], int]],
        call_kwargs: dict[str, Any],
        progress: str,
        return_exceptions: bool,
    ) -> list[Any]:
        """Send (messages, n) requests; raise the first failure unless asked not to."""
        responses: list[Any]
        if len(requests) == 1:
            messages, size = requests[0]
            try:
                responses = [
                    _complete(
                        messages,
                        size,
                        call_kwargs,
                        self._semaphore,
                        retries=self._num_retries,
                    )
                ]
            except Exception as exc:
                if not return_exceptions:
                    raise
                responses = [exc]
        else:
            responses = _batch_completion_with_progress(
                requests,
                call_kwargs,
                semaphore=self._semaphore,
                max_workers=self._max_concurrency or _DEFAULT_MAX_WORKERS,
                progress=progress,
                retries=self._num_retries,
            )
        if not return_exceptions:
            for response in responses:
                if isinstance(response, Exception):
                    raise response
        return responses

    def _record(self, results: list[GenerateResult], *, add_usage: bool = True) -> None:
        """Add each result to history and, with add_usage, to the step report."""
        for result in results:
            self.history.append(result)
            if add_usage:
                _tracking.add_usage(
                    result.usage.get("prompt_tokens", 0) or 0,
                    result.usage.get("completion_tokens", 0) or 0,
                    result.usage.get("cost_usd", 0.0) or 0.0,
                )


# Default cap on generated tokens for a local reasoning model. Reasoning traces
# can be long (tens of thousands of tokens), so the LiteLLM default of "let the
# provider decide" is replaced here by an explicit, generous ceiling.
_DEFAULT_LOCAL_MAX_TOKENS = 32768


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
        dtype: str = "auto",
        enable_thinking: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
        show_progress: bool = False,
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
            tensor_parallel_size: Number of GPUs to shard the model across.
            dtype: Engine dtype ("auto", "bfloat16", "float16", ...).
            enable_thinking: For reasoning models, whether the chat template
                should enable the thinking block. Forwarded via
                `chat_template_kwargs={"enable_thinking": ...}` unless
                `chat_template_kwargs` is given explicitly.
            chat_template_kwargs: Extra kwargs for the chat template. When set,
                it is used verbatim and `enable_thinking` is ignored.
            seed: Sampling seed for reproducible generation.
            show_progress: When True, display vLLM's per-call progress bar (with
                live input/output tokens-per-second) during generation. Useful
                for long batched runs; off by default to keep logs quiet.
            **engine_kwargs: Extra keyword arguments forwarded to `vllm.LLM`
                (e.g. `max_num_seqs` to cap concurrent sequences and avoid KV
                cache over-admission on long-generation workloads).
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
        generation length. `show_progress` is always left out.
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

    def _engine(self) -> Any:
        """Lazily construct and cache the vLLM engine."""
        if self._llm is None:
            self._llm = _import_vllm().LLM(
                model=self._model_id,
                gpu_memory_utilization=self._gpu_memory_utilization,
                tensor_parallel_size=self._tensor_parallel_size,
                dtype=self._dtype,
                seed=self._seed,
                max_model_len=self._max_model_len,
                **self._kwargs,
            )
        return self._llm

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
    ) -> list[GenerateResult]:
        """Generate `n` completions per prompt with the local vLLM engine.

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
            outputs = self._engine().chat(
                conversations,
                sampling_params,
                chat_template_kwargs=self._template_kwargs(),
                use_tqdm=(
                    self._show_progress if show_progress is None else show_progress
                ),
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
            # raw is dropped: vLLM's RequestOutput holds per-token id lists that
            # would pin gigabytes in self.history over a long sampling run.
            result = GenerateResult(
                completions,
                usage=usage,
                reasoning=[None] * len(completions),
                model_id=self._model_id,
                raw=None,
            )
            results.append(result)

        self._record(results)
        if n != 1:
            _check_completion_count(self._model_id, n, results, fanned_out=False)
        return results


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
