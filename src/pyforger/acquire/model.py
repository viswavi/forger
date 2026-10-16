"""LLM model wrapper for Forger.

Provides a simplified interface to LiteLLM for generating text
completions from any supported LLM provider.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import warnings
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Union

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
# line at most this often instead of a redrawn bar.
_PROGRESS_LOG_INTERVAL_S = 30.0

_progress_logger = get_formatted_logger(__name__)


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


def _log_progress(futures: Iterable[Future], total: int, desc: str) -> Iterator[Future]:
    """Yield `futures` and log "<desc>: 120/500 done, 31s" at most every 30s.

    A batch that finishes within the interval logs nothing.
    """
    start = last = time.monotonic()
    done = 0
    for future in futures:
        done += 1
        now = time.monotonic()
        if now - last >= _PROGRESS_LOG_INTERVAL_S or (done == total and last > start):
            _progress_logger.info(
                "%s: %d/%d done, %.0fs", desc, done, total, now - start
            )
            last = now
        yield future


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


def _complete(
    messages: list[Message],
    n: int,
    call_kwargs: dict[str, Any],
    semaphore: threading.BoundedSemaphore | None,
) -> Any:
    """One litellm.completion call, holding a `max_concurrency` slot if any."""
    if semaphore is None:
        return litellm.completion(messages=messages, n=n, **call_kwargs)
    with semaphore:
        return litellm.completion(messages=messages, n=n, **call_kwargs)


def _batch_completion_with_progress(
    requests: list[tuple[list[Message], int]],
    call_kwargs: dict[str, Any],
    *,
    semaphore: threading.BoundedSemaphore | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    progress: str = "bar",
) -> list[Any]:
    """Send (messages, n) requests concurrently; return responses or errors.

    Results are in request order.

    A replacement for `litellm.batch_completion`, which retrieves
    `future.result()` in submission order, so the whole call looks frozen until
    the *slowest* early-submitted item finishes. This reports progress via
    `as_completed` as each call actually finishes: a tqdm bar (`progress="bar"`),
    a log line every 30s (`"log"`), or nothing (`"off"`). Each request holds a
    slot of `semaphore` (the Model's `max_concurrency`) while in flight.
    """
    results: list[Any] = [None] * len(requests)
    total = len(requests)
    desc = f"LLM calls ({call_kwargs.get('model', '?')})"
    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
        future_to_idx = {
            executor.submit(_complete, msgs, n, call_kwargs, semaphore): i
            for i, (msgs, n) in enumerate(requests)
        }
        finished: Iterable[Future] = as_completed(future_to_idx)
        if progress == "bar":
            from tqdm import tqdm

            finished = tqdm(finished, total=total, desc=desc, leave=False)
        elif progress == "log":
            finished = _log_progress(finished, total, desc)
        for future in finished:
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = exc
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
            num_retries: Retries with exponential backoff on transient
                failures.
            show_progress: Progress for a generate_batch() call that sends
                more than one request (several prompts, or n > 1 on a model
                that does not support n): a tqdm bar when stderr is a
                terminal, otherwise a log line every 30 seconds. None (default)
                shows it unless the environment sets FORGER_PROGRESS=0.
            max_concurrency: Most requests this instance has in flight at once,
                across all generate_batch() calls and threads (each n fan-out
                copy counts). None (default) caps each call at 100 requests.
                Not part of cache_key().
            **kwargs: Passed through to litellm.completion()
                (e.g. api_base, timeout, organization).
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
        requests = [
            ([dict(m) for m in messages] if copies > 1 else messages, size)
            for messages in map(_to_messages, prompts)
            for size in sizes
        ]
        call_kwargs = dict(
            model=self._model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            api_key=self._api_key,
            num_retries=self._num_retries,
            **self._kwargs,
        )
        responses: list[Any]
        if len(requests) == 1:
            messages, size = requests[0]
            try:
                responses = [_complete(messages, size, call_kwargs, self._semaphore)]
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
                progress=_progress_mode(
                    self._show_progress if show_progress is None else show_progress
                ),
            )
        if not return_exceptions:
            for response in responses:
                if isinstance(response, Exception):
                    raise response

        results: list[Any] = []
        partial: set[int] = set()
        for i in range(0, len(responses), copies):
            group = responses[i : i + copies]
            finished = [r for r in group if not isinstance(r, Exception)]
            if not finished:
                results.append(group[0])
                continue
            if len(finished) < len(group):
                partial.add(len(results))
            results.append(_merge_responses(finished, self._model_id))
        self._record([r for r in results if isinstance(r, GenerateResult)])
        if n != 1:
            _check_completion_count(
                self._model_id, n, results, fanned_out=fan_out, skip=partial
            )
        return results

    def _record(self, results: list[GenerateResult]) -> None:
        """Add each result to history and its usage to the step report, once."""
        for result in results:
            self.history.append(result)
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
