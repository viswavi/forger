"""Sample a base model or a trained ``tinker://`` checkpoint remotely on Tinker.

Tinker samples token ids, so prompts are rendered with the model's own HF chat
template and completions decoded with ``skip_special_tokens=True``, exactly as
PostDataBench's evaluator (``coding/tinker_sampling.py``) does. Rendering is
duplicated here rather than shared because the evaluator runs in environments
without pyforger.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import math
import threading
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from pyforger import _tracking
from pyforger.acquire import _request_cache
from pyforger.acquire.model import (
    _DEFAULT_LOCAL_MAX_TOKENS,
    GenerateResult,
    Model,
    Prompt,
    _is_transient,
    _note_truncation,
    _progress_mode,
    _RequestClock,
    _to_messages,
    _watch,
)
from pyforger.op import _NO_CACHE_SCOPE, _REQUEST_SCOPE, _hash
from pyforger.tinker_usage import (
    TINKER_PRICES_PER_MILLION,
    TinkerUsageLedger,
    max_concurrent_samples,
    sample_tokens,
    tinker_cost_usd,
)

logger = logging.getLogger(__name__)

# Context window when Tinker's server capabilities do not list the base model.
_DEFAULT_MAX_CONTEXT = 32768
# Backoff before retry k (0-based) is _RETRY_DELAY_S * 2**k seconds.
_RETRY_DELAY_S = 1.0
_TRANSIENT_ERRORS = ("APIConnectionError", "RateLimitError", "InternalServerError")
_RATE_KINDS = ("prefill", "cached_prefill", "sample")

# The process-wide cap on in-flight sample requests, shared by every
# TinkerModel (and thread) in the process: TINKER_MAX_CONCURRENT_SAMPLES, read
# when a request starts (one semaphore per value the variable has had).
_SLOTS: dict[int, threading.BoundedSemaphore] = {}
_SLOTS_LOCK = threading.Lock()
_PRICE_WARNING_SEEN: set[str] = set()


def _process_slots() -> threading.BoundedSemaphore:
    limit = max_concurrent_samples()
    with _SLOTS_LOCK:
        if limit not in _SLOTS:
            _SLOTS[limit] = threading.BoundedSemaphore(limit)
        return _SLOTS[limit]


class _Cancelled(RuntimeError):
    """The generate_batch call was interrupted; this request is not sent."""


@contextlib.contextmanager
def _acquire(slots: threading.BoundedSemaphore, cancel: threading.Event):
    """Hold a slot; give up waiting (raise `_Cancelled`) once `cancel` is set."""
    while not slots.acquire(timeout=0.1):
        if cancel.is_set():
            raise _Cancelled("generate_batch was interrupted")
    try:
        yield
    finally:
        slots.release()


# Worst-case dollars of this process's in-flight requests per extra `budget`
# object (TinkerModel(budget=...)), as TinkerUsageLedger.reserve does per file.
_BUDGET_RESERVED: dict[int, tuple[int, float]] = {}
_BUDGET_CHANGED = threading.Condition()


@contextlib.contextmanager
def _reserve_on(budget: Any, estimate_usd: float, cancel: threading.Event):
    """Hold a request's worst case against `budget` while it runs.

    `budget.check(estimate)` must raise when spend plus the estimate would
    cross its limit. The estimate checked includes this process's other
    requests in flight on the same budget; a request that does not fit waits
    for them to be recorded, and raises when none are in flight.
    """
    key = id(budget)
    with _BUDGET_CHANGED:
        while True:
            count, reserved = _BUDGET_RESERVED.get(key, (0, 0.0))
            try:
                budget.check(reserved + estimate_usd)
            except Exception as exc:
                # PostDataBench's TinkerBudget raises its own TinkerBudgetExceeded.
                if not count or not type(exc).__name__.endswith("BudgetExceeded"):
                    raise
                if cancel.is_set():
                    raise _Cancelled("generate_batch was interrupted") from None
                _BUDGET_CHANGED.wait(timeout=0.1)
                continue
            _BUDGET_RESERVED[key] = (count + 1, reserved + estimate_usd)
            break
    try:
        yield
    finally:
        with _BUDGET_CHANGED:
            count, reserved = _BUDGET_RESERVED.pop(key)
            if count > 1:
                _BUDGET_RESERVED[key] = (count - 1, reserved - estimate_usd)
            _BUDGET_CHANGED.notify_all()


def _import_tinker() -> Any:
    """Import tinker, with an install hint when it is not installed."""
    try:
        import tinker
    except ModuleNotFoundError as e:
        if e.name != "tinker":  # tinker is installed but one of its imports failed
            raise
        raise ImportError(
            "tinker is required for TinkerModel generation. "
            'Install it with: pip install "pyforger[tinker]"'
        ) from e
    return tinker


@dataclass
class _Runtime:
    """Clients and model facts resolved on the first generation call."""

    types: Any
    sampler: Any
    tokenizer: Any
    base_model: str
    stop_ids: list[int]
    max_context: int
    session_id: str | None
    ledger: TinkerUsageLedger | None
    rates: dict[str, float] | None
    transient: tuple[type[BaseException], ...]


class TinkerModel(Model):
    """A base model or ``tinker://`` checkpoint sampled on Tinker, like `Model`.

    `n` samples of one prompt are a single Tinker request (``num_samples=n``),
    so they share one prefill: the prompt is billed once at the prefill rate and
    the other ``n - 1`` copies at the prefix-cache rate. Prompts run
    concurrently, at most ``TINKER_MAX_CONCURRENT_SAMPLES`` (default 600)
    requests in flight per process.

    Like :class:`LocalModel`, reasoning models (e.g. Qwen3) return their
    ``<think>...</think>`` block inline and ``reasoning`` is ``None`` per
    choice; use :func:`split_reasoning` to separate it.

    Spend is charged to the shared ledger named by ``FORGER_TINKER_USAGE_FILE``
    and ``FORGER_TINKER_USD_LIMIT`` when both are set, and to ``budget`` when
    given; calls that would exceed either limit raise `TinkerBudgetExceeded`.

    Example:
        >>> model = TinkerModel("Qwen/Qwen3-8B")
        >>> model.generate("What is 2+2?", n=2)  # doctest: +SKIP
        ['<think>...</think>4', '<think>...</think>4']
    """

    def __init__(
        self,
        model_id: str,
        *,
        temperature: float = 0.6,
        max_tokens: int | None = None,
        top_p: float = 0.95,
        top_k: int = 20,
        enable_thinking: bool | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
        num_retries: int = 2,
        max_concurrent: int | None = None,
        prices: dict[str, float] | None = None,
        budget: Any = None,
        show_progress: bool | None = None,
        request_cache: bool = True,
    ) -> None:
        """Create a Tinker-backed model (clients are created lazily).

        Args:
            model_id: A base model name (e.g. "Qwen/Qwen3-8B") or a sampler
                checkpoint path ("tinker://<run>/sampler_weights/<name>").
            temperature: Default sampling temperature. The defaults for
                temperature, top_p and top_k are Qwen3's thinking-mode settings.
            max_tokens: Default cap on generated tokens per completion
                (default 32768). Each request is clamped to the context window
                left after its prompt.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling (-1 for no limit).
            enable_thinking: Passed to the chat template when not None; None
                keeps the template's default (thinking on for Qwen3).
            chat_template_kwargs: Extra chat-template kwargs. When set, it is
                used verbatim and `enable_thinking` is ignored.
            seed: Optional sampling seed.
            num_retries: Retries per request on transient Tinker errors
                (connection errors, rate limits, server errors).
            max_concurrent: In-flight requests per process. Defaults to
                TINKER_MAX_CONCURRENT_SAMPLES, else 600.
            prices: USD per million tokens for "prefill", "cached_prefill" and
                "sample", overriding the built-in table for `cost_usd`. A
                shared ledger always charges its own built-in rates.
            budget: Optional extra ledger with ``check(estimate_usd)`` and
                ``record(tokens, session_id)``, e.g. PostDataBench's
                ``TinkerBudget``. Charged in addition to the environment ledger.
            show_progress: Progress for a call with several prompts: a tqdm bar
                when stderr is a terminal, otherwise a log line every 30
                seconds. None (default) shows it unless FORGER_PROGRESS=0.
            request_cache: Keep each sample on disk and reuse it when the same
                op call asks again (see generate_batch); False, or
                FORGER_REQUEST_CACHE=0, turns it off. Not part of cache_key().
        """
        super().__init__(
            model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=num_retries,
            show_progress=show_progress,
            request_cache=request_cache,
        )
        if not model_id:
            raise ValueError("model_id must name a base model or a tinker:// path")
        if prices is not None:
            missing = [kind for kind in _RATE_KINDS if kind not in prices]
            if missing:
                raise ValueError(f"prices is missing {missing}")
        self._top_p = top_p
        self._top_k = top_k
        self._enable_thinking = enable_thinking
        self._chat_template_kwargs = chat_template_kwargs
        self._seed = seed
        self._max_concurrent = (
            max_concurrent if max_concurrent is not None else max_concurrent_samples()
        )
        if self._max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self._instance_slots = threading.BoundedSemaphore(self._max_concurrent)
        self._prices = dict(prices) if prices is not None else None
        self._budget = budget
        # Read now so a half-configured environment fails before any call.
        self._ledger_config = TinkerUsageLedger.env_config()
        self._client_lock = threading.Lock()
        self._runtime: _Runtime | None = None

    def __repr__(self) -> str:
        """Show model id for debugging."""
        return f"TinkerModel({self._model_id!r})"

    def __getstate__(self) -> dict[str, Any]:
        """Omit clients and locks; they are recreated on the next call."""
        state = super().__getstate__()
        state.pop("_client_lock", None)
        state.pop("_instance_slots", None)
        state["_runtime"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state with fresh locks."""
        super().__setstate__(state)
        self._client_lock = threading.Lock()
        self._instance_slots = threading.BoundedSemaphore(self._max_concurrent)

    def cache_key(self) -> tuple:
        """Stable hash key for @op caching from the sampling configuration.

        `model_id` tells a base model from a ``tinker://`` checkpoint. Retries,
        concurrency, prices, budgets and progress bars do not change outputs.
        """
        return (
            type(self).__name__,
            self._model_id,
            self._temperature,
            self._max_tokens,
            self._top_p,
            self._top_k,
            self._template_kwargs(),
            self._seed,
        )

    def request_scope_key(self) -> tuple:
        """This model as an op argument in request-cache scopes.

        The sampling configuration of cache_key(); a ``tinker://`` checkpoint
        path names its weights, so another checkpoint never reuses samples.
        """
        return self.cache_key()

    def _check_params(self, **params: Any) -> None:
        """No-op: Tinker model ids are not in LiteLLM's parameter registry."""

    def _calculate_token_units(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """NaN: Tinker spend is in dollars, even for a model family with weights."""
        return math.nan

    def _make_token_budget(self) -> None:
        # Token units cover LiteLLM generation; Tinker spend is in dollars.
        return None

    def _template_kwargs(self) -> dict[str, Any]:
        if self._chat_template_kwargs is not None:
            return self._chat_template_kwargs
        if self._enable_thinking is None:
            return {}
        return {"enable_thinking": self._enable_thinking}

    def _ensure_runtime(self) -> _Runtime:
        """Open the sampling client and load the tokenizer once."""
        with self._client_lock:
            if self._runtime is None:
                self._runtime = self._open()
            return self._runtime

    def _open(self) -> _Runtime:
        tinker = _import_tinker()
        from transformers import AutoTokenizer, GenerationConfig

        service = tinker.ServiceClient(user_metadata={"client": "pyforger.TinkerModel"})
        if self._model_id.startswith("tinker://"):
            sampler = service.create_sampling_client(model_path=self._model_id)
        else:
            sampler = service.create_sampling_client(base_model=self._model_id)
        base_model = sampler.get_base_model()
        ledger = None
        if self._ledger_config is not None:
            path, limit = self._ledger_config
            # Unknown prices or a ledger for another model raise here: fail closed.
            ledger = TinkerUsageLedger(path, limit, model=base_model)
            ledger.snapshot()
        rates = self._prices or TINKER_PRICES_PER_MILLION.get(base_model)
        if rates is None and self._budget is not None:
            raise ValueError(
                f"No Tinker prices for {base_model}; pass prices= to use a budget"
            )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        # Stop on generation_config's eos ids, as vLLM does (Qwen3: <|im_end|>,
        # <|endoftext|>).
        eos = GenerationConfig.from_pretrained(base_model).eos_token_id
        stop_ids = eos if isinstance(eos, list) else [eos or tokenizer.eos_token_id]
        max_context = next(
            (
                m.max_context_length
                for m in service.get_server_capabilities().supported_models
                if m.model_name == base_model and m.max_context_length
            ),
            _DEFAULT_MAX_CONTEXT,
        )
        transient = tuple(
            error
            for error in (getattr(tinker, name, None) for name in _TRANSIENT_ERRORS)
            if isinstance(error, type) and issubclass(error, BaseException)
        )
        return _Runtime(
            types=tinker.types,
            sampler=sampler,
            tokenizer=tokenizer,
            base_model=base_model,
            stop_ids=list(stop_ids),
            max_context=max_context,
            # Lets billing records be matched to the ledger's sessions.
            session_id=getattr(getattr(service, "holder", None), "_session_id", None),
            ledger=ledger,
            rates=rates,
            transient=transient,
        )

    def _render(self, runtime: _Runtime, prompt: Prompt) -> list[int]:
        """Prompt token ids; Qwen3 prompts end in the assistant turn header."""
        text = runtime.tokenizer.apply_chat_template(
            _to_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
            **self._template_kwargs(),
        )
        return runtime.tokenizer.encode(text, add_special_tokens=False)

    @contextlib.contextmanager
    def _budgeted(
        self, runtime: _Runtime, estimate: dict[str, int], cancel: threading.Event
    ):
        """Check budgets for a request's worst case; hold it reserved meanwhile.

        Both the environment ledger and `budget` hold the worst case of every
        request in flight in this process, so concurrent requests cannot each
        pass the check on the same spend.
        """
        with contextlib.ExitStack() as stack:
            if runtime.ledger is not None:
                stack.enter_context(
                    runtime.ledger.reserve(runtime.ledger.cost(estimate))
                )
            if self._budget is not None:
                assert runtime.rates is not None
                stack.enter_context(
                    _reserve_on(
                        self._budget, tinker_cost_usd(estimate, runtime.rates), cancel
                    )
                )
            yield

    def _charge(self, runtime: _Runtime, tokens: dict[str, int]) -> None:
        if runtime.ledger is not None:
            runtime.ledger.record(tokens, runtime.session_id)
        if self._budget is not None:
            self._budget.record(tokens, runtime.session_id)

    def _sample(
        self,
        runtime: _Runtime,
        prompt: list[int],
        n: int,
        params: Any,
        max_tokens: int,
        cancel: threading.Event,
    ) -> tuple[Any, dict[str, int], Exception | None]:
        """One request for n samples: check budgets, sample, charge billed tokens.

        Runs in a worker thread with the caller's context, so the charge reaches
        the caller's step report and spend ledger as soon as the request ends.
        Returns the response, its billed tokens, and the error from writing the
        ledgers, if any: the samples are paid for either way, so the caller
        caches them before raising it. Once `cancel` is set (the call was
        interrupted), no further attempt is sent.
        """
        estimate = sample_tokens(len(prompt), n, 0, [max_tokens] * n)
        for attempt in range(self._num_retries + 1):
            try:
                # Checked once a slot is free, before every attempt: neither a
                # queued request nor a retry may start past the limit. The
                # worst case stays reserved on the ledgers until it is charged.
                with _acquire(self._instance_slots, cancel), _acquire(
                    _process_slots(), cancel
                ), self._budgeted(runtime, estimate, cancel):
                    if cancel.is_set():
                        raise _Cancelled("generate_batch was interrupted")
                    _tracking.check_budget()  # max_cost_usd, FORGER_MAX_COST_USD
                    response = runtime.sampler.sample(
                        prompt=runtime.types.ModelInput.from_ints(prompt),
                        num_samples=n,
                        sampling_params=params,
                    ).result()
                    billed = sample_tokens(
                        len(prompt),
                        n,
                        getattr(response, "prompt_cache_hit_tokens", 0) or 0,
                        [len(sequence.tokens) for sequence in response.sequences],
                    )
                    # The step report and totals first: a ledger write that
                    # fails must not hide spend that Tinker already billed.
                    self._account(
                        len(prompt),
                        sum(len(seq.tokens) for seq in response.sequences),
                        self._cost(runtime, billed),
                    )
                    charge_error: Exception | None = None
                    try:
                        self._charge(runtime, billed)
                    except Exception as exc:
                        charge_error = exc
            except runtime.transient as exc:
                if attempt == self._num_retries:
                    raise
                delay = _RETRY_DELAY_S * 2**attempt
                logger.warning(
                    "Tinker sample failed (%s); retry %d/%d in %.0fs",
                    type(exc).__name__,
                    attempt + 1,
                    self._num_retries,
                    delay,
                )
                if cancel.wait(delay):
                    raise _Cancelled("generate_batch was interrupted") from exc
                continue
            return response, billed, charge_error
        raise AssertionError("unreachable")

    def _account(
        self, prompt_tokens: int, completion_tokens: int, cost_usd: float
    ) -> None:
        """Add one request to the instance totals, step report, and spend ledger."""
        units = self._calculate_token_units(prompt_tokens, completion_tokens)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        self._add_totals(usage, cost_usd, units)
        _tracking.add_usage(
            prompt_tokens, completion_tokens, cost_usd, units, model_id=self._model_id
        )

    def _cost(self, runtime: _Runtime, billed: dict[str, int]) -> float:
        if runtime.rates is None:
            if runtime.base_model not in _PRICE_WARNING_SEEN:
                _PRICE_WARNING_SEEN.add(runtime.base_model)
                warnings.warn(
                    f"No Tinker prices for {runtime.base_model}; pass prices= to "
                    "TinkerModel. total_cost undercounts.",
                    UserWarning,
                    stacklevel=3,
                )
            return 0.0
        return tinker_cost_usd(billed, runtime.rates)

    def _request_params(
        self, temperature: float, max_tokens: int, stop: list[str]
    ) -> dict[str, Any]:
        """What the request cache keys a completion on, besides its messages."""
        return {
            "model": self._model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "seed": self._seed,
            "stop": stop,
            "chat_template_kwargs": self._template_kwargs(),
        }

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
        """Generate `n` completions per prompt, one Tinker request per prompt.

        Request cache (as for `Model.generate_batch`, unless off): inside an
        @op, each sample is stored under a key of the op call, the prompt's
        row (`cache_rows`), the messages, the checkpoint (`model_id`), the
        sampling settings and chat-template kwargs, and `cache_salt`, plus a
        slot index. A rerun reuses cached samples and asks Tinker only for the
        missing ones, still in one request per prompt (one prefill). When every
        sample is cached, no Tinker client is opened.

        Args:
            prompts: List of prompts (string or OpenAI-format message list).
            n: Completions per prompt, sampled in one request (one prefill).
            temperature: Sampling temperature. Overrides the instance default.
            max_tokens: Max generated tokens per completion. Overrides the
                instance default; clamped to the context left after the prompt.
            stop: Optional stop strings, in addition to the model's eos tokens.
                Completions end before the first stop string.
            show_progress: Per-call override of the instance's show_progress
                (a bar in a terminal, else a log line every 30 seconds).
            return_exceptions: When True, a prompt whose request failed (after
                retries, or a budget refusal) gets the exception in its slot,
                and the other prompts keep their results. When False (default),
                the first failure in prompt order is raised.
            cache_salt: Part of the request-cache key (see `Model`).
            cache_rows: The row each prompt was built from (see `Model`).

        Returns:
            List of GenerateResult, one per prompt, in input order. Completions
            hold the decoded text, including any ``<think>...</think>`` block;
            `reasoning` is ``None`` per choice. `usage` has prompt, completion
            and cached prompt tokens, Tinker's billed token kinds, `cost_usd`,
            each sample's stop reason ("stop" or "length"), and
            ``cached_completions`` when the request cache supplied some.

        Raises:
            TinkerBudgetExceeded: A request would cross a configured dollar
                limit (the Tinker ledger or `budget`).
            BudgetExceededError: Spend reached ``max_cost_usd`` or
                ``FORGER_MAX_COST_USD``; checked before the batch and each
                request.

            Requests finished before a failure are charged to the ledgers, the
            instance totals, and the step report; `history` gets the results
            only when the call returns.
        """
        if not prompts:
            return []
        if n < 1:
            raise ValueError("n must be positive")
        if cache_rows is not None and len(cache_rows) != len(prompts):
            raise ValueError(
                f"cache_rows has {len(cache_rows)} entries for {len(prompts)} prompts"
            )
        _tracking.check_budget()  # raises BudgetExceededError at a spending cap
        temperature = temperature if temperature is not None else self._temperature
        if max_tokens is None:
            max_tokens = (
                self._max_tokens
                if self._max_tokens is not None
                else _DEFAULT_LOCAL_MAX_TOKENS
            )
        stop_strings = list(stop or [])

        # Request cache: look up every prompt's slots before opening a client.
        conversations = [_to_messages(prompt) for prompt in prompts]
        params = self._request_params(temperature, max_tokens, stop_strings)
        scope = _REQUEST_SCOPE.get()
        caching = scope is not None and _request_cache.enabled(self._request_cache)
        keys: list[str | None] = [None] * len(prompts)
        slot_ranges: list[range] = [range(n)] * len(prompts)
        hits: list[dict[int, dict[str, Any]]] = [{} for _ in prompts]
        if caching:
            assert scope is not None  # mypy hint
            read = not _NO_CACHE_SCOPE.get()
            for i, messages in enumerate(conversations):
                row = None if cache_rows is None else cache_rows[i]
                row_key = _hash(row)[0] if row is not None else scope.rows_key
                key = _request_cache.request_key(
                    scope.key, row_key, messages, params, cache_salt
                )
                first = scope.take_slots(key, n)
                keys[i], slot_ranges[i] = key, range(first, first + n)
                if read:
                    for slot in slot_ranges[i]:
                        entry = _request_cache.read_slot(key, slot)
                        if entry is not None:
                            hits[i][slot] = entry
            scope.count(sum(len(h) for h in hits), n * len(prompts))
        missing = [
            [s for s in slot_ranges[i] if s not in hits[i]] for i in range(len(prompts))
        ]

        responses: list[Any] = [None] * len(prompts)
        errors: dict[int, Exception] = {}
        jobs: dict[int, tuple[list[int], int, Any]] = {}
        runtime: _Runtime | None = None
        if any(missing):
            runtime = self._ensure_runtime()
            # SamplingParams takes either token ids or strings, not both.
            stop_param: list[Any] = (
                [runtime.tokenizer.decode([i]) for i in runtime.stop_ids] + stop_strings
                if stop_strings
                else list(runtime.stop_ids)
            )
            for i, prompt in enumerate(prompts):
                if not missing[i]:
                    continue
                # Render in this thread: HF tokenizers are not reliably thread-safe.
                rendered = self._render(runtime, prompt)
                # Keep prompt + completion inside the context window.
                limit = min(max_tokens, runtime.max_context - len(rendered))
                if limit < 1:
                    errors[i] = ValueError(
                        f"Prompt {i} has {len(rendered)} tokens; the context "
                        f"window is {runtime.max_context}"
                    )
                    continue
                sampling = runtime.types.SamplingParams(
                    max_tokens=limit,
                    temperature=temperature,
                    top_p=self._top_p,
                    top_k=self._top_k,
                    stop=stop_param,
                    seed=self._seed,
                )
                jobs[i] = (rendered, limit, sampling)

        cancel = threading.Event()

        def run(i: int) -> tuple[Any, dict[str, int]]:
            """Sample prompt i's missing slots; cache them as soon as they arrive."""
            assert runtime is not None  # mypy hint
            rendered, limit, sampling = jobs[i]
            response, billed, charge_error = self._sample(
                runtime, rendered, len(missing[i]), sampling, limit, cancel
            )
            key = keys[i]
            if key is not None:
                try:
                    for slot, sequence in zip(missing[i], response.sequences):
                        text = self._decode(runtime, sequence, stop_strings)
                        # Empty text is cached only when it ran out of tokens
                        # (see model._cache_entry).
                        if text or sequence.stop_reason == "length":
                            _request_cache.write_slot(
                                key,
                                slot,
                                {
                                    "text": text,
                                    "reasoning": None,
                                    "finish_reason": sequence.stop_reason,
                                },
                            )
                except Exception as exc:  # a full disk must not fail the request
                    logger.warning("request cache write failed: %s", exc)
            if charge_error is not None:
                raise charge_error  # after caching: the samples are paid for
            return response, billed

        progress = _progress_mode(
            self._show_progress if show_progress is None else show_progress
        )
        if jobs:
            executor = ThreadPoolExecutor(
                max_workers=min(self._max_concurrent, len(jobs))
            )
            try:
                # One context copy per request, so the worker's add_usage and
                # check_budget see the caller's op / step / pipeline.
                futures: dict[Future, int] = {}
                for i in jobs:
                    in_context: Callable[..., Any] = contextvars.copy_context().run
                    futures[executor.submit(in_context, run, i)] = i
                desc = f"Tinker samples ({self._model_id})"
                # The clock never sees a request start, so it warns about no
                # stragglers: Tinker requests have no per-attempt timeout.
                finished: Iterable[Future] = _watch(
                    futures,
                    len(futures),
                    desc,
                    len(futures) > 1 and progress == "log",
                    _RequestClock(desc, None),
                )
                if len(futures) > 1 and progress == "bar":
                    from tqdm import tqdm

                    finished = tqdm(
                        finished, total=len(futures), desc=desc, leave=False
                    )
                for future in finished:
                    i = futures[future]
                    try:
                        responses[i] = future.result()
                    except Exception as exc:
                        errors[i] = exc
            except BaseException:
                cancel.set()  # workers waiting for a slot, budget, or retry give up
                # Ctrl-C (or any error here): send no queued request; requests in
                # flight finish in the background and are still charged.
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            executor.shutdown()

        results: list[Any] = []
        for i in range(len(prompts)):
            # A budget refusal or bad request is not a short result to retry:
            # the prompt gets the exception even when some samples were cached.
            if i in errors and (not hits[i] or not _is_transient(errors[i])):
                results.append(errors[i])
                continue
            results.append(
                self._merge(
                    runtime,
                    slot_ranges[i],
                    hits[i],
                    missing[i],
                    responses[i],
                    jobs.get(i),
                    max_tokens,
                    stop_strings,
                )
            )
        if errors and not return_exceptions:
            raise errors[min(errors)]
        self._record(
            [r for r in results if isinstance(r, GenerateResult)], add_usage=False
        )
        _note_truncation(results, self._model_id, max_tokens)
        return results

    def _decode(self, runtime: _Runtime, sequence: Any, stop: list[str]) -> str:
        return _cut_at_stop(
            runtime.tokenizer.decode(sequence.tokens, skip_special_tokens=True), stop
        )

    def _merge(
        self,
        runtime: _Runtime | None,
        slots: range,
        hits: dict[int, dict[str, Any]],
        missing: list[int],
        outcome: tuple[Any, dict[str, int]] | None,
        job: tuple[list[int], int, Any] | None,
        max_tokens: int,
        stop: list[str],
    ) -> GenerateResult:
        """One prompt's GenerateResult: cached and sampled completions in slot order.

        Usage counts the fresh request only; a cached sample costs nothing.
        """
        by_slot = {
            slot: (entry["text"], entry.get("finish_reason"))
            for slot, entry in hits.items()
        }
        usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            # Thinking is inline in the text, so it is not counted apart.
            "reasoning_tokens": 0,
            "cached_prompt_tokens": 0,
            "billed_tokens": {kind: 0 for kind in _RATE_KINDS},
            "cost_usd": 0.0,
        }
        clamped = False
        if outcome is not None and job is not None and runtime is not None:
            response, billed = outcome
            rendered, limit, _ = job
            for slot, sequence in zip(missing, response.sequences):
                by_slot[slot] = (
                    self._decode(runtime, sequence, stop),
                    sequence.stop_reason,
                )
            usage["prompt_tokens"] = len(rendered)
            usage["completion_tokens"] = sum(len(s.tokens) for s in response.sequences)
            usage["cached_prompt_tokens"] = min(
                getattr(response, "prompt_cache_hit_tokens", 0) or 0, len(rendered)
            )
            usage["billed_tokens"] = billed
            usage["cost_usd"] = self._cost(runtime, billed)
            clamped = limit < max_tokens
        ordered = [by_slot[slot] for slot in slots if slot in by_slot]
        usage["stop_reasons"] = [reason for _, reason in ordered]
        usage["max_tokens_clamped"] = clamped
        usage["token_units"] = self._calculate_token_units(
            usage["prompt_tokens"], usage["completion_tokens"]
        )
        if hits:
            usage["cached_completions"] = len(hits)
        # raw is dropped: responses hold per-token arrays, which would pin
        # memory in self.history over a long sampling run.
        return GenerateResult(
            [text for text, _ in ordered],
            usage=usage,
            reasoning=[None] * len(ordered),
            model_id=self._model_id,
            raw=None,
            finish_reasons=[reason for _, reason in ordered],
        )


def _cut_at_stop(text: str, stop: list[str]) -> str:
    """End text before the first stop string, as vLLM does."""
    cuts = [index for index in (text.find(s) for s in stop if s) if index >= 0]
    return text[: min(cuts)] if cuts else text
