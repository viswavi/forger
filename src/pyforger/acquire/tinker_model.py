"""Sample a base model or a trained ``tinker://`` checkpoint remotely on Tinker.

Tinker samples token ids, so prompts are rendered with the model's own HF chat
template and completions decoded with ``skip_special_tokens=True``, exactly as
PostDataBench's evaluator (``coding/tinker_sampling.py``) does. Rendering is
duplicated here rather than shared because the evaluator runs in environments
without pyforger.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from pyforger.acquire.model import (
    _DEFAULT_LOCAL_MAX_TOKENS,
    GenerateResult,
    Model,
    Prompt,
    _to_messages,
)
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

# Process-wide caps on in-flight sample requests, one semaphore per limit, so
# several TinkerModels (or threads) in one process share the same cap.
_SLOTS: dict[int, threading.BoundedSemaphore] = {}
_SLOTS_LOCK = threading.Lock()
_PRICE_WARNING_SEEN: set[str] = set()


def _sample_slots(limit: int) -> threading.BoundedSemaphore:
    with _SLOTS_LOCK:
        if limit not in _SLOTS:
            _SLOTS[limit] = threading.BoundedSemaphore(limit)
        return _SLOTS[limit]


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
        show_progress: bool = False,
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
            show_progress: When True, multi-prompt calls show a tqdm bar.
        """
        super().__init__(
            model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=num_retries,
            show_progress=show_progress,
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
        state["_runtime"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state with fresh locks."""
        super().__setstate__(state)
        self._client_lock = threading.Lock()

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

    def _check_params(self, **params: Any) -> None:
        """No-op: Tinker model ids are not in LiteLLM's parameter registry."""

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
    def _budgeted(self, runtime: _Runtime, estimate: dict[str, int]):
        """Check budgets for a request's worst case; hold it on the ledger meanwhile."""
        if runtime.ledger is None:
            reservation: Any = contextlib.nullcontext()
        else:
            reservation = runtime.ledger.reserve(runtime.ledger.cost(estimate))
        with reservation:
            if self._budget is not None:
                assert runtime.rates is not None
                self._budget.check(tinker_cost_usd(estimate, runtime.rates))
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
    ) -> tuple[Any, dict[str, int]]:
        """One request for n samples: check budgets, sample, charge billed tokens."""
        estimate = sample_tokens(len(prompt), n, 0, [max_tokens] * n)
        slots = _sample_slots(self._max_concurrent)
        for attempt in range(self._num_retries + 1):
            try:
                # Checked once a slot is free, before every attempt: neither a
                # queued request nor a retry may start past the limit. The
                # worst case stays reserved on the ledger until it is charged.
                with slots, self._budgeted(runtime, estimate):
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
                    self._charge(runtime, billed)
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
                time.sleep(delay)
                continue
            return response, billed
        raise AssertionError("unreachable")

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

    def generate_batch(
        self,
        prompts: list[Prompt],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        show_progress: bool | None = None,
    ) -> list[GenerateResult]:
        """Generate `n` completions per prompt, one Tinker request per prompt.

        Args:
            prompts: List of prompts (string or OpenAI-format message list).
            n: Completions per prompt, sampled in one request (one prefill).
            temperature: Sampling temperature. Overrides the instance default.
            max_tokens: Max generated tokens per completion. Overrides the
                instance default; clamped to the context left after the prompt.
            stop: Optional stop strings, in addition to the model's eos tokens.
                Completions end before the first stop string.
            show_progress: Per-call override of the instance's show_progress.

        Returns:
            List of GenerateResult, one per prompt, in input order. Completions
            hold the decoded text, including any ``<think>...</think>`` block;
            `reasoning` is ``None`` per choice. `usage` has prompt, completion
            and cached prompt tokens, Tinker's billed token kinds, `cost_usd`,
            and each sample's stop reason ("stop" or "length").

        Raises:
            TinkerBudgetExceeded: A request would cross a configured dollar
                limit. Results finished before it are still recorded in
                `history` and charged to the ledger.
        """
        if not prompts:
            return []
        if n < 1:
            raise ValueError("n must be positive")
        runtime = self._ensure_runtime()
        temperature = temperature if temperature is not None else self._temperature
        if max_tokens is None:
            max_tokens = (
                self._max_tokens
                if self._max_tokens is not None
                else _DEFAULT_LOCAL_MAX_TOKENS
            )
        stop_strings = list(stop or [])
        # SamplingParams takes either token ids or strings, not both.
        stop_param: list[Any] = (
            [runtime.tokenizer.decode([i]) for i in runtime.stop_ids] + stop_strings
            if stop_strings
            else list(runtime.stop_ids)
        )
        # Render in this thread: HF tokenizers are not reliably thread-safe.
        rendered = [self._render(runtime, prompt) for prompt in prompts]

        responses: list[Any] = [None] * len(prompts)
        errors: dict[int, Exception] = {}
        jobs: dict[int, tuple[list[int], int, Any]] = {}
        for i, prompt in enumerate(rendered):
            # Keep prompt + completion inside the context window (vLLM rejects it).
            limit = min(max_tokens, runtime.max_context - len(prompt))
            if limit < 1:
                errors[i] = ValueError(
                    f"Prompt {i} has {len(prompt)} tokens; the context window "
                    f"is {runtime.max_context}"
                )
                continue
            params = runtime.types.SamplingParams(
                max_tokens=limit,
                temperature=temperature,
                top_p=self._top_p,
                top_k=self._top_k,
                stop=stop_param,
                seed=self._seed,
            )
            jobs[i] = (prompt, limit, params)

        from tqdm import tqdm

        use_progress = self._show_progress if show_progress is None else show_progress
        if jobs:
            with ThreadPoolExecutor(
                max_workers=min(self._max_concurrent, len(jobs))
            ) as executor:
                futures = {
                    executor.submit(self._sample, runtime, prompt, n, params, limit): i
                    for i, (prompt, limit, params) in jobs.items()
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Tinker samples ({self._model_id})",
                    leave=False,
                    disable=not (use_progress and len(futures) > 1),
                ):
                    i = futures[future]
                    try:
                        responses[i] = future.result()
                    except Exception as exc:
                        errors[i] = exc

        results: list[GenerateResult] = []
        for i, outcome in enumerate(responses):
            if outcome is None:
                continue
            response, billed = outcome
            prompt, limit, _ = jobs[i]
            completions = [
                _cut_at_stop(
                    runtime.tokenizer.decode(sequence.tokens, skip_special_tokens=True),
                    stop_strings,
                )
                for sequence in response.sequences
            ]
            completion_tokens = sum(len(seq.tokens) for seq in response.sequences)
            usage: dict[str, Any] = {
                "prompt_tokens": len(prompt),
                "completion_tokens": completion_tokens,
                # Thinking is inline in the text, so it is not counted apart.
                "reasoning_tokens": 0,
                "cached_prompt_tokens": min(
                    getattr(response, "prompt_cache_hit_tokens", 0) or 0, len(prompt)
                ),
                "billed_tokens": billed,
                "cost_usd": self._cost(runtime, billed),
                "stop_reasons": [seq.stop_reason for seq in response.sequences],
                "max_tokens_clamped": limit < max_tokens,
            }
            usage["token_units"] = self._calculate_token_units(
                len(prompt), completion_tokens
            )
            # raw is dropped: responses hold per-token arrays, which would pin
            # memory in self.history over a long sampling run.
            result = GenerateResult(
                completions,
                usage=usage,
                reasoning=[None] * len(completions),
                model_id=self._model_id,
                raw=None,
            )
            self._record_result(result)
            results.append(result)
        if errors:
            raise errors[min(errors)]
        return results


def _cut_at_stop(text: str, stop: list[str]) -> str:
    """End text before the first stop string, as vLLM does."""
    cuts = [index for index in (text.find(s) for s in stop if s) if index >= 0]
    return text[: min(cuts)] if cuts else text
