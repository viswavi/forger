"""LLM model wrapper for Forger.

Provides a simplified interface to LiteLLM for generating text
completions from any supported LLM provider.
"""

from __future__ import annotations

import warnings
from typing import Any, Union

import litellm

# Type aliases
Message = dict[str, str]
Prompt = Union[str, list[Message]]


class GenerateResult(list):
    """list[str] with ``usage`` / ``reasoning`` / ``model_id`` / ``raw`` attributes."""

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
    "o1": {"temperature": 1.0, "n": 1},
    "o3": {"temperature": 1.0, "n": 1},
    "o4": {"temperature": 1.0, "n": 1},
    "gpt-5": {"temperature": 1.0, "n": 1},
    "gemini-3": {"temperature": 1.0, "n": 1},
}

_PROVIDER_PREFIXES = ("openai/", "anthropic/", "gemini/", "xai/", "deepseek/")


def _family(model_id: str) -> str:
    """Strip ``litellm_proxy/`` / ``openrouter/`` gateway + provider prefix."""
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


def _validate_params(model_id: str, **params: Any) -> None:
    """Raise ``ValueError`` on unsupported params or constraint violations."""
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
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        required.setdefault("temperature", 1.0)

    for name, required_value in required.items():
        got = params.get(name)
        if got is not None and got != required_value:
            raise ValueError(
                f"{model_id} requires {name}={required_value} (got {got}). "
                f"Pass {name}={required_value} or omit the parameter."
            )


def _extract_reasoning(response: Any, model_id: str) -> list[str | None]:
    """Extract reasoning per choice; ``None`` when absent.

    Skips ``litellm.supports_reasoning`` (false negatives on proxy aliases).
    OpenRouter uses ``message.reasoning`` instead of ``reasoning_content``.
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
    """Cost in USD; ``0.0`` when unavailable. Warns once per ``model_id`` on miss."""
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
    """Dict with ``prompt_tokens`` / ``completion_tokens`` / ``reasoning_tokens``."""
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
            **kwargs: Passed through to litellm.completion()
                (e.g. api_base, timeout, organization).
        """
        self._model_id = model_id
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_retries = num_retries
        self._kwargs = kwargs
        self.history: list[GenerateResult] = []
        _validate_params(
            model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=kwargs.get("thinking"),
        )

    def __repr__(self) -> str:
        """Show model id for debugging."""
        return f"Model({self._model_id!r})"

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
    ) -> list[GenerateResult]:
        """Generate n responses for each prompt in a batch.

        Args:
            prompts: List of prompts. Each prompt is either a string or
                a list of chat messages in OpenAI format.
            n: Number of completions per prompt.
            temperature: Sampling temperature. Overrides the instance default.
            max_tokens: Max tokens. Overrides the instance default.
            stop: Stop sequences.

        Returns:
            List of GenerateResult, one per prompt. Each GenerateResult
            is a list of n completion strings carrying usage / reasoning
            / model_id / raw as attributes.
        """
        if not prompts:
            return []

        temperature = temperature if temperature is not None else self._temperature
        max_tokens = max_tokens if max_tokens is not None else self._max_tokens

        validate_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
            "thinking": self._kwargs.get("thinking"),
        }
        if n != 1:
            validate_kwargs["n"] = n
        _validate_params(self._model_id, **validate_kwargs)

        all_messages = [_to_messages(prompt) for prompt in prompts]
        call_kwargs = dict(
            model=self._model_id,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            api_key=self._api_key,
            num_retries=self._num_retries,
            **self._kwargs,
        )
        if len(all_messages) == 1:
            responses = [litellm.completion(messages=all_messages[0], **call_kwargs)]
        else:
            responses = litellm.batch_completion(messages=all_messages, **call_kwargs)

        results: list[GenerateResult] = []
        for response in responses:
            if isinstance(response, BaseException):
                # litellm.batch_completion returns the exception itself (rather
                # than raising) for a prompt that failed after retries -- e.g. a
                # provider-side RateLimitError on one item shouldn't crash every
                # other, otherwise-successful prompt in the same batch. Surface
                # it as an empty/failed result (same shape callers already treat
                # a parse failure as) instead of an AttributeError deep in
                # unrelated code that tries to read .choices off it.
                result = GenerateResult(
                    [""] * n,
                    usage={"cost_usd": 0.0, "error": str(response)},
                    reasoning=[None] * n,
                    model_id=self._model_id,
                    raw=response,
                )
                results.append(result)
                self.history.append(result)
                continue
            completions = [choice.message.content or "" for choice in response.choices]
            usage: dict[str, Any] = dict(_extract_usage(response))
            usage["cost_usd"] = _extract_cost(response, self._model_id)
            reasoning = _extract_reasoning(response, self._model_id)
            result = GenerateResult(
                completions,
                usage=usage,
                reasoning=reasoning,
                model_id=self._model_id,
                raw=response,
            )
            results.append(result)
            self.history.append(result)

        return results


# Default cap on generated tokens for a local reasoning model. Reasoning traces
# can be long (tens of thousands of tokens), so the LiteLLM default of "let the
# provider decide" is replaced here by an explicit, generous ceiling.
_DEFAULT_LOCAL_MAX_TOKENS = 32768


class LocalModel(Model):
    """On-GPU model served locally via vLLM, with the same interface as ``Model``.

    Unlike the LiteLLM-backed :class:`Model`, this loads model weights into GPU
    memory once (lazily, on the first ``generate``/``generate_batch`` call) and
    runs batched inference in-process. This is the efficient path for sampling
    many completions per prompt from an open-weights student model (e.g. for
    on-policy distillation).

    The engine is created lazily so that importing ``pyforger`` -- and even
    constructing a ``LocalModel`` -- never requires ``vllm`` or a GPU. ``vllm``
    is imported only when generation actually happens.

    Chat prompts are rendered with the model's own chat template. For reasoning
    models (e.g. Qwen3) the generated text contains a ``<think>...</think>``
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
                ``chat_template_kwargs={"enable_thinking": ...}`` unless
                ``chat_template_kwargs`` is given explicitly.
            chat_template_kwargs: Extra kwargs for the chat template. When set,
                it is used verbatim and ``enable_thinking`` is ignored.
            seed: Sampling seed for reproducible generation.
            show_progress: When True, display vLLM's per-call progress bar (with
                live input/output tokens-per-second) during generation. Useful
                for long batched runs; off by default to keep logs quiet.
            **engine_kwargs: Extra keyword arguments forwarded to ``vllm.LLM``
                (e.g. ``max_num_seqs`` to cap concurrent sequences and avoid KV
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

        Includes the sampling/template settings that affect outputs so that, for
        example, two ``LocalModel`` instances with different temperatures or
        ``enable_thinking`` do not collide in the disk cache.
        """
        return (
            type(self).__name__,
            self._model_id,
            self._temperature,
            self._max_tokens,
            self._max_model_len,
            self._enable_thinking,
            tuple(sorted((self._chat_template_kwargs or {}).items())),
            self._seed,
        )

    def _engine(self) -> Any:
        """Lazily construct and cache the vLLM engine."""
        if self._llm is None:
            from vllm import LLM

            self._llm = LLM(
                model=self._model_id,
                gpu_memory_utilization=self._gpu_memory_utilization,
                tensor_parallel_size=self._tensor_parallel_size,
                dtype=self._dtype,
                seed=self._seed,
                **(
                    {"max_model_len": self._max_model_len}
                    if self._max_model_len is not None
                    else {}
                ),
                **self._kwargs,
            )
        return self._llm

    def _template_kwargs(self) -> dict[str, Any]:
        """Resolve chat-template kwargs from explicit override or enable_thinking."""
        if self._chat_template_kwargs is not None:
            return self._chat_template_kwargs
        return {"enable_thinking": self._enable_thinking}

    # Returns plain lists, not GenerateResult -- deliberately narrower than Model.
    def generate_batch(  # type: ignore[override]
        self,
        prompts: list[Prompt],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        """Generate ``n`` completions per prompt with the local vLLM engine.

        Args:
            prompts: List of prompts (string or OpenAI-format message list).
            n: Number of completions per prompt.
            temperature: Sampling temperature. Overrides the instance default.
            max_tokens: Max generated tokens per completion. Overrides the
                instance default (which itself defaults to a generous ceiling).
            stop: Optional stop strings.

        Returns:
            List of lists -- one inner list of ``n`` completion strings per
            prompt, in input order. Each completion is the raw generated text
            (including any ``<think>...</think>`` block for reasoning models).
        """
        from vllm import SamplingParams

        temperature = temperature if temperature is not None else self._temperature
        if max_tokens is None:
            max_tokens = (
                self._max_tokens
                if self._max_tokens is not None
                else _DEFAULT_LOCAL_MAX_TOKENS
            )

        self._last_usage = None
        if not prompts:
            self._last_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost": 0.0,
            }
            return []

        conversations = [_to_messages(prompt) for prompt in prompts]
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=self._seed,
        )
        outputs = self._engine().chat(
            conversations,
            sampling_params,
            chat_template_kwargs=self._template_kwargs(),
            use_tqdm=self._show_progress,
        )

        all_results: list[list[str]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for output in outputs:
            completions = [completion.text for completion in output.outputs]
            all_results.append(completions)
            total_prompt_tokens += len(output.prompt_token_ids or [])
            total_completion_tokens += sum(
                len(completion.token_ids) for completion in output.outputs
            )

        # Local inference has no per-call monetary cost; cost is fixed (GPU time).
        self._last_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_cost": 0.0,
        }
        return all_results


# Marks the boundary between a reasoning model's thinking block and its answer.
_THINK_CLOSE = "</think>"
_THINK_OPEN = "<think>"


def split_reasoning(text: str) -> tuple[str, str]:
    """Split a reasoning-model completion into (thinking, answer).

    Reasoning models such as Qwen3 emit ``<think>...</think>`` followed by the
    final answer. This splits on the closing ``</think>`` tag, stripping an
    optional leading ``<think>``. When no closing tag is present (e.g. the model
    ran out of tokens mid-thought, or thinking was disabled), the whole text is
    returned as the answer with an empty thinking string.

    Args:
        text: Raw completion text.

    Returns:
        ``(thinking, answer)`` with surrounding whitespace stripped.
    """
    if _THINK_CLOSE not in text:
        return "", text.strip()
    thinking, _, answer = text.partition(_THINK_CLOSE)
    thinking = thinking.strip()
    if thinking.startswith(_THINK_OPEN):
        thinking = thinking[len(_THINK_OPEN) :].strip()
    return thinking, answer.strip()
