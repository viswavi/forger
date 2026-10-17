"""LLM-based data generation building block.

``sample_from_LLM`` calls a model in batches, parsing each response,
until ``n`` accepted samples are collected (or retries exhausted).
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any

from pyforger.acquire._model_utils import (
    _accepts_kwarg,
    _build_prompt,
    _is_transient,
    _parse_output,
    _reasoning_for,
    _report_provider_failures,
)
from pyforger.acquire.model import Model, Prompt, _cache_kwargs
from pyforger.op import op
from pyforger.utils import get_formatted_logger

logger = get_formatted_logger(__name__)


def _generation_loop(
    prompt: Prompt,
    model: Model,
    n: int,
    output_schema: dict[str, Any] | None,
    temperature: float | None,
    max_retries: int,
    batch_size: int | None,
    errors: list[str] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Yield accepted samples until ``n`` collected or retries exhausted.

    A round whose request fails at the provider with a transient error (e.g. a
    504) accepts nothing and counts toward ``max_retries``; its error type name
    is appended to ``errors``. Other errors (a bad key, a 400) raise.
    """
    # Only Model.generate_batch can hand back an error instead of raising it;
    # duck-typed fakes and models with their own generate keep calling generate.
    generate = getattr(model, "generate", None)
    catch = getattr(generate, "__func__", None) is Model.generate and (
        _accepts_kwarg(getattr(model, "generate_batch", None), "return_exceptions")
    )
    accepted: list[dict[str, Any]] = []
    consecutive_failed_rounds = 0
    round_index = 0

    while len(accepted) < n and consecutive_failed_rounds < max_retries:
        remaining = n - len(accepted)
        current_batch_size = (
            remaining if batch_size is None else min(batch_size, remaining)
        )
        full_prompt = _build_prompt(prompt, output_schema)

        # Each round re-asks the same prompt, so it salts the request cache to
        # get new completions instead of the earlier rounds' (e.g. "round 1").
        raw_responses: Any
        if catch:
            raw_responses = model.generate_batch(
                [full_prompt],
                n=current_batch_size,
                temperature=temperature,
                return_exceptions=True,
                **_cache_kwargs(
                    model.generate_batch,
                    cache_salt=f"round {round_index}" if round_index else None,
                ),
            )[0]
        else:
            raw_responses = model.generate(
                full_prompt,
                n=current_batch_size,
                temperature=temperature,
                **_cache_kwargs(
                    model.generate,
                    cache_salt=f"round {round_index}" if round_index else None,
                ),
            )
        round_index += 1

        if isinstance(raw_responses, BaseException):
            if not _is_transient(raw_responses):
                raise raw_responses
            logger.warning("sample_from_LLM request failed: %s", raw_responses)
            if errors is not None:
                errors.append(type(raw_responses).__name__)
            raw_responses = []
        elif len(raw_responses) < current_batch_size:
            logger.warning(
                "generate returned %d responses for %d requested",
                len(raw_responses),
                current_batch_size,
            )
            if catch and errors is not None:
                # Some of the round's requests failed (n fan-out or n > 16).
                errors.append("missing completions")

        round_accepts = 0
        for raw_text, reasoning_text in zip(
            raw_responses, _reasoning_for(raw_responses)
        ):
            parsed = _parse_output(raw_text, output_schema, reasoning=reasoning_text)
            if parsed is None:
                logger.debug(
                    "Parse failed for output (schema=%s)",
                    output_schema is not None,
                )
                continue
            accepted.append(parsed)
            round_accepts += 1
            yield parsed
            if len(accepted) >= n:
                return

        # Only count completely failed rounds (zero accepts) towards max_retries.
        # Partial successes are treated as successes.
        if round_accepts == 0:
            consecutive_failed_rounds += 1
        else:
            consecutive_failed_rounds = 0


@op(name="acquire.sample_from_LLM", samples_arg="n")
def sample_from_LLM(
    prompt: Prompt,
    model: Model,
    n: int = 100,
    output_schema: dict[str, Any] | None = None,
    temperature: float | None = None,
    max_retries: int = 3,
    batch_size: int | None = None,
    name: str = "sample_from_LLM",
) -> list[dict]:
    """Generate from `model` until `n` samples are generated.

    Args:
        prompt: A string or chat-message list (OpenAI format).
        model: Model to call.
        n: Target number of accepted samples.
        output_schema: Optional dict of expected keys; triggers JSON parse + validation.
        temperature: Sampling temperature; None (default) uses the model's own
            default (1.0 for ``Model``, Qwen3's 0.6 for ``TinkerModel``).
        max_retries: Consecutive zero-acceptance rounds before stopping. A round
            whose request fails at the provider (e.g. a 504) counts as one; when
            failures leave fewer than ``n`` samples, the shorter list is
            returned and not cached.
        batch_size: Number of responses to request per round (one
            ``model.generate`` call with ``n=batch_size``). None (default) requests
            all that are still missing at once.
        name: A string used to distinguish cache keys for otherwise identical calls.

    Returns:
        List of accepted sample dicts. ``{"text": <raw>}`` when no schema;
        otherwise the parsed JSON objects. When the model returns a reasoning
        trace separately in `GenerateResult.reasoning`, it is included under
        the `thinking_trace` key. Inline `<think>` traces (e.g. from `LocalModel`,
        which supplies `None` for that metadata) are not extracted; use
        `split_reasoning` to separate them out as needed.

    Note:
        Because this is decorated with `@op`, outputs are memoized on disk by argument
        hash. Therefore, repeat calls with a non-zero temperature will still produce
        the same samples if all other arguments are the same. If this is not desired,
        force fresh generation by passing `_no_cache=True` or varying `name`. The
        model's request cache follows the same rule: a call that differs in any
        argument but `n` draws new samples, and raising `n` keeps the samples
        already drawn and requests only the extra ones.
    """
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n <= 0:
        raise ValueError("n must be positive")
    errors: list[str] = []
    samples = list(
        _generation_loop(
            prompt=prompt,
            model=model,
            n=n,
            output_schema=output_schema,
            temperature=temperature,
            max_retries=max_retries,
            batch_size=batch_size,
            errors=errors,
        )
    )
    # Short because requests failed: not cached, so a rerun asks again (the
    # request cache replays the completions that arrived).
    if errors and len(samples) < n:
        _report_provider_failures(
            f"acquire.sample_from_LLM: {len(samples)} of {n} samples; "
            f"{len(errors)} request(s) failed",
            errors,
        )
    return samples
