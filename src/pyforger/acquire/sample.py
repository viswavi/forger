"""LLM-based data generation building block.

``sample_from_LLM`` calls a model in batches, parsing each response,
until ``n`` accepted samples are collected (or retries exhausted).
"""
from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

from pyforger.acquire.model import Model, Prompt
from pyforger.op import op
from pyforger.utils import get_formatted_logger

logger = get_formatted_logger(__name__)


def _schema_format_instructions(output_schema: dict[str, Any]) -> str:
    """Build a short JSON-format instruction for the prompt."""
    keys = list(output_schema.keys())
    placeholders: dict[str, str] = {}
    for k, v in output_schema.items():
        if isinstance(v, type) and getattr(v, "__name__", None):
            placeholders[k] = f"<{v.__name__}>"
        elif isinstance(v, str):
            placeholders[k] = "<value>"
        else:
            placeholders[k] = "<value>"
    example_fmt = json.dumps(placeholders)
    return (
        f"Respond with a JSON object only, with no additional text. "
        f"Required keys: {keys}. Example format: {example_fmt}."
    )


def _build_prompt(
    prompt: Prompt,
    output_schema: dict[str, Any] | None = None,
) -> Prompt:
    if output_schema is None:
        return prompt
    schema_text = _schema_format_instructions(output_schema)
    if isinstance(prompt, str):
        return f"{prompt}\n\n{schema_text}"
    return [*prompt, {"role": "user", "content": schema_text}]


def _parse_output(
    raw_text: str,
    output_schema: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if output_schema is None:
        return {"text": raw_text}
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    required_keys = set(output_schema.keys())
    if not required_keys.issubset(parsed.keys()):
        return None
    return parsed


def _generation_loop(
    prompt: Prompt,
    model: Model,
    n: int,
    output_schema: dict[str, Any] | None,
    temperature: float,
    max_retries: int,
    batch_size: int,
) -> Generator[dict[str, Any], None, None]:
    """Yield accepted samples until ``n`` collected or retries exhausted."""
    accepted: list[dict[str, Any]] = []
    consecutive_failed_rounds = 0

    while len(accepted) < n and consecutive_failed_rounds < max_retries:
        remaining = n - len(accepted)
        current_batch_size = min(batch_size, remaining)
        full_prompt = _build_prompt(prompt, output_schema)

        raw_responses = model.generate(
            full_prompt, n=current_batch_size, temperature=temperature
        )

        short_response = len(raw_responses) < current_batch_size
        if short_response:
            logger.warning(
                "generate returned %d responses for %d requested",
                len(raw_responses),
                current_batch_size,
            )

        round_accepts = 0
        for raw_text in raw_responses:
            parsed = _parse_output(raw_text, output_schema)
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


@op(name="acquire.sample_from_LLM")
def sample_from_LLM(
    prompt: Prompt,
    model: Model,
    n: int = 100,
    output_schema: dict[str, Any] | None = None,
    temperature: float = 0.7,
    max_retries: int = 3,
    batch_size: int = 10,
    name: str = "sample_from_LLM",
) -> list[dict]:
    """Generate from `model` until `n` samples are generated.

    Args:
        prompt: A string or chat-message list (OpenAI format).
        model: Model to call.
        n: Target number of accepted samples.
        output_schema: Optional dict of expected keys; triggers JSON parse + validation.
        temperature: Sampling temperature.
        max_retries: Consecutive zero-acceptance rounds before stopping.
        batch_size: Number of responses to generate at a time.
        name: A string used to distinguish cache keys for otherwise identical calls.

    Returns:
        List of accepted sample dicts. ``{"text": <raw>}`` when no schema;
        otherwise the parsed JSON objects.

    Note:
        Because this is decorated with `@op`, outputs are memoized on disk by argument
        hash. Therefore, repeat calls with a non-zero temperature will still produce
        the same samples if all other arguments are the same. If this is not desired,
        force fresh generation by passing `_no_cache=True` or varying `name`.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n <= 0:
        raise ValueError("n must be positive")
    return list(
        _generation_loop(
            prompt=prompt,
            model=model,
            n=n,
            output_schema=output_schema,
            temperature=temperature,
            max_retries=max_retries,
            batch_size=batch_size,
        )
    )
