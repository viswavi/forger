"""LLM-based data generation building block.

sample_from_LLM is a first-class acquire building block that generates data from
an LLM until n accepted samples are collected, with optional structured output
parsing.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import Any

from pyforger.acquire.model import Model, Prompt
from pyforger.core import DataSpec, NodeType


def _get_logger() -> logging.Logger:
    """Lazy logger to avoid circular import with utils."""
    from pyforger.utils import get_formatted_logger as _gf

    return _gf(__name__)


def _schema_format_instructions(output_schema: dict[str, Any]) -> str:
    """Turn output_schema into a short instruction string for prompt injection.

    Instructs the model to produce JSON with the required keys. Used both to
    tell the model what to emit (prompt) and aligned with _parse_output validation.

    Args:
        output_schema: Dict of key -> type or description (e.g. {"answer": str}).

    Returns:
        A string to append to the prompt, e.g. "Respond with a JSON object only..."
    """
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
    """Combine base prompt with optional schema instructions.

    Args:
        prompt: A string or list of chat messages in OpenAI format.
        output_schema: If set, format instructions are appended so the model
            produces JSON matching the schema (same schema used by _parse_output).

    Returns:
        For string prompts: string with schema instructions appended.
        For message lists: copy with an extra user message containing
            schema instructions.
    """
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
    """Parse raw model output into a dict; validate against schema if provided.

    If output_schema is None, returns {"text": raw_text}. Otherwise attempts
    JSON parse and lightweight validation (required keys present).

    Args:
        raw_text: Raw string from the model.
        output_schema: Optional dict describing expected keys (e.g. {"answer": str}).
            Used only to check presence of keys in parsed JSON.

    Returns:
        Parsed dict on success, None on parse or validation failure.
    """
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


def _build_batch(
    prompt: Prompt,
    batch_size: int,
    output_schema: dict[str, Any] | None = None,
) -> list[Prompt]:
    """Prepare a batch of prompts.

    When output_schema is set, schema format instructions are injected into
    the prompt once and replicated across the batch.

    Args:
        prompt: Base prompt (string or message list).
        batch_size: Number of prompts to build.
        output_schema: Optional schema; when set, format instructions are appended.

    Returns:
        List of prompts ready for model.generate_batch().
    """
    full_prompt = _build_prompt(prompt, output_schema)
    return [full_prompt for _ in range(batch_size)]


def _generation_loop(
    prompt: Prompt,
    model: Model,
    n: int,
    output_schema: dict[str, Any] | None,
    temperature: float,
    max_retries: int,
    batch_size: int,
) -> Generator[dict[str, Any], None, None]:
    """Yield accepted samples until *n* collected or retries exhausted.

    Works in batched rounds: each round builds a batch of prompts, calls
    model.generate_batch(), then parses each result and accepts on success.
    consecutive_failed_rounds counts rounds where zero samples were accepted.
    """
    accepted: list[dict[str, Any]] = []
    consecutive_failed_rounds = 0
    log = _get_logger()

    while len(accepted) < n and consecutive_failed_rounds < max_retries:
        remaining = n - len(accepted)
        current_batch_size = min(batch_size, remaining)
        prompts_list = _build_batch(
            prompt,
            current_batch_size,
            output_schema=output_schema,
        )

        batch_results = model.generate_batch(
            prompts_list,
            temperature=temperature,
        )
        raw_responses = [r[0] for r in batch_results]

        if len(raw_responses) != len(prompts_list):
            log.warning(
                "generate_batch returned %d responses for %d prompts; skipping round",
                len(raw_responses),
                len(prompts_list),
            )
            consecutive_failed_rounds += 1
            continue

        round_accepts = 0
        for raw_text in raw_responses:
            parsed = _parse_output(raw_text, output_schema)
            if parsed is None:
                log.debug(
                    "Parse failed for output (schema=%s)",
                    output_schema is not None,
                )
                continue
            accepted.append(parsed)
            round_accepts += 1
            yield parsed
            if len(accepted) >= n:
                return

        if round_accepts == 0:
            consecutive_failed_rounds += 1
        else:
            consecutive_failed_rounds = 0


def sample_from_LLM(
    prompt: Prompt,
    model: Model,
    n: int = 100,
    output_schema: dict[str, Any] | None = None,
    temperature: float = 0.7,
    max_retries: int = 3,
    batch_size: int = 10,
    name: str = "sample_from_LLM",
) -> DataSpec:
    """Build an ACQUIRE node that generates data from an LLM until n accepted samples.

    This is a first-class building block: no inputs, no samples_per_input.
    Generation runs only when the DAG runner evaluates the node (lazy execution).

    Args:
        prompt: A string or list of chat messages in OpenAI format.
            String prompts are wrapped as a single user message by Model.
            Message lists support system prompts, few-shot examples, etc.
        model: Model instance (placeholder or real backend).
        n: Target number of accepted samples.
        output_schema: Optional dict of expected keys for structured output;
            triggers JSON parse and key validation. For string prompts,
            schema instructions are appended to the text. For message
            lists, an extra user message is added.
        temperature: Sampling temperature passed to the model.
        max_retries: Consecutive rounds with zero acceptances before stopping.
        batch_size: Prompts per round sent to model.generate_batch().
        name: Label for the DataSpec node.

    Returns:
        DataSpec whose execute_fn is a generator yielding accepted samples.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n <= 0:
        raise ValueError("n must be positive")

    def _execute() -> list[dict]:
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

    return DataSpec(NodeType.ACQUIRE, [], _execute, name)
