"""LLM-based data generation building block.

sample_from_LLM is a first-class acquire building block that generates data from
an LLM until n accepted samples are collected, with optional few-shot prompting,
structured output parsing.
"""
from __future__ import annotations

import json
import logging
import random
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
    prompt: str,
    examples: list[dict[str, Any]],
    output_schema: dict[str, Any] | None = None,
) -> str:
    """Combine base prompt with optional schema instructions and few-shot examples.

    Args:
        prompt: The base instruction or prompt template.
        examples: List of example dicts; each is formatted and appended.
        output_schema: If set, format instructions are appended so the model
            produces JSON matching the schema (same schema used by _parse_output).

    Returns:
        Single string: prompt, then schema instructions (if any), then examples.
    """
    parts = [prompt]
    if output_schema is not None:
        parts.append(_schema_format_instructions(output_schema))
    for ex in examples:
        if isinstance(ex, dict) and "text" in ex:
            parts.append(str(ex["text"]))
        else:
            parts.append(str(ex))
    return "\n\n".join(parts)


def _sample_from_pool(
    pool: list[dict[str, Any]] | None,
    k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Randomly pick k examples from the pool without replacement.

    Args:
        pool: List of example dicts; may be None.
        k: Number to sample.
        rng: Random instance for reproducibility.

    Returns:
        List of up to k examples, or [] if pool is None or k is 0.
    """
    if not pool or k <= 0:
        return []
    if k >= len(pool):
        return list(rng.sample(pool, len(pool)))
    return list(rng.sample(pool, k))


def _parse_output(
    raw_text: str,
    output_schema: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Parse raw model output into a dict; validate against schema if provided.

    If output_schema is None, returns {"text": raw_text}. Otherwise attempts
    JSON parse and lightweight validation (required keys present).
    TODO: Full JSON Schema validation in future.

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


def _per_call_seed(base_seed: int | None, prompt_idx: int) -> int | None:
    """Compute a unique seed for a single generation request.

    Ensures cache keys differ per request when Model adds caching.
    """
    if base_seed is not None:
        return hash((base_seed, prompt_idx))
    return prompt_idx


def _build_batch(
    prompt: str,
    pool: list[dict[str, Any]] | None,
    pool_sample_k: int,
    batch_size: int,
    prompt_idx_start: int,
    seed: int | None,
    rng: random.Random,
    output_schema: dict[str, Any] | None = None,
) -> tuple[list[Prompt], list[int | None], int]:
    """Prepare a batch of prompt strings and their per-call seeds.

    Each prompt in the batch may have different few-shot examples (re-sampled
    from pool per slot). When output_schema is set, schema format instructions
    are injected into every prompt.

    Args:
        prompt: Base prompt text.
        pool: Optional list of example dicts for few-shot.
        pool_sample_k: Number of examples to sample per prompt in the batch.
        batch_size: Number of prompts to build.
        prompt_idx_start: Starting index for seed computation.
        seed: Base seed; combined with index for per-call seed.
        rng: Random instance for pool sampling.
        output_schema: Optional schema; when set, format instructions are appended.

    Returns:
        (prompts_list, seeds_list, next_prompt_idx).
    """
    prompts_list: list[Prompt] = []
    seeds_list: list[int | None] = []
    for i in range(batch_size):
        idx = prompt_idx_start + i
        examples = _sample_from_pool(pool, pool_sample_k, rng)
        full_prompt = _build_prompt(prompt, examples, output_schema)
        prompts_list.append(full_prompt)
        seeds_list.append(_per_call_seed(seed, idx))
    return prompts_list, seeds_list, prompt_idx_start + batch_size


def _generation_loop(
    prompt: str,
    model: Model,
    n: int,
    output_schema: dict[str, Any] | None,
    pool: list[dict[str, Any]] | None,
    pool_sample_k: int,
    temperature: float,
    max_retries: int,
    seed: int | None,
    batch_size: int,
    rng: random.Random,
) -> Generator[dict[str, Any], None, None]:
    """Yield accepted samples until *n* collected or retries exhausted.

    Works in batched rounds: each round builds a batch of prompts, calls
    model.generate_batch(), then parses each result and accepts on success.
    consecutive_failed_rounds counts rounds where zero samples were accepted.
    """
    accepted: list[dict[str, Any]] = []
    prompt_idx = 0
    consecutive_failed_rounds = 0
    log = _get_logger()

    while len(accepted) < n and consecutive_failed_rounds < max_retries:
        remaining = n - len(accepted)
        current_batch_size = min(batch_size, remaining)
        prompts_list, seeds_list, next_idx = _build_batch(
            prompt,
            pool,
            pool_sample_k,
            current_batch_size,
            prompt_idx,
            seed,
            rng,
            output_schema=output_schema,
        )
        prompt_idx = next_idx

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
    prompt: str,
    model: Model,
    n: int = 100,
    output_schema: dict[str, Any] | None = None,
    pool: list[dict[str, Any]] | None = None,
    pool_sample_k: int = 0,
    temperature: float = 0.7,
    max_retries: int = 3,
    seed: int | None = None,
    batch_size: int = 10,
    name: str = "sample_from_LLM",
) -> DataSpec:
    """Build an ACQUIRE node that generates data from an LLM until n accepted samples.

    This is a first-class building block: no inputs, no samples_per_input.
    Generation runs only when the DAG runner evaluates the node (lazy execution).
    Uses batch generation and seed-per-index for cache-friendly, scalable runs.

    Args:
        prompt: Base instruction or prompt (string; TODO: richer prompt/template types).
        model: Model instance (placeholder or real backend).
        n: Target number of accepted samples.
        output_schema: Optional dict of expected keys for structured output;
            triggers JSON parse and key validation.
        pool: Optional list of example dicts for few-shot prompting.
        pool_sample_k: Number of examples sampled from pool per generation prompt.
        temperature: Sampling temperature passed to the model.
        max_retries: Consecutive rounds with zero acceptances before stopping.
        seed: Base seed; combined with prompt index for per-call seeds.
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
        rng = random.Random(seed)
        return list(
            _generation_loop(
                prompt=prompt,
                model=model,
                n=n,
                output_schema=output_schema,
                pool=pool,
                pool_sample_k=pool_sample_k,
                temperature=temperature,
                max_retries=max_retries,
                seed=seed,
                batch_size=batch_size,
                rng=rng,
            )
        )

    return DataSpec(NodeType.ACQUIRE, [], _execute, name)
