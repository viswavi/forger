"""Conditional LM generation: prompt the model with each upstream item.

Returns one parsed dict per (input item × ``samples_per_item``). When
``output_schema`` is None, each result is ``{"text": <raw>}``; otherwise
results are JSON-parsed and validated against the schema with retries.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Union

from pyforger.acquire.model import Model, Prompt
from pyforger.acquire.sample import _build_prompt, _parse_output
from pyforger.op import op

PromptTemplate = Union[str, Callable[[dict], Prompt]]


def _get_logger() -> logging.Logger:
    from pyforger.utils import get_formatted_logger as _gf

    return _gf(__name__)


def _resolve_prompt(template: PromptTemplate, item: dict) -> Prompt:
    """Format-string templates use ``str.format_map``; callables get the item."""
    if isinstance(template, str):
        return template.format_map(item)
    return template(item)


@op(name="alter.generate_from_context")
def generate_from_context(
    items: list[dict],
    prompt_fn: PromptTemplate,
    model: Model,
    *,
    samples_per_item: int = 1,
    output_schema: dict[str, Any] | None = None,
    temperature: float = 0.7,
    max_retries: int = 3,
    batch_size: int = 10,
) -> list[dict]:
    """Prompt ``model`` once per (item × sample), return the parsed results.

    Args:
        items: Input rows to condition on.
        prompt_fn: Format string filled from item keys, or callable
            returning a string / chat-message list per item.
        model: Model to call.
        samples_per_item: Generations per input item.
        output_schema: Optional dict of expected keys; when set, responses
            are JSON-parsed and retried on failure.
        temperature, max_retries, batch_size: Generation hyperparameters.

    Returns:
        Flat list of parsed dicts. Length is at most
        ``len(items) * samples_per_item``; failed parses (after
        ``max_retries``) are dropped with a warning.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if samples_per_item <= 0:
        raise ValueError("samples_per_item must be positive")

    log = _get_logger()

    all_prompts: list[Prompt] = []
    for item in items:
        full_prompt = _build_prompt(_resolve_prompt(prompt_fn, item), output_schema)
        all_prompts.extend([full_prompt] * samples_per_item)

    results: list[dict] = []
    pending = all_prompts

    for attempt in range(1 + max_retries):
        if not pending:
            break
        next_pending: list[Prompt] = []
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            batch_results = model.generate_batch(batch, temperature=temperature)
            if len(batch_results) != len(batch):
                log.warning(
                    "generate_batch returned %d results for %d prompts; will retry",
                    len(batch_results),
                    len(batch),
                )
                next_pending.extend(batch)
                continue
            for prompt, responses in zip(batch, batch_results):
                raw_text = responses[0]
                parsed = _parse_output(raw_text, output_schema)
                if parsed is None:
                    log.debug("Parse failed (attempt %d); will retry", attempt + 1)
                    next_pending.append(prompt)
                else:
                    results.append(parsed)
        pending = next_pending

    if pending:
        log.warning(
            "%d prompt(s) exhausted %d retries and were dropped",
            len(pending),
            max_retries,
        )
    return results
