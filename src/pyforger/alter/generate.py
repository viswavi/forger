"""Building block for conditional LM generation (conditioned on upstream input)."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Union

from pyforger.acquire._model_utils import _build_prompt, _parse_output
from pyforger.acquire.model import Model, Prompt
from pyforger.op import op

PromptTemplate = Union[str, Callable[[dict], Prompt]]


def _get_logger() -> logging.Logger:
    """Lazy logger to avoid circular import with utils."""
    from pyforger.utils import get_formatted_logger as _gf

    return _gf(__name__)


def _resolve_prompt(template: PromptTemplate, item: dict) -> Prompt:
    """Resolve a prompt template against one input item.

    A string template is formatted with the item's keys (e.g. ``"{text}"``
    is replaced by ``item["text"]``). A callable is called with the item.
    """
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
) -> list[dict | None]:
    r"""Generate LLM outputs conditioned on upstream input items.

    ``prompt_fn`` wraps each input item in a prompt which is passed to ``model``
    for generation. ``items`` must be a single list of dicts; to condition on
    multiple upstream sources, combine them first with ``forger.zip`` (see
    example below).

    Example — self-refine::

        # forger.zip merges items from two lists into one dict per pair
        prompt_response_pairs = forger.zip(prompts, responses)
        critiques = generate_from_context(
            items=prompt_response_pairs,
            prompt_fn="Critique this response:\\nQ: {prompt}\\nA: {response}",
            model=model,
        )

        prompt_response_critique_triples = forger.zip(prompts, responses, critiques)
        refined = generate_from_context(
            items=prompt_response_critique_triples,
            prompt_fn=lambda item: (
                f"Revise the response given the critique:\\n"
                f"Q: {item['prompt']}\\n"
                f"A: {item['response']}\\n"
                f"Critique: {item['critique']}"
            ),
            model=model,
        )

    Args:
        items: Input rows passed one-by-one to ``prompt_fn``. To combine multiple
            upstream sources, merge items from dictionaries before passing here.
        prompt_fn: A format string (keys filled from the input item dict) or a
            callable ``(dict) -> Prompt``. Callables may return a string or a
            list of chat messages in OpenAI format.
        model: Model instance used for generation.
        samples_per_item: Number of responses to generate per input item.
        output_schema: Optional dict of expected keys. When set, responses are
            JSON-parsed and validated; failures are retried. When None, each
            response is stored as ``{"text": ...}``.
        temperature: Sampling temperature passed to the model.
        max_retries: Additional attempts for prompts whose output fails to parse.
        batch_size: Number of prompts per ``model.generate_batch()`` call.

    Returns:
        List of length ``len(items) * samples_per_item`` in input order, grouped
        by item (item ``i``'s samples occupy slots ``i * samples_per_item`` to
        ``(i + 1) * samples_per_item``). Each slot holds the parsed output dict,
        or ``None`` if the prompt failed to parse after ``max_retries``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if samples_per_item <= 0:
        raise ValueError("samples_per_item must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    log = _get_logger()

    all_prompts: list[Prompt] = []
    for item in items:
        full_prompt = _build_prompt(_resolve_prompt(prompt_fn, item), output_schema)
        all_prompts.extend([full_prompt] * samples_per_item)

    results: list[dict | None] = [None] * len(all_prompts)
    pending: list[tuple[int, Prompt]] = list(enumerate(all_prompts))

    for attempt in range(1 + max_retries):
        if not pending:
            break
        next_pending: list[tuple[int, Prompt]] = []
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            batch_prompts = [p for _, p in batch]
            batch_results = model.generate_batch(batch_prompts, temperature=temperature)
            if len(batch_results) != len(batch):
                log.warning(
                    "generate_batch returned %d results for %d prompts; will retry",
                    len(batch_results),
                    len(batch),
                )
                next_pending.extend(batch)
                continue
            for (idx, prompt), responses in zip(batch, batch_results):
                raw_text = responses[0]
                parsed = _parse_output(raw_text, output_schema)
                if parsed is None:
                    if attempt < max_retries:
                        log.debug("Parse failed (attempt %d); will retry", attempt + 1)
                    else:
                        log.warning(
                            "Parse failed after %d attempts; giving up", attempt + 1
                        )
                    next_pending.append((idx, prompt))
                else:
                    results[idx] = parsed
        pending = next_pending

    if pending:
        log.warning(
            "%d prompt(s) exhausted %d retries; returning None for failed prompts",
            len(pending),
            max_retries,
        )
    return results
