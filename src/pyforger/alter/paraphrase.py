"""Response paraphrases for inspecting reward invariance."""

from __future__ import annotations

import json
from typing import Any

from pyforger.acquire.model import Prompt, _cache_kwargs
from pyforger.op import op
from pyforger.utils import get_field

PARAPHRASE_SYSTEM = "Paraphrase the supplied response. Return only the paraphrase."


@op(name="alter.paraphrase", samples_arg="k")
def paraphrase(
    items: list[dict],
    *,
    model: Any,
    response_field: str = "response",
    output_field: str = "paraphrases",
    prompt_field: str | None = "prompt",
    k: int = 2,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> list[dict]:
    """Add candidate paraphrases, retaining the original response.

    Only the task and response are sent to the model, never the rubric or
    reference answer. The prompt is simply "Paraphrase the supplied response.
    Return only the paraphrase." It imposes no per-fact preservation or
    edit-count constraint. High-level meaning and task-relevant quality are
    not guaranteed: review candidates before attributing reward changes to
    oversensitivity.
    Unchanged and duplicate candidates are retained; the inspection excludes
    them from its paraphrase comparisons. An empty or truncated completion is
    represented by None, never silently treated as a faithful rewrite.

    Args:
        items: Input rows, left unchanged.
        model: Model with generate_batch; receives all prompts in one batch.
        response_field: Text to paraphrase.
        output_field: New field containing k strings or None per row.
        prompt_field: Task context field; None sends the response alone.
        k: Number of candidate rewrites per response.
        temperature: Generation temperature, or the model default.
        max_tokens: Completion token limit, or the model default.

    Returns:
        Copies of the rows with the candidate list in output_field.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not output_field or any(output_field in row for row in items):
        raise ValueError("output_field must be a new, nonempty field")
    prompts: list[Prompt] = []
    for row in items:
        response = get_field(row, response_field, "alter.paraphrase")
        if not isinstance(response, str):
            raise TypeError("response_field must contain text")
        payload = {"response": response}
        if prompt_field is not None:
            payload["task"] = get_field(row, prompt_field, "alter.paraphrase")
        prompts.append(
            [
                {"role": "system", "content": PARAPHRASE_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
    if not prompts:
        return []
    results = model.generate_batch(
        prompts,
        n=k,
        temperature=temperature,
        max_tokens=max_tokens,
        **_cache_kwargs(model.generate_batch, cache_rows=list(prompts)),
    )
    if len(results) != len(items):
        raise ValueError("model returned the wrong number of response groups")
    output = []
    for row, result in zip(items, results):
        if not isinstance(result, list) or len(result) != k:
            raise ValueError("model must return k completions per response")
        reasons = getattr(result, "finish_reasons", None) or [None] * k
        if len(reasons) != k:
            raise ValueError("model returned misaligned finish reasons")
        candidates = []
        for text, reason in zip(result, reasons):
            if text is not None and not isinstance(text, str):
                raise TypeError("paraphrase completion must be text or None")
            candidates.append(
                text if text and text.strip() and reason != "length" else None
            )
        output.append({**row, output_field: candidates})
    return output
