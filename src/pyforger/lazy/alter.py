"""Alter ops (lazy). Transform data without removing items."""
from __future__ import annotations

from typing import Any, Callable

from pyforger.lazy.core import op


@op(name="alter.concat")
def concat(*sources: list[dict]) -> list[dict]:
    """Concatenate multiple list[dict] streams into one."""
    out: list[dict] = []
    for s in sources:
        out.extend(s)
    return out


@op(name="alter.generate_from_context")
def generate_from_context(
    items: list[dict],
    *,
    model: Any,
    prompt_fn: Callable[[dict], Any],
    temperature: float = 0.0,
    samples_per_item: int = 1,
) -> list[dict]:
    """For each input item, build a prompt and ask the model.

    Returns one dict per (item, sample) pair with shape ``{"text": ...}``.
    When ``samples_per_item > 1`` the output length is
    ``len(items) * samples_per_item``.
    """
    if not items:
        return []
    prompts = [prompt_fn(item) for item in items for _ in range(samples_per_item)]
    responses = model.generate_batch(prompts, temperature=temperature)
    return [{"text": r[0] if isinstance(r, (list, tuple)) else r} for r in responses]


@op(name="alter.zip_merge")
def zip_merge(left: list[dict], right: list[dict]) -> list[dict]:
    """Merge parallel streams by position. Same length required.

    Useful when ``generate_from_context`` drops source context and you
    want to reattach it alongside the generated fields.
    """
    if len(left) != len(right):
        raise ValueError(f"zip_merge length mismatch: {len(left)} vs {len(right)}")
    return [{**a, **b} for a, b in zip(left, right)]
