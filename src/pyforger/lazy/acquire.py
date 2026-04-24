"""Acquire ops (lazy). Entry points that put data into the graph."""
from __future__ import annotations

from typing import Any

from pyforger.lazy.core import op


@op(name="acquire.from_list")
def from_list(items: list[dict]) -> list[dict]:
    """Wrap a concrete list as a graph leaf."""
    return list(items)


@op(name="acquire.sample_from_LLM")
def sample_from_LLM(prompt: Any, model: Any, n: int = 10) -> list[dict]:
    """Generate n samples from the model for a single prompt.

    Minimal port — supports StubModel for offline demos and any model
    exposing ``generate_batch(prompts, temperature=...)``.
    """
    prompts = [prompt] * n
    responses = model.generate_batch(prompts)
    return [{"text": r[0] if isinstance(r, (list, tuple)) else r} for r in responses]
