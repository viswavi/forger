"""Shared helpers for prompt construction and output parsing.

Internal to ``pyforger``; used by ``acquire.sample`` and ``alter.generate``.
"""
from __future__ import annotations

import json
from typing import Any

from pyforger.acquire.model import Prompt


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


def _reasoning_for(completions: Any) -> list[str | None]:
    """Per-completion reasoning list from a ``GenerateResult``, else all-``None``.

    ``Model.generate``/``generate_batch`` return ``GenerateResult`` (a
    ``list[str]`` carrying a parallel ``.reasoning`` list), but a user-supplied
    or built-in ``Model`` subclass may return a plain list instead (e.g.
    ``LocalModel``, whose reasoning is already inline in the completion text
    rather than a separate field) -- fall back to ``None`` per slot then.
    """
    reasoning = getattr(completions, "reasoning", None)
    return reasoning if reasoning is not None else [None] * len(completions)


def _parse_output(
    raw_text: str,
    output_schema: dict[str, Any] | None,
    reasoning: str | None = None,
) -> dict[str, Any] | None:
    if output_schema is None:
        parsed: dict[str, Any] = {"text": raw_text}
    else:
        try:
            loaded = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(loaded, dict):
            return None
        required_keys = set(output_schema.keys())
        if not required_keys.issubset(loaded.keys()):
            return None
        parsed = loaded
    if reasoning:
        parsed["reasoning"] = reasoning
    return parsed
