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
