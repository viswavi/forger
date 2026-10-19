"""Shared helpers for prompt construction and output parsing.

Internal to ``pyforger``; used by ``acquire.sample`` and ``alter.generate``.
"""
from __future__ import annotations

import inspect
import json
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from pyforger import _tracking
from pyforger.acquire.model import Prompt
from pyforger.op import skip_cache_write
from pyforger.utils import get_formatted_logger

logger = get_formatted_logger(__name__)


def _schema_format_instructions(output_schema: dict[str, Any]) -> str:
    """Build a short JSON-format instruction for the prompt.

    It asks for the object at the end and does not forbid text before it, so it
    does not contradict a prompt that asks for visible reasoning or a code
    block; `_find_json_object` takes the last matching object.
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
    return (
        f"End your response with a JSON object with these keys: {json.dumps(keys)}, "
        f"for example {json.dumps(placeholders)}. Put nothing after the JSON object."
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

    `Model.generate`/`generate_batch` return `GenerateResult` (list[str]) carrying
    a parallel `.reasoning` list), but a user-supplied or built-in `Model` subclass
    may return a plain list instead. Fall back to ``None`` per slot then.
    """
    reasoning = getattr(completions, "reasoning", None)
    return reasoning if reasoning is not None else [None] * len(completions)


# An inline reasoning block before the answer, e.g. "<think>...</think>{...}".
_THINK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)


def _find_json_object(text: str, keys: set[str]) -> dict[str, Any] | None:
    r"""The last JSON object in a completion that has all of ``keys``, else None.

    When the whole text is JSON, it must be such an object. Otherwise every
    top-level object in the text counts, fenced or not, and the last match wins:
    the prompt asks for the object at the end, and reasoning before it may quote
    an example. E.g. with ``keys={"answer"}``,
    'Format: {"answer": "<str>"}. 6*7=42. ```json\n{"answer": "42"}\n```'
    gives ``{"answer": "42"}``. Objects nested in another are not candidates,
    and text that is valid JSON but not an object (e.g. a list) fails. Raw
    newlines inside strings are allowed (e.g. multi-line code in a value).
    """
    text = _THINK_RE.sub("", text, count=1)
    decoder = json.JSONDecoder(strict=False)
    try:
        loaded = decoder.decode(text)
    except json.JSONDecodeError:
        pass
    else:
        return loaded if isinstance(loaded, dict) and keys <= loaded.keys() else None
    found = None
    start = text.find("{")
    while start != -1:
        try:
            loaded, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if keys <= loaded.keys():
            found = loaded
        # Skip the objects nested in this one: {"a": {"answer": 1}} is not a match.
        start = text.find("{", end)
    return found


def _parse_output(
    raw_text: str,
    output_schema: dict[str, Any] | None,
    reasoning: str | None = None,
) -> dict[str, Any] | None:
    """Parse a completion into a row dict, attaching reasoning trace if provided.

    With ``output_schema``, the completion must hold a JSON object with every
    schema key (see `_find_json_object`); keys outside the schema are dropped
    and value types are not checked.
    """
    if output_schema is None:
        parsed: dict[str, Any] = {"text": raw_text}
    else:
        if not isinstance(raw_text, str):
            return None
        loaded = _find_json_object(raw_text, set(output_schema))
        if loaded is None:
            return None
        # e.g. a "crafted_by" key the model added would otherwise reach the rows.
        parsed = {key: loaded[key] for key in output_schema}
    if reasoning:
        if "thinking_trace" in parsed:
            logger.warning(
                "Parsed output already has a 'thinking_trace' field; dropping the "
                "provider reasoning trace instead of overwriting it."
            )
        else:
            parsed["thinking_trace"] = reasoning
    return parsed


# Status codes worth asking again: request timeout, conflict, too early, rate
# limit. Every 5xx is too.
_TRANSIENT_STATUS = frozenset({408, 409, 425, 429})


def _is_transient(exc: BaseException) -> bool:
    """True for provider errors a retry can fix: timeouts, 429s, 5xx, lost connections.

    litellm's and openai's ``Timeout`` / ``APIConnectionError`` and builtin
    ``TimeoutError`` / ``ConnectionError`` are transient; other errors are judged
    by their ``status_code``. A bad key (401), unknown model (404), or bad
    request (400, e.g. context length) is not, nor is an error with no status.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    try:
        import openai
    except ImportError:  # pragma: no cover - openai ships with litellm
        pass
    else:
        if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
            return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS or status >= 500
    return False


def _accepts_kwarg(fn: Callable[..., Any] | None, name: str) -> bool:
    """True when ``fn`` names ``name`` as a parameter (``**kwargs`` does not count).

    Lets ops pass newer ``generate_batch`` keywords (``return_exceptions``) only
    to models that take them; duck-typed fakes and older subclasses do not.
    """
    if fn is None:
        return False
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _report_provider_failures(message: str, error_types: list[str]) -> None:
    """Warn about requests that failed at the provider; keep the op out of the cache.

    ``error_types`` has one exception type name per failure, e.g. three 504s
    give "...: InternalServerError ×3". The warning is logged and shown in the
    step report, and the op (and its step) is not cached, so a rerun asks
    again; the request cache replays the completions that did arrive.
    """
    counts = Counter(error_types)
    summary = ", ".join(f"{name} ×{count}" for name, count in counts.most_common())
    text = f"{message}: {summary} (result not cached)"
    logger.warning(text)
    _tracking.add_op_warning(text)
    skip_cache_write()
