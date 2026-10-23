"""Shared helpers for prompt construction and output parsing.

Internal to ``pyforger``; used by ``acquire.sample``, ``alter.generate``, and
the LLM filters.
"""
from __future__ import annotations

import inspect
import json
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from pyforger import _tracking
from pyforger.acquire.model import Prompt, _is_transient
from pyforger.op import skip_cache_write
from pyforger.utils import get_formatted_logger

logger = get_formatted_logger(__name__)


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

    `Model.generate`/`generate_batch` return `GenerateResult` (list[str]) carrying
    a parallel `.reasoning` list), but a user-supplied or built-in `Model` subclass
    may return a plain list instead. Fall back to ``None`` per slot then.
    """
    reasoning = getattr(completions, "reasoning", None)
    return reasoning if reasoning is not None else [None] * len(completions)


# A ``` or ```json fence; group 1 is its content.
_FENCE_RE = re.compile(r"```[\w-]*[^\S\n]*\n?(.*?)```", re.DOTALL)
# An inline reasoning block before the answer, e.g. "<think>...</think>{...}".
_THINK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)


def _find_json_object(text: str, keys: set[str]) -> dict[str, Any] | None:
    """The JSON object in a completion that has all of ``keys``, else None.

    Tries the whole text, then each ``` fence's content, then every object
    embedded in the text (e.g. 'Here it is: {...} Hope this helps'), first
    match wins. Text that is valid JSON but not an object (e.g. a list) fails.
    Raw newlines inside strings are allowed (e.g. multi-line code in a value).
    """
    text = _THINK_RE.sub("", text, count=1)
    decoder = json.JSONDecoder(strict=False)
    try:
        loaded = decoder.decode(text)
    except json.JSONDecodeError:
        pass
    else:
        return loaded if isinstance(loaded, dict) and keys <= loaded.keys() else None
    for fence in _FENCE_RE.finditer(text):
        try:
            loaded = decoder.decode(fence.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict) and keys <= loaded.keys():
            return loaded
    start = text.find("{")
    while start != -1:
        try:
            loaded, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if keys <= loaded.keys():
            return loaded
        # Skip the objects nested in this one: {"a": {"answer": 1}} is not a match.
        start = text.find("{", end)
    return None


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


def _report_provider_failures(
    message: str, error_types: list[str], last_error: BaseException | None = None
) -> None:
    """Warn about requests that failed at the provider; keep the op out of the cache.

    ``error_types`` has one exception type name per failure, e.g. three 504s
    give "...: InternalServerError ×3", and ``last_error`` adds the last one's
    message. The warning is logged and shown in the
    step report, and the op (and its step) is not cached, so a rerun asks
    again; the request cache replays the completions that did arrive.
    """
    counts = Counter(error_types)
    summary = ", ".join(f"{name} ×{count}" for name, count in counts.most_common())
    if last_error is not None:
        detail = " ".join(str(last_error).split())
        if len(detail) > 200:
            detail = detail[:199] + "…"
        summary += f"; last error: {type(last_error).__name__}: {detail}"
    text = f"{message}: {summary} (result not cached)"
    if counts["RateLimitError"] * 2 >= len(error_types):
        # Mostly rate limits: they outlasted minutes of retries (see Model).
        text += (
            "; the provider is rate-limiting: rerun the step (only the failed "
            "prompts are sent again), lower Model(max_concurrency=...), or run "
            "fewer pipelines at once"
        )
    logger.warning(text)
    _tracking.add_op_warning(text)
    skip_cache_write()


def _catch_kwargs(model: Any) -> dict[str, Any]:
    """``{"return_exceptions": True}`` for a model whose generate_batch takes it.

    Duck-typed fakes and older backends get ``{}`` and raise as before.
    """
    if _accepts_kwarg(getattr(model, "generate_batch", None), "return_exceptions"):
        return {"return_exceptions": True}
    return {}


class _RequestFailures:
    """The failed requests an LLM filter tolerates, reported in one warning.

    ``failed(result)`` is True for a transient error (a 504, a timeout) left in
    a ``generate_batch`` slot by ``return_exceptions=True``; the filter then
    treats it as a missing vote or sample. Any other error (a bad key, a
    spending cap, a token-unit quota) is raised.
    """

    def __init__(self) -> None:
        """Start with no failures."""
        self.types: list[str] = []
        self.last: BaseException | None = None

    def __len__(self) -> int:
        """Number of failed requests (or missing completions) so far."""
        return len(self.types)

    def failed(self, result: Any) -> bool:
        """True (and counted) when ``result`` is a transient error; raise others."""
        if not isinstance(result, BaseException):
            return False
        if not _is_transient(result):
            raise result
        self.types.append(type(result).__name__)
        self.last = result
        return True

    def missing(self, count: int) -> None:
        """Count completions that partly failed requests did not return."""
        self.types.extend(["missing completions"] * count)

    def report(self, message: str) -> None:
        """Warn once, e.g. ``"<message>: Timeout ×2; last error: ..."``."""
        if self.types:
            _report_provider_failures(message, self.types, self.last)
