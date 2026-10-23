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
from pyforger.acquire.model import (
    Prompt,
    _is_row_error,
    _is_transient,
    _row_error_reason,
)
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


# How a warning counts row errors by reason (see _failure_label), e.g. "2
# prompts blocked by the provider's content filter".
_ROW_ERROR_PHRASES = {
    "content filter": "blocked by the provider's content filter",
    "context window": "longer than the model's context window",
}


def _failure_label(exc: BaseException) -> str:
    """A failed request's label in a warning: its row-error reason, else its type."""
    return _row_error_reason(exc) or type(exc).__name__


def _count_text(label: str, count: int) -> str:
    """``"InternalServerError ×2"``, or by reason ``"2 prompts blocked by ..."``."""
    if label in _ROW_ERROR_PHRASES:
        noun = "prompt" if count == 1 else "prompts"
        return f"{count} {noun} {_ROW_ERROR_PHRASES[label]}"
    return f"{label} ×{count}"


def _raise_if_systemic(results: list[Any], total: int) -> None:
    """Raise the first row error if all ``total`` (at least 2) prompts hit one.

    One prompt blocked by the content filter is that row's problem; all of
    them blocked points at the template or the model (e.g. a system prompt
    that trips the filter), so the op fails instead of returning only Nones.
    Only the row errors in ``results`` count.
    """
    row_errors = [
        r for r in results if isinstance(r, BaseException) and _is_row_error(r)
    ]
    if total >= 2 and len(row_errors) >= total:
        raise row_errors[0]


def _report_provider_failures(
    message: str, error_types: list[str], last_error: BaseException | None = None
) -> None:
    """Warn about requests that failed at the provider; keep the op out of the cache.

    ``error_types`` has one label per failure (`_failure_label`): an exception
    type name, e.g. three 504s give "...: InternalServerError ×3", or a row
    error's reason, e.g. "...: 2 prompts blocked by the provider's content
    filter". ``last_error`` adds the last one's message. The warning is logged
    and shown in the step report, and the op (and its step) is not cached, so
    a rerun asks again; the request cache replays the completions that did
    arrive.
    """
    counts = Counter(error_types)
    summary = ", ".join(_count_text(name, n) for name, n in counts.most_common())
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

    ``failed(result)`` is True for a transient error (a 504, a timeout) or a
    row error (a content-filter block, an over-long prompt) left in a
    ``generate_batch`` slot by ``return_exceptions=True``; the filter then
    treats it as a missing vote or sample. Any other error (a bad key, a
    spending cap, a token-unit quota) is raised.
    """

    def __init__(self) -> None:
        """Start with no failures."""
        self.types: list[str] = []
        self.last: BaseException | None = None
        self.row_errors: list[BaseException] = []

    def __len__(self) -> int:
        """Number of failed requests (or missing completions) so far."""
        return len(self.types)

    def failed(self, result: Any) -> bool:
        """True (and counted) for a transient or row error; raise other errors."""
        if not isinstance(result, BaseException):
            return False
        if _is_row_error(result):
            self.row_errors.append(result)
        elif not _is_transient(result):
            raise result
        self.types.append(_failure_label(result))
        self.last = result
        return True

    def raise_if_systemic(self, total: int) -> None:
        """Raise the first row error if all ``total`` (at least 2) requests hit one."""
        _raise_if_systemic(self.row_errors, total)

    def missing(self, count: int) -> None:
        """Count completions that partly failed requests did not return."""
        self.types.extend(["missing completions"] * count)

    def report(self, message: str) -> None:
        """Warn once, e.g. ``"<message>: Timeout ×2; last error: ..."``."""
        if self.types:
            _report_provider_failures(message, self.types, self.last)
