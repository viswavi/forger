"""Arithmetic verification utilities for GSM8K-style math word problems.

These functions check that a generated solution is internally consistent:
every inline ``<<expr=result>>`` calculator annotation must actually evaluate
to its stated result, the solution must be multi-step, and the final
``#### <number>`` answer must match the last annotated calculation.

Verifying arithmetic locally (no LLM call) is the cheapest and strongest
quality lever for math data: it removes solutions with hallucinated
calculations, which a fluent LLM produces surprisingly often.
"""
from __future__ import annotations

import re

# Matches GSM8K calculator annotations: <<expr=result>>
_ANNOT_RE = re.compile(r"<<([^=<>]+)=([^<>]*)>>")
# Matches the final answer line: #### <number>
_FINAL_RE = re.compile(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)")
# Characters allowed inside an arithmetic expression we will eval().
_EXPR_OK = re.compile(r"^[0-9.+\-*/() ]+$")


def _to_number(text: str) -> float | None:
    """Parse a numeric string like '1,200' or '.4' into a float."""
    text = text.strip().replace(",", "")
    if text in ("", ".", "-"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_eval(expr: str) -> float | None:
    """Evaluate a pure-arithmetic expression, or return None if unsafe/invalid."""
    expr = expr.replace(",", "").strip()
    if not expr or not _EXPR_OK.match(expr):
        return None
    try:
        # Only arithmetic chars are present (checked above), so eval is safe here.
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, ValueError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _close(a: float, b: float, rel: float = 1e-3, abs_: float = 1e-4) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def check_answer(answer: str, min_steps: int = 2) -> tuple[bool, str]:
    """Validate a GSM8K-style answer string.

    Returns (is_valid, reason). ``reason`` is "ok" when valid, otherwise a
    short diagnostic useful for inspecting rejection rates while iterating.

    Checks:
      * at least ``min_steps`` ``<<expr=result>>`` annotations (multi-step),
      * every annotation's expression evaluates to its stated result,
      * a ``#### <number>`` final line exists,
      * the final number equals the last annotation's result.
    """
    if not isinstance(answer, str) or not answer.strip():
        return False, "empty"

    annots = _ANNOT_RE.findall(answer)
    if len(annots) < min_steps:
        return False, f"too_few_steps({len(annots)})"

    last_result: float | None = None
    for expr, result_str in annots:
        got = _safe_eval(expr)
        if got is None:
            return False, f"unevaluable_expr:{expr!r}"
        stated = _to_number(result_str)
        if stated is None:
            return False, f"bad_result:{result_str!r}"
        if not _close(got, stated):
            return False, f"mismatch:{expr.strip()}=>{got}!={stated}"
        last_result = stated

    final_match = _FINAL_RE.search(answer)
    if not final_match:
        return False, "no_final_answer"
    final_val = _to_number(final_match.group(1))
    if final_val is None:
        return False, "bad_final_answer"
    if last_result is not None and not _close(final_val, last_result):
        return False, f"final_mismatch:{final_val}!={last_result}"

    return True, "ok"


def has_clean_numbers(answer: str, max_decimals: int = 2) -> bool:
    """Reject solutions containing ugly long/repeating decimals.

    GSM8K answers keep numbers clean; a number like 0.16666666 signals a
    poorly-chosen problem. Returns False if any number in the text has more
    than ``max_decimals`` digits after the decimal point.
    """
    for frac in re.findall(r"\d+\.(\d+)", answer):
        if len(frac) > max_decimals:
            return False
    return True


def check_question(question: str) -> tuple[bool, str]:
    """Light sanity checks on the question text."""
    if not isinstance(question, str) or not question.strip():
        return False, "empty"
    # Need at least two numbers to support multi-step arithmetic.
    if len(re.findall(r"\d", question)) < 2:
        return False, "too_few_numbers"
    if len(question.split()) < 8:
        return False, "too_short"
    return True, "ok"
