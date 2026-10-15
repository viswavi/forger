"""Difficulty filters that ask models to solve each row and grade the answers.

``by_comparative_difficulty`` keeps rows a strong model solves and a weak model
does not. ``by_pass_at_k`` samples one model ``k`` times and keeps rows whose
pass rate falls within ``[min_rate, max_rate]``.

Grades use the format of ``filter.by_success_rate``: ``{group: [True | False |
None, ...]}``, where ``None`` is a grade that could not be made (an empty
completion, a ``<think>`` block that never closes, no extractable answer, or a
verifier returning ``None``) and is excluded, not failed. Predictions and
grades are cached in separate inner ops, so changing only the rate bounds makes
no model calls, and changing only the metric regrades the cached predictions.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from typing import Literal, Union

from rouge_score import rouge_scorer

from pyforger import _tracking
from pyforger.acquire.model import Model, Prompt
from pyforger.alter.generate import (
    PromptTemplate,
    _normalized_equal,
    _resolve_prompt,
    default_extract_answer,
)
from pyforger.op import op
from pyforger.utils import get_field, get_formatted_logger

# A verifier gets the extracted prediction and the full row, e.g. to run tests
# against generated code. None means "could not grade" (excluded, not failed).
Verifier = Callable[[str, dict], Union[bool, None]]
Metric = Union[Literal["exact_match", "rouge_l"], Verifier]

DEFAULT_SOLVE_PROMPT = (
    "Solve the following task. Work through it, then end your response with a "
    "final line of the form 'ANSWER: <answer>'.\n\nTask:\n{task}"
)
_ANSWER_MARKER = re.compile(r"ANSWER\s*:", re.I)
# A whole answer line in \boxed{...}, allowing one level of nested braces.
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def _logger() -> logging.Logger:
    # Looked up per call: a module-level logger read inside an @op body would
    # trigger @op's "reads globals" warning.
    return get_formatted_logger("ModelDifficulty")


@op(name="filter.by_comparative_difficulty")
def by_comparative_difficulty(
    items: list[dict],
    *,
    weak_model: Model,
    strong_model: Model,
    input_field: str,
    ground_truth_field: str,
    metric: Metric = "exact_match",
    rouge_l_threshold: float = 0.7,
    extract_answer: Callable[[str], str | None] | None = None,
    prompt_template: PromptTemplate | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    batch_size: int = 10,
    outcomes_field: str | None = None,
) -> list[dict]:
    """Keep rows the strong model answers correctly and the weak model does not.

    Each model makes one prediction per row. A row whose weak or strong grade is
    ``None`` is dropped: e.g. a weak model that ran out of tokens has not shown
    the row is hard.

    In a ``@step`` report the per-row score is ``strong - weak``: ``1`` (kept),
    ``0`` (both right or both wrong), ``-1`` (only the weak model right), or
    ``None`` when either grade is missing. Many ``-1`` rows suggest a wrong
    ground truth or a metric that is too strict.

    Args:
        items: Input rows.
        weak_model: Model expected to fail on hard rows.
        strong_model: Model expected to succeed on rows worth keeping.
        input_field: Key holding the task, used by the default prompt
            (``DEFAULT_SOLVE_PROMPT``).
        ground_truth_field: Key holding the reference answer.
        metric: ``"exact_match"`` (normalized extracted answer equals the
            reference), ``"rouge_l"`` (ROUGE-L F1 >= ``rouge_l_threshold``), or
            a verifier ``(prediction, row) -> bool | None``.
        rouge_l_threshold: Pass threshold for ``metric="rouge_l"``.
        extract_answer: Applied to each completion before grading; returning
            ``None`` grades ``None``. Default: the first line after the last
            ``ANSWER:`` marker, else the number ``default_extract_answer`` finds;
            ``None`` for a ``<think>`` block that never closes. For a verifier
            that needs the whole completion (e.g. code), pass ``lambda t: t``.
        prompt_template: Format string or ``(row) -> Prompt`` callable replacing
            the default prompt, which asks for a final ``ANSWER: ...`` line.
        temperature: Sampling temperature (each model's default when None).
        max_tokens: Max tokens per completion (each model's default when None).
        batch_size: Prompts per ``model.generate_batch()`` call.
        outcomes_field: When set, kept rows carry
            ``{"weak": [bool], "strong": [bool]}`` under this key, replacing any
            value already there.

    Returns:
        The kept rows, in input order.
    """
    _check_common(metric, rouge_l_threshold, batch_size)
    prompts = _prompts(items, input_field, prompt_template)
    grades: dict[str, list[list[bool | None]]] = {}
    for group, model in (("weak", weak_model), ("strong", strong_model)):
        completions = _predictions(
            prompts,
            model=model,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_size=batch_size,
        )
        grades[group] = _grades(
            items,
            completions,
            ground_truth_field=ground_truth_field,
            metric=metric,
            rouge_l_threshold=rouge_l_threshold,
            extract_answer=extract_answer,
        )

    kept: list[dict] = []
    gaps: list[int | None] = []
    both_right = 0
    for item, weak_row, strong_row in zip(items, grades["weak"], grades["strong"]):
        weak, strong = weak_row[0], strong_row[0]
        if weak is None or strong is None:
            gaps.append(None)
            continue
        gaps.append(int(strong) - int(weak))
        both_right += weak and strong
        if strong and not weak:
            kept.append(
                _with_outcomes(item, outcomes_field, weak=weak_row, strong=strong_row)
            )
    # "between 1 and 1" rather than ">= 1": the report then marks each value
    # kept or not, without what-if columns for thresholds this filter lacks.
    _tracking.record_scores("strong - weak", gaps, keep="between", lower=1, upper=1)
    _logger().info(
        "by_comparative_difficulty: kept %d of %d rows (%d both right, %d ungraded)",
        len(kept),
        len(items),
        both_right,
        gaps.count(None),
    )
    return kept


@op(name="filter.by_pass_at_k")
def by_pass_at_k(
    items: list[dict],
    *,
    model: Model,
    input_field: str,
    ground_truth_field: str,
    k: int = 4,
    min_rate: float = 0.1,
    max_rate: float = 0.9,
    min_valid: int | None = None,
    metric: Metric = "exact_match",
    rouge_l_threshold: float = 0.7,
    extract_answer: Callable[[str], str | None] | None = None,
    prompt_template: PromptTemplate | None = None,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    batch_size: int = 10,
    outcomes_field: str | None = None,
) -> list[dict]:
    """Keep rows whose pass rate over ``k`` samples is within the bounds.

    Each prompt is sent once with ``n=k``. The pass rate is passed / valid
    grades, and a row is kept when ``min_rate <= rate <= max_rate``. As in
    ``filter.by_success_rate``, all ``k`` grades must be valid unless
    ``min_valid`` allows exclusions, and a row with no valid grade never passes.
    E.g. with ``k=4``, grades ``[True, False, None, False]`` are dropped by
    default, and kept with ``min_valid=3`` (rate 1/3).

    In a ``@step`` report the per-row score is the pass rate, or ``None`` when
    too few grades are valid.

    Args:
        items: Input rows.
        model: Model to sample from.
        input_field: Key holding the task, used by the default prompt
            (``DEFAULT_SOLVE_PROMPT``).
        ground_truth_field: Key holding the reference answer.
        k: Samples per row.
        min_rate: Lowest pass rate kept (inclusive); raise it to drop rows the
            model cannot solve.
        max_rate: Highest pass rate kept (inclusive); lower it to drop rows the
            model always solves.
        min_valid: Fewest valid grades a row needs, at most ``k``. None
            requires all ``k``.
        metric: ``"exact_match"``, ``"rouge_l"``, or a verifier
            ``(prediction, row) -> bool | None``; see
            ``by_comparative_difficulty``.
        rouge_l_threshold: Pass threshold for ``metric="rouge_l"``.
        extract_answer: Applied to each completion before grading; see
            ``by_comparative_difficulty``.
        prompt_template: Format string or ``(row) -> Prompt`` callable replacing
            the default prompt.
        temperature: Sampling temperature; must be > 0 when ``k > 1``.
        max_tokens: Max tokens per completion (the model's default when None).
        batch_size: Prompts per ``model.generate_batch()`` call.
        outcomes_field: When set, kept rows carry ``{"model": [bool | None] * k}``
            under this key, replacing any value already there, so
            ``filter.by_success_rate`` can re-select them later.

    Returns:
        The kept rows, in input order.
    """
    _check_common(metric, rouge_l_threshold, batch_size)
    if k < 1:
        raise ValueError("k must be at least 1")
    for name, bound in (("min_rate", min_rate), ("max_rate", max_rate)):
        if not (math.isfinite(bound) and 0.0 <= bound <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {bound}")
    if min_rate > max_rate:
        raise ValueError("min_rate must not exceed max_rate")
    if min_valid is not None and not (1 <= min_valid <= k):
        # A row has only k grades, so min_valid > k would keep nothing.
        raise ValueError(f"min_valid must be between 1 and k ({k}), got {min_valid}")
    if k > 1 and not temperature > 0.0:
        raise ValueError(
            "temperature must be > 0.0 when k > 1; identical samples make the "
            "pass rate 0 or 1"
        )

    prompts = _prompts(items, input_field, prompt_template)
    completions = _predictions(
        prompts,
        model=model,
        n=k,
        temperature=temperature,
        max_tokens=max_tokens,
        batch_size=batch_size,
    )
    grades = _grades(
        items,
        completions,
        ground_truth_field=ground_truth_field,
        metric=metric,
        rouge_l_threshold=rouge_l_threshold,
        extract_answer=extract_answer,
    )

    kept: list[dict] = []
    rates: list[float | None] = []
    for item, row_grades in zip(items, grades):
        rate = _pass_rate(row_grades, min_valid)
        rates.append(rate)
        if rate is not None and min_rate <= rate <= max_rate:
            kept.append(_with_outcomes(item, outcomes_field, model=row_grades))
    _tracking.record_scores(
        "pass rate", rates, keep="between", lower=min_rate, upper=max_rate
    )
    return kept


@op(name="filter.model_difficulty.predictions")
def _predictions(
    prompts: list[Prompt],
    *,
    model: Model,
    n: int,
    temperature: float | None,
    max_tokens: int | None,
    batch_size: int,
) -> list[list[str | None]]:
    """Return ``n`` completions per prompt, ``None`` for missing or empty ones.

    Exceptions from the model propagate, so a failed call is not cached.
    """
    completions: list[list[str | None]] = []
    short = 0
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        results = model.generate_batch(
            batch, n=n, temperature=temperature, max_tokens=max_tokens
        )
        if len(results) != len(batch):
            _logger().warning(
                "generate_batch returned %d results for %d prompts",
                len(results),
                len(batch),
            )
        for i in range(len(batch)):
            got = list(results[i]) if i < len(results) else []
            short += len(got) < n
            got = [c if isinstance(c, str) and c.strip() else None for c in got]
            completions.append((got + [None] * n)[:n])
    if short:
        # E.g. a provider that ignores n: the missing samples grade None, so with
        # min_valid=None every such row is dropped.
        _logger().warning(
            "%d of %d prompts got fewer than n=%d completions; missing ones grade "
            "None",
            short,
            len(prompts),
            n,
        )
    return completions


@op(name="filter.model_difficulty.grades")
def _grades(
    items: list[dict],
    completions: list[list[str | None]],
    *,
    ground_truth_field: str,
    metric: Metric,
    rouge_l_threshold: float,
    extract_answer: Callable[[str], str | None] | None,
) -> list[list[bool | None]]:
    """Grade each row's completions against its ground truth.

    A completion grades ``None`` when it is empty or no answer can be extracted
    from it, e.g. a reasoning model cut off mid-thought.
    """
    extract = extract_answer if extract_answer is not None else _default_extract
    scorer = (
        rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        if metric == "rouge_l"
        else None
    )
    out: list[list[bool | None]] = []
    for item, row in zip(items, completions):
        reference = get_field(item, ground_truth_field, "filter.model_difficulty")
        row_grades: list[bool | None] = []
        for completion in row:
            prediction = None if completion is None else extract(completion)
            if prediction is None:
                row_grades.append(None)
            elif callable(metric):
                grade = metric(prediction, item)
                if grade is not None and not isinstance(grade, bool):
                    raise TypeError(f"metric must return bool or None, got {grade!r}")
                row_grades.append(grade)
            elif scorer is not None:
                f1 = scorer.score(str(reference), str(prediction))["rougeL"].fmeasure
                row_grades.append(f1 >= rouge_l_threshold)
            else:
                row_grades.append(_normalized_equal(prediction, reference))
        out.append(row_grades)
    return out


def _default_extract(text: str) -> str | None:
    r"""First line after the last ``ANSWER:``, else ``default_extract_answer``.

    E.g. ``"... ANSWER: Paris."`` gives ``"Paris"``, ``"**ANSWER:** $1,000$"``
    gives ``"1,000"`` (matched as ``1000``), and ``"ANSWER: \boxed{42}"`` or
    ``"... \boxed{42}"`` gives ``"42"``. A closed ``<think>`` block is ignored;
    one that never closes (the model ran out of tokens) gives ``None``.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return None
    parts = _ANSWER_MARKER.split(text)
    if len(parts) > 1:
        lines = [line.strip(" *$") for line in parts[-1].splitlines()]
        final = next((line for line in lines if line), "").rstrip(".").strip(" *$")
        boxed = _BOXED.fullmatch(final)
        if boxed:
            final = boxed.group(1).strip()
        return final or None
    return default_extract_answer(text)


def _pass_rate(grades: list[bool | None], min_valid: int | None) -> float | None:
    """Pass rate as ``filter.by_success_rate`` computes it, or None if it fails.

    None when no grade is valid or fewer than ``min_valid`` (default: all) are.
    """
    valid = sum(g is not None for g in grades)
    if not valid or valid < (min_valid or len(grades)):
        return None
    return sum(g is True for g in grades) / valid


def _with_outcomes(
    item: dict, outcomes_field: str | None, **groups: list[bool | None]
) -> dict:
    """Return ``item``, or a copy with ``{group: grades}`` under ``outcomes_field``."""
    if outcomes_field is None:
        return item
    return {**item, outcomes_field: {g: list(v) for g, v in groups.items()}}


def _prompts(
    items: list[dict], input_field: str, prompt_template: PromptTemplate | None
) -> list[Prompt]:
    if prompt_template is not None:
        return [_resolve_prompt(prompt_template, item) for item in items]
    return [
        DEFAULT_SOLVE_PROMPT.format(
            task=get_field(item, input_field, "filter.model_difficulty")
        )
        for item in items
    ]


def _check_common(metric: Metric, rouge_l_threshold: float, batch_size: int) -> None:
    if not callable(metric) and metric not in ("exact_match", "rouge_l"):
        raise ValueError(
            f"metric must be 'exact_match', 'rouge_l', or a callable, got {metric!r}"
        )
    if not 0.0 <= rouge_l_threshold <= 1.0:
        raise ValueError(
            f"rouge_l_threshold must be in [0, 1], got {rouge_l_threshold}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
