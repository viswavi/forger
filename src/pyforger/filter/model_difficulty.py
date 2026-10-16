"""Difficulty filters that ask models to solve each row and grade the answers.

``by_comparative_difficulty`` keeps rows a strong model solves and a weak model
does not. ``by_pass_at_k`` samples one model ``k`` times and keeps rows whose
pass rate falls within ``[min_rate, max_rate]``.

No answer format is imposed: the prompt is the task text itself unless
``prompt_template`` is given, and each completion is graded on what
``extract_answer`` takes from it. The default, ``answer_segment``, is the text
after a closed ``<think>...</think>`` block with surrounding whitespace
stripped; pass another function, e.g. one reading an ``ANSWER:`` line, to
replace it.

Grades are stored as ``{group: [True | False | None, ...]}``, one list per
model (``"model"``, or ``"weak"`` and ``"strong"``) with one grade per sample,
where ``None`` is a grade that could not be made (an empty answer, a
``<think>`` block that never closes, ``extract_answer`` returning ``None``, or a
verifier returning ``None``) and is excluded, not failed. Completions are
stored in the same shape. Predictions and grades are cached in separate inner
ops, so changing only the rate bounds makes no model calls, and changing only
the metric regrades the cached predictions.
"""
from __future__ import annotations

import contextvars
import logging
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Union

from rouge_score import rouge_scorer

from pyforger import _tracking
from pyforger.acquire.model import Model, Prompt
from pyforger.alter.generate import PromptTemplate, _normalized_equal, _resolve_prompt
from pyforger.op import op
from pyforger.utils import get_field, get_formatted_logger

# A verifier gets the prediction (what extract_answer takes from the
# completion) and the full row, e.g. to run tests against generated code.
# None means "could not grade" (excluded, not failed).
Verifier = Callable[[str, dict], Union[bool, None]]
Metric = Union[Literal["exact_match", "rouge_l"], Verifier]


def _logger() -> logging.Logger:
    # Looked up per call: a module-level logger read inside an @op body would
    # trigger @op's "reads globals" warning.
    return get_formatted_logger("ModelDifficulty")


def answer_segment(text: str) -> str | None:
    r"""Default ``extract_answer``: the stripped text after a ``<think>`` block.

    E.g. ``"<think>try 3</think>\n 4 \n"`` gives ``"4"``. None when the block
    never closes (``"<think>try 3"``: the model ran out of tokens) or nothing
    follows it.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return None
    return text.strip() or None


@op(name="filter.by_comparative_difficulty", unkeyed=("verifier_workers",))
def by_comparative_difficulty(
    items: list[dict],
    *,
    weak_model: Model,
    strong_model: Model,
    input_field: str,
    ground_truth_field: str | None = None,
    metric: Metric = "exact_match",
    rouge_l_threshold: float = 0.7,
    extract_answer: Callable[[str], str | None] = answer_segment,
    prompt_template: PromptTemplate | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    batch_size: int = 10,
    outcomes_field: str | None = None,
    completions_field: str | None = None,
    verifier_workers: int = 8,
) -> list[dict]:
    r"""Keep rows the strong model answers correctly and the weak model does not.

    Each model makes one prediction per row. A row whose weak or strong grade is
    ``None`` is dropped: e.g. a weak model that ran out of tokens has not shown
    the row is hard.

    The prompt is the task text unless ``prompt_template`` is given, and by
    default the whole answer segment (the completion after its closed
    ``<think>`` block, stripped) is compared. So for ``"exact_match"`` the task
    or ``prompt_template`` should ask for a bare answer, or ``extract_answer``
    should pick it out.

    In a ``@step`` report the per-row score is ``strong - weak``: ``1`` (kept),
    ``0`` (both right or both wrong), ``-1`` (only the weak model right), or
    ``None`` when either grade is missing. Many ``-1`` rows suggest a wrong
    ground truth or a metric that is too strict.

    Args:
        items: Input rows.
        weak_model: Model expected to fail on hard rows.
        strong_model: Model expected to succeed on rows worth keeping.
        input_field: Key holding the task; its text is the default prompt.
        ground_truth_field: Key holding the reference answer. Required for
            ``"exact_match"`` and ``"rouge_l"``; a verifier reads the row
            itself, so it may be left out.
        metric: ``"exact_match"`` (normalized prediction equals the
            reference), ``"rouge_l"`` (ROUGE-L F1 of the prediction against the
            reference >= ``rouge_l_threshold``), or a verifier
            ``(prediction, row) -> bool | None``, e.g.
            ``filter.program_verifier("tests")``, which runs generated programs
            on the row's tests. Verifiers run ``verifier_workers`` at a time,
            so one must be safe to call from several threads.
        rouge_l_threshold: Pass threshold for ``metric="rouge_l"``.
        extract_answer: Takes the prediction from each raw completion;
            returning ``None`` grades ``None``. The default,
            ``answer_segment``, returns the text after a closed ``<think>``
            block, stripped, and ``None`` when the block never closes or
            nothing follows it. A replacement sees the thinking too; e.g.
            ``lambda t: t.rsplit("ANSWER:", 1)[-1].strip()`` keeps what
            follows the last ``ANSWER:``.
        prompt_template: Format string or ``(row) -> Prompt`` callable replacing
            the default prompt (the task text), e.g. to ask for a final
            ``ANSWER: ...`` line.
        temperature: Sampling temperature (each model's default when None).
        max_tokens: Max tokens per completion (each model's default when None).
        batch_size: Prompts per ``model.generate_batch()`` call.
        outcomes_field: When set, kept rows carry
            ``{"weak": [bool], "strong": [bool]}`` under this key, replacing any
            value already there.
        completions_field: When set, kept rows carry each model's raw
            completion as ``{"weak": [str | None], "strong": [str | None]}``
            under this key (``None`` for an empty completion).
        verifier_workers: Verifier calls run at a time when ``metric`` is a
            verifier; 1 runs them one by one.

    Returns:
        The kept rows, in input order.
    """
    _check_common(
        metric,
        rouge_l_threshold,
        batch_size,
        ground_truth_field,
        outcomes_field,
        completions_field,
        verifier_workers,
    )
    prompts = _prompts(items, input_field, prompt_template)
    grades: dict[str, list[list[bool | None]]] = {}
    completions: dict[str, list[list[str | None]]] = {}
    for group, model in (("weak", weak_model), ("strong", strong_model)):
        completions[group] = _predictions(
            prompts,
            model=model,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_size=batch_size,
        )
        grades[group] = _grades(
            items,
            completions[group],
            ground_truth_field=ground_truth_field,
            metric=metric,
            rouge_l_threshold=rouge_l_threshold,
            extract_answer=extract_answer,
            verifier_workers=verifier_workers,
        )

    kept: list[dict] = []
    gaps: list[int | None] = []
    both_right = 0
    for i, (item, weak_row, strong_row) in enumerate(
        zip(items, grades["weak"], grades["strong"])
    ):
        weak, strong = weak_row[0], strong_row[0]
        if weak is None or strong is None:
            gaps.append(None)
            continue
        gaps.append(int(strong) - int(weak))
        both_right += weak and strong
        if strong and not weak:
            kept.append(
                _annotate(
                    item,
                    outcomes_field,
                    {"weak": weak_row, "strong": strong_row},
                    completions_field,
                    {g: completions[g][i] for g in ("weak", "strong")},
                )
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


@op(name="filter.by_pass_at_k", unkeyed=("verifier_workers",))
def by_pass_at_k(
    items: list[dict],
    *,
    model: Model,
    input_field: str,
    ground_truth_field: str | None = None,
    k: int = 4,
    min_rate: float = 0.1,
    max_rate: float = 0.9,
    min_valid: int | None = None,
    metric: Metric = "exact_match",
    rouge_l_threshold: float = 0.7,
    extract_answer: Callable[[str], str | None] = answer_segment,
    prompt_template: PromptTemplate | None = None,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    batch_size: int = 10,
    outcomes_field: str | None = None,
    completions_field: str | None = None,
    verifier_workers: int = 8,
) -> list[dict]:
    """Keep rows whose pass rate over ``k`` samples is within the bounds.

    Each prompt is sent once with ``n=k``. The pass rate is passed / valid
    grades, and a row is kept when ``min_rate <= rate <= max_rate``. All
    ``k`` grades must be valid unless ``min_valid`` allows exclusions, and a
    row with no valid grade never passes.
    E.g. with ``k=4``, grades ``[True, False, None, False]`` are dropped by
    default, and kept with ``min_valid=3`` (rate 1/3).

    In a ``@step`` report the per-row score is the pass rate, or ``None`` when
    too few grades are valid.

    The prompt and answer extraction work as in ``by_comparative_difficulty``:
    by default the whole answer segment is compared.

    For STaR-style targets, ``completions_field`` keeps the sampled
    completions, aligned with the grades in ``outcomes_field``, e.g.
    ``[c for c, ok in zip(row["samples"]["model"], row["grades"]["model"])
    if ok]`` are a kept row's passing samples.

    Args:
        items: Input rows.
        model: Model to sample from.
        input_field: Key holding the task; its text is the default prompt.
        ground_truth_field: Key holding the reference answer; may be left out
            when ``metric`` is a verifier.
        k: Samples per row.
        min_rate: Lowest pass rate kept (inclusive); raise it to drop rows the
            model cannot solve.
        max_rate: Highest pass rate kept (inclusive); lower it to drop rows the
            model always solves.
        min_valid: Fewest valid grades a row needs, at most ``k``. None
            requires all ``k``.
        metric: ``"exact_match"``, ``"rouge_l"``, or a verifier
            ``(prediction, row) -> bool | None`` such as
            ``filter.program_verifier("tests")``; see
            ``by_comparative_difficulty``.
        rouge_l_threshold: Pass threshold for ``metric="rouge_l"``.
        extract_answer: Takes the prediction from each raw completion
            (default ``answer_segment``); see ``by_comparative_difficulty``.
        prompt_template: Format string or ``(row) -> Prompt`` callable replacing
            the default prompt (the task text).
        temperature: Sampling temperature; must be > 0 when ``k > 1``.
        max_tokens: Max tokens per completion (the model's default when None).
        batch_size: Prompts per ``model.generate_batch()`` call.
        outcomes_field: When set, kept rows carry ``{"model": [bool | None] * k}``
            under this key, replacing any value already there, so a later
            filter can re-select them without new model calls.
        completions_field: When set, kept rows carry the ``k`` raw completions
            as ``{"model": [str | None] * k}`` under this key, in the same
            order as the grades (``None`` for an empty completion).
        verifier_workers: Verifier calls run at a time when ``metric`` is a
            verifier; 1 runs them one by one.

    Returns:
        The kept rows, in input order.
    """
    _check_common(
        metric,
        rouge_l_threshold,
        batch_size,
        ground_truth_field,
        outcomes_field,
        completions_field,
        verifier_workers,
    )
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
        verifier_workers=verifier_workers,
    )

    kept: list[dict] = []
    rates: list[float | None] = []
    for item, row_grades, row_completions in zip(items, grades, completions):
        rate = _pass_rate(row_grades, min_valid)
        rates.append(rate)
        if rate is not None and min_rate <= rate <= max_rate:
            kept.append(
                _annotate(
                    item,
                    outcomes_field,
                    {"model": row_grades},
                    completions_field,
                    {"model": row_completions},
                )
            )
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
    """Return ``n`` completions per prompt, ``None`` for empty ones.

    Raises ``ValueError`` when the model returns a different number of results
    than prompts, or of completions than ``n`` (e.g. a provider that ignores
    ``n``): a missing sample is not an empty answer, so it is never padded.
    Exceptions propagate, so a failed call is not cached.
    """
    completions: list[list[str | None]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        results = model.generate_batch(
            batch, n=n, temperature=temperature, max_tokens=max_tokens
        )
        if len(results) != len(batch):
            raise ValueError(
                f"{model!r}.generate_batch returned {len(results)} results for "
                f"{len(batch)} prompts (prompts {start} to {start + len(batch) - 1})"
            )
        for i, result in enumerate(results):
            if len(result) != n:
                raise ValueError(
                    f"{model!r} returned {len(result)} completions for prompt "
                    f"{start + i}, but n={n} were requested"
                )
            completions.append(
                [c if isinstance(c, str) and c.strip() else None for c in result]
            )
    return completions


@op(name="filter.model_difficulty.grades", unkeyed=("verifier_workers",))
def _grades(
    items: list[dict],
    completions: list[list[str | None]],
    *,
    ground_truth_field: str | None,
    metric: Metric,
    rouge_l_threshold: float,
    extract_answer: Callable[[str], str | None],
    verifier_workers: int = 1,
) -> list[list[bool | None]]:
    """Grade each row's completions against its ground truth or a verifier.

    A completion grades ``None`` when ``extract_answer`` returns ``None`` for
    it, e.g. by default a reasoning model cut off mid-thought.
    """
    predictions = [
        [None if c is None else extract_answer(c) for c in row] for row in completions
    ]
    if callable(metric):
        return _verify(items, predictions, metric, verifier_workers)
    assert ground_truth_field is not None  # checked by _check_common
    scorer = (
        rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        if metric == "rouge_l"
        else None
    )
    out: list[list[bool | None]] = []
    for item, row in zip(items, predictions):
        reference = get_field(item, ground_truth_field, "filter.model_difficulty")
        row_grades: list[bool | None] = []
        for prediction in row:
            if prediction is None:
                row_grades.append(None)
            elif scorer is not None:
                f1 = scorer.score(str(reference), str(prediction))["rougeL"].fmeasure
                row_grades.append(f1 >= rouge_l_threshold)
            else:
                row_grades.append(_normalized_equal(prediction, reference))
        out.append(row_grades)
    return out


def _verify(
    items: list[dict],
    predictions: list[list[str | None]],
    verifier: Verifier,
    workers: int,
) -> list[list[bool | None]]:
    """Run ``verifier`` on every prediction, ``workers`` calls at a time.

    E.g. 111 rows x 4 samples of a verifier that runs tests take minutes one by
    one. Each call runs in a copy of the caller's context, so ops it calls are
    cached and reported as they would be without threads.
    """
    out: list[list[bool | None]] = [[None] * len(row) for row in predictions]
    jobs = [
        (i, j)
        for i, row in enumerate(predictions)
        for j, prediction in enumerate(row)
        if prediction is not None
    ]

    def grade(i: int, j: int) -> bool | None:
        result = verifier(predictions[i][j], items[i])  # type: ignore[arg-type]
        if result is not None and not isinstance(result, bool):
            raise TypeError(f"metric must return bool or None, got {result!r}")
        return result

    def grade_in(context: contextvars.Context, i: int, j: int) -> bool | None:
        return context.run(grade, i, j)

    if workers <= 1 or len(jobs) <= 1:
        for i, j in jobs:
            out[i][j] = grade(i, j)
        return out
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = [
            pool.submit(grade_in, contextvars.copy_context(), i, j) for i, j in jobs
        ]
        try:
            for (i, j), future in zip(jobs, futures):
                out[i][j] = future.result()
        except BaseException:
            for future in futures:
                future.cancel()  # the rest would be thrown away
            raise
    return out


def _pass_rate(grades: list[bool | None], min_valid: int | None) -> float | None:
    """Pass rate (passed / valid grades), or None if the row fails ``min_valid``.

    None when no grade is valid or fewer than ``min_valid`` (default: all) are.
    """
    valid = sum(g is not None for g in grades)
    if not valid or valid < (min_valid or len(grades)):
        return None
    return sum(g is True for g in grades) / valid


def _annotate(
    item: dict,
    outcomes_field: str | None,
    grades: dict[str, list[bool | None]],
    completions_field: str | None,
    completions: dict[str, list[str | None]],
) -> dict:
    """Return ``item``, or a copy carrying its grades and completions.

    E.g. ``{**item, "ev": {"model": [True, False]}, "samples": {"model":
    ["4", "5"]}}``; fields left as None are not added.
    """
    if outcomes_field is None and completions_field is None:
        return item
    out = dict(item)
    if outcomes_field is not None:
        out[outcomes_field] = {g: list(v) for g, v in grades.items()}
    if completions_field is not None:
        out[completions_field] = {g: list(v) for g, v in completions.items()}
    return out


def _prompts(
    items: list[dict], input_field: str, prompt_template: PromptTemplate | None
) -> list[Prompt]:
    if prompt_template is not None:
        return [_resolve_prompt(prompt_template, item) for item in items]
    return [
        str(get_field(item, input_field, "filter.model_difficulty")) for item in items
    ]


def _check_common(
    metric: Metric,
    rouge_l_threshold: float,
    batch_size: int,
    ground_truth_field: str | None,
    outcomes_field: str | None,
    completions_field: str | None,
    verifier_workers: int,
) -> None:
    if not callable(metric) and metric not in ("exact_match", "rouge_l"):
        raise ValueError(
            f"metric must be 'exact_match', 'rouge_l', or a callable, got {metric!r}"
        )
    if not callable(metric) and ground_truth_field is None:
        raise ValueError(
            f"metric={metric!r} compares with a reference answer, so "
            "ground_truth_field is required (a verifier metric does not need it)"
        )
    if not 0.0 <= rouge_l_threshold <= 1.0:
        raise ValueError(
            f"rouge_l_threshold must be in [0, 1], got {rouge_l_threshold}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if verifier_workers < 1:
        raise ValueError("verifier_workers must be at least 1")
    if outcomes_field is not None and outcomes_field == completions_field:
        raise ValueError("outcomes_field and completions_field must differ")
