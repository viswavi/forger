"""LLM-as-a-judge filter.

Each item is rated by N independent judge calls; kept iff at least
``agreement_threshold`` of judges score it ≥ ``score_threshold``.
"""
from __future__ import annotations

import json

from pyforger import _tracking
from pyforger.acquire.model import Model
from pyforger.op import op
from pyforger.utils import get_field, get_formatted_logger

logger = get_formatted_logger("Judge")


@op(name="filter.by_llm_score")
def by_llm_score(
    items: list[dict],
    *,
    model: Model,
    input_field: str,
    output_field: str,
    score_threshold: int = 3,
    max_score: int = 5,
    num_judges: int = 1,
    agreement_threshold: float = 1.0,
    criteria: str = "overall quality, relevance, and correctness",
    temperature: float = 0.0,
) -> list[dict]:
    """Keep items where ``num_judges`` LLM judges sufficiently agree on quality.

    Judge scores are cached separately from the thresholds, so re-running with a
    different ``score_threshold`` or ``agreement_threshold`` makes no new calls.
    ``_no_cache=True`` also re-asks the judges.

    In a ``@step`` report the per-row score is the judge's score when
    ``num_judges == 1``, and otherwise the fraction of judges scoring at least
    ``score_threshold`` (compared with ``agreement_threshold``).
    """
    if num_judges < 1:
        raise ValueError("num_judges must be at least 1")
    if not (0.0 < agreement_threshold <= 1.0):
        raise ValueError("agreement_threshold must be in (0, 1]")
    if score_threshold > max_score:
        raise ValueError(
            f"score_threshold ({score_threshold}) must not exceed "
            f"max_score ({max_score})"
        )
    if num_judges > 1 and temperature == 0.0:
        raise ValueError(
            "temperature must be > 0.0 when num_judges > 1; "
            "deterministic sampling defeats multi-judge consensus"
        )

    all_scores = _judge_scores(
        items,
        model=model,
        input_field=input_field,
        output_field=output_field,
        criteria=criteria,
        max_score=max_score,
        num_judges=num_judges,
        temperature=temperature,
    )
    kept: list[dict] = []
    # One judge: its score. Several: the fraction of judges approving, which is
    # what agreement_threshold is compared with. None when nothing parsed.
    row_scores: list[float | int | None] = []
    parse_failures = 0
    for item, scores in zip(items, all_scores):
        parsed = [score for score in scores if score is not None]
        parse_failures += len(scores) - len(parsed)
        approvals = sum(1 for score in parsed if score >= score_threshold)
        if approvals / num_judges >= agreement_threshold:
            kept.append(item)
        if not parsed:
            row_scores.append(None)
        elif num_judges == 1:
            row_scores.append(parsed[0])
        else:
            row_scores.append(approvals / num_judges)
    if parse_failures:
        logger.warning("by_llm_score: %d judge responses unparseable", parse_failures)
    if num_judges == 1:
        _tracking.record_scores(
            "score", row_scores, keep="ge", threshold=float(score_threshold)
        )
    else:
        _tracking.record_scores(
            f"approval >={score_threshold}",
            row_scores,
            keep="ge",
            threshold=agreement_threshold,
        )
    return kept


@op(name="filter.by_llm_score.scores")
def _judge_scores(
    items: list[dict],
    *,
    model: Model,
    input_field: str,
    output_field: str,
    criteria: str,
    max_score: int,
    num_judges: int,
    temperature: float,
) -> list[list[int | None]]:
    """Return each item's ``num_judges`` parsed scores (None when unparseable)."""
    all_scores: list[list[int | None]] = []
    for item in items:
        prompt = _build_judge_prompt(
            item, input_field, output_field, criteria, max_score
        )
        responses = model.generate(prompt, n=num_judges, temperature=temperature)
        all_scores.append([_parse_judge_response(raw, max_score) for raw in responses])
    return all_scores


def _parse_judge_response(raw: str, max_score: int) -> int | None:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    score = parsed.get("score")
    if not isinstance(score, (int, float)):
        return None
    int_score = int(score)
    if int_score < 1 or int_score > max_score:
        return None
    return int_score


def _build_judge_prompt(
    item: dict,
    input_field: str,
    output_field: str,
    criteria: str,
    max_score: int,
) -> str:
    input_text = get_field(item, input_field, "_build_judge_prompt")
    output_text = get_field(item, output_field, "_build_judge_prompt")
    return (
        f"Rate the quality of the following output on a scale of 1 to {max_score}.\n"
        f"Criteria: {criteria}\n\n"
        f"Input: {input_text}\n\n"
        f"Output: {output_text}\n\n"
        f'Respond with JSON: {{"score": <integer>, "explanation": "<text>"}}'
    )
