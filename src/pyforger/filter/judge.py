"""LLM-as-a-judge filter.

Each item is rated by N independent judge calls; kept iff at least
``agreement_threshold`` of judges score it ≥ ``score_threshold``.
"""
from __future__ import annotations

import json

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
    temperature: float = 1.0,
) -> list[dict]:
    """Keep items where ``num_judges`` LLM judges sufficiently agree on quality."""
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

    kept: list[dict] = []
    parse_failures = 0
    for item in items:
        prompt = _build_judge_prompt(
            item, input_field, output_field, criteria, max_score
        )
        responses = model.generate(prompt, n=num_judges, temperature=temperature)
        approvals = 0
        for raw in responses:
            score = _parse_judge_response(raw, max_score)
            if score is None:
                parse_failures += 1
                continue
            if score >= score_threshold:
                approvals += 1
        if approvals / num_judges >= agreement_threshold:
            kept.append(item)
    if parse_failures:
        logger.warning("by_llm_score: %d judge responses unparseable", parse_failures)
    return kept


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
