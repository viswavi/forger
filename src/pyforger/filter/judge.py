"""LLM-as-a-judge filter operations for Forger.

Filters that use LLM evaluation to remove instruction-tuning examples
based on quality scores or multi-judge consensus.
"""
from __future__ import annotations

import json

from pyforger.acquire.model import Model
from pyforger.core import DataSpec, NodeType
from pyforger.utils import get_field, get_formatted_logger

logger = get_formatted_logger("Judge")


def by_llm_score(
    source: DataSpec,
    *,
    model: Model,
    threshold: int = 3,
    max_score: int = 5,
    criteria: str = "overall quality, relevance, and correctness",
    instruction_field: str,
    response_field: str,
    temperature: float = 0.0,
) -> DataSpec:
    """Filter items by LLM judge score.

    Each item is evaluated by a single LLM call that produces a numeric
    score. Items with a score below the threshold are removed.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        model: Model instance for judge evaluation.
        threshold: Minimum score to keep an item (inclusive).
        max_score: Maximum score the judge can assign.
        criteria: Description of evaluation criteria included in the prompt.
        instruction_field: Key containing the instruction text.
        response_field: Key containing the response text.
        temperature: Sampling temperature for judge calls. Defaults to 0.0
            for deterministic scoring.

    Returns:
        A FILTER DataSpec that keeps only items scoring at or above threshold.

    Raises:
        ValueError: If threshold > max_score.
    """
    if threshold > max_score:
        raise ValueError(
            f"threshold ({threshold}) must not exceed max_score ({max_score})"
        )

    def _execute(data: list[dict]) -> list[dict]:
        kept: list[dict] = []
        parse_failures = 0
        for item in data:
            prompt = _build_judge_prompt(
                item, instruction_field, response_field, criteria, max_score
            )
            raw = model.generate(prompt, n=1, temperature=temperature)[0]
            score = _parse_judge_response(raw, max_score)
            if score is None:
                parse_failures += 1
                continue
            if score >= threshold:
                kept.append(item)
        if parse_failures > 0:
            logger.warning(
                "by_llm_score: %d/%d items dropped due to unparseable judge responses",
                parse_failures,
                len(data),
            )
        return kept

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_llm_score")


def by_llm_agreement(
    source: DataSpec,
    *,
    model: Model,
    num_judges: int = 3,
    agreement_threshold: float = 0.5,
    score_threshold: int = 3,
    max_score: int = 5,
    criteria: str = "overall quality, relevance, and correctness",
    instruction_field: str,
    response_field: str,
    temperature: float = 0.7,
) -> DataSpec:
    """Filter items by multi-judge LLM agreement.

    Each item is evaluated by multiple independent LLM calls. An item is
    kept only if at least agreement_threshold fraction of judges score it
    at or above score_threshold.

    Uses temperature > 0 by default to get diverse judge opinions.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        model: Model instance for judge evaluation.
        num_judges: Number of independent judge evaluations per item.
        agreement_threshold: Fraction of judges that must score at or
            above score_threshold for the item to be kept.
        score_threshold: Minimum score for a judge to "approve" an item.
        max_score: Maximum score the judge can assign.
        criteria: Description of evaluation criteria included in the prompt.
        instruction_field: Key containing the instruction text.
        response_field: Key containing the response text.
        temperature: Sampling temperature for judge calls. Defaults to 0.7
            for diverse opinions across judges.

    Returns:
        A FILTER DataSpec that keeps only items with sufficient judge agreement.

    Raises:
        ValueError: If num_judges < 1, agreement_threshold not in (0, 1],
            or score_threshold > max_score.
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

    def _execute(data: list[dict]) -> list[dict]:
        kept: list[dict] = []
        total_parse_failures = 0
        for item in data:
            prompt = _build_judge_prompt(
                item, instruction_field, response_field, criteria, max_score
            )
            responses = model.generate(prompt, n=num_judges, temperature=temperature)
            approvals = 0
            for raw in responses:
                score = _parse_judge_response(raw, max_score)
                if score is None:
                    total_parse_failures += 1
                    continue
                if score >= score_threshold:
                    approvals += 1
            fraction = approvals / num_judges
            if fraction >= agreement_threshold:
                kept.append(item)
        if total_parse_failures > 0:
            logger.warning(
                "by_llm_agreement: %d judge responses were unparseable",
                total_parse_failures,
            )
        return kept

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_llm_agreement")


def _parse_judge_response(raw: str, max_score: int) -> int | None:
    """Extract numeric score from a judge model JSON response.

    Returns None if the response is not valid JSON, missing a score key,
    has a non-numeric score, or the score is outside [1, max_score].

    Args:
        raw: Raw string response from the judge model.
        max_score: Maximum valid score.

    Returns:
        Parsed integer score, or None on any parse/validation failure.
    """
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
    instruction_field: str,
    response_field: str,
    criteria: str,
    max_score: int,
) -> str:
    """Build an evaluation prompt for the judge model."""
    instruction = get_field(item, instruction_field, "_build_judge_prompt")
    response = get_field(item, response_field, "_build_judge_prompt")
    return (
        f"Rate the quality of the following response on a scale of 1 to {max_score}.\n"
        f"Criteria: {criteria}\n\n"
        f"Instruction: {instruction}\n\n"
        f"Response: {response}\n\n"
        f'Respond with JSON: {{"score": <integer>, "explanation": "<text>"}}'
    )
