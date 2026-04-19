"""LLM-as-a-judge filter operations for Forger.

Filters that use LLM evaluation to remove input-output pairs based on quality scores
or multi-judge consensus.
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
    input_field: str,
    output_field: str,
    score_threshold: int = 3,
    max_score: int = 5,
    num_judges: int = 1,
    agreement_threshold: float = 1.0,
    criteria: str = "overall quality, relevance, and correctness",
    temperature: float = 0.0,
) -> DataSpec:
    """Filter items by LLM judge score, optionally requiring multi-judge agreement.

    Each item is evaluated by ``num_judges`` independent LLM calls. An item
    is kept only when at least ``agreement_threshold`` fraction of judges
    score it at or above ``score_threshold``. The defaults
    (``num_judges=1, agreement_threshold=1.0``) reduce to single-judge
    scoring; pass ``num_judges>1`` (and consider raising ``temperature``)
    for diverse multi-judge consensus.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        model: Model instance for judge evaluation.
        input_field: Key containing the input/prompt text.
        output_field: Key containing the output/response text.
        score_threshold: Minimum score for a judge to "approve" an item.
        max_score: Maximum score the judge can assign.
        num_judges: Number of independent judge evaluations per item.
        agreement_threshold: Fraction of judges that must score at or
            above ``score_threshold`` for the item to be kept.
        criteria: Description of evaluation criteria included in the prompt.
        temperature: Sampling temperature for judge calls. Defaults to 0.0
            for deterministic single-judge scoring; raise for diverse
            multi-judge opinions.

    Returns:
        A FILTER DataSpec keeping only items with sufficient judge approval.

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
                item, input_field, output_field, criteria, max_score
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
            if approvals / num_judges >= agreement_threshold:
                kept.append(item)
        if total_parse_failures > 0:
            logger.warning(
                "by_llm_score: %d judge responses were unparseable",
                total_parse_failures,
            )
        return kept

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_llm_score")


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
    input_field: str,
    output_field: str,
    criteria: str,
    max_score: int,
) -> str:
    """Build an evaluation prompt for the judge model."""
    input_text = get_field(item, input_field, "_build_judge_prompt")
    output_text = get_field(item, output_field, "_build_judge_prompt")
    return (
        f"Rate the quality of the following output on a scale of 1 to {max_score}.\n"
        f"Criteria: {criteria}\n\n"
        f"Input: {input_text}\n\n"
        f"Output: {output_text}\n\n"
        f'Respond with JSON: {{"score": <integer>, "explanation": "<text>"}}'
    )
