"""Agreement-based filter operations.

Keep items where two fields agree, by exact match or ROUGE-L similarity.
"""
from __future__ import annotations

from rouge_score import rouge_scorer

from pyforger.op import op
from pyforger.utils import get_field


@op(name="filter.by_exact_agreement")
def by_exact_agreement(
    items: list[dict],
    *,
    field_a: str,
    field_b: str,
) -> list[dict]:
    """Keep items where two fields agree or disagree, by exact match.

    Useful for comparing outputs from different models or annotators
    on tasks that produce short, deterministic answers (numbers, labels,
    short phrases).

    Args:
        items: List of input data items.
        field_a: Key for the first value to compare.
        field_b: Key for the second value to compare.
        disagree: If False (default), keep only items where the fields match.
            If True, keep only items where they differ.

    Returns:
        A filtered list of data items where the specified fields all agree.
    """
    return [
        item
        for item in items
        if str(get_field(item, field_a, "by_exact_agreement")).strip()
        == str(get_field(item, field_b, "by_exact_agreement")).strip()
    ]


@op(name="filter.by_rouge_l_agreement")
def by_rouge_l_agreement(
    items: list[dict],
    *,
    field_a: str,
    field_b: str,
    threshold: float = 0.7,
) -> list[dict]:
    """Keep items where ROUGE-L F1 between two fields meets or misses threshold.

    Useful for comparing natural-language outputs from different models
    or annotators where exact matching is too strict.

    Args:
        items: List of input data items.
        field_a: Key for the first text to compare.
        field_b: Key for the second text to compare.
        threshold: ROUGE-L F1 threshold. Items scoring at or above are
            considered "in agreement".
        disagree: If False (default), keep only items that meet the threshold.
            If True, keep only the items that fall below it.

    Returns:
        A filtered list of data items where the specified fields all agree.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return [
        item
        for item in items
        if scorer.score(
            get_field(item, field_a, "by_rouge_l_agreement"),
            get_field(item, field_b, "by_rouge_l_agreement"),
        )["rougeL"].fmeasure
        >= threshold
    ]
