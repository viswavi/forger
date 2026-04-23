"""Agreement-based filter operations for Forger.

Filters that remove instruction-tuning examples where two fields within
each item disagree, as measured by exact matching or ROUGE-L similarity.
"""
from __future__ import annotations

from rouge_score import rouge_scorer

from pyforger.core import DataSpec, NodeType
from pyforger.utils import get_field


def by_exact_agreement(
    source: DataSpec,
    *,
    field_a: str,
    field_b: str,
    disagree: bool = False,
) -> DataSpec:
    """Keep items where two fields agree or disagree, by exact match.

    Useful for comparing outputs from different models or annotators
    on tasks that produce short, deterministic answers (numbers, labels,
    short phrases).

    Args:
        source: Upstream DataSpec providing list[dict] data.
        field_a: Key for the first value to compare.
        field_b: Key for the second value to compare.
        disagree: If False (default), keep only items where the fields match.
            If True, keep only items where they differ.

    Returns:
        A FILTER DataSpec that keeps only the selected items.
    """

    def _execute(data: list[dict]) -> list[dict]:
        return [
            item
            for item in data
            if (
                str(get_field(item, field_a, "by_exact_agreement")).strip()
                != str(get_field(item, field_b, "by_exact_agreement")).strip()
            )
            == disagree
        ]

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_exact_agreement")


def by_rouge_l_agreement(
    source: DataSpec,
    *,
    field_a: str,
    field_b: str,
    threshold: float = 0.7,
    disagree: bool = False,
) -> DataSpec:
    """Keep items where ROUGE-L F1 between two fields meets or misses threshold.

    Useful for comparing natural-language outputs from different models
    or annotators where exact matching is too strict.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        field_a: Key for the first text to compare.
        field_b: Key for the second text to compare.
        threshold: ROUGE-L F1 threshold. Items scoring at or above are
            considered "in agreement".
        disagree: If False (default), keep items that meet the threshold.
            If True, keep items that fall below it.

    Returns:
        A FILTER DataSpec that keeps only the selected items.
    """

    def _execute(data: list[dict]) -> list[dict]:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        return [
            item
            for item in data
            if (
                scorer.score(
                    get_field(item, field_a, "by_rouge_l_agreement"),
                    get_field(item, field_b, "by_rouge_l_agreement"),
                )["rougeL"].fmeasure
                < threshold
            )
            == disagree
        ]

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_rouge_l_agreement")
