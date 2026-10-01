"""Agreement-based filter operations.

Keep items where two fields agree, by exact match or ROUGE-L similarity,
or groups of items whose outputs agree.
"""
from __future__ import annotations

from typing import Any

from rouge_score import rouge_scorer

from pyforger.op import op
from pyforger.utils import canonical, get_field


@op(name="filter.by_exact_agreement")
def by_exact_agreement(
    items: list[dict],
    *,
    field_a: str,
    field_b: str,
    disagree: bool = False,
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
        A filtered list of data items where the specified fields all agree (or disagree
            if `disagree=True`) by exact match.
    """
    return [
        item
        for item in items
        if (
            str(get_field(item, field_a, "by_exact_agreement")).strip()
            != str(get_field(item, field_b, "by_exact_agreement")).strip()
        )
        == disagree
    ]


@op(name="filter.by_rouge_l_agreement")
def by_rouge_l_agreement(
    items: list[dict],
    *,
    field_a: str,
    field_b: str,
    threshold: float = 0.7,
    disagree: bool = False,
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
        A filtered list of data items where the specified fields all agree (or disagree
            if `disagree=True`) by ROUGE-L F1.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return [
        item
        for item in items
        if (
            scorer.score(
                get_field(item, field_a, "by_rouge_l_agreement"),
                get_field(item, field_b, "by_rouge_l_agreement"),
            )["rougeL"].fmeasure
            < threshold
        )
        == disagree
    ]


@op(name="filter.by_group_agreement")
def by_group_agreement(
    items: list[dict],
    *,
    group_by_input_field: str,
    match_output_field: str,
    max_count: int | None = None,
) -> list[dict]:
    """Keep only groups of items whose outputs all agree.

    Values are compared by type and structure (``1 != "1"``); strings are
    stripped, and dict and set order is ignored.

    Args:
        items: List of input data items.
        group_by_input_field: Key whose value defines each group.
        match_output_field: Key whose value must match within a group.
        max_count: If set, keep at most this many items per group.

    Returns:
        Items from agreeing groups, in their original order.
    """
    keys = [
        _normalize(get_field(item, group_by_input_field, "by_group_agreement"))
        for item in items
    ]
    first_seen: dict[Any, Any] = {}
    conflicting: set[Any] = set()
    for key, item in zip(keys, items):
        val = _normalize(get_field(item, match_output_field, "by_group_agreement"))
        if first_seen.setdefault(key, val) != val:
            conflicting.add(key)

    counts: dict[Any, int] = {}
    result: list[dict] = []
    for key, item in zip(keys, items):
        if key in conflicting:
            continue
        counts[key] = counts.get(key, 0) + 1
        if max_count is None or counts[key] <= max_count:
            result.append(item)
    return result


def _normalize(value: Any) -> Any:
    """Hashable, type-tagged form of ``canonical(value)`` for comparison."""
    if isinstance(value, str):
        value = value.strip()
    return _tag(canonical(value))


def _tag(obj: Any) -> Any:
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, tuple(_tag(x) for x in obj))
    return (type(obj).__name__, obj)
