"""Agreement-based filter operations.

Keep items where two fields agree, by exact match or ROUGE-L similarity,
or groups of items whose outputs agree.
"""
from __future__ import annotations

import pickle
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
) -> list[dict]:
    """Drop every group of items whose output values disagree.

    Items are grouped by ``group_by_input_field``, and a group is kept only if
    all of its ``match_output_field`` values are identical. Useful for
    filtering (input, output) pairs where one input has conflicting outputs,
    e.g. checking self-consistency across several samples per input. Chain
    with ``by_exact_match`` to cap the rows kept per group.

    String values of both fields are stripped of surrounding whitespace.
    Values of different types (``1`` and ``"1"``) disagree.

    Args:
        items: List of input data items.
        group_by_input_field: Key whose value defines each group.
        match_output_field: Key whose value must be identical within a group.

    Returns:
        The items from groups whose outputs all agree, in their original order.
    """

    def group_key(item: dict) -> Any:
        key = get_field(item, group_by_input_field, "by_group_agreement")
        return key.strip() if isinstance(key, str) else key

    first_seen: dict[Any, bytes] = {}
    conflicting: set[Any] = set()
    for item in items:
        key = group_key(item)
        val = _canonicalize_value(
            get_field(item, match_output_field, "by_group_agreement")
        )
        if first_seen.setdefault(key, val) != val:
            conflicting.add(key)
    return [item for item in items if group_key(item) not in conflicting]


def _canonicalize_value(value: Any) -> bytes:
    """Normalize an output value the way ``@op`` keys its cache."""
    if isinstance(value, str):
        value = value.strip()
    return pickle.dumps(canonical(value), protocol=4)
