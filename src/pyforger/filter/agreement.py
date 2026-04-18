"""Agreement-based filter operations for Forger.

Filters that remove instruction-tuning examples where values disagree,
either across two fields of the same item or across items sharing a group
key. Supports exact matching and ROUGE-L similarity.
"""
from __future__ import annotations

from typing import Any

from rouge_score import rouge_scorer

from pyforger.core import DataSpec, NodeType
from pyforger.utils import get_field


def by_exact_agreement(
    source: DataSpec,
    *,
    field_a: str | None = None,
    field_b: str | None = None,
    group_by: str | None = None,
    agree_on: str | None = None,
) -> DataSpec:
    """Keep items where values agree, within each item or across a group.

    Two modes, selected by which parameters are passed:

    Within-item (field_a + field_b):
        Keep items where item[field_a] == item[field_b]. Useful for
        comparing two annotations or model outputs carried on the same
        record.

    Cross-item group (group_by + agree_on):
        Group items by group_by. For each group, if all agree_on values
        are identical, keep one representative; otherwise drop the entire
        group. Useful for filtering instance pairs with conflicting
        answers for the same input.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        field_a: Within-item mode -- key for the first value to compare.
        field_b: Within-item mode -- key for the second value to compare.
        group_by: Group mode -- key to group items by.
        agree_on: Group mode -- key whose values must match within each group.

    Returns:
        A FILTER DataSpec applying the chosen agreement rule.
    """
    any_within = field_a is not None or field_b is not None
    any_group = group_by is not None or agree_on is not None
    within_item = field_a is not None and field_b is not None
    group_mode = group_by is not None and agree_on is not None
    if (any_within and any_group) or not (within_item or group_mode):
        raise ValueError(
            "Pass either (field_a, field_b) for within-item mode "
            "or (group_by, agree_on) for group mode, not both or neither."
        )

    if within_item:
        assert field_a is not None and field_b is not None
        fa, fb = field_a, field_b

        def _within_item(data: list[dict]) -> list[dict]:
            return [
                item
                for item in data
                if str(get_field(item, fa, "by_exact_agreement")).strip()
                == str(get_field(item, fb, "by_exact_agreement")).strip()
            ]

        return DataSpec(
            NodeType.FILTER, source, _within_item, "filter_by_exact_agreement"
        )

    assert group_by is not None and agree_on is not None
    gb, ao = group_by, agree_on

    def _group(data: list[dict]) -> list[dict]:
        groups: dict[Any, list[dict]] = {}
        for item in data:
            key = get_field(item, gb, "by_exact_agreement")
            groups.setdefault(key, []).append(item)
        out: list[dict] = []
        for items in groups.values():
            values = {
                str(get_field(i, ao, "by_exact_agreement")).strip() for i in items
            }
            if len(values) == 1:
                out.append(items[0])
        return out

    return DataSpec(NodeType.FILTER, source, _group, "filter_by_exact_agreement")


def by_rouge_l_agreement(
    source: DataSpec,
    *,
    field_a: str,
    field_b: str,
    threshold: float = 0.7,
) -> DataSpec:
    """Keep items where ROUGE-L F1 between two fields meets the threshold.

    Useful for comparing natural-language outputs from different models
    or annotators where exact matching is too strict.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        field_a: Key for the first text to compare.
        field_b: Key for the second text to compare.
        threshold: Minimum ROUGE-L F1 score to consider the fields
            in agreement.

    Returns:
        A FILTER DataSpec that keeps only items where the two fields
        have sufficient ROUGE-L similarity.
    """

    def _execute(data: list[dict]) -> list[dict]:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        return [
            item
            for item in data
            if scorer.score(
                get_field(item, field_a, "by_rouge_l_agreement"),
                get_field(item, field_b, "by_rouge_l_agreement"),
            )["rougeL"].fmeasure
            >= threshold
        ]

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_rouge_l_agreement")
