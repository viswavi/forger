"""Diversity-based filter operations.

Remove duplicate and near-duplicate items by exact matching, ROUGE-L
similarity, or n-gram overlap against the kept pool.
"""
from __future__ import annotations

from pyforger import _tracking
from pyforger.filter._rouge import RougePool, RougeText
from pyforger.op import op
from pyforger.utils import get_field


@op(name="filter.by_exact_match")
def by_exact_match(
    items: list[dict],
    *,
    field: str,
    max_count: int = 1,
) -> list[dict]:
    """Keep up to ``max_count`` occurrences of each unique ``field`` value."""
    counts: dict[str, int] = {}
    occurrences: list[int] = []
    result: list[dict] = []
    for item in items:
        val = get_field(item, field, "by_exact_match")
        counts[val] = counts.get(val, 0) + 1
        occurrences.append(counts[val])
        if counts[val] <= max_count:
            result.append(item)
    _tracking.record_scores(
        "occurrence", occurrences, keep="le", threshold=float(max_count)
    )
    return result


@op(name="filter.by_rouge_l")
def by_rouge_l(
    items: list[dict],
    *,
    field: str,
    threshold: float = 0.7,
) -> list[dict]:
    """Drop items whose ROUGE-L F1 against any kept item exceeds threshold."""
    kept: list[dict] = []
    pool = RougePool()
    max_scores: list[float] = []
    for item in items:
        text = RougeText(get_field(item, field, "by_rouge_l"))
        # Max similarity to the kept pool; 0.0 for the first item.
        max_score = pool.max_f_measure(text)
        max_scores.append(max_score)
        # The first item is always kept, even with threshold <= 0.
        if not kept or max_score < threshold:
            kept.append(item)
            pool.add(text)
    _tracking.record_scores(
        "rouge_l", max_scores, keep="lt", threshold=threshold, pooled=True
    )
    return kept


@op(name="filter.by_n_gram_overlap")
def by_n_gram_overlap(
    items: list[dict],
    *,
    field: str,
    threshold: float = 0.7,
    n: int = 2,
) -> list[dict]:
    """Drop items whose word n-gram overlap with the kept pool exceeds threshold."""
    pool: set[tuple[str, ...]] = set()
    kept: list[dict] = []
    overlaps: list[float | None] = []
    for item in items:
        item_ngrams = _word_ngrams(get_field(item, field, "by_n_gram_overlap"), n)
        if not item_ngrams:
            overlaps.append(None)
            continue
        overlap = sum(1 for ng in item_ngrams if ng in pool) / len(item_ngrams)
        overlaps.append(overlap)
        if overlap >= threshold:
            continue
        kept.append(item)
        pool.update(item_ngrams)
    _tracking.record_scores(
        "overlap", overlaps, keep="lt", threshold=threshold, pooled=True
    )
    return kept


def _word_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """Extract word-level n-grams. Pads short texts with '_' so they participate."""
    words = text.split()
    if not words:
        return []
    if len(words) < n:
        words = words + ["_"] * (n - len(words))
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
