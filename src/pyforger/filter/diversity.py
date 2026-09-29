"""Diversity-based filter operations.

Remove duplicate and near-duplicate items by exact matching, ROUGE-L
similarity, or n-gram overlap against the kept pool.
"""
from __future__ import annotations

from pyforger.op import op
from pyforger.utils import get_field


@op(name="filter.by_exact_match")
def by_exact_match(
    items: list[dict],
    *,
    field: str,
    max_count: int = 1,
    agree_on: str | None = None,
) -> list[dict]:
    """Keep up to ``max_count`` occurrences of each unique ``field`` value.

    When ``agree_on`` is set, also drop every item whose ``field`` group
    contains disagreeing ``agree_on`` values (useful for filtering
    (input, output) pairs where the same input has conflicting outputs).
    Both keys are compared after stripping surrounding whitespace.
    """
    if agree_on is None:
        counts: dict[str, int] = {}
        result: list[dict] = []
        for item in items:
            val = get_field(item, field, "by_exact_match")
            counts[val] = counts.get(val, 0) + 1
            if counts[val] <= max_count:
                result.append(item)
        return result

    groups: dict[str, list[dict]] = {}
    for item in items:
        key = str(get_field(item, field, "by_exact_match")).strip()
        groups.setdefault(key, []).append(item)
    result = []
    for group in groups.values():
        values = {str(get_field(i, agree_on, "by_exact_match")).strip() for i in group}
        if len(values) == 1:
            result.extend(group[:max_count])
    return result


@op(name="filter.by_rouge_l")
def by_rouge_l(
    items: list[dict],
    *,
    field: str,
    threshold: float = 0.7,
) -> list[dict]:
    """Drop items whose ROUGE-L F1 against any kept item exceeds threshold."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score is required for by_rouge_l. "
            "Install it with: pip install rouge-score"
        ) from None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    kept: list[dict] = []
    kept_texts: list[str] = []
    for item in items:
        text = get_field(item, field, "by_rouge_l")
        is_dup = any(
            scorer.score(other, text)["rougeL"].fmeasure >= threshold
            for other in kept_texts
        )
        if not is_dup:
            kept.append(item)
            kept_texts.append(text)
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
    for item in items:
        item_ngrams = _word_ngrams(get_field(item, field, "by_n_gram_overlap"), n)
        if not item_ngrams:
            continue
        overlap = sum(1 for ng in item_ngrams if ng in pool) / len(item_ngrams)
        if overlap >= threshold:
            continue
        kept.append(item)
        pool.update(item_ngrams)
    return kept


def _word_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """Extract word-level n-grams. Pads short texts with '_' so they participate."""
    words = text.split()
    if not words:
        return []
    if len(words) < n:
        words = words + ["_"] * (n - len(words))
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
