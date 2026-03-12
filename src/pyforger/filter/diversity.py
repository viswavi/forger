"""Diversity-based filter operations for Forger.

Filters that remove duplicate and near-duplicate instruction-tuning
examples based on exact matching, ROUGE-L similarity, and n-gram overlap.
"""
from __future__ import annotations

from pyforger.core import DataSpec, NodeType
from pyforger.utils import get_field


def by_exact_match(
    source: DataSpec,
    *,
    field: str,
    max_count: int = 1,
) -> DataSpec:
    """Remove items with duplicate field values.

    Keeps up to max_count occurrences of each unique value and discards
    the rest.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        field: Key in each dict to check for duplicates.
        max_count: Maximum number of items with the same field value
            to keep (e.g. 5 allows each value to appear up to 5 times).

    Returns:
        A FILTER DataSpec that keeps only items within the count limit.
    """

    def _execute(data: list[dict]) -> list[dict]:
        counts: dict[str, int] = {}
        result: list[dict] = []
        for item in data:
            val = get_field(item, field, "by_exact_match")
            counts[val] = counts.get(val, 0) + 1
            if counts[val] <= max_count:
                result.append(item)
        return result

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_exact_match")


def by_rouge_l(
    source: DataSpec,
    *,
    field: str,
    threshold: float = 0.7,
) -> DataSpec:
    """Remove items whose ROUGE-L F1 against any kept item exceeds threshold.

    Compares each item against all previously-kept items. If the ROUGE-L
    F1 score exceeds the threshold for any pair, the item is discarded.
    Default threshold of 0.7.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        threshold: ROUGE-L F1 above which an item is considered
            a near-duplicate.
        field: Key in each dict containing the text to compare.

    Returns:
        A FILTER DataSpec that keeps only sufficiently distinct items.
    """

    def _execute(data: list[dict]) -> list[dict]:
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
        for item in data:
            text = get_field(item, field, "by_rouge_l")
            is_dup = any(
                scorer.score(other, text)["rougeL"].fmeasure >= threshold
                for other in kept_texts
            )
            if not is_dup:
                kept.append(item)
                kept_texts.append(text)
        return kept

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_rouge_l")


def by_n_gram_overlap(
    source: DataSpec,
    *,
    field: str,
    threshold: float = 0.7,
    n: int = 2,
) -> DataSpec:
    """Remove items with high n-gram overlap against the kept pool.

    For each item, computes the fraction of its word n-grams that
    already appear in the pool of kept items. If that fraction exceeds
    the threshold, the item is discarded. The pool accumulates n-grams
    from all kept items.

    Args:
        source: Upstream DataSpec providing list[dict] data.
        threshold: Maximum fraction of n-grams allowed to overlap
            with the existing pool.
        n: Size of n-grams (number of words).
        field: Key in each dict containing the text to compare.

    Returns:
        A FILTER DataSpec that keeps only items with low n-gram overlap.
    """

    def _execute(data: list[dict]) -> list[dict]:
        pool: set[tuple[str, ...]] = set()
        kept: list[dict] = []
        for item in data:
            item_ngrams = _word_ngrams(get_field(item, field, "by_n_gram_overlap"), n)
            if not item_ngrams:
                continue
            overlap = sum(1 for ng in item_ngrams if ng in pool) / len(item_ngrams)
            if overlap >= threshold:
                continue
            kept.append(item)
            pool.update(item_ngrams)
        return kept

    return DataSpec(NodeType.FILTER, source, _execute, "filter_by_n_gram_overlap")
<<<<<<< HEAD
=======


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute length of the longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _rouge_l_f1(a: list[str], b: list[str]) -> float:
    """Compute ROUGE-L F1 between two word-token lists."""
    lcs = _lcs_length(a, b)
    if lcs == 0:
        return 0.0
    recall = lcs / len(b)
    precision = lcs / len(a)
    return 2 * precision * recall / (precision + recall)
>>>>>>> edb1680 (Black format fixes)


def _word_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """Extract word-level n-grams from text.

    When the text has fewer than n words, pads with "_" placeholders
    so short texts still participate in overlap checking.
    """
    words = text.split()
    if not words:
        return []
    if len(words) < n:
        words = words + ["_"] * (n - len(words))
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
