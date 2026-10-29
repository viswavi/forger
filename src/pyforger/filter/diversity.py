"""Diversity-based filter operations.

Remove duplicate and near-duplicate items by exact matching, ROUGE-L
similarity, n-gram overlap, or embedding similarity against the kept pool.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pyforger import _tracking
from pyforger.filter._rouge import RougePool, RougeText
from pyforger.inspect.cluster import _DEFAULT_EMBEDDING_MODEL, _resolve_embedding_model
from pyforger.inspect.dispersion import _TOL, _embedding_label, _snippet, _unit_vectors
from pyforger.op import op
from pyforger.utils import get_field

_EMBEDDING_OP = "by_embedding_similarity"
# Entries of one block of the similarity matrix (4M float32 = 16 MB).
_BLOCK_ENTRIES = 4_000_000
_NOTE_CHARS = 120  # characters of the matched row quoted in a dropped row's note


@op(name="filter.by_exact_match")
def by_exact_match(
    items: list[dict],
    *,
    field: str,
    max_count: int = 1,
) -> list[dict]:
    """Keep up to ``max_count`` occurrences of each unique ``field`` value.

    Returns:
        The kept rows, in input order and unchanged.
    """
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
    """Drop items whose ROUGE-L F1 against any kept item exceeds threshold.

    Returns:
        The kept rows, in input order and unchanged.
    """
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


def by_embedding_similarity(
    items: list[dict],
    *,
    field: str,
    threshold: float = 0.95,
    reference: list[dict] | None = None,
    reference_field: str | None = None,
    dedupe: bool = True,
    embed_fn: Callable[[list[str]], Any] | None = None,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    seed: int = 0,
    **op_controls: Any,
) -> list[dict]:
    """Drop rows whose embedding is too close to a kept row or a reference row.

    Rows are compared by the cosine similarity of ``field``'s embedding. Walking
    the rows in input order, a row is dropped when it is at least ``threshold``
    similar to any ``reference`` row (e.g. an eval set to decontaminate
    against), or, with ``dedupe``, to a row kept before it; the first copy of
    a near-duplicate is kept, as in ``by_rouge_l``. With a ``reference``,
    ``dedupe=False`` only removes rows near the reference. Rows that embed to
    zero (e.g. empty text) are 0.0 similar to everything; use
    ``by_exact_match`` for exact copies of those.

    The near-duplicates are those ``inspect.embedding_dispersion`` counts with
    the same ``duplicate_threshold``, ``embedding_model``, and ``seed``: each row
    it pairs with an earlier one is dropped, except that a row similar only to
    an already dropped row is kept (A ≈ B ≈ C with A and C apart keeps A and C).
    Numbers depend on the embedding: 0.95 counts near-verbatim copies of long
    rows under Qwen3-Embedding-0.6B; one-line rows score lower, so use about 0.9
    for them (see ``embedding_dispersion``).

    Embeddings come from ``embed_fn`` (``list[str]`` to an array of shape
    ``(len(docs), dim)``), else ``embedding_model``, as in
    ``embedding_dispersion``: a sentence-transformers model id, or ``"tfidf"``
    (the fallback, with a warning, when sentence-transformers cannot be
    imported). Rows and reference are embedded in one call, so TF-IDF shares
    one vocabulary.

    Args:
        items: The rows to filter.
        field: The field to embed; non-string values are converted with ``str``.
        threshold: Cosine similarity at or above which a row is dropped, in (0, 1].
        reference: Rows to drop near matches of; never returned.
        reference_field: The reference's field; defaults to ``field``.
        dedupe: Also drop rows near an earlier kept row. Needs a ``reference``
            when False.
        embed_fn: A custom embedder; overrides ``embedding_model``.
        embedding_model: A sentence-transformers model id, or ``"tfidf"``.
        seed: Seed for TF-IDF's SVD.
        op_controls: Per-call ``@op`` controls (``_no_cache``, ``_cache_only``,
            ``min_yield``, ``max_yield``).

    Returns:
        The kept rows, in input order and unchanged.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be a cosine similarity in (0, 1]")
    if not dedupe and reference is None:
        raise ValueError(
            "dedupe=False needs a reference; without one nothing would be dropped"
        )
    ref_texts: list[str] | None = None
    if reference is not None:
        ref_field = field if reference_field is None else reference_field
        ref_texts = [str(get_field(row, ref_field, _EMBEDDING_OP)) for row in reference]
    # Resolved before the cache lookup, so a TF-IDF result is cached as "tfidf".
    embedding_model = _resolve_embedding_model(embed_fn, embedding_model, _EMBEDDING_OP)
    return _by_embedding_similarity(
        items,
        ref_texts,
        field=field,
        threshold=threshold,
        dedupe=dedupe,
        embed_fn=embed_fn,
        embedding_model=embedding_model,
        seed=seed,
        **op_controls,
    )


@op(name="filter.by_embedding_similarity")
def _by_embedding_similarity(
    items: list[dict],
    ref_texts: list[str] | None,
    *,
    field: str,
    threshold: float,
    dedupe: bool,
    embed_fn: Callable[[list[str]], Any] | None,
    embedding_model: str,
    seed: int,
) -> list[dict]:
    """The cached body of ``by_embedding_similarity``, with the embedder resolved."""
    import numpy as np

    texts = [str(get_field(item, field, _EMBEDDING_OP)) for item in items]
    refs = ref_texts or []
    n = len(texts)
    # The score of a row with nothing to compare with, as in by_rouge_l.
    ref_best = np.zeros(n, dtype=np.float32)
    ref_arg = np.full(n, -1)
    gen: Any = None
    if n and (refs or (dedupe and n > 1)):
        vectors = _unit_vectors(
            texts + refs, embed_fn, embedding_model, seed, _EMBEDDING_OP
        )
        gen, ref = vectors[:n], vectors[n:]
        for start, end in _blocks(n, len(refs)) if refs else []:
            sims = gen[start:end] @ ref.T
            ref_arg[start:end] = sims.argmax(axis=1)
            ref_best[start:end] = sims.max(axis=1)

    # Same slack as embedding_dispersion, so a pair on the threshold agrees.
    cut = threshold - _TOL
    kept = np.zeros(n, dtype=bool)
    scores: list[float] = []
    notes: list[str | None] = []
    near_kept = near_ref = 0
    for start, end in _blocks(n, n if dedupe and gen is not None else 0):
        # Each row of the block against every row before the block's end.
        sims = gen[start:end] @ gen[:end].T if dedupe and gen is not None else None
        for j in range(start, end):
            best, match, source = -np.inf, -1, ""
            if ref_arg[j] >= 0:
                best, match, source = float(ref_best[j]), int(ref_arg[j]), "reference"
            if sims is not None and best < cut and kept[:j].any():
                prior = np.where(kept[:j], sims[j - start, :j], -np.inf)
                i = int(prior.argmax())
                if prior[i] > best:
                    best, match, source = float(prior[i]), i, "kept"
            # float32 puts a row's copy at e.g. 1.0000001.
            best = min(best, 1.0) if match >= 0 else 0.0
            if best >= cut:
                near_ref += source == "reference"
                near_kept += source == "kept"
                matched = refs[match] if source == "reference" else texts[match]
                notes.append(
                    f"cosine {best:.2f} to {source} row {match}: "
                    f'"{_snippet(matched)[:_NOTE_CHARS]}"'
                )
                # Within _TOL of the threshold, so the report agrees it dropped.
                scores.append(max(best, threshold))
            else:
                kept[j] = True
                notes.append(None)
                scores.append(best)

    summary = f"dropped: {near_kept} near a kept row"
    if ref_texts is not None:
        summary += f", {near_ref} near the reference ({len(refs)} rows)"
    summary += f" · embedding {_embedding_label(embed_fn, embedding_model)}"
    _tracking.record_scores(
        "cosine",
        scores,
        keep="lt",
        threshold=threshold,
        # Without dedupe a score is the nearest reference row's, which another
        # threshold would not change.
        pooled=dedupe,
        notes=notes,
        previews=texts,
        note=summary,
    )
    return [item for item, keep in zip(items, kept) if keep]


def _blocks(n: int, width: int) -> list[tuple[int, int]]:
    """Split ``n`` rows into ``(start, end)`` blocks of a similarity matrix.

    A block times ``width`` columns has at most ``_BLOCK_ENTRIES`` entries, and
    at least one row.
    """
    size = max(1, _BLOCK_ENTRIES // max(width, 1))
    return [(start, min(start + size, n)) for start in range(0, n, size)]


@op(name="filter.by_n_gram_overlap")
def by_n_gram_overlap(
    items: list[dict],
    *,
    field: str,
    threshold: float = 0.7,
    n: int = 2,
) -> list[dict]:
    """Drop items whose word n-gram overlap with the kept pool exceeds threshold.

    Returns:
        The kept rows, in input order and unchanged.
    """
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
