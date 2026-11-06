"""How spread out rows are in embedding space, relative to a reference set.

Coverage and off-distribution share follow the k-NN balls of Naeem et al. 2020
("Reliable Fidelity and Diversity Metrics for Generative Models"), and the
diversity number is the Vendi score of Friedman & Dieng 2023.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.text import Text

from pyforger._render import WIDTH, _clean, _console, _emit, _plain, _rule
from pyforger.inspect.cluster import (
    _DEFAULT_EMBEDDING_MODEL,
    _embed,
    _resolve_embedding_model,
)
from pyforger.op import op
from pyforger.utils import get_field

_OP = "embedding_dispersion"
_EXAMPLE_CHARS = 300  # characters of an example row kept in the report
_CHUNK = 1024  # rows per block of the similarity matrix
_PAIR_CANDIDATES = 50  # most similar pairs kept per block, for the examples
_TOL = 1e-6  # float slack when a row sits exactly on a k-NN radius
# An example at least this similar to one already listed is skipped, so the
# examples show different gaps rather than three rows of one template.
_EXAMPLE_OVERLAP = 0.5
# A row is "near the reference" when it is more similar to its nearest
# reference row than this share of reference rows are to their nearest other
# one: about 10% of rows drawn like the reference are. Flagged at 25%, as
# compare_to_reference flags 25% outside the reference's 5th-95th percentiles.
_NEAR_QUANTILE = 0.9
_NEAR_FLAG = 0.25
# Flagged like the near-reference share: a quarter of rows with a near-copy.
_DUPLICATE_FLAG = 0.25


class DispersionExample(BaseModel):
    """One example row.

    ``index`` is its position in its own list (``rows`` or ``reference``),
    ``text`` its value cut to 300 characters, and ``similarity`` its cosine
    similarity to the nearest row of the other set.
    """

    index: int
    text: str
    similarity: float


class NearDuplicatePair(BaseModel):
    """Two rows at or above the near-duplicate threshold.

    ``first`` and ``second`` index ``rows``; texts are cut to 300 characters.
    """

    first: int
    second: int
    similarity: float
    first_text: str
    second_text: str


class NearestReference(BaseModel):
    """A reference row and the rows near it.

    ``rows`` rows are nearer to reference row ``index`` than 90% of reference
    rows are to their nearest other one, and it is their nearest reference row.
    ``example_index`` (in ``rows``) is the most similar of them, at
    ``similarity``. Texts are cut to 300 characters.
    """

    index: int
    text: str
    rows: int
    similarity: float
    example_index: int
    example_text: str


class DispersionReport(BaseModel):
    """What ``embedding_dispersion`` measured; ``model_dump()`` gives a dict.

    ``embedded`` (``reference_embedded``) rows of ``n_rows`` (``n_reference``)
    were embedded, a random sample when there were more than ``max_rows``.
    ``compared`` is the number of rows of each set used for coverage and both
    Vendi scores: equal-size random samples, since both grow with the number
    of rows. Off-distribution share, near-reference share, and near-duplicates
    use every embedded row. Without a reference, ``compared`` is ``embedded``
    and the reference fields are None.

    ``nearest_p50``/``nearest_p90`` are percentiles of each row's similarity
    to its nearest reference row; ``reference_nearest_p50``/``_p90`` the same
    for each reference row's nearest other reference row. ``near_reference``
    is the share of rows above ``reference_nearest_p90`` (about 0.10 for rows
    drawn like the reference), ``near_reference_targets`` how many reference
    rows those rows are nearest to, ``nearest_references`` the reference rows
    with the most of them, and ``reference_copies`` the rows at or above
    ``duplicate_threshold`` to some reference row.
    """

    field: str
    embedding: str
    n_rows: int
    embedded: int
    compared: int
    n_reference: int | None = None
    reference_embedded: int | None = None
    k: int
    coverage: float | None = None
    uncovered: list[DispersionExample] = Field(default_factory=list)
    off_distribution: float | None = None
    off_distribution_rows: list[DispersionExample] = Field(default_factory=list)
    nearest_p50: float | None = None
    nearest_p90: float | None = None
    reference_nearest_p50: float | None = None
    reference_nearest_p90: float | None = None
    near_reference: float | None = None
    near_reference_rows: int = 0
    near_reference_targets: int = 0
    nearest_references: list[NearestReference] = Field(default_factory=list)
    reference_copies: int = 0
    duplicate_threshold: float
    near_duplicate_pairs: int = 0
    near_duplicate_rows: int = 0
    # Rows linked by near-duplicate pairs, directly or through other rows,
    # form a group: e.g. 40 instances of one template are one group of 40.
    near_duplicate_groups: int = 0
    largest_group: int = 0
    exact_duplicates: int = 0
    near_duplicates: list[NearDuplicatePair] = Field(default_factory=list)
    vendi: float
    reference_vendi: float | None = None
    notes: list[str] = Field(default_factory=list)


def embedding_dispersion(
    rows: list[dict],
    field: str,
    *,
    reference: list[dict] | None = None,
    reference_field: str | None = None,
    embed_fn: Callable[[list[str]], Any] | None = None,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    k: int = 5,
    duplicate_threshold: float = 0.95,
    max_rows: int = 2000,
    examples: int = 3,
    seed: int = 0,
    **op_controls: Any,
) -> DispersionReport:
    """Print and return how spread out one field's values are in embedding space.

    With a ``reference`` (e.g. retrieved or human-written rows of the kind you
    want), reports *coverage*: the share of reference rows that have a row of
    ``rows`` within their k-NN radius (the distance to their ``k``-th nearest
    reference row), and *off-distribution share*: the share of ``rows`` outside
    every reference row's radius, each with the farthest examples. It also
    reports the *near-reference share*: the share of ``rows`` more similar to
    their nearest reference row than 90% of reference rows are to their
    nearest other one. Rows drawn like the reference score about 0.10; rows
    that restate or vary individual reference rows score higher, and the
    report lists the reference rows they cluster on, each beside its closest
    row. Always
    reports near-duplicate pairs (cosine similarity at least
    ``duplicate_threshold``), the groups they link rows into (e.g. "415 rows
    form 13 groups, largest 40: about 13 distinct rows", flagged when a
    quarter of rows have a near-copy), and the Vendi score, the effective number of
    distinct rows (from 1 when all are the same up to the number of rows),
    for ``rows`` and the reference on equal-size random samples. Numbers
    depend on the embedding: compare datasets under the same embedding.
    Rows are not modified.

    Embeddings come from ``embed_fn`` (``list[str]`` to an array of shape
    ``(len(docs), dim)``), else ``embedding_model``, as in
    ``topic_based_clustering``: a sentence-transformers model id, or
    ``"tfidf"`` for TF-IDF + SVD on the CPU (the fallback, with a warning,
    when sentence-transformers cannot be imported). Both sets are embedded
    in one call, so TF-IDF shares one vocabulary (and its numbers for the same
    rows change with the reference).

    Args:
        rows: The rows to measure.
        field: The field to embed; non-string values are converted with ``str``.
        reference: Rows to compare against.
        reference_field: The reference's field; defaults to ``field``.
        embed_fn: A custom embedder; overrides ``embedding_model``.
        embedding_model: A sentence-transformers model id, or ``"tfidf"``.
        k: Neighbors that set a reference row's radius (Naeem et al. use 5).
        duplicate_threshold: Cosine similarity at or above which two rows (or
            a row and a reference row) are near-duplicates. 0.95 counts
            near-verbatim copies: with Qwen3-Embedding-0.6B, a 220-word coding
            problem with a few numbers or one sentence changed scores 0.95 to
            1.00, and distinct problems rarely reach 0.95. Short rows score
            lower: a one-line math problem with one number changed scores 0.93
            (median; 0.83 under TF-IDF), so lower it (e.g. 0.9) for short rows.
        max_rows: Embed at most this many rows of each set (a random sample).
        examples: Example rows (or pairs) listed per metric.
        seed: Seed for the samples and for TF-IDF's SVD.
        op_controls: Per-call ``@op`` controls (``_no_cache``, ``_cache_only``).

    Returns:
        A ``DispersionReport``, also printed.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if max_rows < 2:
        raise ValueError("max_rows must be at least 2")
    if not -1.0 <= duplicate_threshold <= 1.0:
        raise ValueError("duplicate_threshold must be a cosine similarity in [-1, 1]")
    texts = [str(get_field(row, field, _OP)) for row in rows]
    if len(texts) < 2:
        raise ValueError(f"{_OP} needs at least 2 rows, got {len(texts)}")
    ref_texts = None
    if reference is not None:
        ref_field = field if reference_field is None else reference_field
        ref_texts = [str(get_field(row, ref_field, _OP)) for row in reference]
        if len(ref_texts) < 2:
            raise ValueError(
                f"{_OP} needs at least 2 reference rows, got {len(ref_texts)}"
            )
    # Resolved before the cache lookup, so a TF-IDF result is cached as "tfidf".
    embedding_model = _resolve_embedding_model(embed_fn, embedding_model, _OP)
    report = _embedding_dispersion(
        texts,
        ref_texts,
        field=field,
        embed_fn=embed_fn,
        embedding_model=embedding_model,
        k=k,
        duplicate_threshold=duplicate_threshold,
        max_rows=max_rows,
        examples=examples,
        seed=seed,
        **op_controls,
    )
    console = _console()
    _emit(_report_renderable(report, width=console.width), console)
    return report


@op(name="inspect.embedding_dispersion")
def _embedding_dispersion(
    texts: list[str],
    ref_texts: list[str] | None,
    *,
    field: str,
    embed_fn: Callable[[list[str]], Any] | None,
    embedding_model: str,
    k: int,
    duplicate_threshold: float,
    max_rows: int,
    examples: int,
    seed: int,
) -> DispersionReport:
    """The cached body of ``embedding_dispersion``, with the embedder resolved."""
    import numpy as np

    rng = np.random.default_rng(seed)
    # A random order; the first `compared` rows of each are the comparison sample.
    gen_idx = rng.permutation(len(texts))[:max_rows]
    refs = ref_texts or []
    ref_idx = rng.permutation(len(refs))[:max_rows]
    docs = [texts[i] for i in gen_idx] + [refs[i] for i in ref_idx]
    vectors = _unit_vectors(docs, embed_fn, embedding_model, seed, _OP)
    gen = vectors[: len(gen_idx)]

    report = DispersionReport(
        field=field,
        embedding=_embedding_label(embed_fn, embedding_model),
        n_rows=len(texts),
        embedded=len(gen_idx),
        compared=len(gen_idx),
        k=k,
        duplicate_threshold=duplicate_threshold,
        vendi=0.0,
    )
    _near_duplicates(report, gen, gen_idx, texts, examples)
    if len(gen_idx) < len(texts):
        report.notes.append(
            f"near-duplicates are counted among {len(gen_idx):,} sampled rows"
        )

    if ref_texts is None:
        report.vendi = round(_vendi(gen), 4)
        return report

    ref = vectors[len(gen_idx) :]
    m = min(len(gen), len(ref))
    report.n_reference = len(refs)
    report.reference_embedded = len(ref_idx)
    report.compared = m
    report.vendi = round(_vendi(gen[:m]), 4)
    report.reference_vendi = round(_vendi(ref[:m]), 4)
    if m <= k:
        report.notes.append(
            f"coverage needs more than k={k} rows of each set; compared {m}"
        )
    else:
        # Equal sizes: with fewer rows than the reference, coverage is lower
        # even for rows drawn from the reference's own distribution.
        covered, best = _inside(ref[:m], gen[:m], k, axis=0)
        report.coverage = round(float(covered.mean()), 4)
        report.uncovered = _examples(
            covered, best, ref[:m], gen[:m], ref_idx, refs, examples
        )
    if len(ref) > k:
        # Every embedded row: the more reference rows, the tighter the balls.
        inside, best = _inside(ref, gen, k, axis=1)
        report.off_distribution = round(float(1.0 - inside.mean()), 4)
        report.off_distribution_rows = _examples(
            inside, best, gen, ref, gen_idx, texts, examples
        )
    _near_reference(report, gen, ref, gen_idx, ref_idx, texts, refs, examples)
    return report


def _near_reference(
    report: DispersionReport,
    gen: Any,
    ref: Any,
    gen_idx: Any,
    ref_idx: Any,
    texts: list[str],
    refs: list[str],
    examples: int,
) -> None:
    """How many rows sit closer to one reference row than reference rows do.

    A reference row's nearest other reference row is the baseline: a row
    drawn like the reference is as similar to its nearest reference row as a
    reference row is to its nearest other one (one candidate fewer), so about
    10% of such rows exceed the reference's 90th percentile.
    """
    import numpy as np

    ref_sims = ref @ ref.T
    np.fill_diagonal(ref_sims, -np.inf)
    ref_best = ref_sims.max(axis=1)
    best = np.empty(len(gen), dtype=np.float32)
    nearest = np.empty(len(gen), dtype=np.int64)
    for start in range(0, len(gen), _CHUNK):
        sims = gen[start : start + _CHUNK] @ ref.T
        nearest[start : start + _CHUNK] = sims.argmax(axis=1)
        best[start : start + _CHUNK] = sims.max(axis=1)
    line = float(np.quantile(ref_best, _NEAR_QUANTILE))
    near = best > line + _TOL
    report.nearest_p50 = round(float(np.quantile(best, 0.5)), 4)
    report.nearest_p90 = round(float(np.quantile(best, 0.9)), 4)
    report.reference_nearest_p50 = round(float(np.quantile(ref_best, 0.5)), 4)
    report.reference_nearest_p90 = round(line, 4)
    report.near_reference = round(float(near.mean()), 4)
    report.near_reference_rows = int(near.sum())
    report.reference_copies = int((best >= report.duplicate_threshold - _TOL).sum())
    counts = np.bincount(nearest[near], minlength=len(ref))
    report.near_reference_targets = int((counts > 0).sum())
    for j in np.argsort(-counts, kind="stable")[:examples]:
        if counts[j] == 0:
            break
        members = np.flatnonzero(near & (nearest == j))
        top = int(members[np.argmax(best[members])])
        report.nearest_references.append(
            NearestReference(
                index=int(ref_idx[j]),
                text=_snippet(refs[int(ref_idx[j])]),
                rows=int(counts[j]),
                similarity=round(min(float(best[top]), 1.0), 4),
                example_index=int(gen_idx[top]),
                example_text=_snippet(texts[int(gen_idx[top])]),
            )
        )


def _unit_vectors(
    docs: list[str],
    embed_fn: Callable[[list[str]], Any] | None,
    embedding_model: str,
    seed: int,
    op_name: str,
) -> Any:
    """Embed ``docs`` in one call: one float32 unit vector per doc, in order.

    A text repeated in ``docs`` is embedded once, so TF-IDF sees each text
    once. Cosine similarity is then a dot product; an all-zero vector (e.g.
    empty text) stays zero, so it is 0.0 similar to everything.
    ``filter.by_embedding_similarity`` uses it too, so both ops embed alike.
    """
    import numpy as np

    unique = list(dict.fromkeys(docs))
    vectors = _unit_rows(
        np.asarray(
            _embed(unique, embed_fn, embedding_model, seed, op_name=op_name),
            dtype=np.float32,
        ),
        op_name,
    )
    if vectors.shape[0] != len(unique):
        raise ValueError(
            f"{op_name}: the embedder returned {vectors.shape[0]} vectors "
            f"for {len(unique)} documents"
        )
    position = {doc: i for i, doc in enumerate(unique)}
    return vectors[[position[doc] for doc in docs]]


def _unit_rows(vectors: Any, op_name: str = _OP) -> Any:
    """Scale each row to length 1; all-zero rows (e.g. empty text) stay zero."""
    import numpy as np

    if vectors.ndim != 2:
        raise ValueError(
            f"{op_name}: embeddings must have shape (n_docs, dim), got {vectors.shape}"
        )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def _embedding_label(
    embed_fn: Callable[[list[str]], Any] | None, embedding_model: str
) -> str:
    """Name the embedder, e.g. "tfidf" or "embed_fn my_embed"."""
    if embed_fn is None:
        return embedding_model
    name = getattr(embed_fn, "__qualname__", None) or type(embed_fn).__name__
    return f"embed_fn {name}"


def _snippet(text: str) -> str:
    return " ".join(text.split())[:_EXAMPLE_CHARS]


def _near_duplicates(
    report: DispersionReport,
    gen: Any,
    gen_idx: Any,
    texts: list[str],
    examples: int,
) -> None:
    """Count pairs at or above the threshold, block by block to bound memory."""
    import numpy as np

    n = len(gen)
    # Rows with the same text share an id; their pairs are exact copies.
    ids: dict[str, int] = {}
    doc = np.array([ids.setdefault(texts[i], len(ids)) for i in gen_idx])
    threshold = report.duplicate_threshold - _TOL
    in_pair = np.zeros(n, dtype=bool)
    parent = list(range(n))  # union-find over near-duplicate pairs

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = 0
    candidates: list[tuple[bool, float, int, int]] = []  # (exact, -sim, a, b)
    for start in range(0, n, _CHUNK):
        sims = gen[start : start + _CHUNK] @ gen.T
        rows_i, cols = np.nonzero(sims >= threshold)
        rows_i = rows_i + start
        upper = cols > rows_i  # each pair once, not a row with itself
        rows_i, cols = rows_i[upper], cols[upper]
        pairs += len(rows_i)
        in_pair[rows_i] = True
        in_pair[cols] = True
        for a, b in zip(rows_i.tolist(), cols.tolist()):
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[ra] = rb
        values = sims[rows_i - start, cols]
        exact = doc[rows_i] == doc[cols]
        # Copies with edits first: exact copies are counted on their own.
        for keep in (~exact, exact):
            top = np.flatnonzero(keep)[np.argsort(-values[keep], kind="stable")]
            candidates += [
                (bool(exact[t]), -float(values[t]), int(rows_i[t]), int(cols[t]))
                for t in top[:_PAIR_CANDIDATES]
            ]
    report.near_duplicate_pairs = pairs
    report.near_duplicate_rows = int(in_pair.sum())
    sizes: dict[int, int] = {}
    for i in np.flatnonzero(in_pair).tolist():
        r = root(i)
        sizes[r] = sizes.get(r, 0) + 1
    report.near_duplicate_groups = len(sizes)
    report.largest_group = max(sizes.values(), default=0)
    sampled = [texts[i] for i in gen_idx]
    report.exact_duplicates = len(sampled) - len(set(sampled))
    seen: set[tuple[str, str]] = set()
    for _, neg_sim, a, b in sorted(candidates):
        sim = -neg_sim
        if len(report.near_duplicates) >= examples:
            break
        first, second = sorted((int(gen_idx[a]), int(gen_idx[b])))
        key = (texts[first], texts[second])
        if key in seen:
            continue
        seen.add(key)
        report.near_duplicates.append(
            NearDuplicatePair(
                first=first,
                second=second,
                similarity=round(min(sim, 1.0), 4),
                first_text=_snippet(texts[first]),
                second_text=_snippet(texts[second]),
            )
        )


def _inside(ref: Any, gen: Any, k: int, *, axis: int) -> tuple[Any, Any]:
    """Which rows fall inside the reference rows' k-NN balls, and by what margin.

    A reference row's radius is the cosine similarity to its k-th most similar
    other reference row; a row is inside its ball when at least that similar.
    ``axis=0``: per reference row, whether some generated row is inside its
    ball. ``axis=1``: per generated row, whether it is inside some ball. The
    margin (best similarity minus radius) orders the examples, farthest first.
    A radius is at least just above 0, so a row with nothing in common with a
    reference row (cosine 0, e.g. no shared words under TF-IDF) is never in its
    ball, even when that row's k-th neighbor also shares nothing with it.
    """
    import numpy as np

    ref_sims = ref @ ref.T
    np.fill_diagonal(ref_sims, -np.inf)
    radius = -np.partition(-ref_sims, k - 1, axis=1)[:, k - 1]
    radius = np.maximum(radius, 2 * _TOL)
    best = (gen @ ref.T - radius[None, :]).max(axis=axis) + _TOL
    return best >= 0, best


def _examples(
    hit: Any,
    best: Any,
    own: Any,
    other: Any,
    idx: Any,
    texts: list[str],
    examples: int,
) -> list[DispersionExample]:
    """The rows that missed (``~hit``), farthest outside first, unlike each other."""
    import numpy as np

    out: list[DispersionExample] = []
    listed: list[int] = []
    for j in np.argsort(best, kind="stable"):
        if hit[j] or len(out) >= examples:
            break
        if listed and float((own[listed] @ own[j]).max()) >= _EXAMPLE_OVERLAP:
            continue
        listed.append(int(j))
        i = int(idx[j])
        out.append(
            DispersionExample(
                index=i,
                text=_snippet(texts[i]),
                similarity=round(float((other @ own[j]).max()), 4),
            )
        )
    return out


def _vendi(vectors: Any) -> float:
    """Vendi score with the cosine kernel.

    That is exp of the entropy of the eigenvalues of K / n, where K is the
    rows' cosine-similarity matrix. K = X X^T shares its nonzero eigenvalues
    with X^T X, so the smaller of the two is decomposed (at most dim x dim).
    """
    import numpy as np

    x = np.asarray(vectors, dtype=np.float64)
    gram = x.T @ x if x.shape[1] < x.shape[0] else x @ x.T
    eig = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
    total = eig.sum()
    if total <= 0:  # every row embedded to zero: nothing tells them apart
        return 1.0
    p = eig[eig > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def render_dispersion_report(report: DispersionReport) -> str:
    """Render an ``embedding_dispersion`` report as plain text."""
    return _plain(_report_renderable(report, width=WIDTH))


def _quote(text: str, chars: int) -> str:
    text = _clean(text)
    return f'"{text}"' if len(text) <= chars else f'"{text[: chars - 1]}…"'


def _metric(name: str, value: str, detail: str) -> Text:
    line = Text()
    line.append(name, style="bold")
    line.append(f" {value}")
    line.append(f"  {detail}", style="dim")
    return line


def _example_lines(
    label: str, rows: list[DispersionExample], room: int
) -> list[RenderableType]:
    """One line per example: label, text, similarity to the nearest other row."""
    lines: list[RenderableType] = []
    for ex in rows:
        line = Text(f"{label}: ", style="dim")
        line.append(_quote(ex.text, room - len(label) - 14))
        line.append(f"  nearest {ex.similarity:.2f}", style="dim")
        lines.append(Padding(line, (0, 0, 0, 4)))
    return lines


def _near_reference_lines(report: DispersionReport, room: int) -> list[RenderableType]:
    """The near-reference share, its percentiles, and where the rows cluster."""
    n = report.embedded
    share = report.near_reference or 0.0
    flag = "! " if share >= _NEAR_FLAG else ""
    targets = report.near_reference_targets
    lines: list[RenderableType] = [
        _metric(
            f"{flag}near reference",
            f"{share:.2f}",
            f"{report.near_reference_rows:,}/{n:,} rows, on {targets:,} reference "
            f"row{'s' if targets != 1 else ''} (about 0.10 for rows drawn like "
            "the reference)",
        )
    ]
    details = [
        "counts rows more similar to their nearest reference row than 90% of "
        f"reference rows are to theirs ({report.reference_nearest_p90:.2f})",
        f"nearest reference p50 {report.nearest_p50:.2f} / p90 "
        f"{report.nearest_p90:.2f} · reference to nearest other p50 "
        f"{report.reference_nearest_p50:.2f} / p90 {report.reference_nearest_p90:.2f}",
    ]
    if report.reference_copies:
        threshold = report.duplicate_threshold
        details[-1] += f" · {report.reference_copies:,} at cosine ≥ {threshold:g}"
    lines += [Padding(Text(d, style="dim"), (0, 0, 0, 4)) for d in details]
    half = (room - 34) // 2
    for near in report.nearest_references:
        line = Text(
            f"{near.rows:,} row{'s' if near.rows != 1 else ''} near ", style="dim"
        )
        line.append(_quote(near.text, half))
        line.append(f"  closest {near.similarity:.2f}: ", style="dim")
        line.append(_quote(near.example_text, half))
        lines.append(Padding(line, (0, 0, 0, 4)))
    return lines


def _report_renderable(report: DispersionReport, *, width: int) -> RenderableType:
    indent = (0, 0, 0, 4)
    room = max(width - 10, 40)  # characters of an example row on its line
    head = Text()
    head.append(_clean(report.field), style="bold")
    sizes = f"  {report.n_rows:,} rows"
    if report.embedded < report.n_rows:
        sizes += f" ({report.embedded:,} sampled)"
    if report.n_reference is not None:
        sizes += f" · reference {report.n_reference:,} rows"
        ref_embedded = report.reference_embedded or 0
        if ref_embedded < report.n_reference:
            sizes += f" ({ref_embedded:,} sampled)"
    head.append(sizes + f" · embedding {_clean(report.embedding)}", style="dim")
    body: list[RenderableType] = [head]

    m = report.compared
    if report.coverage is not None:
        body.append(
            _metric(
                "coverage",
                f"{report.coverage:.2f}",
                f"{round(report.coverage * m):,}/{m:,} reference rows have a row "
                f"within their {report.k}-NN radius ({m:,} random rows of each)",
            )
        )
        body += _example_lines("not covered", report.uncovered, room)
    if report.off_distribution is not None:
        n = report.embedded
        body.append(
            _metric(
                "off-distribution",
                f"{report.off_distribution:.2f}",
                f"{round(report.off_distribution * n):,}/{n:,} rows are outside "
                f"every reference row's {report.k}-NN radius",
            )
        )
        body += _example_lines("outside", report.off_distribution_rows, room)
    if report.near_reference is not None:
        body += _near_reference_lines(report, room)

    dup = (
        f"{report.near_duplicate_rows:,}/{report.embedded:,} rows in "
        f"{report.near_duplicate_pairs:,} pairs with cosine ≥ "
        f"{report.duplicate_threshold:g}"
    )
    if report.exact_duplicates:
        dup += f" ({report.exact_duplicates:,} rows repeat an earlier row exactly)"
    share = report.near_duplicate_rows / max(report.embedded, 1)
    flag = "! " if share >= _DUPLICATE_FLAG else ""
    body.append(_metric(f"{flag}near-duplicates", f"{share:.2f}", dup))
    if report.near_duplicate_groups:
        distinct = (
            report.embedded - report.near_duplicate_rows + report.near_duplicate_groups
        )
        body.append(
            Padding(
                Text(
                    f"they form {report.near_duplicate_groups:,} group"
                    f"{'s' if report.near_duplicate_groups != 1 else ''}, the "
                    f"largest of {report.largest_group:,} rows: the "
                    f"{report.embedded:,} rows hold about {distinct:,} distinct ones",
                    style="dim",
                ),
                indent,
            )
        )
    for pair in report.near_duplicates:
        line = Text(f"{pair.similarity:.2f} ", style="dim")
        if pair.first_text == pair.second_text:
            line.append(_quote(pair.first_text, room - 20))
            line.append("  exact copy", style="dim")
        else:
            half = (room - 10) // 2
            line.append(
                f"{_quote(pair.first_text, half)} ≈ {_quote(pair.second_text, half)}"
            )
        body.append(Padding(line, indent))

    vendi = f"effective number of distinct rows among {m:,}"
    if report.reference_vendi is not None:
        vendi += f" (reference {report.reference_vendi:.1f})"
    body.append(_metric("Vendi", f"{report.vendi:.1f}", vendi))
    for note in report.notes:
        body.append(Text(f"note: {note}", style="dim"))
    return Group(_rule("inspect", _OP), Padding(Group(*body), (0, 0, 0, 2)))
