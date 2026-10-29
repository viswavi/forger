"""Tests for filter.by_embedding_similarity (TF-IDF or fake embedders only)."""

from __future__ import annotations

import numpy as np
import pytest
from test_inspect_dispersion import _fixture

from pyforger import _tracking
from pyforger.filter import by_embedding_similarity
from pyforger.inspect import cluster as cluster_mod
from pyforger.inspect import embedding_dispersion
from pyforger.op import clear_trace, trace
from pyforger.steps import step

_OP = "filter.by_embedding_similarity"


class FakeEmbed:
    """Embed each text as a fixed vector; records every call's documents."""

    def __init__(self, vectors: dict[str, list[float]]):
        """Embed ``vectors[text]`` for each text."""
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def __call__(self, docs: list[str]) -> np.ndarray:
        """Return the documents' vectors, one row each."""
        self.calls.append(list(docs))
        return np.array([self.vectors[d] for d in docs], dtype=np.float64)

    def cache_key(self):
        """Key the embedder by its vectors."""
        return sorted((k, tuple(v)) for k, v in self.vectors.items())


def _rows(*texts: str) -> list[dict]:
    return [{"text": t} for t in texts]


def _unit(angle_deg: float) -> list[float]:
    a = np.deg2rad(angle_deg)
    return [float(np.cos(a)), float(np.sin(a))]


def test_greedy_dedup_keeps_first_copy_in_input_order():
    """A later near-copy of a kept row is dropped; kept rows are the input dicts."""
    embed = FakeEmbed({"a": [1, 0], "a2": [1, 0.01], "b": [0, 1], "c": [-1, 0]})
    rows = _rows("b", "a", "c", "a2", "a")

    kept = by_embedding_similarity(rows, field="text", embed_fn=embed)

    assert kept == _rows("b", "a", "c")
    assert all(k is r for k, r in zip(kept, [rows[0], rows[1], rows[2]]))
    # One call, each text once.
    assert len(embed.calls) == 1 and sorted(embed.calls[0]) == ["a", "a2", "b", "c"]


def test_greedy_compares_with_kept_rows_only():
    """A ≈ B ≈ C with A and C apart keeps A and C: B is dropped, C is near only B."""
    embed = FakeEmbed({"A": _unit(0), "B": _unit(15), "C": _unit(30)})  # cos 0.97, 0.87

    kept = by_embedding_similarity(
        _rows("A", "B", "C"), field="text", threshold=0.95, embed_fn=embed
    )

    assert kept == _rows("A", "C")


def test_threshold_is_inclusive():
    """A pair exactly at the threshold is dropped; just above it, kept."""
    embed = FakeEmbed({"x": [1.0, 0.0], "y": [0.6, 0.8]})  # cosine 0.6
    rows = _rows("x", "y")

    assert by_embedding_similarity(
        rows, field="text", threshold=0.6, embed_fn=embed
    ) == (_rows("x"))
    assert (
        by_embedding_similarity(rows, field="text", threshold=0.6001, embed_fn=embed)
        == rows
    )


def test_reference_drops_near_rows_and_dedupes():
    """Rows near any reference row are dropped; dedupe still runs by default."""
    embed = FakeEmbed(
        {
            "eval q": _unit(0),
            "leaked q": _unit(5),  # cos 0.996 to "eval q"
            "fresh": _unit(90),
            "fresh copy": _unit(92),
            "other": _unit(180),
        }
    )
    rows = _rows("fresh", "leaked q", "fresh copy", "other")
    reference = [{"question": "eval q"}]

    kept = by_embedding_similarity(
        rows,
        field="text",
        reference=reference,
        reference_field="question",
        embed_fn=embed,
    )
    assert kept == _rows("fresh", "other")
    # Rows and reference are embedded in one call.
    assert len(embed.calls) == 1
    assert set(embed.calls[0]) == {"eval q", "leaked q", "fresh", "fresh copy", "other"}

    only_ref = by_embedding_similarity(
        rows,
        field="text",
        reference=reference,
        reference_field="question",
        dedupe=False,
        embed_fn=embed,
    )
    assert only_ref == _rows("fresh", "fresh copy", "other")


def test_a_row_dropped_near_the_reference_does_not_shadow_later_rows():
    """Only kept rows count for dedupe: B is near the reference, C near only B."""
    embed = FakeEmbed({"R": _unit(-15), "B": _unit(0), "C": _unit(15)})

    kept = by_embedding_similarity(
        _rows("B", "C"), field="text", reference=_rows("R"), embed_fn=embed
    )

    assert kept == _rows("C")


def _later_members(rows: list[dict], field: str, **kwargs) -> set[int]:
    """Rows embedding_dispersion pairs with an earlier row (texts must be distinct)."""
    report = embedding_dispersion(rows, field, examples=10_000, **kwargs)
    # Every pair is listed, so the second members are every later member.
    assert len(report.near_duplicates) == report.near_duplicate_pairs
    return {pair.second for pair in report.near_duplicates}


def _dropped(rows: list[dict], kept: list[dict]) -> set[int]:
    kept_ids = {id(row) for row in kept}
    return {i for i, row in enumerate(rows) if id(row) not in kept_ids}


def test_agrees_with_embedding_dispersion_tfidf(capsys):
    """Same rows, field, threshold, and backend: same near-duplicates.

    The variants differ from their originals only in case and spacing, which
    TF-IDF ignores; one comes before its original, so the original is dropped.
    """
    rows, _ = _fixture()
    rows = list({r["problem"]: r for r in rows}.values())  # distinct texts
    variants = [
        (3, {"problem": rows[0]["problem"].upper()}),
        (8, {"problem": rows[5]["problem"].replace(" ", "  ")}),
        (12, {"problem": rows[20]["problem"].lower()}),
        (len(rows), {"problem": " " + rows[0]["problem"]}),
    ]
    for at, row in sorted(variants, key=lambda v: -v[0]):
        rows.insert(at, row)

    for threshold in (0.95, 0.9):
        kept = by_embedding_similarity(
            rows, field="problem", threshold=threshold, embedding_model="tfidf"
        )
        expected = _later_members(
            rows, "problem", embedding_model="tfidf", duplicate_threshold=threshold
        )
        assert _dropped(rows, kept) == expected
        assert len(expected) == 4


def test_agrees_with_embedding_dispersion_fake_embedder(capsys):
    """Clusters of near-copies under a fake embedder, rows in a random order."""
    rng = np.random.default_rng(0)
    vectors: dict[str, list[float]] = {}
    for c in range(25):
        base = rng.normal(size=64)
        for m in range(1 + c % 3):
            vectors[f"c{c}-m{m}"] = list(base + 0.03 * rng.normal(size=64))
    texts = list(vectors)
    rng.shuffle(texts)
    rows = _rows(*texts)
    embed = FakeEmbed(vectors)

    kept = by_embedding_similarity(rows, field="text", threshold=0.95, embed_fn=embed)
    expected = _later_members(rows, "text", embed_fn=embed, duplicate_threshold=0.95)

    assert _dropped(rows, kept) == expected
    assert len(kept) == 25  # one row per cluster


def test_step_report_counts_and_notes(monkeypatch):
    """The score record says how many rows went for which reason, with pairs."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    embed = FakeEmbed(
        {"eval": _unit(0), "leak": _unit(3), "x": _unit(90), "x again": _unit(91)}
    )

    @step
    def decontaminate(rows):
        return by_embedding_similarity(
            rows, field="text", reference=_rows("eval"), embed_fn=embed
        )

    assert decontaminate(_rows("x", "leak", "x again")) == _rows("x")
    scores = _tracking.step_records()["decontaminate"].ops[0].scores
    assert scores is not None
    assert scores.label == "cosine" and scores.keep == "lt" and scores.pooled
    assert scores.note == (
        "dropped: 1 near a kept row, 1 near the reference (1 rows) · "
        "embedding embed_fn FakeEmbed"
    )
    assert [(d.preview, d.note) for d in scores.dropped] == [
        ("leak", 'cosine 1.00 to reference row 0: "eval"'),
        ("x again", 'cosine 1.00 to kept row 0: "x"'),
    ]
    # A dropped score is at least the threshold, a kept one below it.
    assert [v >= 0.95 for v in scores.values] == [False, True, True]


def test_cache_key_includes_embedder_and_threshold(monkeypatch):
    """A rerun is a cache hit; another embedder or threshold is a miss."""
    rows, _ = _fixture()
    clear_trace()

    def hits() -> list[bool]:
        return [r.cache_hit for r in trace() if r.op_name == _OP]

    by_embedding_similarity(rows, field="problem", embedding_model="tfidf")
    by_embedding_similarity(rows, field="problem", embedding_model="tfidf")
    by_embedding_similarity(
        rows, field="problem", embedding_model="tfidf", threshold=0.9
    )
    vectors = {r["problem"]: [float(len(r["problem"])), 1.0] for r in rows}
    by_embedding_similarity(rows, field="problem", embed_fn=FakeEmbed(vectors))
    vectors[rows[0]["problem"]] = [0.0, 1.0]
    by_embedding_similarity(rows, field="problem", embed_fn=FakeEmbed(vectors))
    assert hits() == [False, True, False, False, False]

    # Without sentence-transformers the default model falls back to TF-IDF and
    # is cached as "tfidf", so it reuses the explicit TF-IDF result.
    monkeypatch.setattr(cluster_mod, "_sentence_transformers_importable", lambda: False)
    clear_trace()
    by_embedding_similarity(rows, field="problem")
    assert hits() == [True]


def test_small_inputs_skip_embedding():
    """No rows, or one row without a reference, needs no embedding."""
    embed = FakeEmbed({"a": [1, 0]})

    assert by_embedding_similarity([], field="text", embed_fn=embed) == []
    assert by_embedding_similarity(_rows("a"), field="text", embed_fn=embed) == _rows(
        "a"
    )
    assert embed.calls == []


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"threshold": 0.0}, "threshold"),
        ({"threshold": 1.5}, "threshold"),
        ({"dedupe": False}, "needs a reference"),
    ],
)
def test_invalid_arguments(kwargs, match):
    """A threshold outside (0, 1], or dedupe=False without a reference, raises."""
    with pytest.raises(ValueError, match=match):
        by_embedding_similarity(_rows("a", "b"), field="text", **kwargs)


def test_missing_field_names_the_op():
    """A missing field's error names the filter."""
    with pytest.raises(Exception, match="by_embedding_similarity"):
        by_embedding_similarity(_rows("a", "b"), field="nope", embedding_model="tfidf")
