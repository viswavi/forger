"""Tests for inspect.embedding_dispersion (TF-IDF or fake embedders only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyforger.inspect import DispersionReport, embedding_dispersion
from pyforger.inspect.dispersion import _vendi, render_dispersion_report

TEMPLATES = {
    "probability": [
        "A fair die is rolled {a} times. What is the probability that a six "
        "appears at least once?",
        "Two cards are drawn from a standard deck. What is the probability both "
        "are {suit}?",
    ],
    "geometry": [
        "A triangle has sides {a}, {b}, and {c}. Find its area.",
        "A circle of radius {a} is inscribed in a square. Find the area between them.",
    ],
    "number theory": [
        "Find the remainder when {a}^{b} is divided by 7.",
        "How many positive divisors does {a}{b}0 have?",
    ],
    "algebra": [
        "Solve for x: {a}x + {b} = {c}x - 4.",
        "Find all real roots of x^2 - {a}x + {b} = 0.",
    ],
    "combinatorics": [
        "In how many ways can {a} people sit around a round table?",
        "How many {a}-letter strings use only A and B with no two adjacent Bs?",
    ],
    "calculus": [
        "Evaluate the integral of x^{a} e^x from 0 to 1.",
        "Find the limit of (1 + {a}/k)^k as k goes to infinity.",
    ],
}
SUITS = ("hearts", "aces", "red", "spades", "kings", "face cards", "black", "queens")


def _problem(template: str, i: int) -> dict:
    text = template.format(a=i + 2, b=i + 3, c=i + 5, suit=SUITS[i])
    return {"problem": text}


def _fixture() -> tuple[list[dict], list[dict]]:
    """Return rows from 2 of 6 reference topics, plus copies and 2 unlike rows.

    Also the fixture for the example report in docs/agents/inspect.md.
    """
    reference = [
        _problem(t, i) for ts in TEMPLATES.values() for t in ts for i in range(8)
    ]
    rows = [
        _problem(t, i)
        for t in TEMPLATES["probability"] + TEMPLATES["algebra"]
        for i in range(6)
    ]
    rows += [dict(rows[0]), dict(rows[0]), dict(rows[7])]
    rows += [
        {"problem": "Write a haiku about the ocean at dawn."},
        {"problem": "Translate 'good morning' into French."},
    ]
    return rows, reference


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path))


def test_reference_metrics_on_fixture(capsys):
    """Coverage, off-distribution, copies, and Vendi on the docs fixture."""
    rows, reference = _fixture()

    report = embedding_dispersion(
        rows, "problem", reference=reference, embedding_model="tfidf"
    )

    assert report.embedding == "tfidf"
    assert (report.n_rows, report.embedded, report.n_reference) == (29, 29, 96)
    assert report.compared == 29  # equal-size samples for coverage and Vendi
    # Rows from 2 of 6 topics leave most of the reference uncovered.
    assert report.coverage is not None and report.coverage < 0.6
    assert report.uncovered
    assert all(
        not any(w in ex.text for w in ("die", "cards", "Solve", "roots"))
        for ex in report.uncovered
    )
    # Exactly the two rows from outside the reference are off-distribution.
    assert report.off_distribution == pytest.approx(2 / 29, abs=1e-4)
    assert {ex.index for ex in report.off_distribution_rows} == {27, 28}
    # The three copies: rows 0, 24, 25 share a text, and rows 7, 26 another.
    assert report.exact_duplicates == 3
    assert report.near_duplicate_pairs == 4
    assert report.near_duplicate_rows == 5
    assert report.near_duplicates[0].first_text == report.near_duplicates[0].second_text
    # Two topics are less diverse than six.
    assert report.vendi < report.reference_vendi
    out = capsys.readouterr().out
    assert "inspect embedding_dispersion" in out
    assert "coverage" in out and "off-distribution" in out
    assert rows == _fixture()[0]  # rows untouched


def test_report_is_json_serializable_and_renders():
    """The report round-trips through JSON and renders within 120 columns."""
    rows, reference = _fixture()
    report = embedding_dispersion(
        rows, "problem", reference=reference, embedding_model="tfidf"
    )

    data = report.model_dump()
    assert json.loads(json.dumps(data))["coverage"] == report.coverage
    assert DispersionReport.model_validate(data) == report
    text = render_dispersion_report(report)
    assert "Vendi" in text and "(reference" in text
    assert "outside: " in text and "not covered: " in text
    assert "exact copy" in text
    assert all(len(line) <= 120 for line in text.splitlines())


def test_without_reference_reports_duplicates_and_vendi():
    """Without a reference, only near-duplicates and Vendi are reported."""
    rows, _ = _fixture()
    report = embedding_dispersion(rows, "problem", embedding_model="tfidf")

    assert report.coverage is None and report.off_distribution is None
    assert report.reference_vendi is None and report.n_reference is None
    assert report.compared == report.embedded == 29
    assert 1 < report.vendi < 29
    text = render_dispersion_report(report)
    assert "coverage" not in text and "off-distribution" not in text


def _orthogonal(docs):
    """Each distinct document gets its own axis; copies share it."""
    axis = {d: i for i, d in enumerate(dict.fromkeys(docs))}
    vectors = np.zeros((len(docs), len(axis)))
    for row, doc in enumerate(docs):
        vectors[row, axis[doc]] = 1.0
    return vectors


def test_vendi_counts_distinct_rows():
    """Vendi is the number of distinct rows for orthogonal ones, 1 for copies."""
    assert _vendi(np.eye(7)) == pytest.approx(7.0)
    assert _vendi(np.ones((5, 3)) / np.sqrt(3)) == pytest.approx(1.0)
    assert _vendi(np.zeros((4, 2))) == 1.0
    # Two groups of 3 identical rows: 2 effective rows.
    assert _vendi(np.repeat(np.eye(2), 3, axis=0)) == pytest.approx(2.0)


def test_fake_embed_fn_identical_sets_cover_each_other():
    """Rows equal to the reference cover all of it and none is outside."""
    rows = [{"q": f"q{i}"} for i in range(12)]
    rng = np.random.default_rng(1)
    table = {f"q{i}": rng.normal(size=8) for i in range(12)}

    def embed(docs):
        return np.array([table[d] for d in docs])

    report = embedding_dispersion(rows, "q", reference=rows, embed_fn=embed, k=3)

    assert report.coverage == 1.0
    assert report.off_distribution == 0.0
    assert report.vendi == pytest.approx(report.reference_vendi)
    assert report.embedding == (
        "embed_fn test_fake_embed_fn_identical_sets_cover_each_other.<locals>.embed"
    )


def test_both_sets_embedded_in_one_call():
    """TF-IDF needs one vocabulary, so the embedder sees both sets at once."""
    calls = []

    def embed(docs):
        calls.append(list(docs))
        return _orthogonal(docs)

    rows = [{"t": f"row {i}"} for i in range(8)]
    reference = [{"text": f"ref {i}"} for i in range(8)]
    embedding_dispersion(
        rows, "t", reference=reference, reference_field="text", embed_fn=embed, k=2
    )

    assert len(calls) == 1
    assert sorted(calls[0]) == sorted(
        [f"row {i}" for i in range(8)] + [f"ref {i}" for i in range(8)]
    )


def test_max_rows_samples_each_set_and_is_seeded():
    """max_rows samples each set; the same seed gives the same report."""
    rows = [{"t": f"row number {i}"} for i in range(50)]
    reference = [{"t": f"reference item {i}"} for i in range(80)]
    kwargs = dict(reference=reference, embed_fn=_orthogonal, max_rows=20)

    a = embedding_dispersion(rows, "t", **kwargs)
    b = embedding_dispersion(rows, "t", **kwargs, _no_cache=True)
    c = embedding_dispersion(rows, "t", **kwargs, seed=1)

    assert (a.embedded, a.reference_embedded, a.compared) == (20, 20, 20)
    assert a == b
    assert a.notes == ["near-duplicates are counted among 20 sampled rows"]
    assert c.vendi == pytest.approx(20.0)  # orthogonal rows: all distinct
    assert a.off_distribution == 1.0  # nothing shared with the reference


def test_equal_size_comparison_and_small_inputs():
    """Coverage needs more than k rows of each set; it is left out otherwise."""
    reference = [{"t": f"ref {i}"} for i in range(30)]
    rows = [{"t": "ref 0"}, {"t": "ref 1"}, {"t": "ref 2"}]

    report = embedding_dispersion(
        rows, "t", reference=reference, embed_fn=_orthogonal, k=5
    )

    assert report.compared == 3
    assert report.coverage is None
    assert report.notes == ["coverage needs more than k=5 rows of each set; compared 3"]
    # Off-distribution still uses every reference row's ball.
    assert report.off_distribution is not None


def test_near_duplicate_threshold():
    """Pairs count as near-duplicates at or above duplicate_threshold."""
    vectors = {
        "a": [1.0, 0.0, 0.0],
        "a edited": [0.97, 0.243, 0.0],  # cosine 0.97 with "a"
        "b": [0.0, 1.0, 0.0],
        "c": [0.0, 0.0, 1.0],
    }
    rows = [{"t": t} for t in vectors]

    def embed(docs):
        return np.array([vectors[d] for d in docs])

    strict = embedding_dispersion(rows, "t", embed_fn=embed, duplicate_threshold=0.99)
    loose = embedding_dispersion(rows, "t", embed_fn=embed)

    assert strict.near_duplicate_pairs == 0 and not strict.near_duplicates
    assert loose.near_duplicate_pairs == 1 and loose.near_duplicate_rows == 2
    pair = loose.near_duplicates[0]
    assert (pair.first, pair.second) == (0, 1)
    assert pair.similarity == pytest.approx(0.97, abs=1e-3)


def test_cached_result_is_printed_again(capsys):
    """A cache hit returns the same report and prints it again."""
    from pyforger.op import clear_trace, trace

    rows, reference = _fixture()
    clear_trace()
    kwargs = dict(reference=reference, embed_fn=_orthogonal)
    first = embedding_dispersion(rows, "problem", **kwargs)
    second = embedding_dispersion(rows, "problem", **kwargs)

    assert first == second
    hits = [r.cache_hit for r in trace() if r.op_name == "inspect.embedding_dispersion"]
    assert hits == [False, True]
    assert capsys.readouterr().out.count("inspect embedding_dispersion") == 2


def test_falls_back_to_tfidf_without_sentence_transformers(monkeypatch, caplog):
    """Without sentence-transformers, the op warns and embeds with TF-IDF."""
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    rows, reference = _fixture()
    with caplog.at_level("WARNING", logger="Inspect"):
        report = embedding_dispersion(rows, "problem", reference=reference)

    assert report.embedding == "tfidf"
    assert any(
        "embedding_dispersion: sentence-transformers cannot be imported" in r.message
        for r in caplog.records
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"k": 0}, "k must be"),
        ({"max_rows": 1}, "max_rows"),
        ({"duplicate_threshold": 1.5}, "duplicate_threshold"),
        ({"reference": [{"t": "x"}]}, "2 reference rows"),
    ],
)
def test_invalid_arguments(kwargs, match):
    """Bad arguments raise ValueError before anything is embedded."""
    rows = [{"t": "a"}, {"t": "b"}]
    with pytest.raises(ValueError, match=match):
        embedding_dispersion(rows, "t", embed_fn=_orthogonal, **kwargs)


def test_missing_field_names_the_op():
    """A missing field raises a KeyError naming the op."""
    with pytest.raises(KeyError, match="embedding_dispersion"):
        embedding_dispersion([{"a": 1}, {"a": 2}], "b", embed_fn=_orthogonal)
