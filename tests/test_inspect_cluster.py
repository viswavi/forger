"""Tests for the topic_based_clustering inspect operation."""

import hashlib
import json

import numpy as np
import pytest

from pyforger.acquire.model import Model
from pyforger.inspect import topic_based_clustering
from pyforger.inspect.cluster import _build_codebook, _match_topic, _parse_json

CONTEXT = "short hobby notes. Topics should describe the HOBBY the note is about."

COOKING_DOCS = [
    "I baked a sourdough loaf and the crumb was perfect.",
    "My new recipe for garlic butter shrimp pasta.",
    "How to properly sear a steak in a cast-iron pan.",
]
SPACE_DOCS = [
    "The telescope resolved the rings of Saturn last night.",
    "Watching the rocket launch from the causeway was incredible.",
    "Charting the phases of the moon for the next month.",
]

_COOKING_WORDS = ("baked", "recipe", "sear", "loaf", "pasta", "steak")


def _is_cooking(text):
    return any(word in text.lower() for word in _COOKING_WORDS)


def _fake_embed(docs):
    """Two well-separated 2D groups with deterministic per-doc jitter."""
    vectors = []
    for doc in docs:
        base = [1.0, 0.0] if _is_cooking(doc) else [0.0, 1.0]
        digest = hashlib.md5(doc.encode()).digest()
        vectors.append(
            [
                base[0] + (digest[0] / 255 - 0.5) * 0.1,
                base[1] + (digest[1] / 255 - 0.5) * 0.1,
            ]
        )
    return np.array(vectors)


class _FakeClusterModel(Model):
    """Fake Model that answers the three ClusterFusion prompt types."""

    def __init__(self, broken=()):
        """``broken``: prompt markers whose responses are unparseable."""
        super().__init__("fake-model")
        self.broken = broken
        self.summarize_calls = 0
        self.assign_calls = 0
        self.summarized_records = []

    def generate_batch(
        self, prompts, *, n=1, temperature=None, max_tokens=None, stop=None
    ):
        return [[self._respond(prompt)] * max(n, 1) for prompt in prompts]

    def _respond(self, prompt):
        user_content = prompt[-1]["content"]
        if any(marker in user_content for marker in self.broken):
            return "not json at all"
        if "reviewing records" in user_content:
            self.summarize_calls += 1
            records = user_content.split("Records: ", 1)[1].split("\n\nYour goal")[0]
            self.summarized_records.extend(records.split(" ; "))
            return json.dumps(
                {"cooking": "Notes about food.", "space": "Notes about the sky."}
            )
        if "candidate topics" in user_content.lower():
            return json.dumps(
                {"cooking": "Preparing food.", "space": "Astronomy and rockets."}
            )
        if "classify the record" in user_content:
            self.assign_calls += 1
            record = user_content.split("Record: ", 1)[1]
            return json.dumps({"topic": "cooking" if _is_cooking(record) else "space"})
        raise AssertionError(f"unexpected prompt: {user_content[:100]}")


def _rows(docs):
    return [{"text": doc, "id": i} for i, doc in enumerate(docs)]


def _cluster(items, model=None, **kwargs):
    kwargs.setdefault("num_clusters", 2)
    return topic_based_clustering(
        items,
        field="text",
        model=model or _FakeClusterModel(),
        context=CONTEXT,
        embed_fn=_fake_embed,
        **kwargs,
    )


def test_returns_aligned_cluster_names():
    """Each row gets its cluster name; unique docs are processed once."""
    docs = COOKING_DOCS + SPACE_DOCS + [COOKING_DOCS[0]]  # last row is a duplicate
    items = _rows(docs)
    model = _FakeClusterModel()

    clusters = _cluster(items, model)

    assert clusters == ["cooking" if _is_cooking(doc) else "space" for doc in docs]
    assert items == _rows(docs)  # rows untouched
    # Unique docs are classified once each, not once per row.
    assert model.assign_calls == len(set(docs))
    # A small corpus is summarized in one shard, each unique doc exactly once.
    assert model.summarize_calls == 1
    assert sorted(model.summarized_records) == sorted(set(docs))


def test_tiny_tokens_per_shard_splits_into_shards():
    """A tiny token budget puts each record in its own shard."""
    docs = COOKING_DOCS + SPACE_DOCS
    model = _FakeClusterModel()
    _cluster(_rows(docs), model, tokens_per_shard=1)
    # One shard per token, capped at one record per shard.
    assert model.summarize_calls == len(docs)
    assert sorted(model.summarized_records) == sorted(docs)


def test_empty_items_returns_empty():
    """Empty input yields an empty result."""
    assert _cluster([]) == []


def test_missing_field_raises_key_error():
    """A row without the target field raises KeyError."""
    with pytest.raises(KeyError, match="text"):
        _cluster([{"other": "value"}])


def test_num_clusters_below_two_raises():
    """Fewer than two clusters is rejected."""
    with pytest.raises(ValueError, match="num_clusters"):
        _cluster(_rows(COOKING_DOCS), num_clusters=1)


@pytest.mark.parametrize(
    "param, value",
    [
        ("sample_size", 0),
        ("tokens_per_shard", 0),
        ("tokens_per_shard", -1),
        ("summarize_truncation", 0),
        ("assign_truncation", -1),
        ("max_assign_retries", -1),
    ],
)
def test_invalid_count_parameters_raise(param, value):
    """Non-positive sizes and budgets, and negative retry counts, are rejected."""
    with pytest.raises(ValueError, match=param):
        _cluster(_rows(COOKING_DOCS), **{param: value})


def test_too_few_unique_values_raises():
    """More clusters than unique values is rejected."""
    with pytest.raises(ValueError, match="unique"):
        _cluster(_rows(COOKING_DOCS), num_clusters=3)


def test_all_unparseable_summaries_raise():
    """With no parseable shard summary, there is nothing to build a codebook from."""
    with pytest.raises(RuntimeError, match="every shard summary"):
        _cluster(
            _rows(COOKING_DOCS + SPACE_DOCS),
            _FakeClusterModel(broken=("reviewing records",)),
        )


def test_unclassifiable_docs_fall_back_to_first_topic():
    """Docs that never classify fall back to the first topic."""
    items = _rows(COOKING_DOCS + SPACE_DOCS)
    clusters = _cluster(
        items, _FakeClusterModel(broken=("classify the record",)), max_assign_retries=2
    )
    assert clusters == ["cooking"] * len(items)  # first codebook topic


def test_zero_assign_retries_still_classifies():
    """max_assign_retries counts retries, so 0 still makes one attempt."""
    docs = COOKING_DOCS + SPACE_DOCS
    clusters = _cluster(_rows(docs), max_assign_retries=0)
    assert clusters == ["cooking" if _is_cooking(doc) else "space" for doc in docs]


_TWO_TOPICS = {"cooking": "Preparing food.", "space": "Astronomy and rockets."}
_THREE_TOPICS = {**_TWO_TOPICS, "gardening": "Growing plants."}


class _ScriptedCodebookModel(Model):
    """Returns scripted codebooks in order, recording each prompt."""

    def __init__(self, codebooks):
        super().__init__("fake-model")
        self.codebooks = list(codebooks)
        self.prompts = []

    def generate_batch(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        return [[json.dumps(self.codebooks.pop(0))] for _ in prompts]


def test_codebook_with_wrong_topic_count_is_sent_back():
    """A codebook with the wrong number of topics is returned for correction."""
    model = _ScriptedCodebookModel([_THREE_TOPICS, _TWO_TOPICS])
    codebook = _build_codebook([_THREE_TOPICS], model, CONTEXT, num_clusters=2)
    assert codebook == _TWO_TOPICS
    assert "That is 3 topics" in model.prompts[-1][-1]["content"]


def test_codebook_keeps_last_attempt_when_count_never_matches(caplog):
    """If the count is never corrected, the last codebook is used with a warning."""
    model = _ScriptedCodebookModel([_THREE_TOPICS] * 3)
    with caplog.at_level("WARNING", logger="Inspect"):
        codebook = _build_codebook([_THREE_TOPICS], model, CONTEXT, num_clusters=2)
    assert codebook == _THREE_TOPICS
    assert any("3 topics instead of 2" in record.message for record in caplog.records)


def test_match_topic_is_lenient():
    """Topic matching tolerates case and surrounding text."""
    codebook = {"Number Theory": "...", "Geometry": "..."}
    assert _match_topic("Number Theory", codebook) == "Number Theory"
    assert _match_topic("number theory", codebook) == "Number Theory"
    assert _match_topic("Topic: Geometry problems", codebook) == "Geometry"
    assert _match_topic("Algebra", codebook) is None
    assert _match_topic("", codebook) is None


def test_match_topic_prefers_longest_whole_word_match():
    """The most specific topic wins, and topic names must match whole words."""
    codebook = {"Data": "...", "Data Science": "...", "Art": "..."}
    assert _match_topic("Topic: Data Science", codebook) == "Data Science"
    assert _match_topic("Partial derivatives", codebook) is None


def test_match_topic_rejects_ambiguous_partial_label():
    """A label contained in several topic names is ambiguous, so it's unmatched."""
    codebook = {"Data Science": "...", "Data Engineering": "...", "Art": "..."}
    assert _match_topic("Data", codebook) is None


def test_default_embed_warns_without_gpu(monkeypatch, caplog):
    """The default embedder warns when sentence-transformers lands on the CPU."""
    import sys
    import types

    from pyforger.inspect.cluster import _default_embed

    class _FakeEncoder:
        device = types.SimpleNamespace(type="cpu")

        def __init__(self, model_id):
            pass

        def encode(self, docs, **kwargs):
            return _fake_embed(docs)

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _FakeEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    with caplog.at_level("WARNING", logger="Inspect"):
        embeddings = _default_embed(COOKING_DOCS, "some/embedding-model")

    assert embeddings.shape == (len(COOKING_DOCS), 2)
    assert any("no GPU detected" in record.message for record in caplog.records)


def test_parse_json_handles_fences_and_prose():
    """JSON parsing strips code fences and surrounding prose."""
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json('Sure! Here you go: {"topic": "space"} hope that helps') == {
        "topic": "space"
    }
    with pytest.raises(ValueError):
        _parse_json("no json here")


def test_parse_json_returns_first_complete_object():
    """Only the first complete object is parsed, skipping stray braces."""
    assert _parse_json('{"topic": "a"} or maybe {"topic": "b"}') == {"topic": "a"}
    assert _parse_json('Using {braces} loosely: {"topic": "a"}') == {"topic": "a"}
