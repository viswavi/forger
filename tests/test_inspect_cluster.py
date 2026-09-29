"""Tests for the topic_based_clustering inspect operation."""

import hashlib
import json

import numpy as np
import pytest

from pyforger.acquire.model import Model
from pyforger.inspect import topic_based_clustering
from pyforger.inspect.cluster import _match_topic, _parse_json

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

    def __init__(self):
        super().__init__("fake-model")
        self.summarize_calls = 0
        self.assign_calls = 0
        self.summarized_records = []

    def generate_batch(
        self, prompts, *, n=1, temperature=None, max_tokens=None, stop=None
    ):
        return [[self._respond(prompt)] * max(n, 1) for prompt in prompts]

    def _respond(self, prompt):
        user_content = prompt[-1]["content"] if isinstance(prompt, list) else prompt
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


def test_too_few_unique_values_raises():
    """More clusters than unique values is rejected."""
    with pytest.raises(ValueError, match="unique"):
        _cluster(_rows(COOKING_DOCS), num_clusters=3)


class _GarbageAssignModel(_FakeClusterModel):
    """Summarizes and aggregates fine but never returns a valid assignment."""

    def _respond(self, prompt):
        user_content = prompt[-1]["content"] if isinstance(prompt, list) else prompt
        if "classify the record" in user_content:
            return "not json at all"
        return super()._respond(prompt)


def test_unclassifiable_docs_fall_back_to_first_topic():
    """Docs that never classify fall back to the first topic."""
    items = _rows(COOKING_DOCS + SPACE_DOCS)
    clusters = _cluster(items, _GarbageAssignModel(), max_assign_retries=2)
    assert clusters == ["cooking"] * len(items)  # first codebook topic


def test_match_topic_is_lenient():
    """Topic matching tolerates case and surrounding text."""
    codebook = {"Number Theory": "...", "Geometry": "..."}
    assert _match_topic("Number Theory", codebook) == "Number Theory"
    assert _match_topic("number theory", codebook) == "Number Theory"
    assert _match_topic("Topic: Geometry problems", codebook) == "Geometry"
    assert _match_topic("Algebra", codebook) is None
    assert _match_topic("", codebook) is None


def test_default_embed_warns_without_gpu(monkeypatch, caplog):
    """The default embedder warns and uses CPU when no GPU is present."""
    import sys
    import types

    from pyforger.inspect.cluster import _default_embed

    class _FakeEncoder:
        def __init__(self, model_id, device=None):
            assert device == "cpu"

        def encode(self, docs, **kwargs):
            return _fake_embed(docs)

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
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
