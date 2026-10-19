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


class _RecordingClusterModel(_FakeClusterModel):
    """Fake cluster model that keeps every user prompt it answers."""

    def __init__(self):
        """Start with no prompts."""
        super().__init__()
        self.user_prompts = []

    def _respond(self, prompt):
        self.user_prompts.append(prompt[-1]["content"])
        return super()._respond(prompt)


def _first_lines(model):
    """The first line of each prompt kind: summarize, aggregate, assign."""
    lines = {prompt.split("\n", 1)[0] for prompt in model.user_prompts}
    return sorted(lines)


def test_context_defaults_to_the_field_values():
    """Without context, every prompt describes the corpus by its field."""
    docs = COOKING_DOCS + SPACE_DOCS
    model = _RecordingClusterModel()
    clusters = topic_based_clustering(
        _rows(docs), field="text", model=model, num_clusters=2, embed_fn=_fake_embed
    )
    assert clusters == ["cooking" if _is_cooking(doc) else "space" for doc in docs]
    assert _first_lines(model) == [
        "Below are candidate topics proposed independently from 1 different "
        "samples of a corpus of 'text' values from a dataset",
        "Following record is about 'text' values from a dataset",
        "You are now tasked with reviewing records related to 'text' values "
        "from a dataset",
    ]


def test_explicit_context_prompts_and_cache_are_unchanged():
    """An explicit context is used verbatim and cached apart from the default."""
    docs = COOKING_DOCS + SPACE_DOCS
    model = _RecordingClusterModel()
    _cluster(_rows(docs), model)
    assert _first_lines(model) == [
        "Below are candidate topics proposed independently from 1 different "
        f"samples of a corpus of {CONTEXT}",
        f"Following record is about {CONTEXT}",
        f"You are now tasked with reviewing records related to {CONTEXT}",
    ]
    calls = len(model.user_prompts)
    _cluster(_rows(docs), model)
    assert len(model.user_prompts) == calls  # cache hit
    topic_based_clustering(
        _rows(docs), field="text", model=model, num_clusters=2, embed_fn=_fake_embed
    )
    assert len(model.user_prompts) > calls  # the default is a different key


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


class _PromptRecordingModel(_FakeClusterModel):
    """_FakeClusterModel that also keeps every user prompt."""

    def __init__(self):
        """Start with no prompts."""
        super().__init__()
        self.prompts = []

    def _respond(self, prompt):
        self.prompts.append(prompt[-1]["content"])
        return super()._respond(prompt)


def test_fewer_unique_values_than_clusters_clamps_with_a_warning(caplog):
    """Two unique values and num_clusters=5: two clusters, and a warning."""
    docs = [COOKING_DOCS[0], SPACE_DOCS[0]] * 3
    model = _PromptRecordingModel()
    with caplog.at_level("WARNING", logger="Inspect"):
        clusters = _cluster(_rows(docs), model, num_clusters=5)
    assert clusters == ["cooking", "space"] * 3
    assert (
        "only 2 unique 'text' values, so using 2 clusters instead of num_clusters=5"
        in caplog.text
    )
    assert any("Extract at most 2 topics" in p for p in model.prompts)
    assert any("into exactly 2 final" in p for p in model.prompts)


def test_as_many_unique_values_as_clusters_runs_without_a_warning(caplog):
    """num_clusters equal to the unique count is used as given."""
    docs = [COOKING_DOCS[0], SPACE_DOCS[0], COOKING_DOCS[0]]
    with caplog.at_level("WARNING", logger="Inspect"):
        clusters = _cluster(_rows(docs), num_clusters=2)
    assert clusters == ["cooking", "space", "cooking"]
    assert "unique" not in caplog.text


def test_one_unique_value_is_one_topic_without_llm_calls():
    """Every row gets the value itself as its topic; the model is not called."""

    class _NoCalls(Model):
        def generate_batch(self, prompts, **kwargs):
            raise AssertionError("the model should not be called")

    long_note = "I baked a sourdough loaf.\n" + "x" * 200
    clusters = _cluster(_rows([long_note] * 4), _NoCalls("fake-model"), num_clusters=5)
    assert clusters == [("I baked a sourdough loaf. " + "x" * 200)[:80]] * 4


def test_no_rows_with_many_clusters_returns_empty():
    """Zero rows is not too few unique values: the result is just empty."""
    assert _cluster([], num_clusters=50) == []


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
    """Returns scripted codebooks in order (raising error entries); records prompts."""

    def __init__(self, codebooks):
        super().__init__("fake-model")
        self.codebooks = list(codebooks)
        self.prompts = []

    def generate_batch(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        out = []
        for _ in prompts:
            codebook = self.codebooks.pop(0)
            if isinstance(codebook, Exception):
                raise codebook
            out.append([json.dumps(codebook)])
        return out


def test_codebook_with_wrong_topic_count_is_sent_back():
    """A codebook with the wrong number of topics is returned for correction."""
    model = _ScriptedCodebookModel([_THREE_TOPICS, _TWO_TOPICS])
    codebook = _build_codebook([_THREE_TOPICS], model, CONTEXT, num_clusters=2)
    assert codebook == _TWO_TOPICS
    assert "That is 3 distinct topic names" in model.prompts[-1][-1]["content"]


def test_codebook_keeps_last_attempt_when_count_never_matches(caplog):
    """If the count is never corrected, the last codebook is used with a warning."""
    model = _ScriptedCodebookModel([_THREE_TOPICS] * 3)
    with caplog.at_level("WARNING", logger="Inspect"):
        codebook = _build_codebook([_THREE_TOPICS], model, CONTEXT, num_clusters=2)
    assert codebook == _THREE_TOPICS
    assert any("3 topics instead of 2" in record.message for record in caplog.records)


def test_failed_codebook_request_is_not_asked_again():
    """A codebook request that fails after the model's retries is not re-sent."""
    model = _ScriptedCodebookModel([TimeoutError("slow"), _TWO_TOPICS])
    with pytest.raises(RuntimeError, match="codebook request failed"):
        _build_codebook([_TWO_TOPICS], model, CONTEXT, num_clusters=2)
    assert len(model.prompts) == 1


def test_failed_codebook_correction_keeps_last_codebook():
    """A failed correction request ends the corrections; the last codebook is used."""
    model = _ScriptedCodebookModel([_THREE_TOPICS, TimeoutError("slow"), _TWO_TOPICS])
    codebook = _build_codebook([_THREE_TOPICS], model, CONTEXT, num_clusters=2)
    assert codebook == _THREE_TOPICS
    assert len(model.prompts) == 2


def test_codebook_case_duplicates_count_once():
    """Names differing only in case are one topic, so the count gets corrected."""
    duplicated = {"Cooking": "Food.", "cooking ": "Also food.", " ": "Blank."}
    model = _ScriptedCodebookModel([duplicated, _TWO_TOPICS])
    codebook = _build_codebook([_TWO_TOPICS], model, CONTEXT, num_clusters=2)
    assert codebook == _TWO_TOPICS
    assert "That is 1 distinct topic names" in model.prompts[-1][-1]["content"]


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


def _block_sentence_transformers(monkeypatch):
    """Make `import sentence_transformers` raise ImportError."""
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)


def _install_fake_sentence_transformers(monkeypatch):
    """Install a fake sentence-transformers whose encoder records its calls."""
    import sys
    import types

    calls = []

    class _FakeEncoder:
        device = types.SimpleNamespace(type="cuda")

        def __init__(self, model_id):
            calls.append(model_id)

        def encode(self, docs, **kwargs):
            return _fake_embed(docs)

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _FakeEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    return calls


def _cluster_with(items, model=None, **kwargs):
    """Cluster without the fake embed_fn, so embedding_model decides."""
    kwargs.setdefault("num_clusters", 2)
    return topic_based_clustering(
        items,
        field="text",
        model=model or _FakeClusterModel(),
        context=CONTEXT,
        **kwargs,
    )


def test_tfidf_embed_shape_and_normalization():
    """TF-IDF vectors are SVD-reduced to < n_rows dims and unit length."""
    from pyforger.inspect.cluster import _tfidf_embed

    docs = COOKING_DOCS + SPACE_DOCS
    embeddings = _tfidf_embed(docs)
    assert embeddings.shape == (len(docs), len(docs))
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_tfidf_embed_caps_dimension(monkeypatch):
    """More documents than the cap (256, lowered here to 4) reduce to the cap."""
    from pyforger.inspect import cluster

    assert cluster._TFIDF_MAX_DIM == 256
    monkeypatch.setattr(cluster, "_TFIDF_MAX_DIM", 4)
    docs = [f"note {i} about topic{i} and item{i * 7}" for i in range(10)]
    embeddings = cluster._tfidf_embed(docs)
    assert embeddings.shape == (10, 4)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_tfidf_embed_is_deterministic():
    """The same documents and seed give the same vectors."""
    from pyforger.inspect.cluster import _tfidf_embed

    docs = COOKING_DOCS + SPACE_DOCS
    np.testing.assert_array_equal(_tfidf_embed(docs), _tfidf_embed(docs))
    np.testing.assert_array_equal(
        _tfidf_embed(docs, seed=3), _tfidf_embed(docs, seed=3)
    )


def test_tfidf_embed_separates_topics():
    """Docs sharing words are closer to each other than to unrelated docs."""
    from pyforger.inspect.cluster import _tfidf_embed

    docs = [
        "bake bread with sourdough starter",
        "bake bread with rye flour",
        "rocket launch to orbit the moon",
        "rocket launch to orbit mars",
    ]
    vectors = _tfidf_embed(docs)
    similarity = vectors @ vectors.T
    assert similarity[0, 1] > similarity[0, 2]
    assert similarity[2, 3] > similarity[2, 1]


@pytest.mark.parametrize(
    "docs",
    [
        ["only one document"],
        ["two documents", "second one"],
        ["a", "b", "c"],  # one-character words
        ["?!", "...", "!!"],  # no word characters: character n-grams
        ["", "", ""],  # nothing at all: zero vectors
        ["same", "same"],  # a single feature
        ["one two", "one two"],  # identical: zero variance
    ],
)
def test_tfidf_embed_tiny_inputs(docs):
    """Tiny inputs and empty vocabularies give one finite vector per doc."""
    from pyforger.inspect.cluster import _tfidf_embed

    embeddings = _tfidf_embed(docs)
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == len(docs)
    assert embeddings.shape[1] >= 1
    assert np.isfinite(embeddings).all()
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.all(np.isclose(norms, 1.0) | (norms < 1e-8))


def test_clustering_with_tfidf_embedder():
    """embedding_model="tfidf" clusters with the fake LLM, no embed_fn needed."""
    docs = COOKING_DOCS + SPACE_DOCS
    model = _FakeClusterModel()
    clusters = _cluster_with(_rows(docs), model, embedding_model="tfidf")
    assert clusters == ["cooking" if _is_cooking(doc) else "space" for doc in docs]
    assert sorted(model.summarized_records) == sorted(docs)


def test_falls_back_to_tfidf_without_sentence_transformers(monkeypatch, caplog):
    """A missing sentence-transformers falls back to TF-IDF with one warning."""
    from pyforger.op import trace

    _block_sentence_transformers(monkeypatch)
    docs = COOKING_DOCS + SPACE_DOCS
    with caplog.at_level("WARNING", logger="Inspect"):
        clusters = _cluster_with(_rows(docs))
    assert clusters == ["cooking" if _is_cooking(doc) else "space" for doc in docs]
    warnings = [r.message for r in caplog.records if "TF-IDF" in r.message]
    assert len(warnings) == 1
    assert "pyforger[clustering]" in warnings[0]

    # The fallback is cached under the TF-IDF key: an explicit "tfidf" call hits it.
    _cluster_with(_rows(docs), embedding_model="tfidf")
    first, second = trace()
    assert (first.cache_hit, second.cache_hit) == (False, True)
    assert first.input_hash == second.input_hash


def test_fallback_result_is_not_reused_for_sentence_transformers(monkeypatch):
    """Once sentence-transformers is installed, the TF-IDF result is not reused."""
    from pyforger.op import trace

    docs = COOKING_DOCS + SPACE_DOCS
    _block_sentence_transformers(monkeypatch)
    _cluster_with(_rows(docs))

    encoder_calls = _install_fake_sentence_transformers(monkeypatch)
    _cluster_with(_rows(docs))
    first, second = trace()
    assert second.cache_hit is False
    assert first.input_hash != second.input_hash
    assert encoder_calls == ["Qwen/Qwen3-Embedding-0.6B"]


def test_embed_fn_wins_without_sentence_transformers(monkeypatch, caplog):
    """An explicit embed_fn is used as-is, with no fallback warning."""
    _block_sentence_transformers(monkeypatch)
    embedded = []

    def recording_embed(docs):
        embedded.append(list(docs))
        return _fake_embed(docs)

    docs = COOKING_DOCS + SPACE_DOCS
    with caplog.at_level("WARNING", logger="Inspect"):
        _cluster_with(_rows(docs), embed_fn=recording_embed)
    assert embedded == [docs]
    assert not any("TF-IDF" in r.message for r in caplog.records)


def test_cache_only_is_forwarded():
    """Per-call @op controls pass through to the cached computation."""
    with pytest.raises(FileNotFoundError):
        _cluster_with(
            _rows(COOKING_DOCS + SPACE_DOCS), embedding_model="tfidf", _cache_only=True
        )
