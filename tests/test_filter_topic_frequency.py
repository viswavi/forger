"""Tests for the by_topic_frequency filter."""

import hashlib
import json

import numpy as np
import pytest

from pyforger import _tracking
from pyforger.acquire.model import Model
from pyforger.filter import by_topic_frequency
from pyforger.inspect import topic_based_clustering
from pyforger.steps import pipeline, step

CONTEXT = "short hobby notes. Topics should describe the HOBBY the note is about."

# 7 cooking, 2 space, 1 music: topic shares 0.7, 0.2, 0.1.
COOKING_DOCS = [
    "I baked a sourdough loaf and the crumb was perfect.",
    "My new recipe for garlic butter shrimp pasta.",
    "How to properly sear a steak in a cast-iron pan.",
    "I baked cinnamon rolls for the whole street.",
    "A quick recipe for weeknight vegetable curry.",
    "Learning to sear scallops without overcooking them.",
    "Homemade pasta dough needs only flour and eggs.",
]
SPACE_DOCS = [
    "The telescope resolved the rings of Saturn last night.",
    "Watching the rocket launch from the causeway was incredible.",
]
MUSIC_DOCS = ["Practicing guitar scales for an hour every morning."]
DOCS = COOKING_DOCS + SPACE_DOCS + MUSIC_DOCS

_TOPIC_WORDS = {
    "cooking": ("baked", "recipe", "sear", "pasta"),
    "music": ("guitar",),
}
_CODEBOOK = {
    "cooking": "Preparing food.",
    "space": "Astronomy and rockets.",
    "music": "Playing instruments.",
}


def _topic(text):
    for topic, words in _TOPIC_WORDS.items():
        if any(word in text.lower() for word in words):
            return topic
    return "space"


class _CountingEmbed:
    """Fake embed_fn: one well-separated 3D group per topic, with jitter."""

    def __init__(self):
        """Start with no calls."""
        self.calls = 0

    def cache_key(self):
        """Exclude the call counter from the @op cache key."""
        return ("_CountingEmbed",)

    def __call__(self, docs):
        """Embed ``docs`` by topic."""
        self.calls += 1
        vectors = []
        for doc in docs:
            base = np.eye(3)[list(_CODEBOOK).index(_topic(doc))]
            digest = hashlib.md5(doc.encode()).digest()
            vectors.append(
                base + (np.frombuffer(digest[:3], np.uint8) / 255 - 0.5) / 10
            )
        return np.array(vectors)


class _FakeTopicModel(Model):
    """Fake Model that answers the three ClusterFusion prompt types."""

    def __init__(self):
        """Count every call; Model.cache_key leaves the counter out."""
        super().__init__("fake-model")
        self.calls = 0

    def generate(self, prompt, *, n=1, temperature=None, max_tokens=None, stop=None):
        """Answer one prompt."""
        return self.generate_batch([prompt], n=n, temperature=temperature)[0]

    def generate_batch(
        self, prompts, *, n=1, temperature=None, max_tokens=None, stop=None
    ):
        """Answer each prompt."""
        self.calls += len(prompts)
        return [[self._respond(prompt)] * max(n, 1) for prompt in prompts]

    def _respond(self, prompt):
        user_content = prompt[-1]["content"]
        if "reviewing records" in user_content or "candidate topics" in user_content:
            return json.dumps(_CODEBOOK)
        if "classify the record" in user_content:
            return json.dumps({"topic": _topic(user_content.split("Record: ", 1)[1])})
        raise AssertionError(f"unexpected prompt: {user_content[:100]}")


def _rows(docs=DOCS):
    return [{"text": doc, "id": i} for i, doc in enumerate(docs)]


def _filter(items, model=None, embed=None, **kwargs):
    kwargs.setdefault("num_clusters", 3)
    return by_topic_frequency(
        items,
        field="text",
        model=model or _FakeTopicModel(),
        context=CONTEXT,
        embed_fn=embed or _CountingEmbed(),
        **kwargs,
    )


def _topics_of(rows):
    return [_topic(row["text"]) for row in rows]


def test_downsamples_over_represented_topic():
    """A topic above max_frequency keeps max_frequency of the input rows."""
    kept = _filter(_rows(), max_frequency=0.5)
    topics = _topics_of(kept)
    # 0.5 of 10 rows: 5 of the 7 cooking rows, plus every other row.
    assert topics.count("cooking") == 5
    assert topics[-3:] == ["space", "space", "music"]


def test_default_max_frequency_is_twice_the_uniform_share():
    """With 3 topics the default cap is 2/3, so the 0.7 topic keeps 6 of 10."""
    assert _topics_of(_filter(_rows())) == ["cooking"] * 6 + ["space", "space", "music"]
    assert len(_filter(_rows(), num_clusters=3, max_frequency=0.7)) == len(DOCS)


def test_min_frequency_drops_rare_topics():
    """Rows of a topic below min_frequency are dropped too."""
    kept = _filter(_rows(), min_frequency=0.15, max_frequency=1.0)
    assert set(_topics_of(kept)) == {"cooking", "space"}
    assert len(kept) == 9


def test_bounds_are_inclusive():
    """A topic exactly at min_frequency or max_frequency is kept."""
    assert len(_filter(_rows(), min_frequency=0.1, max_frequency=0.7)) == len(DOCS)
    # Cooking is capped at 0.2 (2 rows), which is still >= min_frequency.
    capped = _filter(_rows(), min_frequency=0.2, max_frequency=0.2)
    assert _topics_of(capped) == ["cooking", "cooking", "space", "space"]


def test_downsample_caps_over_represented_topic():
    """A large topic keeps max_frequency of the input rows."""
    kept = _filter(_rows(), max_frequency=0.3)
    topics = _topics_of(kept)
    # 0.3 of 10 rows: 3 of the 7 cooking rows, plus every other row.
    assert topics.count("cooking") == 3
    assert topics.count("space") == 2 and topics.count("music") == 1
    ids = [row["id"] for row in kept]
    assert ids == sorted(ids)  # input order is kept


def test_downsample_is_seeded():
    """The same sample_seed keeps the same rows, even when the clustering reruns."""
    model, embed = _FakeTopicModel(), _CountingEmbed()
    first = _filter(_rows(), model, embed, max_frequency=0.3)
    assert _filter(_rows(), max_frequency=0.3, _no_cache=True) == first
    calls = (model.calls, embed.calls)
    other_seeds = [
        _filter(_rows(), model, embed, max_frequency=0.3, sample_seed=seed)
        for seed in range(1, 6)
    ]
    assert any(kept != first for kept in other_seeds)
    assert (model.calls, embed.calls) == calls  # no re-clustering


def test_downsample_drops_cap_below_min_frequency():
    """A cap under min_frequency leaves too few rows, so the topic is dropped."""
    # The cap is 2 of 10 rows (0.2), below min_frequency 0.25.
    assert _filter(_rows(), min_frequency=0.25, max_frequency=0.25) == []


def test_topic_field_names_each_kept_row():
    """topic_field adds the topic name to kept rows; input rows are untouched."""
    items = _rows()
    kept = _filter(items, max_frequency=1.0, topic_field="topic")
    assert [row["topic"] for row in kept] == _topics_of(items)
    assert [{k: v for k, v in row.items() if k != "topic"} for row in kept] == items
    assert items == _rows()


def test_changing_thresholds_makes_no_new_calls():
    """New frequency bounds, sample_seed, or topic_field reuse the clustering."""
    model, embed = _FakeTopicModel(), _CountingEmbed()
    _filter(_rows(), model, embed, max_frequency=0.5)
    calls = (model.calls, embed.calls)
    assert calls[0] > 0 and calls[1] == 1
    _filter(_rows(), model, embed, max_frequency=0.8)
    _filter(_rows(), model, embed, min_frequency=0.15, max_frequency=0.9)
    _filter(_rows(), model, embed, max_frequency=0.3, sample_seed=7)
    _filter(_rows(), model, embed, max_frequency=0.5, topic_field="topic")
    # Rows that differ only outside the clustered field reuse it too.
    relabeled = [{**row, "id": -row["id"]} for row in _rows()]
    _filter(relabeled, model, embed, max_frequency=0.5)
    assert (model.calls, embed.calls) == calls
    # A different num_clusters is a different clustering.
    with pytest.raises(ValueError, match="need more than num_clusters"):
        _filter(_rows(), model, embed, num_clusters=10)


def test_changing_clustered_values_reclusters():
    """A new value in the clustered field is a new clustering, not a cache hit."""
    model, embed = _FakeTopicModel(), _CountingEmbed()
    _filter(_rows(), model, embed, max_frequency=0.5)
    calls = (model.calls, embed.calls)
    edited = _rows(DOCS[:-1] + ["Practicing guitar chords for an hour every night."])
    _filter(edited, model, embed, max_frequency=0.5)
    assert model.calls > calls[0] and embed.calls == calls[1] + 1


def test_topics_match_topic_based_clustering():
    """The filter's topics are topic_based_clustering's, which is unchanged."""
    items = _rows()
    model, embed = _FakeTopicModel(), _CountingEmbed()
    kept = _filter(items, model, embed, max_frequency=1.0, topic_field="topic")
    calls = model.calls
    clusters = topic_based_clustering(
        [{"text": row["text"]} for row in items],
        field="text",
        model=model,
        context=CONTEXT,
        num_clusters=3,
        embed_fn=embed,
    )
    assert model.calls == calls  # the filter cached this very clustering
    assert [row["topic"] for row in kept] == clusters == _topics_of(items)


def test_scores_recorded_in_step(monkeypatch):
    """Each row's score is its topic's share, kept between the bounds."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")

    @step
    def balanced(rows):
        return _filter(rows, min_frequency=0.15, max_frequency=0.5)

    kept = balanced(_rows())
    scores = _tracking.step_records()["balanced"].ops[0].scores
    assert scores is not None
    assert (scores.label, scores.keep, scores.lower, scores.upper) == (
        "topic share, capped",
        "between",
        0.15,
        0.5,
    )
    # Cooking (0.7) keeps 5 rows at the capped 0.5; music (0.1) is too rare.
    assert sorted(scores.values[:7]) == pytest.approx([0.5] * 5 + [0.7] * 2)
    assert scores.values[7:] == pytest.approx([0.2, 0.2, 0.1])
    assert len(kept) == 7


def test_downsample_scores_match_kept_rows(monkeypatch):
    """Kept rows of a capped topic score at the capped share."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")

    @step
    def capped(rows):
        return _filter(rows, max_frequency=0.3)

    kept = capped(_rows())
    scores = _tracking.step_records()["capped"].ops[0].scores
    assert scores is not None and scores.label == "topic share, capped"
    cooking = sorted(scores.values[:7])
    assert cooking == pytest.approx([0.3] * 3 + [0.7] * 4)
    in_bounds = [v for v in scores.values if scores.lower <= v <= scores.upper]
    assert len(in_bounds) == len(kept) == 6


def test_counts_as_a_diversity_filter(monkeypatch):
    """A pipeline that runs it gets no "no diversity filter ran" warning."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")

    @step
    def balanced_topics(rows):
        return _filter(rows, max_frequency=0.5)

    @pipeline
    def run(rows):
        return balanced_topics(rows)

    run(_rows())
    rec = _tracking.last_pipeline()
    assert rec is not None
    joined = " | ".join(rec.warnings)
    assert "no diversity filter ran" not in joined
    assert "no quality filter ran" in joined


def test_empty_input_makes_no_calls():
    """Empty input returns [] without clustering."""
    model, embed = _FakeTopicModel(), _CountingEmbed()
    assert _filter([], model, embed) == []
    assert (model.calls, embed.calls) == (0, 0)


def test_missing_field_raises_key_error():
    """A row without the clustered field raises KeyError."""
    with pytest.raises(KeyError, match="by_topic_frequency"):
        _filter([{"other": "x"}])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(num_clusters=1), "num_clusters"),
        (dict(max_frequency=0.0), "max_frequency"),
        (dict(max_frequency=1.5), "max_frequency"),
        (dict(min_frequency=-0.1), "min_frequency"),
        (dict(min_frequency=0.6, max_frequency=0.5), "min_frequency"),
    ],
)
def test_invalid_parameters_raise(kwargs, match):
    """Out-of-range frequencies or num_clusters raise before any model call."""
    model = _FakeTopicModel()
    with pytest.raises(ValueError, match=match):
        _filter(_rows(), model, **kwargs)
    assert model.calls == 0
