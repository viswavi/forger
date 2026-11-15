"""Topic-distribution filter.

Cluster rows into LLM-named topics (``inspect.topic_based_clustering``),
downsample topics that hold too large a share of the rows, and drop topics
that hold too small a share.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from typing import Any

from pyforger import _tracking
from pyforger.acquire.model import Model
from pyforger.inspect.cluster import topic_based_clustering
from pyforger.op import op
from pyforger.utils import get_field


@op(name="filter.by_topic_frequency")
def by_topic_frequency(
    items: list[dict],
    *,
    field: str,
    model: Model,
    context: str | None = None,
    num_clusters: int = 20,
    min_frequency: float = 0.0,
    max_frequency: float | None = None,
    topic_field: str | None = None,
    embed_fn: Callable[[list[str]], Any] | None = None,
    seed: int = 0,
    sample_seed: int = 0,
    **cluster_kwargs: Any,
) -> list[dict]:
    """Downsample over-represented topics and drop rare ones.

    Rows are clustered by ``field`` into ``num_clusters`` topics with
    ``inspect.topic_based_clustering`` (``context``, ``embed_fn``, ``seed``,
    and ``cluster_kwargs`` such as ``embedding_model=`` go to it). A topic's
    frequency is its share of the input rows, so ``1 / num_clusters`` is the
    uniform share. A topic above ``max_frequency`` keeps a random subset of its
    rows, as many as fit in ``max_frequency`` of the input (e.g. 10 of 100 rows
    at 0.1); ``max_frequency`` defaults to ``2 / num_clusters``. A topic below
    ``min_frequency`` is dropped, and so is an over-represented topic whose
    capped share would fall below it. Kept rows stay in input order.
    ``sample_seed`` picks the subset; changing it does not re-cluster.
    ``topic_field="topic"`` adds each kept row's topic name to it.

    The clustering makes one LLM call per unique ``field`` value (2,000
    distinct rows are 2,000 calls), each with every topic's description in its
    prompt, so its cost grows with the rows, and a reasoning model pays for
    its reasoning on each call.

    The clustering is cached by its own op, so changing ``min_frequency``,
    ``max_frequency``, ``sample_seed``, or ``topic_field`` makes no new model
    or embedding calls. In a ``@step`` report each row's score is its topic's
    share; the rows kept from a capped topic are scored at the capped share, so
    the report's kept count matches the rows returned.
    """
    if num_clusters < 2:
        raise ValueError("num_clusters must be at least 2")
    if max_frequency is None:
        max_frequency = 2 / num_clusters
    if not 0.0 < max_frequency <= 1.0:
        raise ValueError("max_frequency must be in (0, 1]")
    if not 0.0 <= min_frequency <= max_frequency:
        raise ValueError(
            f"min_frequency must be in [0, max_frequency], got {min_frequency} "
            f"with max_frequency {max_frequency}"
        )
    if not items:
        _tracking.record_scores(
            "topic share", [], keep="between", lower=min_frequency, upper=max_frequency
        )
        return []

    # Cluster only the field's values, so rows that differ elsewhere (e.g. a
    # field added by a later alter) reuse the cached topics.
    values = [get_field(item, field, "by_topic_frequency") for item in items]
    topics = topic_based_clustering(
        [{field: value} for value in values],
        field=field,
        model=model,
        context=context,
        num_clusters=num_clusters,
        embed_fn=embed_fn,
        seed=seed,
        **cluster_kwargs,
    )
    n = len(items)
    share = {topic: count / n for topic, count in Counter(topics).items()}
    keep = [min_frequency <= share[topic] <= max_frequency for topic in topics]
    scores = [share[topic] for topic in topics]
    # The most rows a topic may keep, e.g. 10 of 100 rows at 0.1.
    cap = sum(1 for count in range(1, n + 1) if count / n <= max_frequency)
    rng = random.Random(sample_seed)
    over = sorted(topic for topic, s in share.items() if s > max_frequency)
    if cap / n < min_frequency:
        # The capped share is itself too rare (e.g. 2 of 7 rows = 0.29 with
        # min_frequency 0.3), so these topics are dropped.
        over = []
    for topic in over:
        rows = [i for i, t in enumerate(topics) if t == topic]
        for i in rng.sample(rows, cap):
            keep[i] = True
            scores[i] = cap / n
    label = "topic share, capped" if over else "topic share"
    _tracking.record_scores(
        label, scores, keep="between", lower=min_frequency, upper=max_frequency
    )
    return [
        {**item, topic_field: topic} if topic_field is not None else item
        for item, topic, kept in zip(items, topics, keep)
        if kept
    ]
