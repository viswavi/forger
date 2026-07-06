"""Topic-based clustering via ClusterFusion (arXiv:2512.04350):
embedding-guided partition -> sharded LLM topic summarization -> LLM
topic assignment.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

import tiktoken

from pyforger.acquire.model import Message, Model
from pyforger.op import op
from pyforger.utils import get_field, get_formatted_logger

logger = get_formatted_logger("Inspect")

_DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

_SUMMARIZE_SYSTEM = (
    "You are an intelligent assistant skilled in summarizing and extracting insights."
)
_SUMMARIZE_USER = """You are now tasked with reviewing records related to {context}
Each record is separated by ";" symbols.

Records: {records}

Your goal is to extract the key topics from these records. Extract at most {num_clusters} topics.

Return the result as a single JSON object mapping each topic name (a short keyphrase) to a one-sentence description of that topic. Output only the JSON object."""

_AGGREGATE_SYSTEM = (
    "You are an intelligent assistant skilled in organizing and consolidating taxonomies."
)
_AGGREGATE_USER = """Below are candidate topics proposed independently from {num_shards} different samples of a corpus of {context}
Each entry is "topic name: description". Many topics overlap or are near-duplicates.

Candidate topics:
{candidates}

Consolidate these into exactly {num_clusters} final, mutually-distinct topics that together cover the corpus. Merge synonyms/overlaps, prefer clear keyphrase names, and keep topics at a consistent level of granularity. The total number of final topics must be exactly {num_clusters}, not more and not fewer.

Return the result as a single JSON object mapping each final topic name (a short keyphrase) to a one-sentence description. Output only the JSON object."""

_ASSIGN_SYSTEM = (
    "You are a helpful assistant, that can help me label each record into topics."
)
_ASSIGN_USER = """Following record is about {context}
Please classify the record into exactly one of the following topics, which are represented as a dictionary whose keys are topic names and values are descriptions:
{topics}

Record: {record}

Return the result in JSON format with a single key 'topic' whose value is exactly one of the topic names above (copied verbatim)."""


@op(name="inspect.topic_based_clustering")
def topic_based_clustering(
    items: list[dict],
    *,
    field: str,
    model: Model,
    context: str,
    num_clusters: int = 10,
    embed_fn: Callable[[list[str]], Any] | None = None,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    sample_size: int = 1200,
    tokens_per_shard: int = 100_000,
    summarize_truncation: int = 700,
    assign_truncation: int = 1500,
    max_assign_retries: int = 5,
    seed: int = 0,
) -> list[str]:
    """Cluster rows into ``num_clusters`` LLM-named topics by one field.

    Returns one topic name per row, aligned with ``items``. ``context``
    describes the corpus and what topics should capture, e.g.
    "competition math problems. Topics should describe the technique,
    NOT the numeric answer.". Embeddings default to ``embedding_model``
    via sentence-transformers (GPU when available); pass ``embed_fn``
    to use another backend.
    """
    if num_clusters < 2:
        raise ValueError("num_clusters must be at least 2")
    if not items:
        return []

    row_docs = [
        str(get_field(item, field, "topic_based_clustering")) for item in items
    ]
    unique_docs = list(dict.fromkeys(row_docs))
    if len(unique_docs) <= num_clusters:
        raise ValueError(
            f"topic_based_clustering: need more than num_clusters "
            f"({num_clusters}) unique '{field}' values, got {len(unique_docs)}"
        )

    embeddings = (
        embed_fn(unique_docs)
        if embed_fn is not None
        else _default_embed(unique_docs, embedding_model)
    )
    shards = _partition(
        unique_docs,
        embeddings,
        num_clusters,
        sample_size,
        tokens_per_shard,
        summarize_truncation,
        seed,
    )
    shard_topics = _summarize_shards(shards, model, context, num_clusters)
    codebook = _build_codebook(shard_topics, model, context, num_clusters)
    logger.info("topic_based_clustering: codebook topics: %s", list(codebook))
    topic_of = _assign_topics(
        unique_docs, model, context, codebook, assign_truncation, max_assign_retries
    )
    return [topic_of[doc] for doc in row_docs]


def _default_embed(docs: list[str], embedding_model: str) -> Any:
    """Embed documents with sentence-transformers, preferring the GPU."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "topic_based_clustering needs its default embedding backend: "
            "`pip install pyforger[clustering]`, or pass embed_fn=."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning(
            "topic_based_clustering: no GPU detected; embedding %d documents "
            "with %s on CPU may be slow. Pass embed_fn= to use a different "
            "embedding backend.",
            len(docs),
            embedding_model,
        )
    encoder = SentenceTransformer(embedding_model, device=device)
    return encoder.encode(
        docs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
    )


def _partition(
    docs: list[str],
    embeddings: Any,
    num_clusters: int,
    sample_size: int,
    tokens_per_shard: int,
    truncation: int,
    seed: int,
) -> list[list[str]]:
    """Step 1: sample a KMeans-group-ordered subset, stride it into shards."""
    try:
        import numpy as np
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(
            "topic_based_clustering requires numpy and scikit-learn: "
            "`pip install pyforger[clustering]`."
        ) from e

    num_groups = min(2 * num_clusters, len(docs))
    labels = KMeans(n_clusters=num_groups, random_state=seed, n_init=10).fit_predict(
        np.asarray(embeddings)
    )
    rng = np.random.default_rng(seed)
    per_group = max(1, sample_size // num_groups)
    sample_idx: list[int] = []  # group-ordered (Eq. 4 of the paper)
    for group in range(num_groups):
        group_idx = np.where(labels == group)[0]
        take = min(per_group, len(group_idx))
        sample_idx.extend(int(i) for i in rng.choice(group_idx, size=take, replace=False))

    records = [docs[i].replace("\n", " ")[:truncation] for i in sample_idx]
    encoding = tiktoken.get_encoding("o200k_base")
    total_tokens = sum(len(encoding.encode(record)) for record in records)
    num_shards = max(1, min(-(-total_tokens // tokens_per_shard), len(records)))
    logger.info(
        "topic_based_clustering: summarizing %d records (~%d tokens) in %d shard(s)",
        len(records),
        total_tokens,
        num_shards,
    )
    return [records[i::num_shards] for i in range(num_shards)]


def _summarize_shards(
    shards: list[list[str]], model: Model, context: str, num_clusters: int
) -> list[dict[str, str]]:
    """Step 2a: summarize each shard into candidate topics."""
    prompts = [
        [
            {"role": "system", "content": _SUMMARIZE_SYSTEM},
            {
                "role": "user",
                "content": _SUMMARIZE_USER.format(
                    context=context,
                    records=" ; ".join(shard),
                    num_clusters=num_clusters,
                ),
            },
        ]
        for shard in shards
    ]
    responses = model.generate_batch(prompts, temperature=0.2)
    shard_topics = [_parse_topics(shard_responses[0]) for shard_responses in responses]
    unparseable = sum(1 for topics in shard_topics if not topics)
    if unparseable:
        logger.warning(
            "topic_based_clustering: %d shard summaries unparseable", unparseable
        )
    return [topics for topics in shard_topics if topics]


def _build_codebook(
    shard_topics: list[dict[str, str]],
    model: Model,
    context: str,
    num_clusters: int,
) -> dict[str, str]:
    """Step 2b: aggregate candidate topics into a codebook of topics."""
    candidates = "\n".join(
        f"- {name}: {description}"
        for topics in shard_topics
        for name, description in topics.items()
    )
    messages: list[Message] = [
        {"role": "system", "content": _AGGREGATE_SYSTEM},
        {
            "role": "user",
            "content": _AGGREGATE_USER.format(
                num_shards=len(shard_topics),
                context=context,
                candidates=candidates,
                num_clusters=num_clusters,
            ),
        },
    ]
    for _ in range(3):  # retry on unparseable JSON
        try:
            parsed = _parse_json(model.generate(messages, temperature=0.0)[0])
        except ValueError:
            continue
        codebook = {str(name): str(desc) for name, desc in parsed.items()}
        if codebook:
            if len(codebook) != num_clusters:
                logger.warning(
                    "topic_based_clustering: codebook has %d topics instead of %d",
                    len(codebook),
                    num_clusters,
                )
            return codebook
    raise RuntimeError(
        "topic_based_clustering: model never returned a parseable codebook"
    )


def _assign_topics(
    docs: list[str],
    model: Model,
    context: str,
    codebook: dict[str, str],
    truncation: int,
    max_retries: int,
) -> dict[str, str]:
    """Step 3: classify every document into one codebook topic."""
    topics_str = json.dumps(codebook, indent=2)
    topic_of: dict[str, str] = {}
    pending = list(docs)
    for _ in range(max_retries):  # rerun documents whose label was invalid
        prompts = [
            [
                {"role": "system", "content": _ASSIGN_SYSTEM},
                {
                    "role": "user",
                    "content": _ASSIGN_USER.format(
                        context=context,
                        topics=topics_str,
                        record=doc.replace("\n", " ")[:truncation],
                    ),
                },
            ]
            for doc in pending
        ]
        responses = model.generate_batch(prompts, temperature=0.0)
        still_pending = []
        for doc, doc_responses in zip(pending, responses):
            try:
                label = str(_parse_json(doc_responses[0]).get("topic", "")).strip()
            except ValueError:
                label = ""
            matched = _match_topic(label, codebook)
            if matched:
                topic_of[doc] = matched
            else:
                still_pending.append(doc)
        pending = still_pending
        if not pending:
            break
    if pending:
        fallback = next(iter(codebook))
        logger.warning(
            "topic_based_clustering: %d documents could not be classified; "
            "assigning fallback topic %r",
            len(pending),
            fallback,
        )
        topic_of.update(dict.fromkeys(pending, fallback))
    return topic_of


def _match_topic(label: str, codebook: dict[str, str]) -> str | None:
    """Match a label to a codebook key: case-insensitive, then substring."""
    low = label.lower()
    if not low:
        return None
    for name in codebook:
        if name.lower() == low:
            return name
    for name in codebook:
        if name.lower() in low or low in name.lower():
            return name
    return None


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    stripped = re.sub(
        r"```$", "", re.sub(r"^```(?:json)?", "", text.strip()).strip()
    ).strip()
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in response: {text[:200]!r}")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            raise ValueError(f"invalid JSON in response: {text[:200]!r}")
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got: {text[:200]!r}")
    return parsed


def _parse_topics(text: str) -> dict[str, str]:
    """Parse a shard summary into {topic: description}, {} when unparseable."""
    try:
        parsed = _parse_json(text)
    except ValueError:
        return {}
    topics: dict[str, str] = {}
    for name, value in parsed.items():
        if isinstance(value, dict):  # tolerate nested {name: description}
            topics.update((str(k), str(v)) for k, v in value.items())
        else:
            topics[str(name)] = str(value)
    return topics
