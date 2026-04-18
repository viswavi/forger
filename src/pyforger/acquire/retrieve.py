"""Generic retrieve and convenience wrappers for acquire operations.

retrieve() builds an ACQUIRE DataSpec that calls a RetrievalSource at run time;
convenience wrappers construct the appropriate source.
"""

from __future__ import annotations

from typing import Any

from pyforger.acquire.sources import (
    ArxivRetrievalSource,
    HFRetrievalSource,
    RetrievalSource,
    RetrieveMode,
    S2RetrievalSource,
    WebRetrievalSource,
    WikiRetrievalSource,
)
from pyforger.core import DataSpec, NodeType


def retrieve(
    source: RetrievalSource,
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    name: str = "retrieve",
    **kwargs: Any,
) -> DataSpec:
    """Build an ACQUIRE node that, when run, calls source.fetch() and returns the list.

    Each returned row is stamped with a ``"query"`` field indicating which
    query produced it.  source.fetch() is called once per query string and the
    results are concatenated into a single flat ``list[dict]``.

    For a single query, pass a one-element list: ``queries=["my query"]``.

    Args:
        source: The corpus or API wrapper; must implement fetch().
        mode: How to select segments (QUERY or RANDOM).
        queries: List of query strings.  Pass a single-element list for one
            query.  ``None`` (default) means no query is used (e.g. random
            sampling).
        n: Number of segments to return **per query**.
        seed: Random seed for reproducible sampling when mode is RANDOM.
        name: Human-readable label for the node (for debugging).
        **kwargs: Passed through to source.fetch().

    Returns:
        DataSpec whose execute_fn returns a flat list of dicts, each with a
        ``"query"`` key set to the originating query string (or ``None``).
    """
    resolved_queries: list[str | None] = (
        list(queries) if queries is not None else [None]
    )

    def _execute():
        all_rows: list[dict[str, Any]] = []
        for q in resolved_queries:
            rows = source.fetch(mode, query=q, n=n, seed=seed, **kwargs)
            for row in rows:
                row["query"] = q
            all_rows.extend(rows)
        return all_rows

    return DataSpec(NodeType.ACQUIRE, [], _execute, name)


def retrieve_from_web(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    base_url: str | None = None,
    name: str = "retrieve_from_web",
    **kwargs: Any,
) -> DataSpec:
    """Retrieve segments from a web corpus (e.g. crawl/search).

    Returns:
        DataSpec whose execute_fn returns the fetched list.
    """
    source = WebRetrievalSource(base_url=base_url, **kwargs)
    return retrieve(
        source,
        mode=mode,
        queries=queries,
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )


def retrieve_from_wiki(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    language: str = "en",
    name: str = "retrieve_from_wiki",
    **kwargs: Any,
) -> DataSpec:
    """Build an ACQUIRE node that fetches articles from Wikipedia when the graph runs.

    Args:
        mode: QUERY (full-text search) or RANDOM (random article sampling).
        query: Search string for QUERY mode; defaults to "Wikipedia" if empty.
        n: Number of articles to return.
        seed: Accepted for interface compatibility; ignored by Wikipedia
            RANDOM mode (the MediaWiki API does not support seeded random).
        language: Wikipedia language code (e.g. "en", "de").
        name: Label for the node.
        **kwargs: Session options (e.g. timeout).

    Returns:
        DataSpec whose execute_fn returns list of dicts in DPR/BEIR format
        with text (intro extract), id (page ID), title, summary.
    """
    source = WikiRetrievalSource(language=language, **kwargs)
    return retrieve(
        source,
        mode=mode,
        queries=queries,
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )


def retrieve_from_hf(
    dataset: str,
    config: str | None = None,
    split: str = "train",
    text_column: str | None = None,
    mode: RetrieveMode = RetrieveMode.RANDOM,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    name: str = "retrieve_from_hf",
    **kwargs: Any,
) -> DataSpec:
    """Retrieve from Hugging Face datasets (load_dataset + shuffle/select).

    Returns:
        DataSpec whose execute_fn returns the fetched list.
    """
    source = HFRetrievalSource(
        dataset=dataset,
        config=config,
        split=split,
        text_column=text_column,
        **kwargs,
    )
    return retrieve(
        source,
        mode=mode,
        queries=queries,
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )


def retrieve_from_arxiv(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    name: str = "retrieve_from_arxiv",
    **kwargs: Any,
) -> DataSpec:
    """Build an ACQUIRE node that fetches papers from arXiv when the graph runs.

    Args:
        mode: QUERY (text search) or RANDOM (sample from category).
        query: Search string for QUERY mode; default "all:*" if empty.
        n: Number of papers to return.
        seed: Random seed for RANDOM mode (optional; for reproducibility).
        name: Label for the node.
        **kwargs: Arxiv options: sort_by, sort_order, random_category (RANDOM),
            page_size, delay_seconds, num_retries (Client).

    Returns:
        DataSpec whose execute_fn returns list of dicts with text (title +
        abstract), id, title, summary.
    """
    source = ArxivRetrievalSource(**kwargs)
    return retrieve(
        source,
        mode=mode,
        queries=queries,
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )


def retrieve_from_s2(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    name: str = "retrieve_from_s2",
    **kwargs: Any,
) -> DataSpec:
    """Build an ACQUIRE node that fetches papers from S2 when the graph runs.

    Args:
        mode: QUERY (text search) or RANDOM (sample from pool).
        query: Search string for QUERY mode; default "machine learning" if empty.
        n: Number of papers to return.
        seed: Random seed for RANDOM mode (optional; for reproducibility).
        name: Label for the node.
        **kwargs: S2 client options: api_key, timeout, api_url, debug, retry.
            For RANDOM mode: fields_of_study (e.g. ["Computer Science"]) to
            build the search pool.

    Returns:
        DataSpec whose execute_fn returns list of dicts with text (title +
        abstract), id, title, summary.
    """
    source = S2RetrievalSource(**kwargs)
    return retrieve(
        source,
        mode=mode,
        queries=queries,
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )
