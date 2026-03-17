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
    query: str | None = None,
    n: int = 10,
    seed: int | None = None,
    criteria: dict[str, Any] | None = None,
    name: str = "retrieve",
    **kwargs: Any,
) -> DataSpec:
    """Build an ACQUIRE node that, when run, calls source.fetch() and returns the list.

    Args:
        source: The corpus or API wrapper; must implement fetch().
        mode: How to select segments (QUERY, RANDOM, or STRUCTURED).
        query: Search or query string; used when mode is QUERY.
        n: Number of segments to return.
        seed: Random seed for reproducible sampling when mode is RANDOM.
        criteria: Field constraints when mode is STRUCTURED.
        name: Human-readable label for the node (for debugging).
        **kwargs: Passed through to source.fetch().

    Returns:
        DataSpec whose execute_fn returns the list from source.fetch(...).
    """

    def _execute():
        return source.fetch(
            mode, query=query, n=n, seed=seed, criteria=criteria, **kwargs
        )

    # ACQUIRE nodes have no inputs; execute_fn is called at run().
    return DataSpec(NodeType.ACQUIRE, [], _execute, name)


def retrieve_from_web(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
    n: int = 10,
    seed: int | None = None,
    criteria: dict[str, Any] | None = None,
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
        query=query,
        n=n,
        seed=seed,
        criteria=criteria,
        name=name,
        **kwargs,
    )


def retrieve_from_wiki(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
    n: int = 10,
    seed: int | None = None,
    criteria: dict[str, Any] | None = None,
    language: str = "en",
    name: str = "retrieve_from_wiki",
    **kwargs: Any,
) -> DataSpec:
    """Retrieve from Wikipedia (e.g. wikipedia-api / MediaWiki).

    Returns:
        DataSpec whose execute_fn returns the fetched list.
    """
    source = WikiRetrievalSource(language=language, **kwargs)
    return retrieve(
        source,
        mode=mode,
        query=query,
        n=n,
        seed=seed,
        criteria=criteria,
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
    query: str | None = None,
    n: int = 10,
    seed: int | None = None,
    criteria: dict[str, Any] | None = None,
    name: str = "retrieve_from_hf",
    **kwargs: Any,
) -> DataSpec:
    """Retrieve from Hugging Face datasets (load_dataset + shuffle/filter/select).

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
        query=query,
        n=n,
        seed=seed,
        criteria=criteria,
        name=name,
        **kwargs,
    )


def retrieve_from_arxiv(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
    n: int = 10,
    seed: int | None = None,
    criteria: dict[str, Any] | None = None,
    name: str = "retrieve_from_arxiv",
    **kwargs: Any,
) -> DataSpec:
    """Build an ACQUIRE node that fetches papers from arXiv when the graph runs.

    Args:
        mode: QUERY (text search), RANDOM (sample from category), or STRUCTURED
            (filter by criteria).
        query: Search string for QUERY mode; default "all:*" if empty.
        n: Number of papers to return.
        seed: Random seed for RANDOM mode (optional; for reproducibility).
        criteria: For STRUCTURED mode only. Keys (all optional): id_list or ids
            (list of arXiv ids), category (e.g. "cs.LG"), author, title, query
            (raw string). Sort via kwargs sort_by/sort_order, not here.
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
        query=query,
        n=n,
        seed=seed,
        criteria=criteria,
        name=name,
        **kwargs,
    )


def retrieve_from_s2(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
    n: int = 10,
    seed: int | None = None,
    criteria: dict[str, Any] | None = None,
    name: str = "retrieve_from_s2",
    **kwargs: Any,
) -> DataSpec:
    """Build an ACQUIRE node that fetches papers from Semantic Scholar when the graph runs.

    Args:
        mode: QUERY (text search), RANDOM (sample from pool), or STRUCTURED
            (filter by criteria).
        query: Search string for QUERY mode; default "machine learning" if empty.
        n: Number of papers to return.
        seed: Random seed for RANDOM mode (optional; for reproducibility).
        criteria: For STRUCTURED mode only. Keys (all optional): query, author
            (folded into query), year, venue, fields_of_study. Same pagination
            as QUERY; cap 1000.
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
        query=query,
        n=n,
        seed=seed,
        criteria=criteria,
        name=name,
        **kwargs,
    )
