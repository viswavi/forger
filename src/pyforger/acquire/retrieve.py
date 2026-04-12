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
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    name: str = "retrieve",
    **kwargs: Any,
) -> DataSpec:
    """Build an ACQUIRE node that, when run, calls source.fetch() and returns the list.

    Each returned row is stamped with a ``"query"`` field indicating which
    query produced it.  When *queries* is provided, source.fetch() is called
    once per query string and the results are concatenated into a single flat
    ``list[dict]``.

    Args:
        source: The corpus or API wrapper; must implement fetch().
        mode: How to select segments (QUERY or RANDOM).
        query: Single search/query string. Mutually exclusive with *queries*.
        queries: List of query strings. Mutually exclusive with *query*.
        n: Number of segments to return **per query**.
        seed: Random seed for reproducible sampling when mode is RANDOM.
        name: Human-readable label for the node (for debugging).
        **kwargs: Passed through to source.fetch().

    Returns:
        DataSpec whose execute_fn returns a flat list of dicts, each with a
        ``"query"`` key set to the originating query string (or ``None``).

    Raises:
        ValueError: If both *query* and *queries* are provided.
    """
    if query is not None and queries is not None:
        raise ValueError("Provide either 'query' or 'queries', not both.")

    resolved_queries: list[str | None]
    if queries is not None:
        resolved_queries = list(queries)
    else:
        resolved_queries = [query]

    def _execute():
        all_rows: list[dict[str, Any]] = []
        for q in resolved_queries:
            rows = source.fetch(mode, query=q, n=n, seed=seed, **kwargs)
            for row in rows:
                row["query"] = q
            all_rows.extend(rows)
        return all_rows

    # ACQUIRE nodes have no inputs; execute_fn is called at run().
    return DataSpec(NodeType.ACQUIRE, [], _execute, name)


def retrieve_from_web(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
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
        query=query,
        queries=queries,
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )


def retrieve_from_wiki(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
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
    query: str | None = None,
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
        query=query,
        queries=queries,
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )


def retrieve_from_arxiv(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    name: str = "retrieve_from_arxiv",
    **kwargs: Any,
) -> DataSpec:
    """Retrieve from arXiv (arxiv package Search + Client).

    Returns:
        DataSpec whose execute_fn returns the fetched list.
    """
    source = ArxivRetrievalSource(**kwargs)
    return retrieve(
        source,
        mode=mode,
        query=query,
        queries=queries,
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )


def retrieve_from_s2(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    name: str = "retrieve_from_s2",
    **kwargs: Any,
) -> DataSpec:
    """Retrieve from Semantic Scholar.

    Returns:
        DataSpec whose execute_fn returns the fetched list.
    """
    source = S2RetrievalSource(**kwargs)
    return retrieve(
        source,
        mode=mode,
        query=query,
        queries=queries,
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )
