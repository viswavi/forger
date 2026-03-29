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
    name: str = "retrieve",
    **kwargs: Any,
) -> DataSpec:
    """Build an ACQUIRE node that, when run, calls source.fetch() and returns the list.

    Args:
        source: The corpus or API wrapper; must implement fetch().
        mode: How to select segments (QUERY or RANDOM).
        query: Search or query string; used when mode is QUERY.
        n: Number of segments to return.
        seed: Random seed for reproducible sampling when mode is RANDOM.
        name: Human-readable label for the node (for debugging).
        **kwargs: Passed through to source.fetch().

    Returns:
        DataSpec whose execute_fn returns the list from source.fetch(...).
    """

    def _execute():
        return source.fetch(mode, query=query, n=n, seed=seed, **kwargs)

    # ACQUIRE nodes have no inputs; execute_fn is called at run().
    return DataSpec(NodeType.ACQUIRE, [], _execute, name)


def retrieve_from_web(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
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
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )


def retrieve_from_wiki(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
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
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )


def retrieve_from_hf(
    dataset: str,
    config: str | None = None,
    split: str = "train",
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
    n: int = 10,
    seed: int | None = None,
    name: str = "retrieve_from_hf",
    **kwargs: Any,
) -> DataSpec:
    """Retrieve rows from a Hugging Face dataset (hybrid backend).

    Uses a hybrid backend depending on mode:
        QUERY      — hosted Dataset Viewer API (/search, BM25 full-text).
        RANDOM     — local datasets.load_dataset().shuffle().select().
        STRUCTURED — local datasets.load_dataset().filter() with
                     exact-match criteria.

    Args:
        dataset: HF dataset identifier (e.g. "ibm/duorc").
        config: Optional dataset config/subset name (e.g. "SelfRC").
        split: Dataset split (default "train").
        mode: QUERY, RANDOM, or STRUCTURED.
        query: Search string; required when mode is QUERY.
        n: Maximum number of rows to return.
        seed: Random seed for reproducible sampling in RANDOM mode.
        criteria: Exact-match field constraints for STRUCTURED mode.
        name: Label for the node.
        **kwargs: Passed to HFRetrievalSource — base_url, timeout,
            token, headers (API); remaining kwargs forwarded to
            load_dataset() for local modes.

    Returns:
        DataSpec whose execute_fn returns a list of flat dicts with
        row_idx and the dataset's column values.
    """
    source = HFRetrievalSource(
        dataset=dataset,
        config=config,
        split=split,
        **kwargs,
    )
    return retrieve(
        source,
        mode=mode,
        query=query,
        n=n,
        seed=seed,
        name=name,
    )


def retrieve_from_arxiv(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
    n: int = 10,
    seed: int | None = None,
    name: str = "retrieve_from_arxiv",
    **kwargs: Any,
) -> DataSpec:
    """Build an ACQUIRE node that fetches papers from arXiv when the graph runs.

    Args:
        mode: QUERY (text search) andRANDOM (sample from category).
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
        query=query,
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )


def retrieve_from_s2(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
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
        n=n,
        seed=seed,
        name=name,
        **kwargs,
    )
