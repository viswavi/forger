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
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    query: str | None = None,
    n: int = 10,
    seed: int | None = None,
    dataset: str | None = None,
    config: str | None = None,
    split: str = "train",
    token: str | None = None,
    name: str = "retrieve_from_hf",
    **kwargs: Any,
) -> DataSpec:
    """Discover a Hugging Face dataset and return raw rows.

    Two-stage pipeline:
        1. **Discovery** -- searches the HF Hub for datasets matching
           *query* via ``HfApi.list_datasets(search=query)``.
        2. **Loading** -- loads the top match with
           ``datasets.load_dataset()`` and returns up to *n* rows.

    When *dataset* is provided, stage 1 is skipped and the given
    dataset is loaded directly.

    Args:
        mode: QUERY (load in dataset order) or RANDOM (shuffle first).
        query: Search string for Hub discovery; required when *dataset*
            is not provided.
        n: Maximum number of rows to return.
        seed: Random seed for reproducible sampling in RANDOM mode.
        dataset: Optional HF dataset identifier (e.g. "ibm/duorc").
            Skips Hub discovery when provided.
        config: Optional dataset config/subset name (e.g. "SelfRC").
        split: Dataset split (default "train").
        token: HF auth token for gated datasets and API calls.
        name: Label for the node.
        **kwargs: Forwarded to ``datasets.load_dataset()``.

    Returns:
        DataSpec whose execute_fn returns a list of flat dicts with
        ``row_idx``, ``dataset_id``, and the dataset's column values.
    """
    source = HFRetrievalSource(
        dataset=dataset,
        config=config,
        split=split,
        token=token,
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
