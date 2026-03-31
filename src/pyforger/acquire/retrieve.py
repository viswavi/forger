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
    fetch_full_content: bool = False,
    name: str = "retrieve_from_web",
    **kwargs: Any,
) -> DataSpec:
    """Build an ACQUIRE node that searches the web via the Brave Search API.

    Only QUERY mode is supported; RANDOM raises ``NotImplementedError``
    because web search has no reproducible random corpus.

    Args:
        mode: Must be QUERY.
        query: Search string (required; raises ``ValueError`` if empty).
        n: Number of results to return.
        seed: Unused (RANDOM mode unsupported).
        base_url: Optional domain restriction.  When set,
            ``site:{base_url}`` is prepended to the query.
        fetch_full_content: When True, fetch full page text from each
            result URL after search.  When False
            (default), ``text`` is the short search snippet.
        name: Label for the node.
        **kwargs: Passed to WebRetrievalSource and fetch().  Recognised
            keys: ``api_key`` (str, required -- Brave subscription token),
            ``timeout`` (float), ``max_content_length`` (int, default 1 MB).

    Returns:
        DataSpec whose execute_fn returns list of dicts with ``text``
        (snippet or full article), ``id`` (URL), ``title``, ``url``.

    Licensing note:
        Brave's ToS require a plan with storage rights if results are used
        for LLM training.  See https://brave.com/search/api/ for details.
    """
    source_kwargs = {k: v for k, v in kwargs.items() if k not in ("api_key",)}
    source = WebRetrievalSource(
        base_url=base_url,
        api_key=kwargs.get("api_key"),
        fetch_full_content=fetch_full_content,
        **source_kwargs,
    )
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
    """Build an ACQUIRE node that fetches articles from Wikipedia when the graph runs.

    Args:
        mode: QUERY (full-text search) or RANDOM (random article sampling).
        query: Search string for QUERY mode; defaults to "Wikipedia" if empty.
        n: Number of articles to return.
        seed: Random seed for RANDOM mode (optional; for reproducibility).
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
    text_column: str | None = None,
    mode: RetrieveMode = RetrieveMode.RANDOM,
    *,
    query: str | None = None,
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
        n=n,
        seed=seed,
        name=name,
        **kwargs,
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
