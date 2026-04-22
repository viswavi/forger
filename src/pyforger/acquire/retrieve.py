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
    fetch_full_content: bool = False,
    name: str = "retrieve_from_web",
    **kwargs: Any,
) -> DataSpec:
    """Build an ACQUIRE node that searches the web via Brave Search at run time.

    Only QUERY mode is supported; RANDOM raises ``NotImplementedError``
    because web search has no reproducible random corpus.

    Args:
        mode: Must be QUERY.
        queries: Search strings (required; raises ``ValueError`` if empty).
            ``n`` applies per query.
        n: Number of results per query.
        seed: Ignored (RANDOM mode unsupported).
        base_url: Optional domain restriction; when set, ``site:{base_url}``
            is prepended to every query.
        fetch_full_content: When True, fetch full page text from each result
            URL; when False (default), ``text`` is the Brave snippet.
        name: Node label.
        **kwargs: Web options — ``api_key`` (required, Brave subscription
            token), ``timeout``, ``max_content_length`` (default 1 MB).

    Returns:
        DataSpec producing dicts with ``text`` (snippet or full article),
        ``id`` (URL), ``title``, ``url``, and a ``"query"`` key per row.

    Licensing note:
        Brave's ToS require a plan with storage rights if results are used
        for LLM training. See https://brave.com/search/api/ for details.
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
    """Build an ACQUIRE node that fetches Wikipedia articles at run time.

    Args:
        mode: QUERY (full-text search) or RANDOM (random article sampling).
        queries: Query strings for QUERY mode; ``None`` falls back to
            ``"Wikipedia"``. ``n`` applies per query.
        n: Number of articles per query.
        seed: Ignored in RANDOM mode (MediaWiki has no seeded random).
        language: Wikipedia language code (e.g. "en", "de").
        name: Node label.
        **kwargs: Session options (e.g. timeout).

    Returns:
        DataSpec producing DPR/BEIR dicts (``text`` intro, ``id``, ``title``,
        ``summary``) with a ``"query"`` key per row.
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
    """Build an ACQUIRE node that loads rows from a Hugging Face dataset at run time.

    Unlike the other wrappers, ``queries`` drives Hub discovery (not content
    search) and is only used when ``dataset`` is ``None``; otherwise each query
    just re-triggers a fetch from the same dataset.

    Args:
        dataset: Explicit HF dataset id; skips Hub discovery.
        config: Dataset config name for ``load_dataset``.
        split: Dataset split (default ``"train"``).
        text_column: Reserved; unused.
        mode: QUERY (in order) or RANDOM (shuffle + sample). Default RANDOM.
        queries: Hub discovery strings; raises ``ValueError`` at run time if
            ``dataset`` is also ``None``. ``n`` applies per query.
        n: Max rows per query.
        seed: Seed for RANDOM shuffling.
        name: Node label.
        **kwargs: Forwarded to ``HFRetrievalSource`` and ``load_dataset``.

    Returns:
        DataSpec producing dicts with ``row_idx``, ``dataset_id``, raw columns,
        and a ``"query"`` key per row.
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
    """Build an ACQUIRE node that fetches papers from arXiv at run time.

    Args:
        mode: QUERY (text search) or RANDOM (sample from category).
        queries: arXiv query strings (e.g. ``"all:quantum"``); ``None`` falls
            back to ``"all:*"``. ``n`` applies per query.
        n: Number of papers per query.
        seed: Seed for RANDOM mode.
        name: Node label.
        **kwargs: arXiv options — sort_by, sort_order, random_category,
            page_size, delay_seconds, num_retries.

    Returns:
        DataSpec producing dicts with ``text`` (title + abstract), ``id``,
        ``title``, ``summary``, and a ``"query"`` key per row.
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
    """Build an ACQUIRE node that fetches papers from Semantic Scholar at run time.

    Args:
        mode: QUERY (text search) or RANDOM (sample from pool).
        queries: Query strings; ``None`` falls back to ``"machine learning"``.
            ``n`` applies per query.
        n: Number of papers per query.
        seed: Seed for RANDOM mode.
        name: Node label.
        **kwargs: S2 client options (api_key, timeout, api_url, debug, retry);
            plus ``fields_of_study`` for RANDOM pool.

    Returns:
        DataSpec producing dicts with ``text`` (title + abstract), ``id``,
        ``title``, ``summary``, and a ``"query"`` key per row.
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
