"""Retrieval wrappers: call a RetrievalSource and return a flat list of dicts.

``retrieve()`` is the generic entry point; the convenience wrappers
construct the appropriate source for a backend (web, wiki, HF, arXiv, S2).
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
from pyforger.op import tracked


@tracked("acquire.retrieve")
def retrieve(
    source: RetrievalSource,
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Call source.fetch() once per query and concatenate results.

    Each row is stamped with a ``"query"`` field. With ``queries=None``,
    a single fetch with no query is performed (e.g. random sampling).
    Calls are not cached; inside a ``@step`` they are listed as
    ``acquire.retrieve``.

    Returns:
        The source's rows for each query in turn, each with a ``query`` field
        added (``None`` when ``queries`` is None).
    """
    resolved_queries: list[str | None] = (
        list(queries) if queries is not None else [None]
    )
    all_rows: list[dict[str, Any]] = []
    for q in resolved_queries:
        rows = source.fetch(mode, query=q, n=n, seed=seed, **kwargs)
        for row in rows:
            row["query"] = q
        all_rows.extend(rows)
    return all_rows


def retrieve_from_web(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Search the web via Brave's LLM Context API. QUERY mode only.

    Returns:
        Rows with ``text``, ``id`` (the page URL), ``title``, ``url``, and
        ``query``.
    """
    source_kwargs = {k: v for k, v in kwargs.items() if k not in ("api_key",)}
    source = WebRetrievalSource(
        base_url=base_url,
        api_key=kwargs.get("api_key"),
        **source_kwargs,
    )
    return retrieve(source, mode=mode, queries=queries, n=n, seed=seed, **source_kwargs)


def retrieve_from_wiki(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    language: str = "en",
    **kwargs: Any,
) -> list[dict]:
    """Fetch Wikipedia articles. QUERY (full-text) or RANDOM (random article).

    Returns:
        Rows with ``text`` (the intro extract), ``id`` (the page ID),
        ``title``, ``summary`` (same as ``text``), and ``query``.
    """
    source = WikiRetrievalSource(language=language, **kwargs)
    return retrieve(source, mode=mode, queries=queries, n=n, seed=seed, **kwargs)


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
    **kwargs: Any,
) -> list[dict]:
    """Load rows from a Hugging Face dataset.

    Returns:
        Rows with ``row_idx``, ``dataset_id``, the dataset's column values, and
        ``query``.
    """
    source = HFRetrievalSource(
        dataset=dataset,
        config=config,
        split=split,
        text_column=text_column,
        **kwargs,
    )
    return retrieve(source, mode=mode, queries=queries, n=n, seed=seed, **kwargs)


def retrieve_from_arxiv(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Fetch arXiv papers. QUERY (text search) or RANDOM (sample from category).

    Returns:
        Rows with ``text`` (title and abstract), ``id`` (the arXiv id),
        ``title``, ``summary`` (the abstract), and ``query``.
    """
    source = ArxivRetrievalSource(**kwargs)
    return retrieve(source, mode=mode, queries=queries, n=n, seed=seed, **kwargs)


def retrieve_from_s2(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Fetch papers from Semantic Scholar.

    Returns:
        Rows with ``text`` (title and abstract), ``id`` (the paper id),
        ``title``, ``summary`` (the abstract), and ``query``.
    """
    source = S2RetrievalSource(**kwargs)
    return retrieve(source, mode=mode, queries=queries, n=n, seed=seed, **kwargs)
