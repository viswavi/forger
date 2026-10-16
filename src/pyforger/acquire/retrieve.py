"""Retrieval wrappers: call a RetrievalSource and return a flat list of dicts.

``retrieve()`` is the generic entry point; the convenience wrappers
construct the appropriate source for a backend (web, wiki, HF, arXiv, S2).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    """
    resolved_queries: list[str | None] = (
        list(queries) if queries is not None else [None]
    )
    all_rows: list[dict[str, Any]] = []
    for q in resolved_queries:
        all_rows.extend(_fetch(source, mode, q, n, seed, kwargs))
    return all_rows


def _fetch(
    source: RetrievalSource,
    mode: RetrieveMode,
    query: str | None,
    n: int,
    seed: int | None,
    kwargs: dict[str, Any],
) -> list[dict]:
    """One ``source.fetch`` call, each row stamped with its ``"query"``."""
    rows = source.fetch(mode, query=query, n=n, seed=seed, **kwargs)
    for row in rows:
        row["query"] = query
    return rows


@tracked("acquire.retrieve")
def _retrieve_concurrently(
    source: S2RetrievalSource,
    mode: RetrieveMode,
    *,
    queries: list[str],
    n: int,
    seed: int | None,
    max_workers: int,
    **kwargs: Any,
) -> list[dict]:
    """``retrieve`` with up to ``max_workers`` queries in flight; rows keep order."""
    source._get_client()  # one shared client, created before the threads start

    def fetch(query: str) -> list[dict]:
        return _fetch(source, mode, query, n, seed, kwargs)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(queries))) as pool:
        return [row for rows in pool.map(fetch, queries) for row in rows]


def retrieve_from_web(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Search the web via Brave's LLM Context API. QUERY mode only."""
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
    """Fetch Wikipedia articles. QUERY (full-text) or RANDOM (random article)."""
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
    """Load rows from a Hugging Face dataset."""
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
    """Fetch arXiv papers. QUERY (text search) or RANDOM (sample from category)."""
    source = ArxivRetrievalSource(**kwargs)
    return retrieve(source, mode=mode, queries=queries, n=n, seed=seed, **kwargs)


def retrieve_from_s2(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    max_workers: int = 1,
    **kwargs: Any,
) -> list[dict]:
    """Fetch papers from Semantic Scholar.

    Each row has ``text`` (title and abstract), ``id``, ``title``, ``summary``
    (the abstract), and ``query``. A query asks for only ``n`` results.

    ``max_workers`` queries run at a time; rows keep the order of
    ``queries``. The default is one at a time, because an API key is usually
    limited to 1 request per second and keyless requests share a throttled
    pool; the client retries a rate-limited request after 5 to 60 seconds,
    so more workers than the limit allows are slower, not faster. Raise it
    when your key's limit allows, e.g. ``max_workers=3``.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    source = S2RetrievalSource(**kwargs)
    if max_workers == 1 or queries is None or len(queries) <= 1:
        return retrieve(source, mode=mode, queries=queries, n=n, seed=seed, **kwargs)
    return _retrieve_concurrently(
        source,
        mode,
        queries=list(queries),
        n=n,
        seed=seed,
        max_workers=max_workers,
        **kwargs,
    )
