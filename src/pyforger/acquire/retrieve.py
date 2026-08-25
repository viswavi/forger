"""Retrieval wrappers: call a RetrievalSource and return a flat list of dicts.

``retrieve()`` is the generic entry point; the convenience wrappers
construct the appropriate source for a backend (web, wiki, HF, arXiv, S2).
Each wrapper takes ``cache=False`` by default; pass ``cache=True`` to
memoize results on disk via ``@op``.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
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
from pyforger.op import op


def _cacheable(fn: Callable[..., list[dict]]) -> Callable[..., list[dict]]:
    """Wrap fn like @op, but with a ``cache: bool = False`` kwarg (default off).

    Translates ``cache`` into @op's ``_no_cache`` opt-out toggle, which
    otherwise defaults to caching being enabled.
    """
    cached = op(fn, name=f"pyforger.acquire.retrieve.{fn.__name__}")
    params = list(inspect.signature(fn).parameters.values())
    insert_at = next(
        (i for i, p in enumerate(params) if p.kind is p.VAR_KEYWORD), len(params)
    )
    params.insert(
        insert_at,
        inspect.Parameter("cache", inspect.Parameter.KEYWORD_ONLY, default=False),
    )

    @wraps(fn)
    def wrapper(*args: Any, cache: bool = False, **kwargs: Any) -> list[dict]:
        return cached(*args, _no_cache=not cache, **kwargs)

    wrapper.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


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


@_cacheable
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


@_cacheable
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

    cache=True memoizes results on disk via @op (default False).
    """
    source = WikiRetrievalSource(language=language, **kwargs)
    return retrieve(source, mode=mode, queries=queries, n=n, seed=seed, **kwargs)


@_cacheable
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

    cache=True memoizes results on disk via @op (default False).
    """
    source = HFRetrievalSource(
        dataset=dataset,
        config=config,
        split=split,
        text_column=text_column,
        **kwargs,
    )
    return retrieve(source, mode=mode, queries=queries, n=n, seed=seed, **kwargs)


@_cacheable
def retrieve_from_arxiv(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Fetch arXiv papers. QUERY (text search) or RANDOM (sample from category).

    cache=True memoizes results on disk via @op (default False).
    """
    source = ArxivRetrievalSource(**kwargs)
    return retrieve(source, mode=mode, queries=queries, n=n, seed=seed, **kwargs)


@_cacheable
def retrieve_from_s2(
    mode: RetrieveMode = RetrieveMode.QUERY,
    *,
    queries: list[str] | None = None,
    n: int = 10,
    seed: int | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Fetch papers from Semantic Scholar.

    cache=True memoizes results on disk via @op (default False).
    """
    source = S2RetrievalSource(**kwargs)
    return retrieve(source, mode=mode, queries=queries, n=n, seed=seed, **kwargs)
