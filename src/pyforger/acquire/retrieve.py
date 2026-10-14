"""Retrieval wrappers: call a RetrievalSource and return a flat list of dicts.

``retrieve()`` is the generic entry point; the convenience wrappers
construct the appropriate source for a backend (web, wiki, HF, arXiv, S2).
All entry points cache results on disk by default, including RANDOM mode.
Pass ``_no_cache=True`` to skip cache reads and writes, or ``_cache_only=True``
to read an existing result without fetching.
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
from pyforger.op import op


def _resolve_queries(queries: str | list[str] | None) -> list[str | None]:
    """Turn the ``queries`` argument into the list of fetches to make.

    A bare string is one query (``"dp tricks"`` -> ``["dp tricks"]``), not a
    list of characters. ``None`` means one fetch with no query (RANDOM mode).
    """
    if isinstance(queries, RetrieveMode):
        raise TypeError("mode is keyword-only: pass mode=RetrieveMode...")
    if queries is None:
        return [None]
    if isinstance(queries, str):
        return [queries]
    return list(queries)


@op(name="acquire.retrieve")
def retrieve(
    source: RetrievalSource,
    queries: str | list[str] | None = None,
    *,
    mode: RetrieveMode = RetrieveMode.QUERY,
    n: int = 10,
    seed: int | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Call source.fetch() once per query and concatenate results.

    Each row is stamped with a ``"query"`` field. With ``queries=None``,
    a single fetch with no query is performed (e.g. random sampling).
    """
    if not isinstance(mode, RetrieveMode):
        raise TypeError(
            f"mode must be a RetrieveMode, not {mode!r}; "
            "pass search strings as queries"
        )
    all_rows: list[dict[str, Any]] = []
    for q in _resolve_queries(queries):
        rows = source.fetch(mode, query=q, n=n, seed=seed, **kwargs)
        for row in rows:
            row["query"] = q
        all_rows.extend(rows)
    return all_rows


# Backend wrappers own their cache boundary; call the unwrapped retrieval
# implementation so bypassing a wrapper cannot hit an inner cache.
@op(name="acquire.retrieve_from_web")
def retrieve_from_web(
    queries: str | list[str] | None = None,
    *,
    mode: RetrieveMode = RetrieveMode.QUERY,
    n: int = 10,
    seed: int | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Search the web via Brave's LLM Context API. QUERY mode only.

    The API key comes from ``api_key=`` or, failing that, ``BRAVE_API_KEY``.
    """
    source_kwargs = {k: v for k, v in kwargs.items() if k not in ("api_key",)}
    source = WebRetrievalSource(
        base_url=base_url,
        api_key=kwargs.get("api_key"),
        **source_kwargs,
    )
    return retrieve.__wrapped__(
        source, mode=mode, queries=queries, n=n, seed=seed, **source_kwargs
    )


@op(name="acquire.retrieve_from_wiki")
def retrieve_from_wiki(
    queries: str | list[str] | None = None,
    *,
    mode: RetrieveMode = RetrieveMode.QUERY,
    n: int = 10,
    seed: int | None = None,
    language: str = "en",
    **kwargs: Any,
) -> list[dict]:
    """Fetch Wikipedia articles. QUERY (full-text) or RANDOM (random article)."""
    source = WikiRetrievalSource(language=language, **kwargs)
    return retrieve.__wrapped__(
        source, mode=mode, queries=queries, n=n, seed=seed, **kwargs
    )


@op(name="acquire.retrieve_from_hf")
def retrieve_from_hf(
    dataset: str,
    queries: str | list[str] | None = None,
    *,
    config: str | None = None,
    split: str = "train",
    text_column: str | None = None,
    mode: RetrieveMode = RetrieveMode.RANDOM,
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
    return retrieve.__wrapped__(
        source, mode=mode, queries=queries, n=n, seed=seed, **kwargs
    )


@op(name="acquire.retrieve_from_arxiv")
def retrieve_from_arxiv(
    queries: str | list[str] | None = None,
    *,
    mode: RetrieveMode = RetrieveMode.QUERY,
    n: int = 10,
    seed: int | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Fetch arXiv papers. QUERY (text search) or RANDOM (sample from category)."""
    source = ArxivRetrievalSource(**kwargs)
    return retrieve.__wrapped__(
        source, mode=mode, queries=queries, n=n, seed=seed, **kwargs
    )


@op(name="acquire.retrieve_from_s2")
def retrieve_from_s2(
    queries: str | list[str] | None = None,
    *,
    mode: RetrieveMode = RetrieveMode.QUERY,
    n: int = 10,
    seed: int | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Fetch papers from Semantic Scholar."""
    source = S2RetrievalSource(**kwargs)
    return retrieve.__wrapped__(
        source, mode=mode, queries=queries, n=n, seed=seed, **kwargs
    )
