"""Retrieval wrappers: call a RetrievalSource and return a flat list of dicts.

``retrieve()`` is the generic entry point; the convenience wrappers
construct the appropriate source for a backend (web, wiki, HF, arXiv, S2,
PubMed).
All entry points cache results on disk by default, including RANDOM mode.
Pass ``_no_cache=True`` to skip cache reads and writes, or ``_cache_only=True``
to read an existing result without fetching.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pyforger import _tracking
from pyforger.acquire.pubmed import PubMedRetrievalSource
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

# The wrapper that builds each source, named in the pipeline's warning when
# retrieved text reaches no prompt.
_WRAPPER_OF = {
    WebRetrievalSource: "acquire.retrieve_from_web",
    WikiRetrievalSource: "acquire.retrieve_from_wiki",
    HFRetrievalSource: "acquire.retrieve_from_hf",
    ArxivRetrievalSource: "acquire.retrieve_from_arxiv",
    S2RetrievalSource: "acquire.retrieve_from_s2",
    PubMedRetrievalSource: "acquire.retrieve_from_pubmed",
}


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


def _register(source: RetrievalSource, rows: list[dict]) -> None:
    """Fingerprint ``rows`` for the pipeline's check, under the wrapper's name."""
    label = next(
        (name for cls, name in _WRAPPER_OF.items() if isinstance(source, cls)),
        "acquire.retrieve",
    )
    _tracking.register_documents(label, rows)


@op(name="acquire.retrieve", smoke=("queries", "n"))
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
    Inside a ``@pipeline`` the rows' text is fingerprinted, and the pipeline
    warns when none of it reaches a model prompt or the pipeline's output.
    In a limited run (``FORGER_LIMIT=3``) only the first 3 queries run,
    with ``n`` capped to 3.

    Returns:
        The source's rows for each query in turn, each with a ``query`` field
        added (``None`` when ``queries`` is None).
    """
    if not isinstance(mode, RetrieveMode):
        raise TypeError(
            f"mode must be a RetrieveMode, not {mode!r}; "
            "pass search strings as queries"
        )
    all_rows: list[dict[str, Any]] = []
    for q in _resolve_queries(queries):
        all_rows.extend(_fetch(source, mode, q, n, seed, kwargs))
    _register(source, all_rows)
    return all_rows


# Backend wrappers own their cache boundary; call the unwrapped retrieval
# implementation so bypassing a wrapper cannot hit an inner cache.
@op(name="acquire.retrieve_from_web", smoke=("queries", "n"))
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
    return retrieve.__wrapped__(
        source, mode=mode, queries=queries, n=n, seed=seed, **source_kwargs
    )


@op(name="acquire.retrieve_from_wiki", smoke=("queries", "n"))
def retrieve_from_wiki(
    queries: str | list[str] | None = None,
    *,
    mode: RetrieveMode = RetrieveMode.QUERY,
    n: int = 10,
    seed: int | None = None,
    language: str = "en",
    **kwargs: Any,
) -> list[dict]:
    """Fetch Wikipedia articles. QUERY (full-text) or RANDOM (random article).

    ``text`` is the article's intro. Pass ``full_text=True`` for the whole
    article as plain text (one request per result; e.g. Quicksort is about
    50,000 characters, against about 1,400 for its intro).

    Returns:
        Rows with ``text`` (the intro, or the whole article with
        ``full_text=True``), ``id`` (the page ID), ``title``, ``summary`` (same
        as ``text``), and ``query``.
    """
    source = WikiRetrievalSource(language=language, **kwargs)
    return retrieve.__wrapped__(
        source, mode=mode, queries=queries, n=n, seed=seed, **kwargs
    )


@op(name="acquire.retrieve_from_hf", smoke=("queries", "n"))
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
    """Load rows from a Hugging Face dataset.

    Returns:
        Rows with ``row_idx``, ``dataset_id``, the dataset's column values, and
        ``query``, which overwrites a dataset column of that name (``None`` when
        ``queries`` is None).
    """
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


@op(name="acquire.retrieve_from_arxiv", smoke=("queries", "n"))
def retrieve_from_arxiv(
    queries: str | list[str] | None = None,
    *,
    mode: RetrieveMode = RetrieveMode.QUERY,
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
    return retrieve.__wrapped__(
        source, mode=mode, queries=queries, n=n, seed=seed, **kwargs
    )


# max_workers only changes how fast the queries run (rows keep query order),
# so it stays out of the cache key.
@op(name="acquire.retrieve_from_s2", unkeyed=("max_workers",), smoke=("queries", "n"))
def retrieve_from_s2(
    queries: str | list[str] | None = None,
    *,
    mode: RetrieveMode = RetrieveMode.QUERY,
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

    Returns:
        Rows with ``text`` (title and abstract), ``id`` (the paper id),
        ``title``, ``summary`` (the abstract), and ``query``.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    source = S2RetrievalSource(**kwargs)
    resolved = _resolve_queries(queries)
    if max_workers == 1 or len(resolved) <= 1:
        return retrieve.__wrapped__(
            source, mode=mode, queries=queries, n=n, seed=seed, **kwargs
        )
    if not isinstance(mode, RetrieveMode):
        raise TypeError(
            f"mode must be a RetrieveMode, not {mode!r}; "
            "pass search strings as queries"
        )
    source._get_client()  # one shared client, created before the threads start

    def fetch(query: str | None) -> list[dict]:
        return _fetch(source, mode, query, n, seed, kwargs)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(resolved))) as pool:
        all_rows = [row for rows in pool.map(fetch, resolved) for row in rows]
    _register(source, all_rows)
    return all_rows


# The API key changes how fast requests may go, not what they return, so it
# stays out of the cache key (and out of the cache files).
@op(name="acquire.retrieve_from_pubmed", unkeyed=("api_key",), smoke=("queries", "n"))
def retrieve_from_pubmed(
    queries: str | list[str] | None = None,
    *,
    n: int = 10,
    api_key: str | None = None,
    sort: str = "relevance",
    **kwargs: Any,
) -> list[dict]:
    """Search PubMed: articles, case reports, reviews, trials and more.

    A query uses PubMed's own search, so plain words work ("lung cancer case
    reports" finds records of the Case Reports publication type) and so do
    field tags (``'"lung neoplasms"[MeSH] AND case reports[pt]'``). PubMed has
    abstracts, not full text; ``pmcid`` names the free PubMed Central copy
    when there is one. ``sort`` is ``"relevance"``, ``"pub_date"``,
    ``"first_author"`` or ``"journal"``.

    The API key comes from ``api_key=`` or ``NCBI_API_KEY`` and is optional.
    Without one NCBI allows 3 requests per second, with one 10. Requests are
    spaced to stay under that limit (2.5 or 6.7 per second, across threads),
    and a rate-limited or failed request is retried up to 5 times (1 to 16
    seconds apart). A query takes 2 requests, plus 1 per 200 results past
    the first 200.

    Returns:
        Rows with ``text`` (title and abstract), ``id`` (the PMID), ``title``,
        ``summary`` (the abstract), ``authors``, ``journal``, ``year``,
        ``publication_types``, ``mesh_terms``, ``doi``, ``pmcid``, ``url``
        and ``query``.
    """
    source = PubMedRetrievalSource(api_key=api_key, sort=sort, **kwargs)
    return retrieve.__wrapped__(
        source, mode=RetrieveMode.QUERY, queries=queries, n=n, **kwargs
    )
