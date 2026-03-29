"""Retrieval sources for acquire.retrieve: abstract base and concrete backends.

Concrete sources (web, Wikipedia, Hugging Face, ArXiv, Semantic Scholar) are stub
implementations that can be replaced with real API calls.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

import arxiv
from semanticscholar import SemanticScholar  # type: ignore[import-untyped]


class RetrieveMode(Enum):
    """How to select segments from the source."""

    QUERY = auto()  # query string (and optional k/top_k)
    RANDOM = auto()  # random sampling (seed, n)


class RetrievalSource(ABC):
    """Abstract base for a corpus or API that can yield text segments.

    Segments may be passages, docs, or rows. Subclasses implement fetch()
    to return a list of items (e.g. dicts with "text", "id").
    """

    @abstractmethod
    def fetch(
        self,
        mode: RetrieveMode,
        *,
        query: str | None = None,
        n: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return list of items (e.g. [{"text": "...", "id": "..."}]).

        Mode-specific args used as needed.

        Args:
            mode: How to select segments (QUERY or RANDOM).
            query: Search or query string; used when mode is QUERY.
            n: Number of segments to return (e.g. top-k or sample size).
            seed: Random seed for reproducible sampling when mode is RANDOM.
            **kwargs: Passed through to backend; mode-specific.

        Returns:
            List of dicts, each typically containing at least "text"
            and optionally "id".
        """
        ...


class WebRetrievalSource(RetrievalSource):
    """Web corpus (e.g. crawl/search). Stub implementation."""

    def __init__(self, base_url: str | None = None, **kwargs: Any) -> None:
        """Store config for web retrieval. Stub does not call any API.

        Args:
            base_url: Optional base URL for the corpus or crawl target.
            **kwargs: Passed through to fetch(); reserved for future options.
        """
        self.base_url = base_url
        self._kwargs = kwargs

    def fetch(
        self,
        mode: RetrieveMode,
        *,
        query: str | None = None,
        n: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return list of items; stub returns empty list.

        Would use requests/crawler with query or random mode.
        """
        return []


class WikiRetrievalSource(RetrievalSource):
    """Wikipedia (e.g. wikipedia-api / MediaWiki). Stub implementation."""

    def __init__(self, language: str = "en", **kwargs: Any) -> None:
        """Store config for Wikipedia retrieval. Stub does not call any API.

        Args:
            language: Wikipedia language code (e.g. "en").
            **kwargs: Passed through to fetch(); reserved for future options.
        """
        self.language = language
        self._kwargs = kwargs

    def fetch(
        self,
        mode: RetrieveMode,
        *,
        query: str | None = None,
        n: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return list of items; stub returns empty list.

        Would use wikipedia-api or MediaWiki with query or random mode.
        """
        return []


class HFRetrievalSource(RetrievalSource):
    """Hugging Face datasets (load_dataset + shuffle/select).

    Stub implementation.
    """

    def __init__(
        self,
        dataset: str,
        config: str | None = None,
        split: str = "train",
        text_column: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Store config for HF dataset retrieval. Stub does not call the Hub.

        Args:
            dataset: Hugging Face dataset name (e.g. "user/repo").
            config: Optional dataset config name.
            split: Split to use (e.g. "train", "validation").
            text_column: Column name containing text; used when
                normalizing rows to dicts.
            **kwargs: Passed through to fetch(); reserved for load_dataset options.
        """
        self.dataset = dataset
        self.config = config
        self.split = split
        self.text_column = text_column
        self._kwargs = kwargs

    def fetch(
        self,
        mode: RetrieveMode,
        *,
        query: str | None = None,
        n: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return list of items; stub returns empty list.

        Would use datasets.load_dataset(), shuffle(seed), select(range(n)).
        """
        return []


class ArxivRetrievalSource(RetrievalSource):
    """Retrieval source that fetches papers from arXiv via the API.

    Uses the arxiv Python package (Client and Search) to run queries in two
    modes: QUERY (text search + n) and RANDOM (sample n from a category with
    seed). Returns a list of item dicts with "text", "id", "title", and
    "summary" per paper.

    Attributes:
        _kwargs: Options passed at construction; used for Client config and
            fetch overrides.
        _client: Lazy-initialized arxiv.Client, reused across fetch() calls.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Store options for arXiv retrieval.

        Args:
            **kwargs: Passed to arxiv.Client (e.g. page_size, delay_seconds,
                num_retries). Also used for fetch() overrides (sort_by, sort_order,
                random_category).
        """
        self._kwargs = kwargs
        self._client: arxiv.Client | None = None

    @staticmethod
    def _parse_sort_criterion(value: Any) -> arxiv.SortCriterion:
        """Normalize sort_by to arxiv.SortCriterion; default Relevance."""
        if isinstance(value, arxiv.SortCriterion):
            return value
        if value is None:
            return arxiv.SortCriterion.Relevance
        m = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
            "submitteddate": arxiv.SortCriterion.SubmittedDate,
        }
        key = str(value).lower() if isinstance(value, str) else value
        return m.get(key, arxiv.SortCriterion.Relevance)

    @staticmethod
    def _parse_sort_order(value: Any) -> arxiv.SortOrder:
        """Normalize sort_order to arxiv.SortOrder; default Descending."""
        if isinstance(value, arxiv.SortOrder):
            return value
        if value is None:
            return arxiv.SortOrder.Descending
        m = {
            "ascending": arxiv.SortOrder.Ascending,
            "descending": arxiv.SortOrder.Descending,
        }
        return m.get(
            str(value).lower() if isinstance(value, str) else value,
            arxiv.SortOrder.Descending,
        )

    @staticmethod
    def _result_to_item(result: arxiv.Result) -> dict[str, Any]:
        """Convert arxiv.Result to item dict (text, id, title, summary)."""
        summary = " ".join(result.summary.split()) if result.summary else ""
        text = f"{result.title}\n\n{summary}".strip() if result.title else summary
        return {
            "text": text,  # Title + normalized abstract for embedding/display
            "id": result.get_short_id(),  # arXiv id e.g. 2107.05580v1
            "title": result.title,
            "summary": result.summary or "",
        }

    def _get_client(self) -> arxiv.Client:
        """Return shared arxiv.Client, creating once with client-safe kwargs."""
        if self._client is None:
            # Only pass Client-supported options: page_size (results per API request),
            # delay_seconds (pause between requests, e.g. 3 for arXiv guidelines),
            # num_retries (retries on failure).
            client_kwargs = {
                k: v
                for k, v in self._kwargs.items()
                if k in ("page_size", "delay_seconds", "num_retries")
            }
            self._client = arxiv.Client(**client_kwargs)
        return self._client

    def fetch(
        self,
        mode: RetrieveMode,
        *,
        query: str | None = None,
        n: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return items from arXiv by mode (QUERY or RANDOM).

        Mode behavior:
        - QUERY: Uses query (default "all:*" when empty) and n. Optional
          kwargs: sort_by (relevance|submittedDate|lastUpdatedDate or
          arxiv.SortCriterion), sort_order (ascending|descending or
          arxiv.SortOrder).
        - RANDOM: Fetches from category cs.AI with max_results=min(n*10, 500),
          then samples n with seed. Optional kwargs: random_category (e.g.
          "cs.CL") to change the category.

        Returns:
            List of dicts; each has "text", "id", "title", "summary".
        """
        # Merge instance options with per-call kwargs; resolve sort from kwargs only
        merged = {**self._kwargs, **kwargs}
        sort_by = self._parse_sort_criterion(merged.get("sort_by"))
        sort_order = self._parse_sort_order(merged.get("sort_order"))
        client = self._get_client()

        if mode == RetrieveMode.QUERY:
            q = query or "all:*"
            search = arxiv.Search(
                query=q,  # arXiv search query (e.g. "quantum" or "all:*")
                max_results=n,
                sort_by=sort_by,  # relevance, submittedDate, lastUpdatedDate
                sort_order=sort_order,
            )
            results = list(client.results(search))
            return [self._result_to_item(r) for r in results]

        if mode == RetrieveMode.RANDOM:
            category = merged.get("random_category", "cs.AI")
            pool_size = min(n * 10, 500)
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=pool_size,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            results = list(client.results(search))
            rng = random.Random(seed)
            sample_size = min(n, len(results))
            sampled = rng.sample(results, sample_size)
            return [self._result_to_item(r) for r in sampled]

        return []


class S2RetrievalSource(RetrievalSource):
    """Retrieval source that fetches papers from Semantic Scholar via the API.

    Uses the semanticscholar library (SemanticScholar.search_paper) in two
    modes: QUERY (up to 1000) and RANDOM (pool up to 1000 or 1000+ with bulk).
    Returns a list of item dicts with "text", "id", "title", and "summary"
    per paper. QUERY is capped at 1000 results due to API limits.
    """

    # Default query when QUERY mode is called with empty query
    _DEFAULT_QUERY = "machine learning"
    # Relevance search returns at most 1000 results; 100 per request
    _PAGE_LIMIT = 100
    _RELEVANCE_CAP = 1000
    # Bulk search: 1000 items per page; cap pool size for RANDOM
    _BULK_PAGE_SIZE = 1000
    _RANDOM_POOL_MAX_BULK = 10_000

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        """Store options for Semantic Scholar retrieval.

        Args:
            api_key: Optional API key for Semantic Scholar (forwarded to
                SemanticScholar(api_key=...)).
            **kwargs: Passed to SemanticScholar (timeout, api_url, retry)
                and to fetch() overrides (e.g. fields_of_study for RANDOM).
        """
        self._api_key = api_key
        self._kwargs = kwargs
        self._client: SemanticScholar | None = None

    @staticmethod
    def _result_to_item(paper: Any) -> dict[str, Any]:
        """Convert semanticscholar Paper to item dict (text, id, title, summary)."""
        abstract = paper.abstract if getattr(paper, "abstract", None) else None
        title = paper.title if getattr(paper, "title", None) else None
        summary = " ".join(abstract.split()) if abstract else ""
        text = f"{title}\n\n{summary}".strip() if title else summary
        paper_id = getattr(paper, "paperId", None) or ""
        return {
            "text": text,
            "id": paper_id,
            "title": title or "",
            "summary": abstract or "",
        }

    def _get_client(self) -> SemanticScholar:
        """Return shared S2 client, creating once with client-safe kwargs."""
        if self._client is None:
            client_kwargs = {
                k: v
                for k, v in self._kwargs.items()
                if k in ("timeout", "api_url", "debug", "retry")
            }
            api_key = self._kwargs.get("api_key", self._api_key)
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            self._client = SemanticScholar(**client_kwargs)
        return self._client

    def _collect_paginated(
        self,
        results: Any,
        n: int,
        cap: int | None = None,
    ) -> list[Any]:
        """Accumulate items from PaginatedResults via next_page() until n or cap."""
        if cap is None:
            cap = self._RELEVANCE_CAP
        accumulated = list(getattr(results, "items", []) or [])
        need = min(n, cap)
        while len(accumulated) < need:
            next_pos = getattr(results, "next", None)
            total = getattr(results, "total", None)
            if next_pos is None or (total is not None and next_pos >= total):
                break
            try:
                results.next_page()
            except Exception:
                break
            new_items = getattr(results, "items", []) or []
            if len(new_items) <= len(accumulated):
                break
            accumulated = list(new_items)
        return accumulated[:n]

    def fetch(
        self,
        mode: RetrieveMode,
        *,
        query: str | None = None,
        n: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return items from Semantic Scholar by mode (QUERY or RANDOM).

        QUERY: When query is empty, uses default "machine learning". Paginates
        (100 per request) up to n or 1000. RANDOM: Builds pool via relevance
        search (paginated, up to 1000) or bulk when requested pool size (n*10)
        exceeds 1000; bulk pool is capped at 10,000. Then samples n with seed.
        Effective max for QUERY: 1000.

        Returns:
            List of dicts; each has "text", "id", "title", "summary".
        """
        merged = {**self._kwargs, **kwargs}
        client = self._get_client()

        if mode == RetrieveMode.QUERY:
            q = query or self._DEFAULT_QUERY
            results = client.search_paper(q, limit=self._PAGE_LIMIT)
            accumulated = self._collect_paginated(results, n)
            return [self._result_to_item(p) for p in accumulated]

        if mode == RetrieveMode.RANDOM:
            use_bulk = (n * 10) > self._RELEVANCE_CAP
            pool_size = (
                min(n * 10, self._RANDOM_POOL_MAX_BULK)
                if use_bulk
                else min(n * 10, self._RELEVANCE_CAP)
            )
            default_query = merged.get("fields_of_study", ["Computer Science"])
            if isinstance(default_query, list):
                query_str = default_query[0] if default_query else "Computer Science"
            else:
                query_str = str(default_query)
            if use_bulk:
                results = client.search_paper(query_str, bulk=True)
                accumulated = []
                need = pool_size
                while len(accumulated) < need:
                    accumulated.extend(getattr(results, "items", []) or [])
                    if len(accumulated) >= need:
                        break
                    try:
                        results.next_page()
                    except Exception:
                        break
                    new_items = getattr(results, "items", []) or []
                    if len(new_items) <= len(accumulated):
                        break
                    accumulated = list(new_items)
                pool = accumulated[:need]
            else:
                results = client.search_paper(query_str, limit=self._PAGE_LIMIT)
                pool = self._collect_paginated(results, pool_size)
            rng = random.Random(seed)
            sample_size = min(n, len(pool))
            sampled = rng.sample(pool, sample_size)
            return [self._result_to_item(p) for p in sampled]

        return []
