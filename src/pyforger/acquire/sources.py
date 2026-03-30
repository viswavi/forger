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
import requests


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
    """Retrieval source that fetches articles from Wikipedia via the MediaWiki API.

    Uses requests.Session to query the MediaWiki Action API for search results
    and page extracts. Supports QUERY (full-text search) and RANDOM (random
    article sampling with seed). Returns item dicts in the DPR/BEIR standard
    format: "id" (page ID), "title" (article title), "text" (intro extract),
    and "summary" (same as text, for arXiv schema compatibility).

    Attributes:
        language: Wikipedia language code (e.g. "en", "de").
        _kwargs: Options passed at construction; used for Session config and
            fetch overrides.
        _session: Lazy-initialized requests.Session, reused across fetch() calls.
    """

    _EXTRACT_BATCH_SIZE = 20

    def __init__(self, language: str = "en", **kwargs: Any) -> None:
        """Store config for Wikipedia retrieval.

        Args:
            language: Wikipedia language code (e.g. "en").
            **kwargs: Session options (e.g. timeout) and fetch overrides.
        """
        self.language = language
        self._kwargs = kwargs  # e.g. WikiRetrievalSource(timeout=10)
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        """Return shared requests.Session, creating once with a proper User-Agent."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {"User-Agent": "pyforger/0.1 (https://github.com/viswavi/forger)"}
            )
        return self._session

    @staticmethod
    def _get_timeout_kw(merged: dict[str, Any]) -> dict[str, Any]:
        """Build keyword args for requests.Session.get when timeout is set."""
        # requests.Session has no session-level timeout; must pass per-call.
        t = merged.get("timeout")
        if t is not None:
            return {"timeout": t}
        return {}

    @staticmethod
    def _result_to_item(page: dict[str, Any]) -> dict[str, Any]:
        """Convert a MediaWiki page dict to DPR/BEIR-standard item dict.

        Args:
            page: Dict from the MediaWiki extracts API with keys "pageid",
                "title", and optionally "extract".

        Returns:
            Dict with "text" (intro extract only), "id" (page ID as string),
            "title", and "summary" (mirrors text).
        """
        title = page.get("title", "")
        extract = page.get("extract", "")
        return {
            "text": extract,
            "id": str(page.get("pageid", "")),
            "title": title,
            "summary": extract,
        }

    def _fetch_extracts(
        self,
        titles: list[str],
        merged: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Fetch intro extracts for a list of page titles, batched by 20.

        Args:
            titles: Wikipedia page titles to fetch extracts for.
            merged: Merged instance and per-call kwargs (e.g. timeout).

        Returns:
            (by_requested_title, by_page_id): Lookups for resolving search/random
            rows. Redirects are applied so a requested title that redirects still
            maps to the target page dict. by_page_id keys are str(pageid).
        """
        session = self._get_session()
        base_url = f"https://{self.language}.wikipedia.org/w/api.php"
        timeout_kw = self._get_timeout_kw(merged)
        title_map: dict[str, dict[str, Any]] = {}
        id_map: dict[str, dict[str, Any]] = {}

        for i in range(0, len(titles), self._EXTRACT_BATCH_SIZE):
            batch = titles[i : i + self._EXTRACT_BATCH_SIZE]
            resp = session.get(
                base_url,
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "exintro": "true",
                    "explaintext": "true",
                    "redirects": "true",
                    "titles": "|".join(batch),
                    "exlimit": str(len(batch)),
                },
                **timeout_kw,
            )
            resp.raise_for_status()
            query = resp.json().get("query", {})
            pages = query.get("pages", {})
            # MediaWiki keys missing pages with negative ids; skip them.
            by_canonical_title: dict[str, dict[str, Any]] = {}
            for page_id, page_data in pages.items():
                if int(page_id) < 0:
                    continue
                pid = str(page_data.get("pageid", page_id))
                id_map[pid] = page_data
                ct = page_data.get("title", "")
                if ct:
                    by_canonical_title[ct] = page_data
            # Map redirected titles back so the *requested* name still resolves.
            for redir in query.get("redirects", []):
                fr = redir.get("from")
                to = redir.get("to")
                if fr and to and to in by_canonical_title:
                    by_canonical_title[fr] = by_canonical_title[to]
            for requested in batch:
                page = by_canonical_title.get(requested)
                if page is not None:
                    title_map[requested] = page

        return title_map, id_map

    def fetch(
        self,
        mode: RetrieveMode,
        *,
        query: str | None = None,
        n: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return items from Wikipedia by mode (QUERY or RANDOM).

        Mode behavior:
        - QUERY: Searches Wikipedia with srsearch={query} and srlimit={n},
          then fetches intro extracts for each result. Empty or None query
          defaults to "Wikipedia". Redirects are resolved so search titles
          still match extracts.
        - RANDOM: Fetches min(n*5, 500) random article titles from the main
          namespace, fetches their extracts, then samples n with seed for
          reproducibility.

        Per-call **kwargs are merged with constructor kwargs (e.g. timeout for
        requests.get on every HTTP call).

        Returns:
            List of dicts in DPR/BEIR format; each has "text" (intro extract),
            "id" (page ID), "title", "summary" (same as text).
        """
        # Per-call kwargs override constructor kwargs (e.g. timeout).
        merged = {**self._kwargs, **kwargs}
        session = self._get_session()
        base_url = f"https://{self.language}.wikipedia.org/w/api.php"
        timeout_kw = self._get_timeout_kw(merged)

        if mode == RetrieveMode.QUERY:
            q = query or "Wikipedia"
            resp = session.get(
                base_url,
                params={
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": q,
                    "srlimit": str(n),
                },
                **timeout_kw,
            )
            resp.raise_for_status()
            search_results = resp.json().get("query", {}).get("search", [])
            if not search_results:
                return []

            titles = [r["title"] for r in search_results]
            title_map, id_map = self._fetch_extracts(titles, merged)
            items = []
            for row in search_results:
                title = row["title"]
                # Fall back to pageid lookup when redirects change the title.
                page = title_map.get(title) or id_map.get(str(row["pageid"]))
                if page is not None:
                    items.append(self._result_to_item(page))
            return items

        if mode == RetrieveMode.RANDOM:
            # Fetch more than n so random.sample with seed is reproducible.
            pool_size = min(n * 5, 500)
            resp = session.get(
                base_url,
                params={
                    "action": "query",
                    "format": "json",
                    "list": "random",
                    "rnnamespace": "0",
                    "rnlimit": str(pool_size),
                },
                **timeout_kw,
            )
            resp.raise_for_status()
            random_results = resp.json().get("query", {}).get("random", [])
            if not random_results:
                return []

            titles = [r["title"] for r in random_results]
            title_map, id_map = self._fetch_extracts(titles, merged)
            items = []
            for row in random_results:
                title = row["title"]
                # random API uses "id" not "pageid" for the page identifier.
                page = title_map.get(title) or id_map.get(str(row["id"]))
                if page is not None:
                    items.append(self._result_to_item(page))

            rng = random.Random(seed)
            sample_size = min(n, len(items))
            return rng.sample(items, sample_size)

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

    Uses the arxiv Python package (Client and Search) for QUERY (text search)
    and RANDOM (sample from a category with seed). Returns item dicts with
    "text", "id", "title", and "summary" per paper.

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
    """Semantic Scholar (semanticscholar library). Stub implementation."""

    def __init__(self, **kwargs: Any) -> None:
        """Store options for Semantic Scholar retrieval. Stub does not call the API.

        Args:
            **kwargs: Passed through to fetch(); reserved for SemanticScholar options.
        """
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

        Would use SemanticScholar().search_paper() with
        query or random mode.
        """
        return []
