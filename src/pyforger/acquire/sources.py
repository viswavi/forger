"""Retrieval sources for acquire.retrieve: abstract base and concrete backends.

Concrete sources: web, Wikipedia, Hugging Face, ArXiv, Semantic Scholar.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

import arxiv
import requests
from semanticscholar import SemanticScholar  # type: ignore[import-untyped]

from pyforger.utils import get_formatted_logger

logger = get_formatted_logger("Retrieval Sources")


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
    """Retrieval source that searches the web via Brave's LLM Context API.

    Supports QUERY mode only.  Returns item dicts with "text" (pre-extracted,
    LLM-ready content chunks joined together), "id" (page URL), "title", and
    "url".  Unlike a plain search API, the LLM Context API returns extracted
    page content directly, so no separate page fetch/HTML-extraction step is
    needed.

    Licensing note:
        Brave's Terms of Service require a plan with explicit *storage rights*
        if search results are stored for LLM training or fine-tuning.  Verify
        your Brave subscription covers your intended use before storing results
        in a training dataset.  See https://brave.com/search/api/ for current
        pricing and plan details.

    Attributes:
        base_url: Optional domain restriction (e.g. ``"example.com"``).
        _api_key: Brave Search API subscription token.
        _kwargs: Options passed at construction; merged with per-call kwargs.
        _session: Lazy-initialized requests.Session for Brave API calls.
    """

    _BRAVE_API_URL = "https://api.search.brave.com/res/v1/llm/context"
    _BRAVE_MAX_RESULTS = 50  # API caps count / maximum_number_of_urls at 50
    _DEFAULT_TIMEOUT = 5  # seconds

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Store config for Brave LLM Context retrieval.

        Args:
            base_url: Optional domain to restrict results to.  When set,
                ``site:{base_url}`` is prepended to every query.
            api_key: Brave Search API subscription token.  Required for
                actual API calls; omitting it will cause a 401 error.
            **kwargs: Session overrides.  ``timeout`` (float) -- per-request
                timeout in seconds (default 5).
        """
        self.base_url = base_url
        self._api_key = api_key
        self._kwargs = kwargs
        self._session: requests.Session | None = None

    # -- session helpers -----------------------------------------------------

    def _get_session(self) -> requests.Session:
        """Return shared requests.Session for Brave API, creating once."""
        if self._session is None:
            self._session = requests.Session()
            headers: dict[str, str] = {
                "User-Agent": "pyforger/0.1 (https://github.com/viswavi/forger)",
                "Accept": "application/json",
            }
            if self._api_key:
                headers["X-Subscription-Token"] = self._api_key
            self._session.headers.update(headers)
        return self._session

    @classmethod
    def _get_timeout_kw(cls, merged: dict[str, Any]) -> dict[str, Any]:
        """Build keyword args for requests.Session.get with timeout."""
        return {"timeout": merged.get("timeout", cls._DEFAULT_TIMEOUT)}

    # -- result conversion ---------------------------------------------------

    @staticmethod
    def _result_to_item(result: dict[str, Any]) -> dict[str, Any]:
        """Convert a ``grounding.generic[]`` entry to a standard item dict.

        Args:
            result: Dict with keys ``url``, ``title``, and ``snippets``
                (list of extracted text chunks).

        Returns:
            Dict with ``text`` (snippets joined with blank lines), ``id``
            (URL), ``title``, ``url``.
        """
        title = result.get("title", "")
        url = result.get("url", "")
        snippets = result.get("snippets", [])
        text = "\n\n".join(snippets)
        return {
            "text": text,
            "id": url,
            "title": title,
            "url": url,
        }

    # -- main fetch ----------------------------------------------------------

    def fetch(
        self,
        mode: RetrieveMode,
        *,
        query: str | None = None,
        n: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return items from the web via Brave's LLM Context API.

        Mode behavior:
        - QUERY: Searches with ``q={query}`` (prepending ``site:{base_url}``
          when set) and returns up to *n* pre-extracted results.  This is a
          single-search endpoint (no offset-based pagination); *n* is capped
          at 50, the API's maximum.
        - RANDOM: Raises ``NotImplementedError`` -- web search has no
          reproducible random corpus.

        Per-call **kwargs are merged with constructor kwargs (e.g. timeout).

        Args:
            mode: QUERY or RANDOM.
            query: Search string.  Required; raises ``ValueError`` if
                empty or None.
            n: Number of results to return (capped at 50).
            seed: Unused (RANDOM mode unsupported).
            **kwargs: Overrides (e.g. timeout).

        Returns:
            List of dicts; each has ``text``, ``id`` (URL), ``title``,
            ``url``.

        Raises:
            ValueError: If *query* is empty or None.
            NotImplementedError: If *mode* is RANDOM.
        """
        if mode == RetrieveMode.RANDOM:
            raise NotImplementedError(
                "WebRetrievalSource does not support RANDOM mode -- "
                "web search has no reproducible random corpus."
            )

        merged = {**self._kwargs, **kwargs}
        session = self._get_session()
        timeout_kw = self._get_timeout_kw(merged)

        if mode == RetrieveMode.QUERY:
            if not query:
                raise ValueError("query is required for WebRetrievalSource QUERY mode")

            q = f"site:{self.base_url} {query}" if self.base_url else query
            result_count = min(n, self._BRAVE_MAX_RESULTS)

            params = {
                "q": q,
                "count": str(result_count),
                "maximum_number_of_urls": str(result_count),
            }

            resp = session.get(self._BRAVE_API_URL, params=params, **timeout_kw)
            resp.raise_for_status()
            generic_results = resp.json().get("grounding", {}).get("generic", [])

            return [self._result_to_item(r) for r in generic_results]

        return []


class WikiRetrievalSource(RetrievalSource):
    """Retrieval source that fetches articles from Wikipedia via the MediaWiki API.

    Uses requests.Session to query the MediaWiki Action API for search results
    and page extracts. Supports QUERY (full-text search) and RANDOM (random
    article sampling). Returns item dicts in the DPR/BEIR standard format:
    "id" (page ID), "title" (article title), "text" (intro extract), and
    "summary" (same as text, for arXiv schema compatibility).

    Note: RANDOM mode is inherently non-deterministic because the MediaWiki
    ``list=random`` endpoint does not support a seed parameter. The ``seed``
    argument is accepted for interface compatibility but ignored.

    Attributes:
        language: Wikipedia language code (e.g. "en", "de").
        _kwargs: Options passed at construction; used for Session config and
            fetch overrides.
        _session: Lazy-initialized requests.Session, reused across fetch() calls.
    """

    _API_LIMIT = 500
    _EXTRACT_BATCH_SIZE = 20
    _DEFAULT_TIMEOUT = 5

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
        """Build keyword args for requests.Session.get with timeout."""
        t = merged.get("timeout", WikiRetrievalSource._DEFAULT_TIMEOUT)
        return {"timeout": t}

    @classmethod
    def _clamp_n(cls, n: int, param_name: str) -> int:
        """Clamp ``n`` to the MediaWiki API maximum, warning once when exceeded.

        Args:
            n: Requested number of results.
            param_name: MediaWiki parameter being clamped (e.g. ``"srlimit"``);
                included in the warning for debuggability.

        Returns:
            ``min(n, cls._API_LIMIT)``.
        """
        if n > cls._API_LIMIT:
            logger.warning(
                "Requested n=%d exceeds Wikipedia API maximum of %d; "
                "clamping %s to %d. Only the first %d results will be returned.",
                n,
                cls._API_LIMIT,
                param_name,
                cls._API_LIMIT,
                cls._API_LIMIT,
            )
            return cls._API_LIMIT
        return n

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
        - RANDOM: Fetches n random articles from the main namespace via
          ``list=random`` and returns their intro extracts. The server
          controls randomness; seed is ignored (see class docstring).

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
                    "srlimit": str(self._clamp_n(n, "srlimit")),
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
            resp = session.get(
                base_url,
                params={
                    "action": "query",
                    "format": "json",
                    "list": "random",
                    "rnnamespace": "0",
                    "rnlimit": str(self._clamp_n(n, "rnlimit")),
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

            return items

        return []


class HFRetrievalSource(RetrievalSource):
    """Hugging Face dataset discovery and retrieval.

    Two-stage pipeline:
        1. **Discovery** -- ``HfApi.list_datasets(search=query)`` finds
           datasets whose name or description match the query.
        2. **Loading** -- ``datasets.load_dataset()`` loads the top match
           and returns up to *n* rows.

    When ``dataset`` is provided at construction, stage 1 is skipped and
    the given dataset is loaded directly.

    Both modes produce the same output shape: a list of flat dicts with
    ``row_idx``, ``dataset_id``, and the dataset's column values.
    """

    def __init__(
        self,
        dataset: str | None = None,
        config: str | None = None,
        split: str = "train",
        *,
        token: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the HF retrieval source.

        Args:
            dataset: Optional HF dataset identifier (e.g. "ibm/duorc").
                When provided, Hub discovery is skipped and this dataset
                is loaded directly.
            config: Optional dataset config/subset name (e.g. "SelfRC").
            split: Split to use (default "train").
            token: HF auth token for gated datasets and API calls.
            **kwargs: Forwarded to ``datasets.load_dataset()``.
        """
        self.dataset = dataset
        self.config = config
        self.split = split
        self._token = token
        self._kwargs = kwargs

    def _discover(self, query: str, limit: int = 1) -> list[Any]:
        """Search the HF Hub for datasets matching *query*.

        Args:
            query: Text search string.
            limit: Maximum number of dataset info objects to return.

        Returns:
            List of ``DatasetInfo`` objects from the Hub API.
        """
        from huggingface_hub import HfApi

        api = HfApi(token=self._token)
        return list(api.list_datasets(search=query, limit=limit))

    def _load_and_select(
        self,
        dataset_id: str,
        n: int,
        seed: int | None = None,
        shuffle: bool = False,
    ) -> list[dict[str, Any]]:
        """Load a dataset and return up to *n* rows as flat dicts.

        Args:
            dataset_id: HF dataset identifier to load.
            n: Maximum number of rows to return.
            seed: Random seed passed to ``shuffle()`` when *shuffle* is True.
            shuffle: Whether to shuffle before selecting rows.

        Returns:
            List of flat dicts with ``row_idx``, ``dataset_id``, and the
            dataset's column values.
        """
        from datasets import load_dataset

        ds = load_dataset(
            dataset_id,
            self.config,
            split=self.split,
            token=self._token,
            **self._kwargs,
        )
        if shuffle:
            ds = ds.shuffle(seed=seed)
        count = min(n, len(ds))
        selected = ds.select(range(count))
        return [
            {"row_idx": i, "dataset_id": dataset_id, **dict(selected[i])}
            for i in range(count)
        ]

    def fetch(
        self,
        mode: RetrieveMode,
        *,
        query: str | None = None,
        n: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return rows from a discovered or specified HF dataset.

        Args:
            mode: QUERY (discover + load in dataset order) or
                RANDOM (discover + shuffle + sample).
            query: Search string for Hub discovery; required when no
                ``dataset`` was provided at construction.
            n: Maximum number of rows to return.
            seed: Random seed for reproducible sampling in RANDOM mode.
            **kwargs: Reserved for future use.

        Returns:
            List of flat dicts, each containing ``row_idx``,
            ``dataset_id``, and the dataset's raw column values.

        Raises:
            ValueError: If *query* is empty/None and no *dataset* was
                provided at construction.
        """
        dataset_id = self.dataset
        if dataset_id is None:
            if not query:
                raise ValueError(
                    "HFRetrievalSource requires a non-empty 'query' "
                    "when no 'dataset' was provided at construction."
                )
            results = self._discover(query, limit=1)
            if not results:
                return []
            dataset_id = results[0].id

        if mode == RetrieveMode.QUERY:
            return self._load_and_select(dataset_id, n)

        if mode == RetrieveMode.RANDOM:
            return self._load_and_select(dataset_id, n, seed=seed, shuffle=True)

        raise ValueError(f"Unsupported RetrieveMode: {mode}")


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
    _RANDOM_POOL_MAX_BULK = 10_000

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        """Store options for Semantic Scholar retrieval.

        Args:
            api_key: Optional API key for Semantic Scholar (forwarded to
                SemanticScholar(api_key=...)). Without a key the client uses
                unauthenticated access, which has lower rate limits.
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
        accumulated = list(results.items)
        target_count = min(n, cap)
        while len(accumulated) < target_count:
            next_pos = results.next
            total = results.total
            if next_pos is None or (total is not None and next_pos >= total):
                break
            try:
                results.next_page()
            except Exception:
                break
            new_items = results.items
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
            if n > self._RELEVANCE_CAP:
                raise ValueError(
                    f"n={n} exceeds Semantic Scholar relevance search cap "
                    f"of {self._RELEVANCE_CAP}. Use RANDOM mode for larger "
                    f"result sets."
                )
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
                accumulated = list(results.items)
                while len(accumulated) < pool_size:
                    try:
                        results.next_page()
                    except Exception:
                        break
                    new_items = list(results.items)
                    if len(new_items) <= len(accumulated):
                        break
                    accumulated = new_items
                pool = accumulated[:pool_size]
            else:
                results = client.search_paper(query_str, limit=self._PAGE_LIMIT)
                pool = self._collect_paginated(results, pool_size)
            rng = random.Random(seed)
            sample_size = min(n, len(pool))
            sampled = rng.sample(pool, sample_size)
            return [self._result_to_item(p) for p in sampled]

        return []
