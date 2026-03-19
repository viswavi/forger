"""Retrieval sources for acquire.retrieve: abstract base and concrete backends.

Concrete sources (web, wiki, HF, arxiv, S2) are stub implementations that can be
replaced with real API calls.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

import arxiv
import requests
from datasets import load_dataset


class RetrieveMode(Enum):
    """How to select segments from the source."""

    QUERY = auto()  # query string (and optional k/top_k)
    RANDOM = auto()  # random sampling (seed, n)
    STRUCTURED = auto()  # criteria/filter (e.g. dict of field constraints)


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
        criteria: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return list of items (e.g. [{"text": "...", "id": "..."}]).

        Mode-specific args used as needed.

        Args:
            mode: How to select segments (QUERY, RANDOM, or STRUCTURED).
            query: Search or query string; used when mode is QUERY.
            n: Number of segments to return (e.g. top-k or sample size).
            seed: Random seed for reproducible sampling when mode is RANDOM.
            criteria: Field constraints or filter when mode is STRUCTURED.
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
        criteria: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return list of items; stub returns empty list.

        Would use requests/crawler with query/random/structured mode.
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
        criteria: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return list of items; stub returns empty list.

        Would use wikipedia-api or MediaWiki with query/random/structured.
        """
        return []


class HFRetrievalSource(RetrievalSource):
    """Hugging Face dataset retrieval with a hybrid backend.

    This source uses two different backends depending on the retrieval mode,
    unlike ArxivRetrievalSource or S2RetrievalSource which each use a single
    backend. The hybrid design is intentional: the hosted Dataset Viewer API
    provides BM25 full-text search, while random sampling and exact-match
    filtering are more natural as local operations via the ``datasets`` library.

    Mode backends:
        QUERY      — hosted Dataset Viewer API at
                     https://datasets-server.huggingface.co/search (BM25).
        RANDOM     — local ``datasets.load_dataset().shuffle().select()``.
        STRUCTURED — local ``datasets.load_dataset().filter()`` with
                     exact-match criteria.

    All three modes produce the same output shape: a list of flat dicts with
    ``row_idx`` and the dataset's column values at the top level.

    Attributes:
        dataset: HF dataset identifier (e.g. "ibm/duorc").
        config: Optional dataset subset/config name.
        split: Dataset split (default "train").
    """

    _DEFAULT_BASE_URL = "https://datasets-server.huggingface.co"
    _DEFAULT_TIMEOUT = 30

    def __init__(
        self,
        dataset: str,
        config: str | None = None,
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        """Configure the HF retrieval source.

        API-related kwargs (base_url, timeout, token, headers) are extracted
        for QUERY mode. Remaining kwargs are passed through to
        ``datasets.load_dataset()`` when RANDOM or STRUCTURED mode triggers
        a local load.

        Args:
            dataset: Hugging Face dataset name (e.g. "ibm/duorc").
            config: Optional dataset config/subset name (e.g. "SelfRC").
            split: Split to use (e.g. "train", "validation").
            **kwargs: Optional overrides — base_url (str), timeout (int),
                token (str for Bearer auth), headers (dict) for the API;
                all other kwargs forwarded to load_dataset().
        """
        self.dataset = dataset
        self.config = config
        self.split = split
        self._base_url: str = kwargs.pop(
            "base_url", self._DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout: int = kwargs.pop("timeout", self._DEFAULT_TIMEOUT)
        self._headers: dict[str, str] = dict(kwargs.pop("headers", {}))
        token: str | None = kwargs.pop("token", None)
        if token:
            self._headers.setdefault("Authorization", f"Bearer {token}")
        self._kwargs = kwargs
        self._dataset_obj: Any = None

    # -- API helpers (QUERY mode) ------------------------------------------

    def _search_rows(
        self, query: str, *, offset: int, length: int
    ) -> dict[str, Any]:
        """Call the Dataset Viewer /search endpoint and return parsed JSON.

        Raises:
            RuntimeError: On non-200 status, malformed JSON, or if the
                dataset is not searchable.
            requests.exceptions.Timeout: On request timeout.
        """
        params: dict[str, Any] = {
            "dataset": self.dataset,
            "split": self.split,
            "query": query,
            "offset": offset,
            "length": length,
        }
        if self.config is not None:
            params["config"] = self.config

        resp = requests.get(
            f"{self._base_url}/search",
            params=params,
            headers=self._headers or None,
            timeout=self._timeout,
        )

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            raise RuntimeError(
                f"HF Dataset Viewer /search failed "
                f"(HTTP {resp.status_code}) for "
                f"dataset={self.dataset!r} config={self.config!r} "
                f"split={self.split!r} query={query!r}: {detail}"
            )

        try:
            return resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed JSON from HF Dataset Viewer /search for "
                f"dataset={self.dataset!r} query={query!r}"
            ) from exc

    # -- Local helpers (RANDOM / STRUCTURED modes) -------------------------

    def _get_dataset(self) -> Any:
        """Lazy-load and cache the local HF dataset for RANDOM/STRUCTURED.

        Calls ``datasets.load_dataset()`` once and reuses the result for
        subsequent fetch() calls on this instance.
        """
        if self._dataset_obj is None:
            self._dataset_obj = load_dataset(
                self.dataset, self.config, split=self.split, **self._kwargs
            )
        return self._dataset_obj

    # -- Shared normalization ----------------------------------------------

    @staticmethod
    def _normalize_row(row_data: dict[str, Any], row_idx: int) -> dict[str, Any]:
        """Flatten a dataset row into a plain dict.

        Returns the dataset columns at the top level with ``row_idx``
        prepended for identification. Used by all three modes to ensure
        consistent output shape.
        """
        return {"row_idx": row_idx, **row_data}

    def fetch(
        self,
        mode: RetrieveMode,
        *,
        query: str | None = None,
        n: int = 10,
        seed: int | None = None,
        criteria: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return rows from a Hugging Face dataset.

        Args:
            mode: QUERY (hosted API search), RANDOM (local shuffle+select),
                or STRUCTURED (local exact-match filter).
            query: Search string; required when mode is QUERY.
            n: Maximum number of rows to return.
            seed: Random seed for reproducible sampling in RANDOM mode.
            criteria: Exact-match field constraints for STRUCTURED mode;
                every (key, value) pair must match.
            **kwargs: Reserved for future use.

        Returns:
            List of flat dicts, each containing ``row_idx`` and the
            dataset's column values.

        Raises:
            ValueError: If query is missing/empty in QUERY mode, if
                criteria is missing/empty in STRUCTURED mode, or if the
                mode is unrecognized.
            RuntimeError: On API errors (QUERY mode).
            requests.exceptions.Timeout: On request timeout (QUERY mode).
        """
        if mode == RetrieveMode.QUERY:
            if not query:
                raise ValueError(
                    "HFRetrievalSource QUERY mode requires a "
                    "non-empty 'query' string."
                )
            data = self._search_rows(query, offset=0, length=n)
            return [
                self._normalize_row(r["row"], r["row_idx"])
                for r in data.get("rows", [])
            ]

        if mode == RetrieveMode.RANDOM:
            ds = self._get_dataset()
            ds = ds.shuffle(seed=seed)
            count = min(n, len(ds))
            selected = ds.select(range(count))
            return [
                self._normalize_row(dict(selected[i]), i)
                for i in range(count)
            ]

        if mode == RetrieveMode.STRUCTURED:
            if not criteria:
                raise ValueError(
                    "HFRetrievalSource STRUCTURED mode requires a "
                    "non-empty 'criteria' mapping."
                )
            ds = self._get_dataset()
            filtered = ds.filter(
                lambda row: all(
                    row.get(k) == v for k, v in criteria.items()
                )
            )
            count = min(n, len(filtered))
            selected = filtered.select(range(count))
            return [
                self._normalize_row(dict(selected[i]), i)
                for i in range(count)
            ]

        raise ValueError(f"Unsupported RetrieveMode: {mode}")


class ArxivRetrievalSource(RetrievalSource):
    """Retrieval source that fetches papers from arXiv via the API.

    Uses the arxiv Python package (Client and Search) to run queries in three
    modes: QUERY (text search + n), RANDOM (sample n from a category with seed),
    and STRUCTURED (criteria-based filters). Returns a list of item dicts with
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
        criteria: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return items from arXiv by mode (QUERY, RANDOM, or STRUCTURED).

        Mode behavior:
        - QUERY: Uses query (default "all:*" when empty) and n. Optional
          kwargs: sort_by (relevance|submittedDate|lastUpdatedDate or
          arxiv.SortCriterion), sort_order (ascending|descending or
          arxiv.SortOrder).
        - RANDOM: Fetches from category cs.AI with max_results=min(n*10, 500),
          then samples n with seed. Optional kwargs: random_category (e.g.
          "cs.CL") to change the category.
        - STRUCTURED: criteria keys: id_list or ids (list of arXiv ids),
          category (e.g. "cs.LG"), author, title, query (raw query string).
          Sort is taken from kwargs (sort_by, sort_order), not from criteria.
          Query parts are ANDed.

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

        if mode == RetrieveMode.STRUCTURED:
            crit = criteria or {}
            id_list = crit.get("id_list") or crit.get("ids") or []
            parts = []
            if crit.get("category"):
                parts.append(f"cat:{crit['category']}")  # arXiv category filter
            if crit.get("author"):
                parts.append(f'au:{crit["author"]}')  # Author name filter
            if crit.get("title"):
                parts.append(f'ti:{crit["title"]}')  # Title text filter
            if crit.get("query"):
                parts.append(crit["query"])  # Raw query string
            search_query = " AND ".join(parts) if parts else ""
            search = arxiv.Search(
                query=search_query,  # AND-combined filters
                id_list=id_list,
                max_results=n,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            results = list(client.results(search))
            return [self._result_to_item(r) for r in results[:n]]

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
        criteria: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return list of items; stub returns empty list.

        Would use SemanticScholar().search_paper() with
        query/random/structured.
        """
        return []
