"""Retrieval sources for acquire.retrieve: abstract base and concrete backends.

Concrete sources (web, wiki, HF, arxiv, S2) are stub implementations that can be
replaced with real API calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any


class RetrieveMode(Enum):
    """How to select segments from the source."""

    QUERY = auto()  # query string (and optional k/top_k)
    RANDOM = auto()  # random sampling (seed, n)
    STRUCTURED = auto()  # criteria/filter (e.g. dict of field constraints)


class RetrievalSource(ABC):
    """Abstract base for a corpus or API that can yield text segments (passages, docs, rows).

    Subclasses implement fetch() to return a list of items (e.g. dicts with "text", "id").
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
        """Return list of items (e.g. [{"text": "...", "id": "..."}]). Mode-specific args used as needed.

        Args:
            mode: How to select segments (QUERY, RANDOM, or STRUCTURED).
            query: Search or query string; used when mode is QUERY.
            n: Number of segments to return (e.g. top-k or sample size).
            seed: Random seed for reproducible sampling when mode is RANDOM.
            criteria: Field constraints or filter when mode is STRUCTURED.
            **kwargs: Passed through to backend; mode-specific.

        Returns:
            List of dicts, each typically containing at least "text" and optionally "id".
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
        # Stub: would use requests/crawler with query/random/structured mode.
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
        # Stub: would use wikipedia-api or MediaWiki with query/random/structured.
        return []


class HFRetrievalSource(RetrievalSource):
    """Hugging Face datasets (load_dataset + shuffle/filter/select). Stub implementation."""

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
            text_column: Column name containing text; used when normalizing rows to dicts.
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
        criteria: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        # Stub: would use datasets.load_dataset(), shuffle(seed), select(range(n)), filter(criteria).
        return []


class ArxivRetrievalSource(RetrievalSource):
    """arXiv (arxiv package Search + Client). Stub implementation."""

    def __init__(self, **kwargs: Any) -> None:
        """Store options for arXiv retrieval. Stub does not call the API.

        Args:
            **kwargs: Passed through to fetch(); reserved for Client/Search options.
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
        # Stub: would use arxiv.Search and Client().results() with query/random/structured.
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
        # Stub: would use SemanticScholar().search_paper() with query/random/structured.
        return []
