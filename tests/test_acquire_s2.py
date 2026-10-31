"""Tests for S2RetrievalSource.fetch() with mocked Semantic Scholar API."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from pyforger.acquire import RetrieveMode, S2RetrievalSource


def _make_mock_paper(
    paper_id: str, title: str = "Title", abstract: str = "Abstract."
) -> MagicMock:
    p = MagicMock()
    p.paperId = paper_id
    p.title = title
    p.abstract = abstract
    return p


def _make_paginated_results(items: list[MagicMock]) -> MagicMock:
    pr = MagicMock()
    pr.items = items
    pr.next = len(items) + 1 if items else None
    pr.total = 1000
    return pr


class TestS2RetrievalSourceFetch:
    """S2RetrievalSource.fetch() with mocked SemanticScholar and search_paper."""

    @patch("pyforger.acquire.sources.SemanticScholar")
    def test_query_mode_returns_items_with_text_and_id(
        self, mock_scholar_cls: MagicMock
    ) -> None:
        """QUERY mode returns n items each with text, id, title, summary."""
        mock_paper1 = _make_mock_paper("abc123", "Quantum ML", "We study quantum.")
        mock_paper2 = _make_mock_paper("def456", "Deep Learning", "We propose a model.")
        mock_results = _make_paginated_results([mock_paper1, mock_paper2])
        mock_results.next = None
        mock_client = MagicMock()
        mock_client.search_paper.return_value = mock_results
        mock_scholar_cls.return_value = mock_client

        source = S2RetrievalSource()
        items = source.fetch(RetrieveMode.QUERY, query="quantum", n=2)

        assert len(items) == 2
        assert items[0]["id"] == "abc123"
        assert items[0]["title"] == "Quantum ML"
        assert items[0]["summary"] == "We study quantum."
        assert "Quantum ML" in items[0]["text"]
        assert "We study quantum." in items[0]["text"]
        assert items[1]["id"] == "def456"
        assert items[1]["title"] == "Deep Learning"
        mock_client.search_paper.assert_called_once()
        call_args, call_kw = mock_client.search_paper.call_args
        assert call_args[0] == "quantum"
        assert call_kw.get("limit") == 100

    @patch("pyforger.acquire.sources.SemanticScholar")
    def test_query_mode_empty_query_uses_default(
        self, mock_scholar_cls: MagicMock
    ) -> None:
        """When query is empty, call uses the default query (machine learning)."""
        mock_results = _make_paginated_results([])
        mock_results.next = None
        mock_client = MagicMock()
        mock_client.search_paper.return_value = mock_results
        mock_scholar_cls.return_value = mock_client

        source = S2RetrievalSource()
        source.fetch(RetrieveMode.QUERY, query="", n=5)
        call_args = mock_client.search_paper.call_args[0]
        assert call_args[0] == "machine learning"

        source.fetch(RetrieveMode.QUERY, query=None, n=5)
        call_args2 = mock_client.search_paper.call_args[0]
        assert call_args2[0] == "machine learning"

    @patch("pyforger.acquire.sources.SemanticScholar")
    def test_random_mode_returns_n_sampled_with_seed(
        self, mock_scholar_cls: MagicMock
    ) -> None:
        """RANDOM mode samples n items using seed; same seed gives same order."""
        results_10 = [
            _make_mock_paper(f"id{i}", f"Paper {i}", f"Abstract {i}") for i in range(10)
        ]
        mock_results = _make_paginated_results(results_10)
        mock_results.next = None
        mock_client = MagicMock()
        mock_client.search_paper.return_value = mock_results
        mock_scholar_cls.return_value = mock_client

        source = S2RetrievalSource()
        items1 = source.fetch(RetrieveMode.RANDOM, n=3, seed=42)
        items2 = source.fetch(RetrieveMode.RANDOM, n=3, seed=42)

        assert len(items1) == 3
        assert len(items2) == 3
        assert [it["id"] for it in items1] == [it["id"] for it in items2]
        call_args, call_kw = mock_client.search_paper.call_args
        assert call_args[0] == "Computer Science"
        assert call_kw.get("limit") == 100

    def test_result_to_item_normalization(self) -> None:
        """_result_to_item produces text, id, title, summary; normalized whitespace."""
        mock_paper = _make_mock_paper("norm-1", "  A Title  ", "Line one.\n\nLine two.")
        item = S2RetrievalSource._result_to_item(mock_paper)
        assert item["id"] == "norm-1"
        assert item["title"] == "  A Title  "
        assert item["summary"] == "Line one.\n\nLine two."
        assert "text" in item and "id" in item and "title" in item and "summary" in item
        assert "A Title" in item["text"]
        assert "Line one. Line two." in item["text"]

    @patch("pyforger.acquire.sources.SemanticScholar")
    def test_query_mode_pagination_n_over_100(
        self, mock_scholar_cls: MagicMock
    ) -> None:
        """When n > 100, next_page() is called until enough items are collected."""
        page1 = [_make_mock_paper(f"p1-{i}", f"Paper {i}", "Abst") for i in range(100)]
        page2 = [
            _make_mock_paper(f"p2-{i}", f"Paper {i+100}", "Abst") for i in range(100)
        ]
        page3 = [
            _make_mock_paper(f"p3-{i}", f"Paper {i+200}", "Abst") for i in range(50)
        ]
        all_250 = page1 + page2 + page3

        mock_results = MagicMock()
        mock_results.items = list(page1)
        mock_results.next = 101
        mock_results.total = 1000
        next_page_calls = [0]

        def side_effect_next_page():
            next_page_calls[0] += 1
            if next_page_calls[0] == 1:
                mock_results.items = list(page1) + list(page2)
                mock_results.next = 201
            elif next_page_calls[0] == 2:
                mock_results.items = list(all_250)
                mock_results.next = None

        mock_results.next_page = side_effect_next_page
        mock_client = MagicMock()
        mock_client.search_paper.return_value = mock_results
        mock_scholar_cls.return_value = mock_client

        source = S2RetrievalSource()
        items = source.fetch(RetrieveMode.QUERY, query="ML", n=250)

        assert len(items) == 250
        assert next_page_calls[0] >= 2

    @patch("pyforger.acquire.sources.SemanticScholar")
    def test_random_mode_uses_bulk_when_pool_exceeds_1000(
        self, mock_scholar_cls: MagicMock
    ) -> None:
        """When n*10 > 1000, bulk search is used automatically (e.g. n=200)."""
        pool_2000 = [
            _make_mock_paper(f"bulk-{i}", f"Bulk {i}", "Abst") for i in range(2000)
        ]
        mock_results = MagicMock()
        mock_results.items = pool_2000[:1000]
        mock_results.next = 1001
        mock_results.total = 5000

        def side_effect_next_page():
            if len(mock_results.items) == 1000:
                mock_results.items = list(pool_2000[:2000])
                mock_results.next = None

        mock_results.next_page = side_effect_next_page
        mock_client = MagicMock()
        mock_client.search_paper.return_value = mock_results
        mock_scholar_cls.return_value = mock_client

        source = S2RetrievalSource()
        items = source.fetch(RetrieveMode.RANDOM, n=200, seed=99)

        assert len(items) == 200
        mock_client.search_paper.assert_called_once()
        call_args, call_kw = mock_client.search_paper.call_args
        assert call_kw.get("bulk") is True

    @patch("pyforger.acquire.sources.SemanticScholar")
    def test_random_mode_uses_relevance_when_pool_at_most_1000(
        self, mock_scholar_cls: MagicMock
    ) -> None:
        """When n*10 <= 1000, relevance search is used (e.g. n=50)."""
        results_100 = [
            _make_mock_paper(f"id{i}", f"Paper {i}", "Abst") for i in range(100)
        ]
        mock_results = _make_paginated_results(results_100)
        mock_results.next = None
        mock_client = MagicMock()
        mock_client.search_paper.return_value = mock_results
        mock_scholar_cls.return_value = mock_client

        source = S2RetrievalSource()
        items = source.fetch(RetrieveMode.RANDOM, n=50, seed=42)

        assert len(items) == 50
        mock_client.search_paper.assert_called_once()
        call_args, call_kw = mock_client.search_paper.call_args
        assert call_kw.get("limit") == 100
        assert "bulk" not in call_kw or call_kw.get("bulk") is not True

    @pytest.mark.skipif(
        not os.environ.get("PYFORGER_S2_LIVE"),
        reason="Set PYFORGER_S2_LIVE=1 to run live Semantic Scholar API test",
    )
    def test_fetch_live_query(self) -> None:
        """Live API test for QUERY mode (skipped unless PYFORGER_S2_LIVE=1)."""
        source = S2RetrievalSource()
        items = source.fetch(RetrieveMode.QUERY, query="machine learning", n=2)
        assert len(items) == 2
        for item in items:
            assert "text" in item
            assert "id" in item
            assert "title" in item
            assert "summary" in item
            assert item["title"] in item["text"]
            assert item["summary"] in item["text"]


class TestS2CitationsMode:
    """CITATIONS mode via get_paper_citations with rich bibliographic items."""

    @staticmethod
    def _edge(paper: MagicMock) -> MagicMock:
        edge = MagicMock()
        edge.paper = paper
        return edge

    @patch("pyforger.acquire.sources.SemanticScholar")
    def test_citations_returns_rich_items(self, mock_scholar_cls: MagicMock) -> None:
        """CITATIONS mode maps rich bibliographic fields onto items."""
        paper = _make_mock_paper("c1", "Citing Paper", "It builds on the seed.")
        paper.year = 2024
        paper.venue = "NeurIPS"
        paper.authors = [{"name": "Ann Author"}, {"name": "Bob Writer"}]
        paper.citationCount = 7
        paper.externalIds = {"DOI": "10.1/abc"}
        paper.url = "https://example.org/c1"
        mock_results = _make_paginated_results([self._edge(paper)])
        mock_results.next = None
        mock_client = MagicMock()
        mock_client.get_paper_citations.return_value = mock_results
        mock_scholar_cls.return_value = mock_client

        source = S2RetrievalSource()
        items = source.fetch(RetrieveMode.CITATIONS, query="DOI:10.0/seed", n=5)

        assert len(items) == 1
        item = items[0]
        assert item["id"] == "c1"
        assert item["title"] == "Citing Paper"
        assert item["summary"] == "It builds on the seed."
        assert item["year"] == 2024
        assert item["venue"] == "NeurIPS"
        assert item["authors"] == ["Ann Author", "Bob Writer"]
        assert item["citation_count"] == 7
        assert item["external_ids"] == {"DOI": "10.1/abc"}
        assert item["url"] == "https://example.org/c1"
        args, kwargs = mock_client.get_paper_citations.call_args
        assert args[0] == "DOI:10.0/seed"
        assert "externalIds" in kwargs["fields"]
        assert kwargs["limit"] == 100

    def test_citations_requires_seed(self) -> None:
        """CITATIONS mode without a seed query is rejected."""
        source = S2RetrievalSource()
        with pytest.raises(ValueError, match="CITATIONS mode requires"):
            source.fetch(RetrieveMode.CITATIONS, query=None, n=5)
