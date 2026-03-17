"""Tests for ArxivRetrievalSource.fetch() with mocked arxiv API."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from pyforger.acquire import ArxivRetrievalSource, RetrieveMode


def _make_mock_result(
    short_id: str, title: str = "Title", summary: str = "Abstract."
) -> MagicMock:
    r = MagicMock()
    r.get_short_id.return_value = short_id
    r.title = title
    r.summary = summary
    r.entry_id = f"http://arxiv.org/abs/{short_id}"
    return r


class TestArxivRetrievalSourceFetch:
    """ArxivRetrievalSource.fetch() with mocked arxiv.Client and Search."""

    @patch("pyforger.acquire.sources.arxiv.Search")
    @patch("pyforger.acquire.sources.arxiv.Client")
    def test_query_mode_returns_items_with_text_and_id(
        self, mock_client_cls: MagicMock, mock_search_cls: MagicMock
    ) -> None:
        """QUERY mode returns n items each with text, id, title, summary."""
        mock_result1 = _make_mock_result(
            "2107.05580v1", "Quantum ML", "We study quantum."
        )
        mock_result2 = _make_mock_result(
            "2301.00001v1", "Deep Learning", "We propose a model."
        )
        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_result1, mock_result2])
        mock_client_cls.return_value = mock_client

        source = ArxivRetrievalSource()
        items = source.fetch(RetrieveMode.QUERY, query="quantum", n=2)

        assert len(items) == 2
        assert items[0]["id"] == "2107.05580v1"
        assert items[0]["title"] == "Quantum ML"
        assert items[0]["summary"] == "We study quantum."
        assert "Quantum ML" in items[0]["text"]
        assert "We study quantum." in items[0]["text"]
        assert items[1]["id"] == "2301.00001v1"
        assert items[1]["title"] == "Deep Learning"
        mock_search_cls.assert_called_once()
        call_kw = mock_search_cls.call_args[1]
        assert call_kw["query"] == "quantum"
        assert call_kw["max_results"] == 2

    @patch("pyforger.acquire.sources.arxiv.Search")
    @patch("pyforger.acquire.sources.arxiv.Client")
    def test_query_mode_empty_query_uses_all_star(
        self, mock_client_cls: MagicMock, mock_search_cls: MagicMock
    ) -> None:
        """Empty or None query uses default 'all:*'."""
        mock_client = MagicMock()
        mock_client.results.return_value = iter([])
        mock_client_cls.return_value = mock_client

        source = ArxivRetrievalSource()
        source.fetch(RetrieveMode.QUERY, query="", n=5)
        call_kw = mock_search_cls.call_args[1]
        assert call_kw["query"] == "all:*"

        source.fetch(RetrieveMode.QUERY, query=None, n=5)
        call_kw2 = mock_search_cls.call_args[1]
        assert call_kw2["query"] == "all:*"

    @patch("pyforger.acquire.sources.arxiv.Search")
    @patch("pyforger.acquire.sources.arxiv.Client")
    def test_random_mode_returns_n_sampled_with_seed(
        self, mock_client_cls: MagicMock, mock_search_cls: MagicMock
    ) -> None:
        """RANDOM mode samples n items using seed; same seed gives same order."""
        results_10 = [
            _make_mock_result(f"2001.0000{i}v1", f"Paper {i}", f"Abstract {i}")
            for i in range(10)
        ]
        mock_client = MagicMock()
        # Fresh iterator each time results() is called (exhausted after first)
        mock_client.results.side_effect = lambda search: iter(results_10)
        mock_client_cls.return_value = mock_client

        source = ArxivRetrievalSource()
        items1 = source.fetch(RetrieveMode.RANDOM, n=3, seed=42)
        items2 = source.fetch(RetrieveMode.RANDOM, n=3, seed=42)

        assert len(items1) == 3
        assert len(items2) == 3
        assert [it["id"] for it in items1] == [it["id"] for it in items2]
        call_kw = mock_search_cls.call_args[1]
        assert "cat:cs.AI" in call_kw["query"] or call_kw["query"] == "cat:cs.AI"
        assert call_kw["max_results"] == min(3 * 10, 500)

    @patch("pyforger.acquire.sources.arxiv.Search")
    @patch("pyforger.acquire.sources.arxiv.Client")
    def test_structured_mode_id_list_builds_search_with_id_list(
        self, mock_client_cls: MagicMock, mock_search_cls: MagicMock
    ) -> None:
        """STRUCTURED mode with criteria id_list builds Search with id_list."""
        mock_result = _make_mock_result("1234.5678v1", "A Paper", "Summary.")
        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_result])
        mock_client_cls.return_value = mock_client

        source = ArxivRetrievalSource()
        items = source.fetch(
            RetrieveMode.STRUCTURED,
            n=10,
            criteria={"id_list": ["1234.5678v1"]},
        )

        assert len(items) == 1
        assert items[0]["id"] == "1234.5678v1"
        assert items[0]["title"] == "A Paper"
        mock_search_cls.assert_called_once()
        call_kw = mock_search_cls.call_args[1]
        assert call_kw["id_list"] == ["1234.5678v1"]

    @patch("pyforger.acquire.sources.arxiv.Search")
    @patch("pyforger.acquire.sources.arxiv.Client")
    def test_structured_mode_ids_alias(
        self, mock_client_cls: MagicMock, mock_search_cls: MagicMock
    ) -> None:
        """STRUCTURED mode accepts 'ids' as alias for id_list."""
        mock_client = MagicMock()
        mock_client.results.return_value = iter([])
        mock_client_cls.return_value = mock_client

        source = ArxivRetrievalSource()
        source.fetch(RetrieveMode.STRUCTURED, n=5, criteria={"ids": ["9999.9999v1"]})

        call_kw = mock_search_cls.call_args[1]
        assert call_kw["id_list"] == ["9999.9999v1"]

    @pytest.mark.skipif(
        not os.environ.get("PYFORGER_ARXIV_LIVE"),
        reason="Set PYFORGER_ARXIV_LIVE=1 to run live arXiv API test",
    )
    def test_fetch_live_query(self) -> None:
        """Live API test for QUERY mode (skipped unless PYFORGER_ARXIV_LIVE=1)."""
        source = ArxivRetrievalSource()
        items = source.fetch(RetrieveMode.QUERY, query="machine learning", n=2)
        assert len(items) == 2
        for item in items:
            assert "text" in item
            assert "id" in item
            assert "title" in item
            assert "summary" in item
            assert item["title"] in item["text"]
            assert item["summary"] in item["text"]
