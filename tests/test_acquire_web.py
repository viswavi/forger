"""Tests for WebRetrievalSource.fetch() with mocked Brave LLM Context API."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from pyforger.acquire import RetrieveMode, WebRetrievalSource


def _make_context_response(results: list[dict]) -> dict:
    """Build a mock Brave LLM Context API JSON response."""
    return {"grounding": {"generic": results, "map": []}, "sources": {}}


def _make_generic_result(
    title: str = "Title",
    url: str = "https://example.com",
    snippets: list[str] | None = None,
) -> dict:
    """Build a single ``grounding.generic[]`` entry."""
    return {"title": title, "url": url, "snippets": snippets or ["A snippet."]}


class TestWebRetrievalSourceFetch:
    """WebRetrievalSource.fetch() tests with mocked HTTP."""

    @patch("pyforger.acquire.sources.requests.Session")
    def test_query_returns_items_with_text_and_id(
        self, mock_session_cls: MagicMock
    ) -> None:
        """QUERY mode returns n items each with text, id, title, url."""
        brave_resp = _make_context_response(
            [
                _make_generic_result("ML Intro", "https://a.com", ["Learn ML."]),
                _make_generic_result("Deep RL", "https://b.com", ["RL overview."]),
                _make_generic_result("NLP 101", "https://c.com", ["NLP basics."]),
            ]
        )

        mock_session = MagicMock()
        resp_mock = MagicMock()
        resp_mock.json.return_value = brave_resp
        resp_mock.raise_for_status = MagicMock()
        mock_session.get.return_value = resp_mock
        mock_session_cls.return_value = mock_session

        source = WebRetrievalSource(api_key="test-key")
        items = source.fetch(RetrieveMode.QUERY, query="machine learning", n=3)

        assert len(items) == 3
        assert items[0]["text"] == "Learn ML."
        assert items[0]["id"] == "https://a.com"
        assert items[0]["title"] == "ML Intro"
        assert items[0]["url"] == "https://a.com"
        assert items[1]["text"] == "RL overview."
        assert items[2]["title"] == "NLP 101"

        call_args = mock_session.get.call_args
        assert call_args[0][0] == "https://api.search.brave.com/res/v1/llm/context"
        params = call_args[1]["params"]
        assert params["q"] == "machine learning"
        assert params["count"] == "3"
        assert params["maximum_number_of_urls"] == "3"

    @patch("pyforger.acquire.sources.requests.Session")
    def test_joins_multiple_snippets(self, mock_session_cls: MagicMock) -> None:
        """Multiple snippets for one URL are joined with blank lines."""
        brave_resp = _make_context_response(
            [_make_generic_result("Page", "https://a.com", ["First.", "Second."])]
        )

        mock_session = MagicMock()
        resp_mock = MagicMock()
        resp_mock.json.return_value = brave_resp
        resp_mock.raise_for_status = MagicMock()
        mock_session.get.return_value = resp_mock
        mock_session_cls.return_value = mock_session

        source = WebRetrievalSource(api_key="k")
        items = source.fetch(RetrieveMode.QUERY, query="test", n=1)

        assert items[0]["text"] == "First.\n\nSecond."

    @patch("pyforger.acquire.sources.requests.Session")
    def test_handles_missing_fields(self, mock_session_cls: MagicMock) -> None:
        """Results with missing snippets or title produce empty strings."""
        brave_resp = _make_context_response(
            [
                {"url": "https://a.com", "title": "Has Title"},
                {"url": "https://b.com", "snippets": ["Has snippet"]},
            ]
        )

        mock_session = MagicMock()
        resp_mock = MagicMock()
        resp_mock.json.return_value = brave_resp
        resp_mock.raise_for_status = MagicMock()
        mock_session.get.return_value = resp_mock
        mock_session_cls.return_value = mock_session

        source = WebRetrievalSource(api_key="k")
        items = source.fetch(RetrieveMode.QUERY, query="test", n=2)

        assert len(items) == 2
        assert items[0]["title"] == "Has Title"
        assert items[0]["text"] == ""
        assert items[1]["title"] == ""
        assert items[1]["text"] == "Has snippet"

    @patch("pyforger.acquire.sources.requests.Session")
    def test_respects_base_url(self, mock_session_cls: MagicMock) -> None:
        """base_url is prepended as site: operator to the query."""
        brave_resp = _make_context_response([_make_generic_result()])

        mock_session = MagicMock()
        resp_mock = MagicMock()
        resp_mock.json.return_value = brave_resp
        resp_mock.raise_for_status = MagicMock()
        mock_session.get.return_value = resp_mock
        mock_session_cls.return_value = mock_session

        source = WebRetrievalSource(base_url="example.com", api_key="k")
        source.fetch(RetrieveMode.QUERY, query="python", n=1)

        params = mock_session.get.call_args[1]["params"]
        assert params["q"] == "site:example.com python"

    def test_random_raises(self) -> None:
        """RANDOM mode raises NotImplementedError with clear message."""
        source = WebRetrievalSource(api_key="k")
        with pytest.raises(NotImplementedError, match="RANDOM mode"):
            source.fetch(RetrieveMode.RANDOM, n=5)

    @patch("pyforger.acquire.sources.requests.Session")
    def test_sets_api_key_header(self, mock_session_cls: MagicMock) -> None:
        """API key is set as X-Subscription-Token in session headers."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        source = WebRetrievalSource(api_key="my-secret-key")
        source._get_session()

        mock_session.headers.update.assert_called_once()
        headers = mock_session.headers.update.call_args[0][0]
        assert headers["X-Subscription-Token"] == "my-secret-key"

    @pytest.mark.parametrize("n,expected", [(25, 25), (100, 50)])
    @patch("pyforger.acquire.sources.requests.Session")
    def test_single_request_capped_at_api_maximum(
        self, mock_session_cls: MagicMock, n: int, expected: int
    ) -> None:
        """One request regardless of n; n above the 50-result max is capped."""
        results = [
            _make_generic_result(f"R{i}", f"https://r{i}.com", [f"Snippet {i}"])
            for i in range(expected)
        ]

        mock_session = MagicMock()
        resp_mock = MagicMock()
        resp_mock.json.return_value = _make_context_response(results)
        resp_mock.raise_for_status = MagicMock()
        mock_session.get.return_value = resp_mock
        mock_session_cls.return_value = mock_session

        source = WebRetrievalSource(api_key="k")
        items = source.fetch(RetrieveMode.QUERY, query="test", n=n)

        assert len(items) == expected
        assert mock_session.get.call_count == 1
        params = mock_session.get.call_args[1]["params"]
        assert params["count"] == str(expected)
        assert params["maximum_number_of_urls"] == str(expected)

    def test_empty_query_raises(self) -> None:
        """Empty or None query raises ValueError."""
        source = WebRetrievalSource(api_key="k")
        with pytest.raises(ValueError, match="query is required"):
            source.fetch(RetrieveMode.QUERY, query="", n=3)
        with pytest.raises(ValueError, match="query is required"):
            source.fetch(RetrieveMode.QUERY, query=None, n=3)

    @patch("pyforger.acquire.sources.requests.Session")
    def test_fetch_passes_timeout_to_get(self, mock_session_cls: MagicMock) -> None:
        """timeout= in constructor is forwarded to session.get."""
        brave_resp = _make_context_response([_make_generic_result()])
        mock_session = MagicMock()
        resp_mock = MagicMock()
        resp_mock.json.return_value = brave_resp
        resp_mock.raise_for_status = MagicMock()
        mock_session.get.return_value = resp_mock
        mock_session_cls.return_value = mock_session

        source = WebRetrievalSource(api_key="k", timeout=15.0)
        source.fetch(RetrieveMode.QUERY, query="test", n=1)

        call_kw = mock_session.get.call_args[1]
        assert call_kw.get("timeout") == 15.0

    @patch("pyforger.acquire.sources.requests.Session")
    def test_fetch_uses_default_timeout(self, mock_session_cls: MagicMock) -> None:
        """Default 5s timeout is applied when none is specified."""
        brave_resp = _make_context_response([_make_generic_result()])
        mock_session = MagicMock()
        resp_mock = MagicMock()
        resp_mock.json.return_value = brave_resp
        resp_mock.raise_for_status = MagicMock()
        mock_session.get.return_value = resp_mock
        mock_session_cls.return_value = mock_session

        source = WebRetrievalSource(api_key="k")
        source.fetch(RetrieveMode.QUERY, query="test", n=1)

        call_kw = mock_session.get.call_args[1]
        assert call_kw.get("timeout") == 5

    @pytest.mark.skipif(
        not os.environ.get("PYFORGER_WEB_LIVE"),
        reason="Set PYFORGER_WEB_LIVE=1 to run live Brave LLM Context API test",
    )
    def test_fetch_live_query(self) -> None:
        """Live API test for QUERY mode (skipped unless PYFORGER_WEB_LIVE=1)."""
        api_key = os.environ.get("BRAVE_API_KEY", "")
        source = WebRetrievalSource(api_key=api_key)
        items = source.fetch(RetrieveMode.QUERY, query="python programming", n=2)
        assert len(items) == 2
        for item in items:
            assert "text" in item
            assert "id" in item
            assert "title" in item
            assert "url" in item
            assert item["url"] == item["id"]
