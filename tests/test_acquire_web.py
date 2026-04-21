"""Tests for WebRetrievalSource.fetch() with mocked Brave Search API."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from pyforger.acquire import RetrieveMode, WebRetrievalSource


def _make_brave_search_response(results: list[dict]) -> dict:
    """Build a mock Brave Search API JSON response."""
    return {"web": {"results": results}}


def _make_brave_result(
    title: str = "Title",
    url: str = "https://example.com",
    description: str = "A snippet.",
) -> dict:
    """Build a single Brave search result dict."""
    return {"title": title, "url": url, "description": description}


class TestWebRetrievalSourceFetch:
    """WebRetrievalSource.fetch() snippet-mode tests with mocked HTTP."""

    @patch("pyforger.acquire.sources.requests.Session")
    def test_query_returns_items_with_text_and_id(
        self, mock_session_cls: MagicMock
    ) -> None:
        """QUERY mode returns n items each with text, id, title, url."""
        brave_resp = _make_brave_search_response(
            [
                _make_brave_result("ML Intro", "https://a.com", "Learn ML."),
                _make_brave_result("Deep RL", "https://b.com", "RL overview."),
                _make_brave_result("NLP 101", "https://c.com", "NLP basics."),
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
        assert call_args[0][0] == "https://api.search.brave.com/res/v1/web/search"
        params = call_args[1]["params"]
        assert params["q"] == "machine learning"
        assert params["count"] == "3"

    @patch("pyforger.acquire.sources.requests.Session")
    def test_handles_missing_fields(self, mock_session_cls: MagicMock) -> None:
        """Results with missing description or title produce empty strings."""
        brave_resp = _make_brave_search_response(
            [
                {"url": "https://a.com", "title": "Has Title"},
                {"url": "https://b.com", "description": "Has Desc"},
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
        assert items[1]["text"] == "Has Desc"

    @patch("pyforger.acquire.sources.requests.Session")
    def test_respects_base_url(self, mock_session_cls: MagicMock) -> None:
        """base_url is prepended as site: operator to the query."""
        brave_resp = _make_brave_search_response([_make_brave_result()])

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

    @patch("pyforger.acquire.sources.requests.Session")
    def test_pagination(self, mock_session_cls: MagicMock) -> None:
        """Multiple paginated requests when n is greater than 20."""
        page1_results = [
            _make_brave_result(f"R{i}", f"https://r{i}.com", f"Snippet {i}")
            for i in range(20)
        ]
        page2_results = [
            _make_brave_result(f"R{i}", f"https://r{i}.com", f"Snippet {i}")
            for i in range(20, 25)
        ]

        mock_session = MagicMock()
        resp1 = MagicMock()
        resp1.json.return_value = _make_brave_search_response(page1_results)
        resp1.raise_for_status = MagicMock()
        resp2 = MagicMock()
        resp2.json.return_value = _make_brave_search_response(page2_results)
        resp2.raise_for_status = MagicMock()
        mock_session.get.side_effect = [resp1, resp2]
        mock_session_cls.return_value = mock_session

        source = WebRetrievalSource(api_key="k")
        items = source.fetch(RetrieveMode.QUERY, query="test", n=25)

        assert len(items) == 25
        assert mock_session.get.call_count == 2
        first_params = mock_session.get.call_args_list[0][1]["params"]
        second_params = mock_session.get.call_args_list[1][1]["params"]
        assert first_params["offset"] == "0"
        assert first_params["count"] == "20"
        assert second_params["offset"] == "1"
        assert second_params["count"] == "5"

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
        brave_resp = _make_brave_search_response([_make_brave_result()])
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
        brave_resp = _make_brave_search_response([_make_brave_result()])
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


class TestWebRetrievalSourceFullContent:
    """WebRetrievalSource.fetch() full-content mode tests."""

    @patch("pyforger.acquire.sources.requests.Session")
    def test_full_content_fetches_urls(self, mock_session_cls: MagicMock) -> None:
        """fetch_full_content=True replaces snippet with extracted page text."""
        brave_resp = _make_brave_search_response(
            [
                _make_brave_result("Page A", "https://a.com", "Snippet A"),
                _make_brave_result("Page B", "https://b.com", "Snippet B"),
            ]
        )

        brave_session = MagicMock()
        brave_resp_mock = MagicMock()
        brave_resp_mock.json.return_value = brave_resp
        brave_resp_mock.raise_for_status = MagicMock()
        brave_session.get.return_value = brave_resp_mock

        content_session = MagicMock()
        page_a_html = "<html><body><p>Full article about A.</p></body></html>"
        page_a_resp = MagicMock()
        page_a_resp.headers = {}
        page_a_resp.raise_for_status = MagicMock()
        page_a_resp.iter_content.return_value = [page_a_html.encode()]
        page_a_resp.apparent_encoding = "utf-8"

        page_b_html = "<html><body><p>Full article about B.</p></body></html>"
        page_b_resp = MagicMock()
        page_b_resp.headers = {}
        page_b_resp.raise_for_status = MagicMock()
        page_b_resp.iter_content.return_value = [page_b_html.encode()]
        page_b_resp.apparent_encoding = "utf-8"

        def content_get(url, **kwargs):
            if url == "https://a.com":
                return page_a_resp
            return page_b_resp

        content_session.get.side_effect = content_get

        # First call to Session() -> brave_session, second -> content_session
        mock_session_cls.side_effect = [brave_session, content_session]

        source = WebRetrievalSource(api_key="k", fetch_full_content=True)
        items = source.fetch(RetrieveMode.QUERY, query="test", n=2)

        assert len(items) == 2
        assert "Full article about A." in items[0]["text"]
        assert "Full article about B." in items[1]["text"]
        assert items[0]["text"] != "Snippet A"
        assert items[1]["text"] != "Snippet B"

    @patch("pyforger.acquire.sources.requests.Session")
    def test_full_content_falls_back_on_fetch_failure(
        self, mock_session_cls: MagicMock
    ) -> None:
        """Failed URL fetch falls back to Brave snippet; no crash."""
        brave_resp = _make_brave_search_response(
            [
                _make_brave_result("Good", "https://good.com", "Good snippet"),
                _make_brave_result("Bad", "https://bad.com", "Bad snippet"),
            ]
        )

        brave_session = MagicMock()
        brave_resp_mock = MagicMock()
        brave_resp_mock.json.return_value = brave_resp
        brave_resp_mock.raise_for_status = MagicMock()
        brave_session.get.return_value = brave_resp_mock

        content_session = MagicMock()
        good_html = "<html><body><p>Full content here.</p></body></html>"
        good_resp = MagicMock()
        good_resp.headers = {}
        good_resp.raise_for_status = MagicMock()
        good_resp.iter_content.return_value = [good_html.encode()]
        good_resp.apparent_encoding = "utf-8"

        bad_resp = MagicMock()
        bad_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Not Found"
        )

        def content_get(url, **kwargs):
            if url == "https://good.com":
                return good_resp
            return bad_resp

        content_session.get.side_effect = content_get
        mock_session_cls.side_effect = [brave_session, content_session]

        source = WebRetrievalSource(api_key="k", fetch_full_content=True)
        items = source.fetch(RetrieveMode.QUERY, query="test", n=2)

        assert len(items) == 2
        assert "Full content here." in items[0]["text"]
        assert items[1]["text"] == "Bad snippet"

    @patch("pyforger.acquire.sources.requests.Session")
    def test_full_content_respects_max_content_length_header(
        self, mock_session_cls: MagicMock
    ) -> None:
        """Pages with Content-Length exceeding limit fall back to snippet."""
        brave_resp = _make_brave_search_response(
            [_make_brave_result("Big", "https://big.com", "Big snippet")]
        )

        brave_session = MagicMock()
        brave_resp_mock = MagicMock()
        brave_resp_mock.json.return_value = brave_resp
        brave_resp_mock.raise_for_status = MagicMock()
        brave_session.get.return_value = brave_resp_mock

        content_session = MagicMock()
        big_resp = MagicMock()
        big_resp.headers = {"Content-Length": "9999999"}
        big_resp.raise_for_status = MagicMock()
        content_session.get.return_value = big_resp
        mock_session_cls.side_effect = [brave_session, content_session]

        source = WebRetrievalSource(
            api_key="k", fetch_full_content=True, max_content_length=100
        )
        items = source.fetch(RetrieveMode.QUERY, query="test", n=1)

        assert len(items) == 1
        assert items[0]["text"] == "Big snippet"
        big_resp.close.assert_called_once()

    @patch("pyforger.acquire.sources.requests.Session")
    def test_full_content_respects_max_content_length_streaming(
        self, mock_session_cls: MagicMock
    ) -> None:
        """Pages exceeding limit during streaming (no Content-Length) fall back."""
        brave_resp = _make_brave_search_response(
            [_make_brave_result("Big", "https://big.com", "Big snippet")]
        )

        brave_session = MagicMock()
        brave_resp_mock = MagicMock()
        brave_resp_mock.json.return_value = brave_resp
        brave_resp_mock.raise_for_status = MagicMock()
        brave_session.get.return_value = brave_resp_mock

        content_session = MagicMock()
        big_resp = MagicMock()
        big_resp.headers = {}
        big_resp.raise_for_status = MagicMock()
        big_resp.iter_content.return_value = [b"x" * 200]
        big_resp.apparent_encoding = "utf-8"
        content_session.get.return_value = big_resp
        mock_session_cls.side_effect = [brave_session, content_session]

        source = WebRetrievalSource(
            api_key="k", fetch_full_content=True, max_content_length=100
        )
        items = source.fetch(RetrieveMode.QUERY, query="test", n=1)

        assert len(items) == 1
        assert items[0]["text"] == "Big snippet"
        content_session.get.assert_called_once_with(
            "https://big.com", stream=True, timeout=5
        )

    @patch("pyforger.acquire.sources.trafilatura")
    def test_extract_text_calls_trafilatura(self, mock_traf: MagicMock) -> None:
        """_extract_text delegates to trafilatura.extract."""
        mock_traf.extract.return_value = "Clean extracted text."

        result = WebRetrievalSource._extract_text("<html><body>raw</body></html>")
        assert result == "Clean extracted text."
        mock_traf.extract.assert_called_once()

    @patch("pyforger.acquire.sources.trafilatura")
    def test_extract_text_returns_none_on_failure(self, mock_traf: MagicMock) -> None:
        """_extract_text returns None when trafilatura returns None."""
        mock_traf.extract.return_value = None

        result = WebRetrievalSource._extract_text("<html><body>raw</body></html>")
        assert result is None

    @pytest.mark.skipif(
        not os.environ.get("PYFORGER_WEB_LIVE"),
        reason="Set PYFORGER_WEB_LIVE=1 to run live Brave Search API test",
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
