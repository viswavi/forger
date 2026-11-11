"""Tests for WikiRetrievalSource.fetch() with mocked MediaWiki API."""
from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from pyforger.acquire import RetrieveMode, WikiRetrievalSource


def _make_search_response(
    titles_and_ids: list[tuple[str, int]],
) -> dict:
    """Build a mock MediaWiki search API JSON response."""
    return {
        "query": {
            "search": [{"pageid": pid, "title": title} for title, pid in titles_and_ids]
        }
    }


def _make_extracts_response(
    pages: list[dict],
    redirects: list[dict] | None = None,
) -> dict:
    """Build a mock MediaWiki extracts API JSON response.

    Args:
        pages: List of dicts each with "pageid", "title", and optionally "extract".
        redirects: Optional list of {"from": ..., "to": ...} redirect entries.
    """
    out: dict = {
        "query": {
            "pages": {
                str(p["pageid"]): {
                    "pageid": p["pageid"],
                    "title": p["title"],
                    "extract": p.get("extract", ""),
                }
                for p in pages
            }
        }
    }
    if redirects:
        out["query"]["redirects"] = redirects
    return out


def _make_random_response(
    titles_and_ids: list[tuple[str, int]],
) -> dict:
    """Build a mock MediaWiki random API JSON response."""
    return {
        "query": {
            "random": [
                {"id": pid, "ns": 0, "title": title} for title, pid in titles_and_ids
            ]
        }
    }


class TestWikiRetrievalSourceFetch:
    """WikiRetrievalSource.fetch() with mocked requests.Session."""

    @patch("pyforger.acquire.sources.requests.Session")
    def test_query_mode_returns_items_with_text_and_id(
        self, mock_session_cls: MagicMock
    ) -> None:
        """QUERY mode returns n items each with text, id, title, summary."""
        search_resp = _make_search_response(
            [("Python (programming language)", 23862), ("History of Python", 21356332)]
        )
        extracts_resp = _make_extracts_response(
            [
                {
                    "pageid": 23862,
                    "title": "Python (programming language)",
                    "extract": "Python is a high-level programming language.",
                },
                {
                    "pageid": 21356332,
                    "title": "History of Python",
                    "extract": (
                        "The programming language Python was conceived "
                        "in the late 1980s."
                    ),
                },
            ]
        )

        mock_session = MagicMock()
        responses = [MagicMock(), MagicMock()]
        responses[0].json.return_value = search_resp
        responses[0].raise_for_status = MagicMock()
        responses[1].json.return_value = extracts_resp
        responses[1].raise_for_status = MagicMock()
        mock_session.get.side_effect = responses
        mock_session_cls.return_value = mock_session

        source = WikiRetrievalSource(language="en")
        items = source.fetch(RetrieveMode.QUERY, query="Python programming", n=2)

        assert len(items) == 2
        assert items[0]["id"] == "23862"
        assert items[0]["title"] == "Python (programming language)"
        assert items[0]["text"] == "Python is a high-level programming language."
        assert items[0]["summary"] == items[0]["text"]
        assert items[1]["id"] == "21356332"
        assert items[1]["title"] == "History of Python"

        search_call = mock_session.get.call_args_list[0]
        search_params = search_call[1]["params"]
        assert search_params["srsearch"] == "Python programming"
        assert search_params["srlimit"] == "2"

    @patch("pyforger.acquire.sources.requests.Session")
    def test_query_mode_empty_query_falls_back(
        self, mock_session_cls: MagicMock
    ) -> None:
        """Empty or None query uses default 'Wikipedia'."""
        search_resp = _make_search_response([("Wikipedia", 5043734)])
        extracts_resp = _make_extracts_response(
            [{"pageid": 5043734, "title": "Wikipedia", "extract": "Wikipedia is..."}]
        )

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        for q_val in ("", None):
            responses = [MagicMock(), MagicMock()]
            responses[0].json.return_value = search_resp
            responses[0].raise_for_status = MagicMock()
            responses[1].json.return_value = extracts_resp
            responses[1].raise_for_status = MagicMock()
            mock_session.get.side_effect = responses

            source = WikiRetrievalSource(language="en")
            source._session = None
            items = source.fetch(RetrieveMode.QUERY, query=q_val, n=1)

            search_call = mock_session.get.call_args_list[0]
            search_params = search_call[1]["params"]
            assert search_params["srsearch"] == "Wikipedia"
            assert len(items) == 1
            mock_session.get.reset_mock()

    @patch("pyforger.acquire.sources.requests.Session")
    def test_random_mode_returns_n_items(self, mock_session_cls: MagicMock) -> None:
        """RANDOM mode returns n items and requests rnlimit=n from the API."""
        pool = [(f"Article {i}", 1000 + i) for i in range(5)]
        random_resp = _make_random_response(pool)
        extracts_resp = _make_extracts_response(
            [
                {
                    "pageid": 1000 + i,
                    "title": f"Article {i}",
                    "extract": f"Content of article {i}.",
                }
                for i in range(5)
            ]
        )

        mock_session = MagicMock()
        responses = [MagicMock(), MagicMock()]
        responses[0].json.return_value = random_resp
        responses[0].raise_for_status = MagicMock()
        responses[1].json.return_value = extracts_resp
        responses[1].raise_for_status = MagicMock()
        mock_session.get.side_effect = responses
        mock_session_cls.return_value = mock_session

        source = WikiRetrievalSource(language="en")
        items = source.fetch(RetrieveMode.RANDOM, n=5)

        assert len(items) == 5
        for item in items:
            assert "text" in item
            assert "id" in item
            assert "title" in item
            assert "summary" in item

        random_call = mock_session.get.call_args_list[0]
        random_params = random_call[1]["params"]
        assert random_params["rnlimit"] == "5"

    @patch("pyforger.acquire.sources.requests.Session")
    def test_query_mode_redirect_resolves_requested_title(
        self, mock_session_cls: MagicMock
    ) -> None:
        """Search title that redirects still resolves to target extract."""
        search_resp = _make_search_response([("Foo", 1)])
        extracts_resp = _make_extracts_response(
            [
                {
                    "pageid": 2,
                    "title": "Bar",
                    "extract": "Resolved body.",
                }
            ],
            redirects=[{"from": "Foo", "to": "Bar"}],
        )

        mock_session = MagicMock()
        responses = [MagicMock(), MagicMock()]
        responses[0].json.return_value = search_resp
        responses[0].raise_for_status = MagicMock()
        responses[1].json.return_value = extracts_resp
        responses[1].raise_for_status = MagicMock()
        mock_session.get.side_effect = responses
        mock_session_cls.return_value = mock_session

        source = WikiRetrievalSource(language="en")
        items = source.fetch(RetrieveMode.QUERY, query="foo", n=1)

        assert len(items) == 1
        assert items[0]["title"] == "Bar"
        assert items[0]["text"] == "Resolved body."
        assert items[0]["id"] == "2"

    @patch("pyforger.acquire.sources.requests.Session")
    def test_fetch_passes_timeout_to_get(self, mock_session_cls: MagicMock) -> None:
        """timeout= in constructor is forwarded to every session.get."""
        search_resp = _make_search_response([("Wikipedia", 5043734)])
        extracts_resp = _make_extracts_response(
            [{"pageid": 5043734, "title": "Wikipedia", "extract": "x"}]
        )

        mock_session = MagicMock()
        responses = [MagicMock(), MagicMock()]
        responses[0].json.return_value = search_resp
        responses[0].raise_for_status = MagicMock()
        responses[1].json.return_value = extracts_resp
        responses[1].raise_for_status = MagicMock()
        mock_session.get.side_effect = responses
        mock_session_cls.return_value = mock_session

        source = WikiRetrievalSource(language="en", timeout=15.0)
        source.fetch(RetrieveMode.QUERY, query="wiki", n=1)

        for call in mock_session.get.call_args_list:
            assert call[1].get("timeout") == 15.0

    @patch("pyforger.acquire.sources.requests.Session")
    def test_fetch_handles_missing_extract(self, mock_session_cls: MagicMock) -> None:
        """Pages with empty or missing extract produce empty text, no crash."""
        search_resp = _make_search_response([("Empty Page", 999)])
        extracts_resp = {
            "query": {
                "pages": {
                    "999": {
                        "pageid": 999,
                        "title": "Empty Page",
                    }
                }
            }
        }

        mock_session = MagicMock()
        responses = [MagicMock(), MagicMock()]
        responses[0].json.return_value = search_resp
        responses[0].raise_for_status = MagicMock()
        responses[1].json.return_value = extracts_resp
        responses[1].raise_for_status = MagicMock()
        mock_session.get.side_effect = responses
        mock_session_cls.return_value = mock_session

        source = WikiRetrievalSource(language="en")
        items = source.fetch(RetrieveMode.QUERY, query="empty", n=1)

        assert len(items) == 1
        assert items[0]["text"] == ""
        assert items[0]["summary"] == ""
        assert items[0]["title"] == "Empty Page"
        assert items[0]["id"] == "999"

    @patch("pyforger.acquire.sources.requests.Session")
    def test_query_mode_caps_srlimit_at_500(
        self, mock_session_cls: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """QUERY mode clamps srlimit to 500 and warns when n exceeds the API max."""
        search_resp = _make_search_response([("Article", 1)])
        extracts_resp = _make_extracts_response(
            [{"pageid": 1, "title": "Article", "extract": "Body."}]
        )

        mock_session = MagicMock()
        responses = [MagicMock(), MagicMock()]
        responses[0].json.return_value = search_resp
        responses[0].raise_for_status = MagicMock()
        responses[1].json.return_value = extracts_resp
        responses[1].raise_for_status = MagicMock()
        mock_session.get.side_effect = responses
        mock_session_cls.return_value = mock_session

        source = WikiRetrievalSource(language="en")
        with caplog.at_level(logging.WARNING, logger="pyforger.acquire.sources"):
            source.fetch(RetrieveMode.QUERY, query="test", n=600)

        search_call = mock_session.get.call_args_list[0]
        search_params = search_call[1]["params"]
        assert search_params["srlimit"] == "500"
        assert any(
            "n=600" in rec.getMessage() and "srlimit" in rec.getMessage()
            for rec in caplog.records
            if rec.levelno == logging.WARNING
        )

    @patch("pyforger.acquire.sources.requests.Session")
    def test_random_mode_caps_rnlimit_at_500(
        self, mock_session_cls: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """RANDOM mode clamps rnlimit to 500 and warns when n exceeds the API max."""
        random_resp = _make_random_response([("Article", 1)])
        extracts_resp = _make_extracts_response(
            [{"pageid": 1, "title": "Article", "extract": "Body."}]
        )

        mock_session = MagicMock()
        responses = [MagicMock(), MagicMock()]
        responses[0].json.return_value = random_resp
        responses[0].raise_for_status = MagicMock()
        responses[1].json.return_value = extracts_resp
        responses[1].raise_for_status = MagicMock()
        mock_session.get.side_effect = responses
        mock_session_cls.return_value = mock_session

        source = WikiRetrievalSource(language="en")
        with caplog.at_level(logging.WARNING, logger="pyforger.acquire.sources"):
            source.fetch(RetrieveMode.RANDOM, n=600)

        random_call = mock_session.get.call_args_list[0]
        random_params = random_call[1]["params"]
        assert random_params["rnlimit"] == "500"
        assert any(
            "n=600" in rec.getMessage() and "rnlimit" in rec.getMessage()
            for rec in caplog.records
            if rec.levelno == logging.WARNING
        )

    @patch("pyforger.acquire.sources.requests.Session")
    def test_fetch_does_not_warn_when_n_within_limit(
        self, mock_session_cls: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning is emitted when n is within the API maximum."""
        search_resp = _make_search_response([("Article", 1)])
        extracts_resp = _make_extracts_response(
            [{"pageid": 1, "title": "Article", "extract": "Body."}]
        )

        mock_session = MagicMock()
        responses = [MagicMock(), MagicMock()]
        responses[0].json.return_value = search_resp
        responses[0].raise_for_status = MagicMock()
        responses[1].json.return_value = extracts_resp
        responses[1].raise_for_status = MagicMock()
        mock_session.get.side_effect = responses
        mock_session_cls.return_value = mock_session

        source = WikiRetrievalSource(language="en")
        with caplog.at_level(logging.WARNING, logger="pyforger.acquire.sources"):
            source.fetch(RetrieveMode.QUERY, query="test", n=10)

        assert not any(rec.levelno == logging.WARNING for rec in caplog.records)

    @staticmethod
    def _session_for(search: dict, extracts: list[dict]) -> MagicMock:
        """A mocked Session whose get() answers the search, then each extracts call."""
        session = MagicMock()
        responses = []
        for body in [search, *extracts]:
            resp = MagicMock()
            resp.json.return_value = body
            responses.append(resp)
        session.get.side_effect = responses
        return session

    @patch("pyforger.acquire.sources.requests.Session")
    def test_intro_is_default_and_batched(self, mock_session_cls: MagicMock) -> None:
        """By default one extracts request covers every title, with exintro set."""
        hits = [(f"T{i}", i) for i in (1, 2, 3)]
        pages = [{"pageid": i, "title": t, "extract": "intro"} for t, i in hits]
        session = self._session_for(
            _make_search_response(hits),
            [_make_extracts_response(pages)],
        )
        mock_session_cls.return_value = session

        items = WikiRetrievalSource().fetch(RetrieveMode.QUERY, query="q", n=3)

        assert [i["title"] for i in items] == ["T1", "T2", "T3"]
        extract_calls = session.get.call_args_list[1:]
        assert len(extract_calls) == 1
        assert extract_calls[0][1]["params"]["exintro"] == "true"

    @pytest.mark.parametrize("where", ["constructor", "fetch"])
    @patch("pyforger.acquire.sources.requests.Session")
    def test_full_text_fetches_one_whole_article_per_request(
        self, mock_session_cls: MagicMock, where: str
    ) -> None:
        """full_text=True drops exintro and sends one extracts request per title.

        Wikipedia returns only one whole-article extract per request.
        """
        hits = [(f"T{i}", i) for i in (1, 2, 3)]
        pages = [{"pageid": i, "title": t, "extract": f"whole {i}"} for t, i in hits]
        session = self._session_for(
            _make_search_response(hits),
            [_make_extracts_response([p]) for p in pages],
        )
        mock_session_cls.return_value = session

        if where == "constructor":
            source = WikiRetrievalSource(full_text=True)
            items = source.fetch(RetrieveMode.QUERY, query="q", n=3)
        else:
            source = WikiRetrievalSource()
            items = source.fetch(RetrieveMode.QUERY, query="q", n=3, full_text=True)

        assert [i["text"] for i in items] == ["whole 1", "whole 2", "whole 3"]
        extract_calls = session.get.call_args_list[1:]
        assert [c[1]["params"]["titles"] for c in extract_calls] == ["T1", "T2", "T3"]
        assert all("exintro" not in c[1]["params"] for c in extract_calls)
        assert all(c[1]["params"]["exlimit"] == "1" for c in extract_calls)

    def test_full_text_changes_source_cache_key(self) -> None:
        """Whole articles and intros never share a cache entry.

        The default key is unchanged: no ``full_text`` in it.
        """
        assert (
            WikiRetrievalSource().cache_key()
            != WikiRetrievalSource(full_text=True).cache_key()
        )
        assert WikiRetrievalSource().cache_key()[-1] == {}

    @pytest.mark.skipif(
        not os.environ.get("PYFORGER_WIKI_LIVE"),
        reason="Set PYFORGER_WIKI_LIVE=1 to run live Wikipedia API test",
    )
    def test_fetch_live_query(self) -> None:
        """Live API test for QUERY mode (skipped unless PYFORGER_WIKI_LIVE=1)."""
        source = WikiRetrievalSource(language="en")
        items = source.fetch(RetrieveMode.QUERY, query="Python programming", n=2)
        assert len(items) == 2
        for item in items:
            assert "text" in item
            assert "id" in item
            assert "title" in item
            assert "summary" in item
            assert item["text"] == item["summary"]
