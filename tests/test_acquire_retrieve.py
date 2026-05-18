"""Tests for Forger acquire.retrieve and retrieval sources."""
from __future__ import annotations

import copy
from typing import Any
from unittest.mock import patch

from pyforger.acquire import (
    RetrievalSource,
    RetrieveMode,
    retrieve,
    retrieve_from_arxiv,
    retrieve_from_hf,
    retrieve_from_s2,
    retrieve_from_web,
    retrieve_from_wiki,
)


class MockRetrievalSource(RetrievalSource):
    """Returns a deep copy of fixed data for testing; ignores mode/query/n/seed."""

    def __init__(self, data: list[dict[str, Any]]) -> None:
        """Store fixed data to return from fetch().

        Args:
            data: List of dicts (e.g. [{"text": "a"}]); deepcopied by fetch().
        """
        self.data = data

    def fetch(
        self,
        mode: RetrieveMode,
        *,
        query: str | None = None,
        n: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return a deep copy of the fixed data list."""
        return copy.deepcopy(self.data)


class TestRetrieve:
    """Retrieve() and wrappers return list with specified length and format."""

    def test_retrieve_returns_list(self):
        """Retrieve with a mock source returns retrieval results."""
        source = MockRetrievalSource([{"text": "a"}, {"text": "b"}])
        result = retrieve(source, mode=RetrieveMode.QUERY, queries=["x"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], dict)

    def test_retrieve_execute_returns_source_result(self):
        """retrieve() returns the list from source.fetch with query stamped."""
        source = MockRetrievalSource([{"text": "a"}])
        results = retrieve(source, mode=RetrieveMode.QUERY)
        assert results == [{"text": "a", "query": None}]

    @patch("pyforger.acquire.sources.HFRetrievalSource.fetch")
    def test_retrieve_from_hf_returns_list(self, mock_fetch):
        """Return a list of datasets from retrieve_from_hf."""
        mock_fetch.return_value = [
            {"row_idx": i, "dataset_id": "some/dataset", "text": f"row{i}"}
            for i in range(5)
        ]
        results = retrieve_from_hf("some/dataset", n=5)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)

    @patch("pyforger.acquire.sources.WebRetrievalSource.fetch")
    def test_retrieve_from_web_returns_list(self, mock_fetch):
        """Return search results from retrieve_from_web."""
        mock_fetch.return_value = [
            {
                "text": f"snippet {i}",
                "id": f"http://source/{i}",
                "title": f"t{i}",
                "url": f"http://source/{i}",
            }
            for i in range(5)
        ]
        results = retrieve_from_web(queries=["test"], n=5)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)

    @patch("pyforger.acquire.sources.WikiRetrievalSource.fetch")
    def test_retrieve_from_wiki_returns_list(self, mock_fetch):
        """Return search results from retrieve_from_wiki."""
        mock_fetch.return_value = [
            {
                "title": f"Python page {i}",
                "url": f"http://wiki/{i}",
                "snippet": f"snippet {i}",
            }
            for i in range(3)
        ]
        results = retrieve_from_wiki(queries=["Python"], n=3)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    @patch("pyforger.acquire.sources.ArxivRetrievalSource.fetch")
    def test_retrieve_from_arxiv_returns_list(self, mock_fetch):
        """Return search results from retrieve_from_arxiv."""
        mock_fetch.return_value = [
            {"title": f"paper {i}", "id": f"arxiv:{i}", "summary": f"summary {i}"}
            for i in range(5)
        ]
        results = retrieve_from_arxiv(queries=["machine learning"], n=5)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)

    @patch("pyforger.acquire.sources.S2RetrievalSource.fetch")
    def test_retrieve_from_s2_returns_list(self, mock_fetch):
        """Return search results from retrieve_from_s2."""
        mock_fetch.return_value = [
            {
                "text": f"abstract {i}",
                "id": f"s2:{i}",
                "title": f"paper {i}",
                "summary": f"summary {i}",
            }
            for i in range(5)
        ]
        results = retrieve_from_s2(queries=["NLP"], n=5)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)


class TestMultiQueryRetrieve:
    """Multi-query retrieval returns flat rows with a 'query' field."""

    def test_multi_query_flat_output(self):
        """Multiple queries produce a flat list with correct query tags."""
        source = MockRetrievalSource([{"text": "x"}, {"text": "y"}])
        result = retrieve(source, mode=RetrieveMode.QUERY, queries=["a", "b"], n=10)
        assert len(result) == 4
        assert all(r["query"] == "a" for r in result[:2])
        assert all(r["query"] == "b" for r in result[2:])
        assert [r["text"] for r in result] == ["x", "y", "x", "y"]

    def test_single_query_stamps_query_field(self):
        """A single-element list of queries stamps 'query' on every row."""
        source = MockRetrievalSource([{"text": "t"}])
        result = retrieve(source, mode=RetrieveMode.QUERY, queries=["hello"])
        assert len(result) == 1
        assert result[0]["query"] == "hello"

    def test_no_query_stamps_none(self):
        """When queries is not provided, query field is None."""
        source = MockRetrievalSource([{"text": "t"}])
        result = retrieve(source, mode=RetrieveMode.RANDOM)
        assert result[0]["query"] is None

    def test_fetch_called_once_per_query(self):
        """source.fetch is called exactly len(queries) times."""
        call_count = 0

        class CountingSource(MockRetrievalSource):
            def fetch(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
                nonlocal call_count
                call_count += 1
                return super().fetch(*args, **kwargs)

        source = CountingSource([{"text": "a"}])
        # We don't care about the retrieval result here here; just checking that
        # fetch() is called the right number of times based on the number of queries.
        retrieve(source, queries=["q1", "q2", "q3"])
        assert call_count == 3

    def test_multi_query_preserves_original_fields(self):
        """Source fields (text, id, etc.) are preserved alongside query."""
        source = MockRetrievalSource([{"text": "hello", "id": "42"}])
        result = retrieve(source, queries=["a"])
        assert result == [{"text": "hello", "id": "42", "query": "a"}]
