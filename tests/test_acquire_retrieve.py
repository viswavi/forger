"""Tests for Forger acquire.retrieve and retrieval sources."""
from __future__ import annotations

import copy
from typing import Any

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
    """Retrieve() and wrappers return Datretrieve functionsaSpec and behave as specified."""

    def test_retrieve_returns_list(self):
        """Retrieve with a mock source returns retrieval results."""
        source = MockRetrievalSource([{"text": "a"}, {"text": "b"}])
        spec = retrieve(source, mode=RetrieveMode.QUERY, queries=["x"])
        assert isinstance(spec, list)
        assert len(spec) == 1
        assert isinstance(spec[0], dict)

    def test_retrieve_no_execution_on_build(self):
        """Building retrieve(...) does not call source.fetch."""
        fetch_called = False

        class TrackingSource(MockRetrievalSource):
            def fetch(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
                nonlocal fetch_called
                fetch_called = True
                return super().fetch(*args, **kwargs)

        source = TrackingSource([{"text": "a"}])
        retrieve(source, mode=RetrieveMode.RANDOM, n=5)
        assert not fetch_called

    def test_retrieve_execute_returns_source_result(self):
        """retrieve() returns the list from source.fetch with query stamped."""
        source = MockRetrievalSource([{"text": "a"}])
        results = retrieve(source, mode=RetrieveMode.QUERY)
        assert results == [{"text": "a", "query": None}]

    def test_retrieve_from_hf_returns_list(self):
        """Return a retrieve function from retrieve_from_web."""
        results = retrieve_from_hf("some/dataset", n=5)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)

    def test_retrieve_from_wiki_returns_list(self):
        """Return search results from retrieve_from_wiki."""
        results = retrieve_from_wiki(queries=["Python"], n=3)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_retrieve_from_arxiv_returns_list(self):
        """Return search results from retrieve_from_arxiv."""
        results = retrieve_from_arxiv(queries=["machine learning"], n=5)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)

    def test_retrieve_from_s2_returns_list(self):
        """Return search results from retrieve_from_s2."""
        results = retrieve_from_s2(queries=["NLP"], n=5)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)


class TestMultiQueryRetrieve:
    """Multiquery retrieval returns flat rows with a 'query' field."""

    def test_multi_query_flat_output(self):
        """Multiple queries produce a flat list with correct query tags."""
        source = MockRetrievalSource([{"text": "x"}, {"text": "y"}])
        spec = retrieve(source, mode=RetrieveMode.QUERY, queries=["a", "b"], n=10)
        result = spec
        assert len(result) == 4
        assert all(r["query"] == "a" for r in result[:2])
        assert all(r["query"] == "b" for r in result[2:])
        assert [r["text"] for r in result] == ["x", "y", "x", "y"]

    def test_single_query_stamps_query_field(self):
        """A singleelement queries list stamps 'query' on every row."""
        source = MockRetrievalSource([{"text": "t"}])
        spec = retrieve(source, mode=RetrieveMode.QUERY, queries=["hello"])
        result = spec
        assert len(result) == 1
        assert result[0]["query"] == "hello"

    def test_no_query_stamps_none(self):
        """When queries is not provided, query field is None."""
        source = MockRetrievalSource([{"text": "t"}])
        spec = retrieve(source, mode=RetrieveMode.RANDOM)
        result = spec
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
        spec = retrieve(source, queries=["q1", "q2", "q3"])
        spec
        assert call_count == 3

    def test_multi_query_preserves_original_fields(self):
        """Source fields (text, id, etc.) are preserved alongside query."""
        source = MockRetrievalSource([{"text": "hello", "id": "42"}])
        spec = retrieve(source, queries=["a"])
        result = spec
        assert result == [{"text": "hello", "id": "42", "query": "a"}]
