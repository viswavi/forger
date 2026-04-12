"""Tests for Forger acquire.retrieve and retrieval sources."""
from __future__ import annotations

import copy
from typing import Any

import pytest

from pyforger import DataSpec, NodeType
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
            data: List of dicts (e.g. [{"text": "a"}]); deep-copied by fetch().
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
    """Retrieve() and wrappers return DataSpec and behave as specified."""

    def test_retrieve_returns_dataspec(self):
        """Retrieve with a mock source returns a DataSpec with node_type ACQUIRE."""
        source = MockRetrievalSource([{"text": "a"}, {"text": "b"}])
        spec = retrieve(source, mode=RetrieveMode.QUERY, query="x")
        assert isinstance(spec, DataSpec)
        assert spec.node_type == NodeType.ACQUIRE

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
        """Execute returns the list from source.fetch with query stamped."""
        source = MockRetrievalSource([{"text": "a"}])
        spec = retrieve(source, mode=RetrieveMode.QUERY)
        result = spec.execute_fn()
        assert result == [{"text": "a", "query": None}]

    def test_retrieve_has_no_dependencies(self):
        """Retrieve node has no upstream dependencies."""
        source = MockRetrievalSource([])
        spec = retrieve(source)
        assert spec.get_dependencies() == []

    def test_retrieve_from_hf_returns_dataspec(self):
        """Return a DataSpec from retrieve_from_hf."""
        spec = retrieve_from_hf("some/dataset", n=5)
        assert isinstance(spec, DataSpec)
        assert spec.node_type == NodeType.ACQUIRE
        assert spec.get_dependencies() == []

    def test_retrieve_from_web_returns_dataspec(self):
        """Return a DataSpec from retrieve_from_web."""
        spec = retrieve_from_web(query="test", n=3)
        assert isinstance(spec, DataSpec)
        assert spec.get_dependencies() == []

    def test_retrieve_from_wiki_returns_dataspec(self):
        """Return a DataSpec from retrieve_from_wiki."""
        spec = retrieve_from_wiki(query="Python", n=3)
        assert isinstance(spec, DataSpec)
        assert spec.get_dependencies() == []

    def test_retrieve_from_arxiv_returns_dataspec(self):
        """Return a DataSpec from retrieve_from_arxiv."""
        spec = retrieve_from_arxiv(query="machine learning", n=5)
        assert isinstance(spec, DataSpec)
        assert spec.get_dependencies() == []

    def test_retrieve_from_s2_returns_dataspec(self):
        """Return a DataSpec from retrieve_from_s2."""
        spec = retrieve_from_s2(query="NLP", n=5)
        assert isinstance(spec, DataSpec)
        assert spec.get_dependencies() == []


class TestMultiQueryRetrieve:
    """Multi-query retrieval returns flat rows with a 'query' field."""

    def test_multi_query_flat_output(self):
        """Multiple queries produce a flat list with correct query tags."""
        source = MockRetrievalSource([{"text": "x"}, {"text": "y"}])
        spec = retrieve(
            source, mode=RetrieveMode.QUERY, queries=["a", "b"], n=10
        )
        result = spec.execute_fn()
        assert len(result) == 4
        assert all(r["query"] == "a" for r in result[:2])
        assert all(r["query"] == "b" for r in result[2:])
        assert [r["text"] for r in result] == ["x", "y", "x", "y"]

    def test_single_query_stamps_query_field(self):
        """A single query string still stamps 'query' on every row."""
        source = MockRetrievalSource([{"text": "t"}])
        spec = retrieve(source, mode=RetrieveMode.QUERY, query="hello")
        result = spec.execute_fn()
        assert len(result) == 1
        assert result[0]["query"] == "hello"

    def test_no_query_stamps_none(self):
        """When neither query nor queries is provided, query is None."""
        source = MockRetrievalSource([{"text": "t"}])
        spec = retrieve(source, mode=RetrieveMode.RANDOM)
        result = spec.execute_fn()
        assert result[0]["query"] is None

    def test_query_and_queries_mutually_exclusive(self):
        """Providing both query and queries raises ValueError."""
        source = MockRetrievalSource([])
        with pytest.raises(ValueError, match="not both"):
            retrieve(source, query="a", queries=["b"])

    def test_multi_query_no_execution_on_build(self):
        """Building with queries does not call source.fetch."""
        fetch_called = False

        class TrackingSource(MockRetrievalSource):
            def fetch(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
                nonlocal fetch_called
                fetch_called = True
                return super().fetch(*args, **kwargs)

        source = TrackingSource([{"text": "a"}])
        retrieve(source, queries=["q1", "q2"])
        assert not fetch_called

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
        spec.execute_fn()
        assert call_count == 3

    def test_multi_query_preserves_original_fields(self):
        """Source fields (text, id, etc.) are preserved alongside query."""
        source = MockRetrievalSource([{"text": "hello", "id": "42"}])
        spec = retrieve(source, queries=["a"])
        result = spec.execute_fn()
        assert result == [{"text": "hello", "id": "42", "query": "a"}]

    def test_wrapper_forwards_queries(self):
        """Convenience wrappers forward queries to retrieve()."""
        spec = retrieve_from_web(queries=["q1", "q2"], n=3)
        assert isinstance(spec, DataSpec)
        assert spec.node_type == NodeType.ACQUIRE
