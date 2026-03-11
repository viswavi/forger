"""Tests for Forger acquire.retrieve and retrieval sources.

MockRetrievalSource is defined here (not in test_helpers/fixtures) per plan.
"""

from typing import Any

from pyforger import DataSpec, NodeType
from pyforger.acquire import (
    RetrieveMode,
    RetrievalSource,
    retrieve,
    retrieve_from_arxiv,
    retrieve_from_hf,
    retrieve_from_s2,
    retrieve_from_web,
    retrieve_from_wiki,
)


class MockRetrievalSource(RetrievalSource):
    """Returns fixed data for testing; ignores mode/query/n/seed/criteria."""

    def __init__(self, data: list[dict[str, Any]]) -> None:
        """Store fixed data to return from fetch().

        Args:
            data: List of dicts (e.g. [{"text": "a"}]); returned as-is by fetch().
        """
        self.data = data

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
        return self.data


class TestRetrieve:
    """retrieve() and wrappers return DataSpec and behave as specified."""

    def test_retrieve_returns_dataspec(self):
        """retrieve with a mock source returns a DataSpec with node_type ACQUIRE."""
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
        """Execute returns the list from source.fetch."""
        source = MockRetrievalSource([{"text": "a"}])
        spec = retrieve(source, mode=RetrieveMode.QUERY)
        result = spec.execute_fn()
        assert result == [{"text": "a"}]

    def test_retrieve_has_no_dependencies(self):
        """retrieve node has no upstream dependencies."""
        source = MockRetrievalSource([])
        spec = retrieve(source)
        assert spec.get_dependencies() == []

    def test_retrieve_from_hf_returns_dataspec(self):
        """retrieve_from_hf returns a DataSpec."""
        spec = retrieve_from_hf("some/dataset", n=5)
        assert isinstance(spec, DataSpec)
        assert spec.node_type == NodeType.ACQUIRE
        assert spec.get_dependencies() == []

    def test_retrieve_from_web_returns_dataspec(self):
        """retrieve_from_web returns a DataSpec."""
        spec = retrieve_from_web(query="test", n=3)
        assert isinstance(spec, DataSpec)
        assert spec.get_dependencies() == []

    def test_retrieve_from_wiki_returns_dataspec(self):
        """retrieve_from_wiki returns a DataSpec."""
        spec = retrieve_from_wiki(query="Python", n=3)
        assert isinstance(spec, DataSpec)
        assert spec.get_dependencies() == []

    def test_retrieve_from_arxiv_returns_dataspec(self):
        """retrieve_from_arxiv returns a DataSpec."""
        spec = retrieve_from_arxiv(query="machine learning", n=5)
        assert isinstance(spec, DataSpec)
        assert spec.get_dependencies() == []

    def test_retrieve_from_s2_returns_dataspec(self):
        """retrieve_from_s2 returns a DataSpec."""
        spec = retrieve_from_s2(query="NLP", n=5)
        assert isinstance(spec, DataSpec)
        assert spec.get_dependencies() == []
