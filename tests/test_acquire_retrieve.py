"""Tests for Forger acquire.retrieve and retrieval sources."""
from __future__ import annotations

import copy
from typing import Any
from unittest.mock import patch

import pytest

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
from pyforger.acquire.sources import (
    ArxivRetrievalSource,
    HFRetrievalSource,
    S2RetrievalSource,
    WebRetrievalSource,
    WikiRetrievalSource,
)
from pyforger.op import trace


class MockRetrievalSource(RetrievalSource):
    """Returns a deep copy of fixed data for testing; ignores mode/query/n/seed."""

    def __init__(self, data: list[dict[str, Any]]) -> None:
        """Store fixed data to return from fetch().

        Args:
            data: List of dicts (e.g. [{"text": "a"}]); deep-copied by fetch().
        """
        self.data = data

    def cache_key(self):
        """Identify fixed test data without serializing local source classes."""
        return (type(self).__qualname__, self.data)

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
        # We don't care about the retrieval result here; just checking that
        # fetch() is called the right number of times based on the number of queries.
        retrieve(source, queries=["q1", "q2", "q3"])
        assert call_count == 3

    def test_multi_query_preserves_original_fields(self):
        """Source fields (text, id, etc.) are preserved alongside query."""
        source = MockRetrievalSource([{"text": "hello", "id": "42"}])
        result = retrieve(source, queries=["a"])
        assert result == [{"text": "hello", "id": "42", "query": "a"}]

    def test_bare_query_string_is_one_query(self):
        """A positional string is a single query, not a list of characters."""
        source = MockRetrievalSource([{"text": "hello"}])
        assert retrieve(source, "dp tricks") == [
            {"text": "hello", "query": "dp tricks"}
        ]

    @patch("pyforger.acquire.sources.WebRetrievalSource.fetch")
    def test_wrapper_takes_query_first(self, mock_fetch):
        """retrieve_from_web("...") searches for that string in QUERY mode."""
        mock_fetch.return_value = [{"text": "t", "url": "u"}]
        assert retrieve_from_web("segment tree", n=3) == [
            {"text": "t", "url": "u", "query": "segment tree"}
        ]
        mock_fetch.assert_called_once_with(
            RetrieveMode.QUERY, query="segment tree", n=3, seed=None
        )

    def test_mode_is_keyword_only(self):
        """Passing a mode positionally, or a string as mode, raises TypeError."""
        source = MockRetrievalSource([{"text": "a"}])
        with pytest.raises(TypeError, match="keyword-only"):
            retrieve_from_web(RetrieveMode.QUERY)
        with pytest.raises(TypeError, match="pass search strings as queries"):
            retrieve(source, mode="segment tree")


@pytest.mark.parametrize(
    "entry,source_type,args",
    [
        (retrieve, MockRetrievalSource, (MockRetrievalSource([{"text": "x"}]),)),
        (retrieve_from_web, WebRetrievalSource, ()),
        (retrieve_from_wiki, WikiRetrievalSource, ()),
        (retrieve_from_hf, HFRetrievalSource, ("some/dataset",)),
        (retrieve_from_arxiv, ArxivRetrievalSource, ()),
        (retrieve_from_s2, S2RetrievalSource, ()),
    ],
)
def test_retrieval_cache_controls(entry, source_type, args, tmp_path):
    """Each public entry caches once, bypasses fully, and supports cache-only."""
    with patch.object(source_type, "fetch", return_value=[{"text": "cached"}]) as fetch:
        with pytest.raises(FileNotFoundError):
            entry(*args, queries=["x"], _cache_only=True)
        fetch.assert_not_called()
        first = entry(*args, queries=["x"])
        assert entry(*args, queries=["x"]) == first
        assert entry(*args, queries=["x"], _cache_only=True) == first
        assert fetch.call_count == 1
        assert [r.cache_hit for r in trace()] == [False, True, True]
        assert len(list(tmp_path.rglob("*.pkl"))) == 1

        fetch.return_value = [{"text": "fresh"}]
        assert entry(*args, queries=["x"], _no_cache=True)[0]["text"] == "fresh"
        assert entry(*args, queries=["x"], _cache_only=True) == first
        entry(*args, queries=["uncached"], _no_cache=True)
        with pytest.raises(FileNotFoundError):
            entry(*args, queries=["uncached"], _cache_only=True)
        assert len(list(tmp_path.rglob("*.pkl"))) == 1
        assert all(
            "_no_cache" not in c.kwargs and "_cache_only" not in c.kwargs
            for c in fetch.call_args_list
        )
        entry(*args, queries=["changed"])
        assert fetch.call_count == 4


@pytest.mark.parametrize(
    "source,changed,runtime_attr",
    [
        (
            WebRetrievalSource(base_url="a"),
            WebRetrievalSource(base_url="b"),
            "_session",
        ),
        (
            WikiRetrievalSource(language="en"),
            WikiRetrievalSource(language="de"),
            "_session",
        ),
        (HFRetrievalSource("a"), HFRetrievalSource("b"), None),
        (
            ArxivRetrievalSource(page_size=10),
            ArxivRetrievalSource(page_size=20),
            "_client",
        ),
        (S2RetrievalSource(timeout=10), S2RetrievalSource(timeout=20), "_client"),
    ],
)
def test_source_cache_ignores_runtime_state(source, changed, runtime_attr):
    """Source configuration determines reuse even after a lazy client is created."""
    same = copy.deepcopy(source)

    def fetch(instance, *args, **kwargs):
        if runtime_attr:
            setattr(instance, runtime_attr, object())
        return [{"text": "x"}]

    with patch.object(
        type(source), "fetch", autospec=True, side_effect=fetch
    ) as mocked:
        first = retrieve(source, queries=["q"])
        assert retrieve(source, queries=["q"]) == first
        assert retrieve(same, queries=["q"]) == first
        assert mocked.call_count == 1
        retrieve(changed, queries=["q"])
        assert mocked.call_count == 2


def test_random_retrieval_caches_without_seed():
    """Random sampling is cached too; callers opt out for fresh samples."""
    with patch.object(
        WikiRetrievalSource, "fetch", return_value=[{"text": "sample"}]
    ) as fetch:
        first = retrieve_from_wiki(mode=RetrieveMode.RANDOM)
        assert retrieve_from_wiki(mode=RetrieveMode.RANDOM) == first
        assert fetch.call_count == 1
        retrieve_from_wiki(mode=RetrieveMode.RANDOM, _no_cache=True)
        retrieve_from_wiki(mode=RetrieveMode.RANDOM, seed=42)
        assert fetch.call_count == 3
