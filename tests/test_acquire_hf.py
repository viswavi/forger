"""Tests for HFRetrievalSource.fetch() with mocked HTTP and local dataset."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from pyforger.acquire import HFRetrievalSource, RetrieveMode


def _make_search_response(rows: list[dict], num_rows_total: int = 100) -> dict:
    """Build a mock /search JSON response body."""
    return {
        "features": [],
        "rows": [
            {"row_idx": i, "row": row, "truncated_cells": []}
            for i, row in enumerate(rows)
        ],
        "num_rows_total": num_rows_total,
        "num_rows_per_page": 100,
        "partial": False,
    }


class TestHFRetrievalSourceFetch:
    """HFRetrievalSource.fetch() with mocked requests.get."""

    @patch("pyforger.acquire.sources.requests.get")
    def test_query_mode_returns_flat_rows(self, mock_get: MagicMock) -> None:
        """QUERY mode returns flat dicts with row_idx and column values."""
        sample_rows = [
            {"question": "What is gravity?", "answer": "A force."},
            {"question": "What is light?", "answer": "EM radiation."},
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_search_response(sample_rows)
        mock_get.return_value = mock_resp

        source = HFRetrievalSource(dataset="test/ds", config="main", split="train")
        items = source.fetch(RetrieveMode.QUERY, query="gravity", n=2)

        assert len(items) == 2
        assert items[0] == {
            "row_idx": 0,
            "question": "What is gravity?",
            "answer": "A force.",
        }
        assert items[1] == {
            "row_idx": 1,
            "question": "What is light?",
            "answer": "EM radiation.",
        }

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["dataset"] == "test/ds"
        assert call_kwargs[1]["params"]["query"] == "gravity"
        assert call_kwargs[1]["params"]["length"] == 2

    @patch("pyforger.acquire.sources.requests.get")
    def test_query_mode_missing_query_raises_value_error(
        self, mock_get: MagicMock
    ) -> None:
        """QUERY mode with None or empty query raises ValueError."""
        source = HFRetrievalSource(dataset="test/ds")

        with pytest.raises(ValueError, match="non-empty 'query'"):
            source.fetch(RetrieveMode.QUERY, query=None, n=5)

        with pytest.raises(ValueError, match="non-empty 'query'"):
            source.fetch(RetrieveMode.QUERY, query="", n=5)

        mock_get.assert_not_called()

    # -- RANDOM mode tests --------------------------------------------------

    @patch("pyforger.acquire.sources.load_dataset")
    def test_random_mode_returns_sampled_rows(self, mock_load: MagicMock) -> None:
        """RANDOM mode returns flat dicts with row_idx and column values."""
        rows = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Carol", "age": 35},
        ]
        mock_ds = MagicMock()
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.select.return_value = mock_ds
        mock_ds.__getitem__ = lambda self, idx: rows[idx]
        mock_load.return_value = mock_ds

        source = HFRetrievalSource(dataset="test/ds")
        items = source.fetch(RetrieveMode.RANDOM, n=2, seed=42)

        assert len(items) == 2
        assert items[0] == {"row_idx": 0, "name": "Alice", "age": 30}
        assert items[1] == {"row_idx": 1, "name": "Bob", "age": 25}

    @patch("pyforger.acquire.sources.load_dataset")
    def test_random_mode_respects_seed(self, mock_load: MagicMock) -> None:
        """RANDOM mode passes seed to dataset.shuffle()."""
        mock_ds = MagicMock()
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.__len__ = lambda self: 5
        mock_ds.select.return_value = mock_ds
        mock_ds.__getitem__ = lambda self, idx: {"val": idx}
        mock_load.return_value = mock_ds

        source = HFRetrievalSource(dataset="test/ds")
        source.fetch(RetrieveMode.RANDOM, n=2, seed=99)

        mock_ds.shuffle.assert_called_once_with(seed=99)

    @patch("pyforger.acquire.sources.load_dataset")
    def test_random_mode_handles_n_larger_than_dataset(
        self, mock_load: MagicMock
    ) -> None:
        """RANDOM mode caps to dataset size when n exceeds available rows."""
        rows = [{"x": 1}, {"x": 2}]
        mock_ds = MagicMock()
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.select.return_value = mock_ds
        mock_ds.__getitem__ = lambda self, idx: rows[idx]
        mock_load.return_value = mock_ds

        source = HFRetrievalSource(dataset="test/ds")
        items = source.fetch(RetrieveMode.RANDOM, n=100, seed=1)

        assert len(items) == 2
        mock_ds.select.assert_called_once_with(range(2))

    # -- STRUCTURED mode tests ---------------------------------------------

    def test_structured_mode_requires_criteria(self) -> None:
        """STRUCTURED mode with None or empty criteria raises ValueError."""
        source = HFRetrievalSource(dataset="test/ds")

        with pytest.raises(ValueError, match="non-empty 'criteria'"):
            source.fetch(RetrieveMode.STRUCTURED, n=5, criteria=None)

        with pytest.raises(ValueError, match="non-empty 'criteria'"):
            source.fetch(RetrieveMode.STRUCTURED, n=5, criteria={})

    @patch("pyforger.acquire.sources.load_dataset")
    def test_structured_mode_filters_exact_match(self, mock_load: MagicMock) -> None:
        """STRUCTURED mode filters rows by exact-match criteria."""
        matching_rows = [
            {"color": "red", "size": 10},
            {"color": "red", "size": 20},
        ]
        mock_ds = MagicMock()
        mock_filtered = MagicMock()
        mock_ds.filter.return_value = mock_filtered
        mock_filtered.__len__ = lambda self: len(matching_rows)
        mock_filtered.select.return_value = mock_filtered
        mock_filtered.__getitem__ = lambda self, idx: matching_rows[idx]
        mock_load.return_value = mock_ds

        source = HFRetrievalSource(dataset="test/ds")
        items = source.fetch(RetrieveMode.STRUCTURED, n=10, criteria={"color": "red"})

        assert len(items) == 2
        assert items[0] == {"row_idx": 0, "color": "red", "size": 10}
        assert items[1] == {"row_idx": 1, "color": "red", "size": 20}
        mock_ds.filter.assert_called_once()

    @patch("pyforger.acquire.sources.load_dataset")
    def test_structured_mode_limits_to_n(self, mock_load: MagicMock) -> None:
        """STRUCTURED mode returns at most n rows from filtered results."""
        filtered_rows = [
            {"val": 1},
            {"val": 2},
            {"val": 3},
        ]
        mock_ds = MagicMock()
        mock_filtered = MagicMock()
        mock_ds.filter.return_value = mock_filtered
        mock_filtered.__len__ = lambda self: len(filtered_rows)
        mock_filtered.select.return_value = mock_filtered
        mock_filtered.__getitem__ = lambda self, idx: filtered_rows[idx]
        mock_load.return_value = mock_ds

        source = HFRetrievalSource(dataset="test/ds")
        items = source.fetch(RetrieveMode.STRUCTURED, n=2, criteria={"val": 1})

        assert len(items) == 2
        mock_filtered.select.assert_called_once_with(range(2))

    # -- Caching test ------------------------------------------------------

    @patch("pyforger.acquire.sources.load_dataset")
    def test_local_dataset_is_loaded_once(self, mock_load: MagicMock) -> None:
        """Repeated RANDOM/STRUCTURED calls reuse the cached dataset."""
        rows = [{"a": 1}]
        mock_ds = MagicMock()
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.select.return_value = mock_ds
        mock_ds.__getitem__ = lambda self, idx: rows[idx]
        mock_load.return_value = mock_ds

        source = HFRetrievalSource(dataset="test/ds")
        source.fetch(RetrieveMode.RANDOM, n=1, seed=1)
        source.fetch(RetrieveMode.RANDOM, n=1, seed=2)

        mock_load.assert_called_once()

    @patch("pyforger.acquire.sources.requests.get")
    def test_api_error_propagation(self, mock_get: MagicMock) -> None:
        """Non-200 API response raises RuntimeError with context."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.json.return_value = {"error": "Not found"}
        mock_get.return_value = mock_resp

        source = HFRetrievalSource(dataset="bad/dataset", config="cfg")
        with pytest.raises(RuntimeError, match="HTTP 404") as exc_info:
            source.fetch(RetrieveMode.QUERY, query="test", n=5)

        msg = str(exc_info.value)
        assert "bad/dataset" in msg
        assert "cfg" in msg

    @patch("pyforger.acquire.sources.requests.get")
    def test_query_mode_passes_config_when_set(self, mock_get: MagicMock) -> None:
        """Config is included in request params when provided."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_search_response([])
        mock_get.return_value = mock_resp

        source = HFRetrievalSource(
            dataset="ibm/duorc", config="SelfRC", split="validation"
        )
        source.fetch(RetrieveMode.QUERY, query="dog", n=3)

        params = mock_get.call_args[1]["params"]
        assert params["config"] == "SelfRC"
        assert params["split"] == "validation"

    @patch("pyforger.acquire.sources.requests.get")
    def test_query_mode_omits_config_when_none(self, mock_get: MagicMock) -> None:
        """Config key is absent from request params when not set."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_search_response([])
        mock_get.return_value = mock_resp

        source = HFRetrievalSource(dataset="some/ds")
        source.fetch(RetrieveMode.QUERY, query="hello", n=1)

        params = mock_get.call_args[1]["params"]
        assert "config" not in params

    @patch("pyforger.acquire.sources.requests.get")
    def test_custom_base_url_and_token(self, mock_get: MagicMock) -> None:
        """Custom base_url and token are used in the request."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_search_response([])
        mock_get.return_value = mock_resp

        source = HFRetrievalSource(
            dataset="test/ds",
            base_url="https://custom.api.example.com/",
            token="hf_abc123",
        )
        source.fetch(RetrieveMode.QUERY, query="test", n=1)

        call_args = mock_get.call_args
        assert call_args[0][0] == "https://custom.api.example.com/search"
        assert call_args[1]["headers"]["Authorization"] == "Bearer hf_abc123"

    @pytest.mark.skipif(
        not os.environ.get("PYFORGER_HF_LIVE"),
        reason="Set PYFORGER_HF_LIVE=1 to run live HF Dataset Viewer test",
    )
    def test_fetch_live_query(self) -> None:
        """Live API test for QUERY mode (skipped unless PYFORGER_HF_LIVE=1)."""
        source = HFRetrievalSource(dataset="ibm/duorc", config="SelfRC")
        items = source.fetch(RetrieveMode.QUERY, query="dog", n=2)
        assert len(items) == 2
        for item in items:
            assert "row_idx" in item
            assert "question" in item
            assert "plot" in item
