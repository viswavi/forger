"""Tests for HFRetrievalSource.fetch() with mocked HfApi and load_dataset."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from pyforger.acquire import HFRetrievalSource, RetrieveMode


def _make_mock_dataset(rows: list[dict]) -> MagicMock:
    """Build a mock HF Dataset object supporting shuffle/select/len/getitem."""
    mock_ds = MagicMock()
    mock_ds.shuffle.return_value = mock_ds
    mock_ds.__len__ = lambda self: len(rows)
    mock_ds.select.return_value = mock_ds
    mock_ds.__getitem__ = lambda self, idx: rows[idx]
    return mock_ds


def _make_mock_dataset_info(dataset_id: str) -> MagicMock:
    """Build a mock DatasetInfo object returned by HfApi.list_datasets."""
    info = MagicMock()
    info.id = dataset_id
    return info


class TestHFRetrievalSourceQueryMode:
    """QUERY mode tests for HFRetrievalSource."""

    @patch("pyforger.acquire.sources.HFRetrievalSource._load_and_select")
    @patch("pyforger.acquire.sources.HFRetrievalSource._discover")
    def test_query_discovers_and_loads_top_dataset(
        self, mock_discover: MagicMock, mock_load: MagicMock
    ) -> None:
        """QUERY mode discovers datasets via HfApi, loads the top match."""
        mock_discover.return_value = [_make_mock_dataset_info("found/dataset")]
        mock_load.return_value = [
            {"row_idx": 0, "dataset_id": "found/dataset", "text": "hello"}
        ]

        source = HFRetrievalSource()
        items = source.fetch(RetrieveMode.QUERY, query="sentiment", n=1)

        assert len(items) == 1
        assert items[0]["dataset_id"] == "found/dataset"
        mock_discover.assert_called_once_with("sentiment", limit=1)
        mock_load.assert_called_once_with("found/dataset", 1)

    @patch("pyforger.acquire.sources.HFRetrievalSource._load_and_select")
    @patch("pyforger.acquire.sources.HFRetrievalSource._discover")
    def test_query_with_explicit_dataset_skips_discovery(
        self, mock_discover: MagicMock, mock_load: MagicMock
    ) -> None:
        """When dataset is provided, discovery is skipped."""
        mock_load.return_value = [{"row_idx": 0, "dataset_id": "known/ds", "val": 42}]

        source = HFRetrievalSource(dataset="known/ds")
        items = source.fetch(RetrieveMode.QUERY, query="anything", n=1)

        assert len(items) == 1
        assert items[0]["dataset_id"] == "known/ds"
        mock_discover.assert_not_called()
        mock_load.assert_called_once_with("known/ds", 1)

    def test_query_empty_query_without_dataset_raises(self) -> None:
        """Empty query with no dataset raises ValueError."""
        source = HFRetrievalSource()

        with pytest.raises(ValueError, match="non-empty 'query'"):
            source.fetch(RetrieveMode.QUERY, query=None, n=5)

        with pytest.raises(ValueError, match="non-empty 'query'"):
            source.fetch(RetrieveMode.QUERY, query="", n=5)

    @patch("pyforger.acquire.sources.HFRetrievalSource._load_and_select")
    def test_query_with_explicit_dataset_and_no_query(
        self, mock_load: MagicMock
    ) -> None:
        """Explicit dataset works even without a query string."""
        mock_load.return_value = [{"row_idx": 0, "dataset_id": "ds", "x": 1}]

        source = HFRetrievalSource(dataset="ds")
        items = source.fetch(RetrieveMode.QUERY, n=1)

        assert len(items) == 1
        mock_load.assert_called_once_with("ds", 1)

    @patch("pyforger.acquire.sources.HFRetrievalSource._discover")
    def test_query_discovery_returns_empty(self, mock_discover: MagicMock) -> None:
        """When discovery returns no results, fetch returns empty list."""
        mock_discover.return_value = []

        source = HFRetrievalSource()
        items = source.fetch(RetrieveMode.QUERY, query="nonexistent", n=5)

        assert items == []


class TestHFRetrievalSourceRandomMode:
    """RANDOM mode tests for HFRetrievalSource."""

    @patch("pyforger.acquire.sources.HFRetrievalSource._load_and_select")
    @patch("pyforger.acquire.sources.HFRetrievalSource._discover")
    def test_random_shuffles_with_seed(
        self, mock_discover: MagicMock, mock_load: MagicMock
    ) -> None:
        """RANDOM mode passes shuffle=True and seed to _load_and_select."""
        mock_discover.return_value = [_make_mock_dataset_info("found/ds")]
        mock_load.return_value = [{"row_idx": 0, "dataset_id": "found/ds", "v": 1}]

        source = HFRetrievalSource()
        items = source.fetch(RetrieveMode.RANDOM, query="test", n=3, seed=42)

        assert len(items) == 1
        mock_load.assert_called_once_with("found/ds", 3, seed=42, shuffle=True)

    @patch("pyforger.acquire.sources.HFRetrievalSource._load_and_select")
    def test_random_with_explicit_dataset(self, mock_load: MagicMock) -> None:
        """RANDOM with explicit dataset skips discovery and shuffles."""
        mock_load.return_value = [{"row_idx": 0, "dataset_id": "my/ds", "a": 1}]

        source = HFRetrievalSource(dataset="my/ds")
        source.fetch(RetrieveMode.RANDOM, n=2, seed=99)

        mock_load.assert_called_once_with("my/ds", 2, seed=99, shuffle=True)


class TestHFRetrievalSourceLoadAndSelect:
    """Integration-level tests for _load_and_select with mocked load_dataset."""

    @patch("datasets.load_dataset")
    def test_returns_flat_dicts_with_row_idx_and_dataset_id(
        self, mock_load: MagicMock
    ) -> None:
        """Rows include row_idx, dataset_id, and raw column values."""
        rows = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        mock_load.return_value = _make_mock_dataset(rows)

        source = HFRetrievalSource()
        items = source._load_and_select("test/ds", n=2)

        assert len(items) == 2
        assert items[0] == {
            "row_idx": 0,
            "dataset_id": "test/ds",
            "name": "Alice",
            "age": 30,
        }
        assert items[1] == {
            "row_idx": 1,
            "dataset_id": "test/ds",
            "name": "Bob",
            "age": 25,
        }

    @patch("datasets.load_dataset")
    def test_caps_to_dataset_size(self, mock_load: MagicMock) -> None:
        """When n exceeds dataset size, returns all available rows."""
        rows = [{"x": 1}]
        mock_load.return_value = _make_mock_dataset(rows)

        source = HFRetrievalSource()
        items = source._load_and_select("test/ds", n=100)

        assert len(items) == 1

    @patch("datasets.load_dataset")
    def test_shuffle_calls_dataset_shuffle(self, mock_load: MagicMock) -> None:
        """When shuffle=True, calls ds.shuffle(seed=seed)."""
        rows = [{"v": 1}, {"v": 2}]
        mock_ds = _make_mock_dataset(rows)
        mock_load.return_value = mock_ds

        source = HFRetrievalSource()
        source._load_and_select("test/ds", n=2, seed=42, shuffle=True)

        mock_ds.shuffle.assert_called_once_with(seed=42)

    @patch("datasets.load_dataset")
    def test_no_shuffle_by_default(self, mock_load: MagicMock) -> None:
        """By default, shuffle is not called."""
        rows = [{"v": 1}]
        mock_ds = _make_mock_dataset(rows)
        mock_load.return_value = mock_ds

        source = HFRetrievalSource()
        source._load_and_select("test/ds", n=1)

        mock_ds.shuffle.assert_not_called()

    @patch("datasets.load_dataset")
    def test_token_forwarded_to_load_dataset(self, mock_load: MagicMock) -> None:
        """Token is passed through to load_dataset."""
        mock_load.return_value = _make_mock_dataset([{"a": 1}])

        source = HFRetrievalSource(token="hf_secret")
        source._load_and_select("test/ds", n=1)

        call_args = mock_load.call_args
        assert call_args[1]["token"] == "hf_secret"

    @patch("datasets.load_dataset")
    def test_config_and_split_forwarded(self, mock_load: MagicMock) -> None:
        """Config and split are passed through to load_dataset."""
        mock_load.return_value = _make_mock_dataset([{"a": 1}])

        source = HFRetrievalSource(config="SelfRC", split="validation")
        source._load_and_select("ibm/duorc", n=1)

        call_args = mock_load.call_args
        assert call_args[0] == ("ibm/duorc", "SelfRC")
        assert call_args[1]["split"] == "validation"


class TestHFRetrievalSourceDiscover:
    """Tests for the _discover method with mocked HfApi."""

    @patch("huggingface_hub.HfApi")
    def test_discover_calls_list_datasets(self, mock_api_cls: MagicMock) -> None:
        """_discover calls HfApi.list_datasets with search and limit."""
        mock_api = MagicMock()
        mock_info = _make_mock_dataset_info("found/ds")
        mock_api.list_datasets.return_value = iter([mock_info])
        mock_api_cls.return_value = mock_api

        source = HFRetrievalSource(token="hf_tok")
        results = source._discover("sentiment", limit=3)

        assert len(results) == 1
        assert results[0].id == "found/ds"
        mock_api_cls.assert_called_once_with(token="hf_tok")
        mock_api.list_datasets.assert_called_once_with(search="sentiment", limit=3)

    @patch("huggingface_hub.HfApi")
    def test_discover_returns_empty_on_no_match(self, mock_api_cls: MagicMock) -> None:
        """_discover returns empty list when Hub has no matches."""
        mock_api = MagicMock()
        mock_api.list_datasets.return_value = iter([])
        mock_api_cls.return_value = mock_api

        source = HFRetrievalSource()
        results = source._discover("zzz_nonexistent_zzz")

        assert results == []


class TestHFRetrievalSourceEdgeCases:
    """Edge case and unsupported mode tests."""

    def test_unsupported_mode_raises(self) -> None:
        """An unrecognized RetrieveMode raises ValueError."""
        source = HFRetrievalSource(dataset="test/ds")
        fake_mode = MagicMock()
        fake_mode.name = "FAKE"

        with pytest.raises(ValueError, match="Unsupported"):
            source.fetch(fake_mode, query="test", n=1)


@pytest.mark.skipif(
    not os.environ.get("PYFORGER_HF_LIVE"),
    reason="Set PYFORGER_HF_LIVE=1 to run live HF Hub test",
)
class TestHFRetrievalSourceLive:
    """Live API tests (skipped unless PYFORGER_HF_LIVE=1)."""

    def test_discover_and_load(self) -> None:
        """Live test: discover a dataset and load rows."""
        source = HFRetrievalSource()
        items = source.fetch(RetrieveMode.QUERY, query="sentiment analysis", n=2)
        assert len(items) <= 2
        if items:
            assert "row_idx" in items[0]
            assert "dataset_id" in items[0]
