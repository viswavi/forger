"""Reusable test fixtures for Forger tests."""

from pyforger.core import DataSpec, NodeType


def make_double_repetition_data(source: DataSpec) -> DataSpec:
    """Alter node that repeats its input list twice."""

    def _execute(data):
        return data + data

    return DataSpec(NodeType.ALTER, source, _execute, "test_duplicate")


def make_empty_filter(source: DataSpec) -> DataSpec:
    """Filter node that always returns empty."""

    def _execute(_data):
        # _data is unused -- this is just here to match the expected signature.
        return []

    return DataSpec(NodeType.FILTER, [source], _execute, "test_empty_filter")


def make_concat(*sources: DataSpec) -> DataSpec:
    """Alter node that concatenates multiple inputs."""

    def _execute(*input_data):
        result = []
        for data in input_data:
            result.extend(data)
        return result

    return DataSpec(NodeType.ALTER, list(sources), _execute, "test_concat")
