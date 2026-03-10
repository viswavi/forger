"""Reusable test fixtures for Forger tests."""

from pyforger.core import DataSpec, NodeType


class CallCounter:
    """Tracks how many times an execute function is called."""

    def __init__(self):
        """Initialize call count to zero."""
        self.count = 0

    def reset(self):
        """Reset call count to zero."""
        self.count = 0


def make_source(data: list, counter: CallCounter = None) -> DataSpec:
    """Source node returning fixed data."""

    def _execute():
        if counter is not None:
            counter.count += 1
        return data

    return DataSpec(NodeType.ACQUIRE, [], _execute, "test_source")


def make_double_length(source: DataSpec) -> DataSpec:
    """Alter node that doubles its input list."""

    def _execute(data):
        return data + data

    return DataSpec(NodeType.ALTER, source, _execute, "test_double")


def make_concat(*sources) -> DataSpec:
    """Alter node that concatenates multiple inputs."""

    def _execute(*input_data):
        result = []
        for data in input_data:
            result.extend(data)
        return result

    return DataSpec(NodeType.ALTER, list(sources), _execute, "test_concat")


def make_empty_filter(source: DataSpec) -> DataSpec:
    """Filter node that always returns empty."""

    def _execute(data):
        return []

    return DataSpec(NodeType.FILTER, [source], _execute, "test_empty_filter")
