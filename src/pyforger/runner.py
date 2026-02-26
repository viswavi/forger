"""Execution engine for Forger computation graphs.

This module implements the run() function that executes a DataSpec DAG
in topological order with caching and early termination support.
"""

from __future__ import annotations

from typing import Any

from core import DataSpec
from utils import get_formatted_logger

logger = get_formatted_logger("Runner")


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the computation graph."""

    pass


class FilterTerminationError(Exception):
    """Raised when a filter operation returns empty results.

    Allows early termination of the pipeline when a filter removes
    all data.
    """

    def __init__(self, operation_name: str, message: str = ""):
        """Create a FilterTerminationError."""
        self.operation_name = operation_name
        super().__init__(message or f"Filter '{operation_name}' returned empty results")


def run(dataset: DataSpec, cache: dict[int, Any] | None = None) -> Any:
    """Execute a dataset creation pipeline.

    Run the data transformation graph in topological order from root
    to leaf and returns the final result.

    Args:
        dataset: The final DataSpec node to execute
        cache: Optional shared cache dict (for testing/advanced usage).
               Keys are DataSpec object ids, values are computed results.

    Returns:
        The computed data (typically list[dict])

    Raises:
        CycleDetectedError: If the graph contains cycles
        NotImplementedError: If an operation stub is executed

    Example:
        >>> retrieved = acquire.HuggingfaceRetrieve(query="math", n=100)
        >>> filtered = filter.Diversity(retrieved(), threshold=0.8)
        >>> result = run(filtered)
    """
    raise NotImplementedError
