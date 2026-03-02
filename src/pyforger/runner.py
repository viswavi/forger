"""Execution engine for Forger computation graphs.

The run() function executes a Forger pipeline.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

from core import DataSpec


def run(
    pipeline: Union[DataSpec, Callable], cache: dict[int, Any] | None = None
) -> Any:
    """Execute a dataset creation pipeline (expressing a DAG of data operations).

    Args:
        pipeline: A DataSpec node, or a function that builds and returns one.
                  Passing a function lets you define pipelines as plain Python
                  functions and pass them directly to run().
        cache: Optional shared cache (for advanced use / testing).

    Returns:
        The computed data (typically List[dict])

    Example:
        >>> def my_pipeline():
        ...     data = forger.acquire.retrieve_from_hf(description="math", max_n=100)
        ...     return forger.filter.by_diversity(data, threshold=0.8)
        >>> result = forger.run(my_pipeline)
    """
    raise NotImplementedError
