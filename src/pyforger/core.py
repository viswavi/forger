"""Core classes for the Forger computation graph.

This module implements the lazy evaluation infrastructure that allows users
to write pipelines as plain Python functions, building a DAG that is only
executed when run() is called.
"""
from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto


class NodeType(Enum):
    """The semantic type of a node in the computation graph."""

    ACQUIRE = auto()  # Generates or retrieves new data
    ALTER = auto()  # Transforms existing data
    FILTER = auto()  # Removes data based on criteria
    DECOMPOSE = auto()  # Generates new sub-pipelines
    RAW = auto()  # Literal/raw data node


class DataSpec:
    """A node in the computation graph.

    Created by functional operations (forger.acquire.*, forger.filter.*, etc.)
    and linked together to form a DAG. Execution is deferred until run().

    Attributes:
        node_type: The semantic type of this node
        inputs: Upstream DataSpec nodes this depends on
        execute_fn: Callable that performs the actual computation
        name: Human-readable label for debugging

    Example:
        >>> retrieved = forger.acquire.retrieve_from_hf(description="math", max_n=100)
        >>> filtered = forger.filter.by_diversity(retrieved, threshold=0.8)
        >>> result = forger.run(filtered)  # executes the graph
    """

    def __init__(
        self,
        node_type: NodeType,
        inputs: list[DataSpec],
        execute_fn: Callable,
        name: str,
    ) -> None:
        """Store basic data in DataSpec object."""
        self.node_type = node_type
        self.inputs = inputs
        self.execute_fn = execute_fn
        self.name = name
