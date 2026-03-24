"""Core classes for the Forger computation graph.

This module implements the lazy evaluation infrastructure that allows users
to write pipelines as plain Python functions, building a DAG that is only
executed when run() is called.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import Enum, auto
from typing import Any


class NodeType(Enum):
    """The semantic type of a node in the computation graph."""

    ACQUIRE = auto()  # Generates or retrieves new data
    ALTER = auto()  # Transforms existing data
    FILTER = auto()  # Removes data based on criteria
    DECOMPOSE = auto()  # Generates new sub-pipelines
    RAW = auto()  # Literal/raw data node


class ExecutionStatus(Enum):
    """Status of a DataSpec node during execution."""

    PENDING = auto()  # The execution of the graph has not yet reached this node
    RUNNING = auto()  # This node is currently being executed
    COMPLETED = auto()  # Execution was successful and the result has been cached
    FAILED = auto()  # Execution failed


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
        inputs: Iterable[DataSpec] | DataSpec,
        execute_fn: Callable[..., Iterable[Any]],
        name: str,
    ) -> None:
        """Store basic data in DataSpec object.

        Args:
            node_type: The semantic type of this node
            inputs: Upstream DataSpec nodes this depends on
            execute_fn: Callable that performs the actual data processing
            name: Human-readable label for debugging
        """
        self.node_type = node_type
        # The inputs, coming from upstream nodes, correspond to the arguments that will
        # be passed to execute_fn at execution time.
        self.inputs: tuple[DataSpec, ...] = self._validate_inputs(inputs)
        # execute_fn is a callable that takes as input the outputs of upstream nodes
        # and returns a list-like iterable of data items (e.g. List[dict])
        self.execute_fn: Callable[..., Iterable[Any]] = execute_fn
        self.name = name
        self._cached_result: Any | None = None
        self._status = ExecutionStatus.PENDING

    @staticmethod
    def _validate_inputs(inputs: Iterable[DataSpec] | DataSpec) -> tuple[DataSpec, ...]:
        """Normalize inputs to an immutable sequence of DataSpec instances."""
        if isinstance(inputs, DataSpec):
            deps = [inputs]
        else:
            try:
                deps = list(inputs)
            except TypeError as exc:
                raise TypeError(
                    "inputs must be a DataSpec or an iterable of DataSpec instances"
                ) from exc
        for dep in deps:
            if not isinstance(dep, DataSpec):
                raise TypeError("All inputs must be DataSpec instances")
        return tuple(deps)

    def __repr__(self) -> str:
        """Show node type and status for debugging."""
        return (
            f"DataSpec({self.name}, status={self._status.name.lower()}, "
            + f"deps={len(self.inputs)})"
        )

    def get_dependencies(self) -> list[DataSpec]:
        """Return list of upstream DataSpec nodes."""
        return list(self.inputs)

    @property
    def is_completed(self) -> bool:
        """Return status of whether this node has been computed."""
        return self._status == ExecutionStatus.COMPLETED

    @property
    def result(self) -> Any:
        """Return the computed result, or raise if not computed yet."""
        if not self.is_completed:
            raise RuntimeError(
                f"DataSpec '{self.name}' has not been executed yet. Call run() first."
            )
        return self._cached_result
