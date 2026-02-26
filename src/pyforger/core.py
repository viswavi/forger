"""Core classes for the Forger computation graph.

This module implements the lazy evaluation infrastructure that allows users
to write code that looks like sequential data transformations, but actually
builds a DAG that can be parallelized and cached.
"""

from __future__ import annotations

from abc import ABC
from enum import Enum, auto
from typing import Any


class ExecutionStatus(Enum):
    """Status of a DataSpec node during execution."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    FILTERED = auto()  # Node was filtered out (early termination)


class DataSpec:
    """Represents a node in the computation graph.

    DataSpec objects are created by operations but don't execute immediately.
    They track dependencies and cache results after execution.

    Attributes:
        operation: The Operation instance that created this node
        inputs: List of upstream DataSpec nodes this depends on
        _cached_result: Stored result after execution (None until run)
        _status: Current execution status

    Example:
        >>> op = SomeOperation(param=42)
        >>> spec = op()  # Creates DataSpec, doesn't execute
        >>> result = run(spec)  # Now it executes
    """

    def __init__(
        self,
        operation: Operation,
        inputs: list[DataSpec] | None = None,
        raw_data: Any | None = None,
    ) -> None:
        """Initialize a DataSpec node."""
        raise NotImplementedError


class Operation(ABC):
    """Abstract base class for all Forger operations.

    Operations define transformations on data, executed just-in-time.

    Subclasses must implement:
        - execute(*input_data): Perform the actual computation

    Example:
        class MyOperation(Operation):
            def __init__(self, threshold: float):
                self.threshold = threshold

            def execute(self, *input_data) -> List[dict]:
                # Actually process the data
                return [row for row in input_data[0] if row['score'] > self.threshold]
    """

    def __call__(self, *args, **kwargs) -> DataSpec:
        """Create a DataSpec node for this operation."""
        raise NotImplementedError


class AcquireOperation(Operation):
    """Operations that generate or retrieve new data.

    Acquire operations are typically "source" nodes in the DAG - they have
    no DataSpec inputs and produce new data from external sources.

    Examples: HuggingfaceRetrieve, GenerateInstance, CritiqueInstance
    """

    expected_inputs = 0  # Acquire operations don't take DataSpec inputs


class AlterOperation(Operation):
    """Operations that transform existing data.

    Alter operations take one or more input datasets and produce a
    transformed output. They don't filter data, just modify it.

    Examples: Refine, Translate, Concatenate
    """

    pass  # Variable inputs depending on operation


class FilterOperation(Operation):
    """Operations that remove data based on criteria.

    Filter operations take an input dataset and return a subset.
    If a filter removes ALL data, the pipeline can terminate early.

    Examples: Diversity, Agreement, Disagreement, Quality
    """

    expected_inputs = 1  # Filters typically take one dataset

    # If True, empty results will cause early pipeline termination
    terminate_on_empty: bool = True


class DecomposeOperation(Operation):
    """Operations that generate new operations or sub-pipelines.

    Decompose operations are meta-operations that create new operations
    from a given sequence of operations, enabling dynamic pipelines.
    """

    expected_inputs = 1
