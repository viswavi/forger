"""Execution engine for Forger computation graphs.

The run() function executes a Forger pipeline.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from pyforger.core import DataSpec, ExecutionStatus, NodeType
from pyforger.utils import get_formatted_logger

logger = get_formatted_logger("Runner")


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the computation graph."""

    pass


class FilterTerminationError(Exception):
    """Raised when a filter operation returns empty results."""

    def __init__(self, operation_name: str, message: str = ""):
        """Initialize with the name of the filter operation that caused the error."""
        self.operation_name = operation_name
        super().__init__(message or f"Filter '{operation_name}' returned empty results")


def run(
    pipeline: DataSpec | Callable[..., DataSpec], cache: dict[int, Any] | None = None
) -> Any:
    """Execute a dataset creation pipeline (expressing a DAG of data operations).

    Accepts either a DataSpec node or a callable that returns one.
    Executes the DAG in topological order with caching.

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
    if callable(pipeline) and not isinstance(pipeline, DataSpec):
        pipeline = pipeline()

    dataset: DataSpec = pipeline

    if cache is None:
        cache = {}

    node_id = id(dataset)
    if node_id in cache:
        return cache[node_id]

    if dataset.is_computed:
        return dataset.result

    execution_order = _topological_sort(dataset)

    for node in execution_order:
        try:
            _execute_node(node, cache)
        except FilterTerminationError as e:
            logger.debug(f"Filter removed all data: {e}")
            return []

    return cache[id(dataset)]


def _topological_sort(root: DataSpec) -> list[DataSpec]:
    """Compute topological execution order via Kahn's algorithm."""
    all_nodes: set[int] = set()
    node_map: dict[int, DataSpec] = {}

    def collect_nodes(node: DataSpec) -> None:
        node_id = id(node)
        if node_id in all_nodes:
            return
        all_nodes.add(node_id)
        node_map[node_id] = node
        for dep in node.inputs:
            collect_nodes(dep)

    collect_nodes(root)

    in_degree: dict[int, int] = {nid: 0 for nid in all_nodes}
    dependents: dict[int, list[int]] = {nid: [] for nid in all_nodes}

    for node_id in all_nodes:
        node = node_map[node_id]
        for dep in node.inputs:
            dep_id = id(dep)
            in_degree[node_id] += 1
            dependents[dep_id].append(node_id)

    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    result: list[DataSpec] = []

    while queue:
        node_id = queue.popleft()
        result.append(node_map[node_id])
        for dependent_id in dependents[node_id]:
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0:
                queue.append(dependent_id)

    if len(result) != len(all_nodes):
        remaining = [node_map[nid].name for nid in all_nodes if in_degree[nid] > 0]
        raise CycleDetectedError(
            f"Cycle detected in computation graph. Nodes involved: {remaining}"
        )

    return result


def _execute_node(node: DataSpec, cache: dict[int, Any]) -> None:
    """Execute a single node, storing result in cache."""
    node_id = id(node)

    if node_id in cache:
        return

    if node.is_computed:
        cache[node_id] = node.result
        return

    input_data = [cache[id(dep)] for dep in node.inputs]
    node._status = ExecutionStatus.RUNNING

    try:
        result = node.execute_fn(*input_data)

        if node.node_type == NodeType.FILTER and _is_empty(result):
            node._status = ExecutionStatus.FILTERED
            raise FilterTerminationError(node.name)

        node._cached_result = result
        node._status = ExecutionStatus.COMPLETED
        cache[node_id] = result

    except NotImplementedError:
        node._status = ExecutionStatus.FAILED
        raise

    except FilterTerminationError:
        raise

    except Exception as e:
        node._status = ExecutionStatus.FAILED
        raise RuntimeError(f"Error executing '{node.name}': {e}") from e


def _is_empty(data: Any) -> bool:
    if data is None:
        return True
    if isinstance(data, (list, tuple, set, frozenset)):
        return len(data) == 0
    elif hasattr(data, "__len__"):
        return len(data) == 0
    elif hasattr(data, "empty"):
        return data.empty
    return False
