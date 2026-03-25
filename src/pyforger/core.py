"""Core classes for the Forger computation graph.

This module implements the lazy evaluation infrastructure that allows users
to write pipelines as plain Python functions, building a DAG that is only
executed when run() is called.
"""
from __future__ import annotations

import inspect
import typing
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto


class NodeType(Enum):
    """The semantic type of a node in the computation graph."""

    ACQUIRE = auto()  # Generates or retrieves new data
    ALTER = auto()  # Transforms existing data
    FILTER = auto()  # Removes data based on criteria
    DECOMPOSE = auto()  # Generates new sub-pipelines
    RAW = auto()  # Literal/raw data node


class ExecuteFn(typing.Protocol):
    """Protocol for the execute function of a DataSpec node."""

    def __call__(self, *args: list[dict]) -> list[dict]:
        """This is a protocol for the function performing actual data processing."""
        ...


class ExecutionStatus(Enum):
    """Status of a DataSpec node during execution."""

    PENDING = auto()  # The execution of the graph has not yet reached this node
    RUNNING = auto()  # This node is currently being executed
    COMPLETED = auto()  # Execution was successful and the result has been cached
    FAILED = auto()  # Execution failed


@dataclass(frozen=True)
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
        >>> source = DataSpec(NodeType.SOURCE, (), lambda: ["a", "a", "b"], "source")
        >>> _execute_fn = lambda inputs: [x * 2 for x in inputs[0]]
        >>> doubled = DataSpec(NodeType.ALTER, source, _execute_fn, "test_double")
        >>> filtered = forger.filter.by_diversity(source, threshold=0.8)
        >>> result = forger.run(filtered)  # executes the graph
    """

    node_type: NodeType
    inputs: tuple[DataSpec, ...]
    execute_fn: ExecuteFn
    name: str
    _cached_result: typing.Any | None
    _status: ExecutionStatus

    def __post_init__(
        self,
        node_type: NodeType,
        inputs: Iterable[DataSpec] | DataSpec,
        execute_fn: ExecuteFn,
        name: str,
    ) -> None:
        """Store basic data in DataSpec object.

        Args:
            node_type: The semantic type of this node
            inputs: Upstream DataSpec nodes this depends on
            execute_fn: Callable that performs the actual data processing
            name: Human-readable label for debugging
        """
        object.__setattr__(self, "node_type", node_type)
        # The inputs, coming from upstream nodes, correspond to the arguments that will
        # be passed to execute_fn at execution time.
        object.__setattr__(self, "inputs", self._validate_inputs(inputs))
        # execute_fn is a callable that takes as input the outputs of upstream nodes
        # and returns a list-like iterable of data items (e.g. List[dict])
        # The arguments to execute_fn must correspond in number and order to the
        # inputs to the node.
        if not callable(execute_fn):
            raise TypeError(
                f"execute_fn must be callable, got {type(execute_fn).__name__!r}"
            )

        # Validate that execute_fn only takes positional arguments, and that the
        # number of arguments matches the number of inputs.
        try:
            sig = inspect.signature(execute_fn)
        except ValueError:
            pass  # can't introspect (e.g. C extensions), skip arity check
        else:
            params = sig.parameters.values()
            if any(
                p.kind
                in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
                for p in params
            ):
                raise TypeError("execute_fn must only take positional arguments")
            if any(p.default is not inspect.Parameter.empty for p in params):
                raise TypeError("execute_fn should not have default arguments")
            if len(params) != len(self.inputs):
                raise TypeError(
                    f"execute_fn takes {len(params)} args but "
                    + f"node expects {len(self.inputs)} inputs"
                )

        # Validate that if execute_fn is annotated with type hints, it takes one or
        # more args of type list[dict] and returns a list[dict].
        # This check should be paired with runtime type validation.
        try:
            hints = typing.get_type_hints(execute_fn)
        except Exception:
            pass  # unannotated or unresolvable, skip
        else:
            for p in params:
                if p.name in hints and hints[p.name] != list[dict]:
                    raise TypeError(
                        f"execute_fn parameter {p.name!r} must be annotated "
                        + f"as list[dict], instead got {hints[p.name]}"
                    )
            if "return" in hints and hints["return"] != list[dict]:
                raise TypeError(
                    f"execute_fn must return list[dict], got {hints['return']}"
                )
        object.__setattr__(self, "execute_fn", execute_fn)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_cached_result", None)
        object.__setattr__(self, "_status", ExecutionStatus.PENDING)

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
            f"DataSpec({self.name}, status={self._status.name.lower()},"
            + f" deps={len(self.inputs)})"
        )

    def get_dependencies(self) -> list[DataSpec]:
        """Return list of upstream DataSpec nodes."""
        return list(self.inputs)
