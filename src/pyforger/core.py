"""Core classes for the Forger computation graph.

This module implements the lazy evaluation infrastructure that allows users
to write pipelines as plain Python functions, building a DAG that is only
executed when run() is called.
"""
from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Iterable
from dataclasses import InitVar, dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class NodeType(Enum):
    """The semantic type of a node in the computation graph."""

    ACQUIRE = auto()  # Generates or retrieves new data
    ALTER = auto()  # Transforms existing data
    FILTER = auto()  # Removes data based on criteria
    DECOMPOSE = auto()  # Generates new sub-pipelines
    RAW = auto()  # Literal/raw data node


ExecuteFn = Callable[..., list[dict]]


class ExecutionStatus(Enum):
    """Status of a DataSpec node during execution."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    FILTERED = auto()


_MUTABLE_FIELDS: frozenset[str] = frozenset({"_status", "_cached_result"})


@dataclass
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
    # inputs_arg is the raw input passed at initialization, which is converted to a
    # tuple of DataSpecs and stored as `inputs`.
    inputs_arg: InitVar[Iterable[DataSpec] | DataSpec]
    execute_fn: ExecuteFn
    name: str
    inputs: tuple[DataSpec, ...] = field(init=False)
    _cached_result: typing.Any | None = field(default=None)
    _status: ExecutionStatus = field(default=ExecutionStatus.PENDING)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutation of fields after initialization with specific exceptions."""
        if hasattr(self, "name") and name not in _MUTABLE_FIELDS:
            raise AttributeError(f"DataSpec field '{name}' is read-only")
        object.__setattr__(self, name, value)

    def __post_init__(self, inputs_arg: Iterable[DataSpec] | DataSpec) -> None:
        """Store basic data in DataSpec object.

        Args:
            inputs_arg: The raw input DataSpec(s) passed at initialization.
        """
        # The inputs, coming from upstream nodes, correspond to the arguments that will
        # be passed to execute_fn at execution time.
        object.__setattr__(self, "inputs", self._validate_inputs(inputs_arg))
        # execute_fn is a callable that takes as input the outputs of upstream nodes
        # and returns a list-like iterable of data items (e.g. List[dict])
        # The arguments to execute_fn must correspond in number and order to the
        # inputs to the node.
        if not callable(self.execute_fn):
            raise TypeError(
                f"execute_fn must be callable, got {type(self.execute_fn).__name__!r}"
            )

        # Validate that execute_fn only takes positional arguments, and that the
        # number of arguments matches the number of inputs.
        try:
            sig = inspect.signature(self.execute_fn)
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
            fixed_params = [
                p
                for p in params
                if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                or p.kind == inspect.Parameter.POSITIONAL_ONLY
            ]
            has_var_positional = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
            )
            if has_var_positional:
                if len(self.inputs) < len(fixed_params):
                    raise TypeError(
                        f"execute_fn requires at least {len(fixed_params)} args but "
                        + f"node has only {len(self.inputs)} inputs"
                    )
            elif len(fixed_params) != len(self.inputs):
                raise TypeError(
                    f"execute_fn takes {len(fixed_params)} args but "
                    + f"node expects {len(self.inputs)} inputs"
                )

        # Validate that if execute_fn is annotated with type hints, it takes one or
        # more args of type list[dict] and returns a list[dict].
        # This check should be paired with runtime type validation.
        try:
            hints = typing.get_type_hints(self.execute_fn)
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

    # Dataclasses are not hashable by default if they are mutable, so we overwrite this
    # behavior to alow custom hashing based on object identity (not its contents).
    __hash__ = object.__hash__

    @staticmethod
    def _validate_inputs(inputs: DataSpec | list[DataSpec]) -> list[DataSpec]:
        """Normalize inputs to a fresh list of DataSpec instances."""
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
        return list(deps)

    # Dataclasses are not hashable by default if they are mutable, so we overwrite this
    # behavior to alow custom hashing based on object identity (not its contents).
    __hash__ = object.__hash__

    def __repr__(self) -> str:
        """Show node type and status for debugging."""
        return (
            f"DataSpec({self.name}, status={self._status.name.lower()}, "
            + f"deps={len(self.inputs)})"
        )

    def get_dependencies(self) -> list[DataSpec]:
        """Return list of upstream DataSpec nodes."""
        return self.inputs.copy()

    @property
    def is_computed(self) -> bool:
        """Return status of whether this node has been computed."""
        return self._status == ExecutionStatus.COMPLETED

    @property
    def is_filtered(self) -> bool:
        """Return status of whether this node has been filtered."""
        return self._status == ExecutionStatus.FILTERED

    @property
    def result(self) -> Any:
        """Return the computed result, or raise if not computed yet."""
        if not (self.is_computed or self.is_filtered):
            raise RuntimeError(
                f"DataSpec '{self.name}' has not been executed yet. Call run() first."
            )
        return self._cached_result
