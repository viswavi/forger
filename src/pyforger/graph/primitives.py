"""Graph primitives: foreach and cond."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pyforger.core import DataSpec, NodeType
from pyforger.graph.symbolic import ExprSpec, ItemSpec


_MUTABLE_GRAPH_FIELDS = frozenset({
    "_foreach_source", "_foreach_body", "_foreach_item_uid",
    "_cond_pred", "_cond_then", "_cond_else",
})


class _GraphNodeMixin:
    def __setattr__(self, name, value):
        if name in _MUTABLE_GRAPH_FIELDS or not hasattr(self, "name"):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)


@dataclass
class ForeachSpec(_GraphNodeMixin, DataSpec):
    _foreach_source: DataSpec | None = field(default=None)
    _foreach_body: DataSpec | None = field(default=None)
    _foreach_item_uid: int = field(default=-1)

    __hash__ = object.__hash__


@dataclass
class CondSpec(_GraphNodeMixin, DataSpec):
    _cond_pred: DataSpec | None = field(default=None)
    _cond_then: DataSpec | None = field(default=None)
    _cond_else: DataSpec | None = field(default=None)

    __hash__ = object.__hash__


def _foreach_not_callable(_source):
    raise RuntimeError("FOREACH nodes are executed by the runner, not execute_fn.")


def _cond_not_callable():
    raise RuntimeError("COND nodes are executed by the runner, not execute_fn.")


def foreach(
    source: DataSpec,
    body_fn: Callable[[ItemSpec], DataSpec],
    *,
    name: str = "foreach",
) -> ForeachSpec:
    """Map ``body_fn`` over each row of ``source`` lazily."""
    item = ItemSpec(name="item")
    body = body_fn(item)
    if not isinstance(body, DataSpec):
        raise TypeError(
            f"foreach body_fn must return a DataSpec, got {type(body).__name__}"
        )

    node = ForeachSpec(
        node_type=NodeType.FOREACH,
        inputs_arg=[source],
        execute_fn=_foreach_not_callable,
        name=name,
    )
    node._foreach_item_uid = item.uid
    node._foreach_source = source
    node._foreach_body = body
    return node


def cond(
    pred: ExprSpec | DataSpec,
    then_fn: Callable[[], DataSpec],
    else_fn: Callable[[], DataSpec],
    *,
    name: str = "cond",
) -> CondSpec:
    """Pick between two sub-DAGs per row based on a symbolic predicate."""
    then_spec = then_fn()
    else_spec = else_fn()
    if not isinstance(then_spec, DataSpec) or not isinstance(else_spec, DataSpec):
        raise TypeError("cond branches must return DataSpecs")

    node = CondSpec(
        node_type=NodeType.COND,
        inputs_arg=[],
        execute_fn=_cond_not_callable,
        name=name,
    )
    node._cond_pred = pred
    node._cond_then = then_spec
    node._cond_else = else_spec
    return node
