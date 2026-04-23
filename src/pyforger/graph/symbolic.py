"""Symbolic per-row handles and sub-expression nodes."""
from __future__ import annotations

import itertools
from typing import Any

from pyforger.core import DataSpec, NodeType


class TracingError(RuntimeError):
    """Raised when a symbolic value is used somewhere that needs a concrete value."""


_item_counter = itertools.count()

_RESERVED_DATASPEC_ATTRS = frozenset({
    "node_type", "inputs", "inputs_arg", "execute_fn", "name",
    "_cached_result", "_status", "is_completed", "result",
    "get_dependencies",
})


def _unsupported_bool(self):
    raise TracingError(
        f"{type(self).__name__} can't be used in a boolean context (if/and/or/not). "
        "Use pyforger.graph.cond(predicate, then_fn, else_fn) — or the AST "
        "compiler's if/else inside a function passed to pyforger.run()."
    )


def _unsupported_iter(self):
    raise TracingError(
        f"{type(self).__name__} can't be iterated directly. "
        "Use pyforger.graph.foreach(source, body_fn) — or a for-loop over a "
        "DataSpec inside a function passed to pyforger.run()."
    )


def _expr(op, args, name=None):
    """Build an EXPR DataSpec wrapping a symbolic operation."""
    dataspec_inputs = [a for a in args if isinstance(a, DataSpec)]

    def _execute(*resolved):
        it = iter(resolved)
        concrete = []
        for a in args:
            concrete.append(next(it) if isinstance(a, DataSpec) else a)
        return _apply_op(op, concrete)

    return ExprSpec(NodeType.EXPR, dataspec_inputs, _execute, name or f"expr_{op}")


def _apply_op(op, args):
    if op == "getitem":
        obj, key = args
        return obj[key]
    if op == "getattr":
        obj, name_ = args
        return getattr(obj, name_)
    if op == "call":
        func, *rest = args
        return func(*rest)
    if op == "method":
        obj, method_name, *rest = args
        return getattr(obj, method_name)(*rest)
    if op == "eq":  return args[0] == args[1]
    if op == "ne":  return args[0] != args[1]
    if op == "lt":  return args[0] <  args[1]
    if op == "le":  return args[0] <= args[1]
    if op == "gt":  return args[0] >  args[1]
    if op == "ge":  return args[0] >= args[1]
    if op == "not": return not args[0]
    if op == "and": return args[0] and args[1]
    if op == "or":  return args[0] or  args[1]
    if op == "in":  return args[0] in args[1]
    if op == "add": return args[0] +  args[1]
    if op == "sub": return args[0] -  args[1]
    if op == "mul": return args[0] *  args[1]
    raise ValueError(f"Unknown symbolic op: {op}")


class ExprSpec(DataSpec):
    """A symbolic expression built from ItemSpec and concrete values."""

    __hash__ = object.__hash__

    def __bool__(self):
        _unsupported_bool(self)

    def __iter__(self):
        _unsupported_iter(self)

    def __getitem__(self, key):
        return _expr("getitem", [self, key], name=f"getitem_{key!r}")

    def __getattr__(self, name):
        if name.startswith("_") or name in _RESERVED_DATASPEC_ATTRS:
            raise AttributeError(name)
        return _BoundAttr(self, name)

    def __eq__(self, other):  return _expr("eq", [self, other])
    def __ne__(self, other):  return _expr("ne", [self, other])
    def __lt__(self, other):  return _expr("lt", [self, other])
    def __le__(self, other):  return _expr("le", [self, other])
    def __gt__(self, other):  return _expr("gt", [self, other])
    def __ge__(self, other):  return _expr("ge", [self, other])
    def __add__(self, other): return _expr("add", [self, other])
    def __sub__(self, other): return _expr("sub", [self, other])
    def __mul__(self, other): return _expr("mul", [self, other])


class _BoundAttr:
    """Intermediate returned by ``expr.attr`` — resolves on call."""
    __slots__ = ("_obj", "_name")

    def __init__(self, obj, name):
        self._obj = obj
        self._name = name

    def __call__(self, *args):
        return _expr("method", [self._obj, self._name, *args],
                     name=f"method_{self._name}")

    def __repr__(self):
        return f"<bound method {self._name!r} on {self._obj!r}>"


class ItemSpec(ExprSpec):
    """A symbolic per-row handle bound to a FOREACH's current iteration."""

    def __init__(self, name="item"):
        uid = next(_item_counter)
        object.__setattr__(self, "_uid", uid)

        def _execute():
            from pyforger.runner import current_frame_value
            return current_frame_value(uid)

        super().__init__(NodeType.ITEM, [], _execute, f"{name}#{uid}")

    @property
    def uid(self):
        return self._uid
