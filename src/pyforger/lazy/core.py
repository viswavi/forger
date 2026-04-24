"""Lazy runtime for Forger — Dask-style delayed execution.

Ops return Lazy[T] handles. Chaining ops stays lazy; the first Python
operation that needs a value (iteration, bool-check, len, equality,
.compute()) materializes the upstream subgraph and caches the result.

The design has no AST walker, no symbolic tracer, no frame stack. Plain
Python control flow (for, if, list comprehensions, generator expressions)
works because materialization is implicit at the dunder boundary.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Generic, TypeVar

logger = logging.getLogger("forger.lazy")

T = TypeVar("T")

CACHE_DIR = Path(os.environ.get("FORGER_CACHE", Path.home() / ".cache" / "forger"))
_MISSING: Any = object()


# ---------------------------------------------------------------------------
# Content-addressed hashing (reused from prior @op prototype).
# ---------------------------------------------------------------------------

_CODE_T = type((lambda: None).__code__)


def _code_key(code: Any) -> list[Any]:
    consts = [_code_key(c) if isinstance(c, _CODE_T) else _canonical(c) for c in code.co_consts]
    return [code.co_code, consts, list(code.co_names), list(code.co_varnames),
            list(code.co_freevars), code.co_argcount]


def _canonical(obj: Any) -> Any:
    """Convert obj to a stable, picklable structure for hashing."""
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    if isinstance(obj, Lazy):
        # Lazies hash by their *plan* (op + inputs), not by identity, so
        # two structurally-identical subgraphs share a cache key.
        return ["__lazy__", obj.op_name, _canonical(list(obj.args)),
                _canonical(obj.kwargs)]
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, type):
        return f"<type {obj.__module__}.{obj.__qualname__}>"
    if hasattr(obj, "cache_key") and callable(obj.cache_key):
        return ["__cache_key__", _canonical(obj.cache_key())]
    if callable(obj):
        ident = f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', repr(obj))}"
        code = getattr(obj, "__code__", None)
        if code is not None:
            closure = getattr(obj, "__closure__", None) or ()
            closure_vals = [_canonical(cell.cell_contents) for cell in closure]
            return ["__callable__", ident, _code_key(code), closure_vals]
        return ["__callable__", ident]
    return repr(obj)


def _hash(obj: Any) -> str:
    return hashlib.sha256(pickle.dumps(_canonical(obj), protocol=4)).hexdigest()


# ---------------------------------------------------------------------------
# Trace — every materialization logs here for post-hoc DAG rendering.
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    op_name: str
    input_hash: str
    cache_hit: bool
    duration_s: float


_TRACE: list[CallRecord] = []


def trace() -> list[CallRecord]:
    return list(_TRACE)


def clear_trace() -> None:
    _TRACE.clear()


# ---------------------------------------------------------------------------
# Lazy — the core type. Wraps a deferred op call.
# ---------------------------------------------------------------------------

class Lazy(Generic[T]):
    """A deferred computation that materializes on demand.

    Materialization is triggered implicitly by Python operations that
    require a value: iteration, len, bool, equality, subscripting. It is
    also triggered explicitly by ``.compute()``.

    Chaining ops stays lazy: ``filter.by_length(alter.generate(...))``
    returns a Lazy whose compute() walks the graph from the leaves.
    """

    __slots__ = ("op_name", "fn", "args", "kwargs", "_body_key", "_cached_result", "_input_hash")

    def __init__(
        self,
        op_name: str,
        fn: Callable[..., T],
        args: tuple,
        kwargs: dict[str, Any],
        body_key: Any = None,
    ) -> None:
        self.op_name = op_name
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self._body_key = body_key if body_key is not None else _canonical(fn)
        self._cached_result: Any = _MISSING
        self._input_hash: str | None = None

    # --- materialization triggers ----------------------------------------

    def compute(self) -> T:
        """Force execution of this node and all upstream Lazies."""
        if self._cached_result is not _MISSING:
            return self._cached_result

        start = time.time()
        resolved_args = tuple(_force(a) for a in self.args)
        resolved_kwargs = {k: _force(v) for k, v in self.kwargs.items()}

        input_hash = _hash([self.op_name, self._body_key, list(resolved_args), resolved_kwargs])
        self._input_hash = input_hash
        cache_path = CACHE_DIR / self.op_name / f"{input_hash}.pkl"

        if os.environ.get("FORGER_HASH_DEBUG"):
            print(f"  [hash] {self.op_name}  input_hash={input_hash[:12]}")
            print(f"    op_name={_hash(self.op_name)[:12]}")
            print(f"    body_key={_hash(self._body_key)[:12]}")
            print(f"    args={_hash(list(resolved_args))[:12]}")
            print(f"    kwargs={_hash(resolved_kwargs)[:12]}  keys={list(resolved_kwargs)}")
            for k, v in resolved_kwargs.items():
                print(f"      kw {k} -> {_hash(v)[:12]}  (type={type(v).__name__})")

        if cache_path.exists():
            with cache_path.open("rb") as f:
                result = pickle.load(f)
            _TRACE.append(CallRecord(self.op_name, input_hash, True, time.time() - start))
            logger.debug("cache HIT  %s %s", self.op_name, input_hash[:8])
        else:
            result = self.fn(*resolved_args, **resolved_kwargs)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with cache_path.open("wb") as f:
                    pickle.dump(result, f, protocol=4)
            except (pickle.PicklingError, TypeError) as e:
                logger.warning("cache skip %s: unpicklable result (%s)", self.op_name, e)
            _TRACE.append(CallRecord(self.op_name, input_hash, False, time.time() - start))
            logger.debug("cache MISS %s %s (%.2fs)", self.op_name, input_hash[:8], time.time() - start)

        self._cached_result = result
        return result

    def __iter__(self):
        return iter(self.compute())

    def __len__(self) -> int:
        return len(self.compute())

    def __bool__(self) -> bool:
        return bool(self.compute())

    def __eq__(self, other: Any) -> bool:
        other_val = other.compute() if isinstance(other, Lazy) else other
        return self.compute() == other_val

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        # Identity hash so Lazy is usable as a dict key during graph walks
        # without forcing materialization.
        return id(self)

    def __getitem__(self, key: Any) -> Any:
        return self.compute()[key]

    def __add__(self, other: Any) -> Any:
        other_val = other.compute() if isinstance(other, Lazy) else other
        return self.compute() + other_val

    def __radd__(self, other: Any) -> Any:
        other_val = other.compute() if isinstance(other, Lazy) else other
        return other_val + self.compute()

    # --- introspection (no materialization) ------------------------------

    def dependencies(self) -> list[Lazy]:
        """Return direct upstream Lazies. Does not materialize."""
        deps: list[Lazy] = []
        for x in (*self.args, *self.kwargs.values()):
            if isinstance(x, Lazy):
                deps.append(x)
            elif isinstance(x, (list, tuple)):
                deps.extend(y for y in x if isinstance(y, Lazy))
        return deps

    def walk(self) -> Iterable[Lazy]:
        """Post-order walk of the full subgraph. Does not materialize."""
        seen: set[int] = set()

        def visit(node: Lazy) -> Iterable[Lazy]:
            if id(node) in seen:
                return
            seen.add(id(node))
            for dep in node.dependencies():
                yield from visit(dep)
            yield node

        yield from visit(self)

    def plan(self) -> str:
        """Render the subgraph as an indented tree. Does not materialize."""
        lines: list[str] = []

        def render(node: Lazy, depth: int, seen: set[int]) -> None:
            status = "done" if node._cached_result is not _MISSING else "pending"
            prefix = "  " * depth + ("└─ " if depth else "")
            marker = " (ref)" if id(node) in seen else ""
            lines.append(f"{prefix}{node.op_name} [{status}]{marker}")
            if id(node) in seen:
                return
            seen.add(id(node))
            for dep in node.dependencies():
                render(dep, depth + 1, seen)

        render(self, 0, set())
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "done" if self._cached_result is not _MISSING else "pending"
        return f"Lazy<{self.op_name} {status}>"


def _force(x: Any) -> Any:
    """Recursively materialize any Lazy inside the given value."""
    if isinstance(x, Lazy):
        return x.compute()
    if isinstance(x, list):
        return [_force(y) for y in x]
    if isinstance(x, tuple):
        return tuple(_force(y) for y in x)
    if isinstance(x, dict):
        return {k: _force(v) for k, v in x.items()}
    return x


# ---------------------------------------------------------------------------
# @op decorator — wrap a plain function so it returns Lazy[T].
# ---------------------------------------------------------------------------

def op(_fn: Callable[..., T] | None = None, *, name: str | None = None) -> Any:
    """Decorator: turn a plain function into a lazy op.

    Calling the wrapped function never executes the body immediately —
    it records inputs and returns a Lazy. The body runs on first
    materialization (iter/bool/len/eq/compute). Results are cached on
    disk by a content hash of (op name, function bytecode + closure,
    realized inputs).

    Usage::

        @op
        def by_length(items, *, min_len, max_len): ...

        @op(name="my.op")
        def weird_thing(x): ...
    """

    def decorate(fn: Callable[..., T]) -> Callable[..., Lazy[T]]:
        op_name = name or f"{fn.__module__}.{fn.__qualname__}"
        body_key = _canonical(fn)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Lazy[T]:
            return Lazy(op_name, fn, args, kwargs, body_key=body_key)

        wrapper.__wrapped__ = fn
        wrapper.op_name = op_name  # type: ignore[attr-defined]
        return wrapper

    if _fn is not None:
        return decorate(_fn)
    return decorate


# ---------------------------------------------------------------------------
# Entry point — run(pipeline) for Dask-style .compute() at the top level.
# ---------------------------------------------------------------------------

def run(pipeline: Callable[..., Lazy[T]] | Lazy[T]) -> T:
    """Execute a pipeline and return the materialized result.

    Accepts either a Lazy directly or a zero-arg callable that returns one.
    """
    result = pipeline() if (callable(pipeline) and not isinstance(pipeline, Lazy)) else pipeline
    if isinstance(result, Lazy):
        return result.compute()
    return result
