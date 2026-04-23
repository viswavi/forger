"""Execution engine for Forger computation graphs.

Handles:
- Regular DAG execution with topological sort + caching
- FOREACH / COND / ITEM / EXPR nodes via a recursive per-row evaluator
  with a frame stack (ContextVar-based, thread- and asyncio-safe) that
  gives ItemSpecs their current-row values
- Auto-compilation of plain Python functions with symbolic for/if

The ``persistent`` flag in ``_evaluate`` is load-bearing:
    persistent=True   — outside a foreach frame; results may be written
                        to node._cached_result for cross-run reuse.
    persistent=False  — inside a foreach body; per-row results stay in
                        the fresh sub-cache only, never on the node.
Never flip this flag on a single node mid-evaluation; do not write
_cached_result from within a FOREACH body.
"""
from __future__ import annotations

import contextvars
from collections.abc import Callable
from typing import Any

from pyforger.core import DataSpec, ExecutionStatus, NodeType
from pyforger.utils import get_formatted_logger

logger = get_formatted_logger("Runner")


# --- frame stack (thread-/asyncio-safe) ---------------------------------------

_frame_stack_var: contextvars.ContextVar[tuple[dict[int, Any], ...]] = (
    contextvars.ContextVar("_pyforger_frame_stack", default=())
)


def _push_frame(frame: dict[int, Any]) -> contextvars.Token:
    return _frame_stack_var.set(_frame_stack_var.get() + (frame,))


def _pop_frame(token: contextvars.Token) -> None:
    _frame_stack_var.reset(token)


def current_frame_value(uid: int) -> Any:
    """Return the current-row binding for an ItemSpec's uid."""
    for frame in reversed(_frame_stack_var.get()):
        if uid in frame:
            return frame[uid]
    raise RuntimeError(
        f"ItemSpec uid={uid} accessed with no active FOREACH frame. "
        "Did you use an ItemSpec outside its foreach()?"
    )


# --- public entry point -------------------------------------------------------

# Only NotImplementedError (compiler doesn't handle this AST shape, but the
# code is safe to eager) triggers fallback. TracingScopeError means the
# compiler recognized the shape AND knows eager execution would produce a
# different result — must propagate.
_COMPILE_SUPPORT_ERRORS = (NotImplementedError,)


def run(
    pipeline: DataSpec | Callable[..., DataSpec],
    cache: dict[int, Any] | None = None,
) -> Any:
    """Execute a Forger pipeline.

    If ``pipeline`` is a plain ``def``-function (not a lambda), it's compiled
    via ``pyforger.graph.compile_function``. Compilation-support gaps fall
    back to eager call; any other exception propagates unchanged so real
    bugs aren't masked.
    """
    if callable(pipeline) and not isinstance(pipeline, DataSpec):
        if _is_lambda(pipeline):
            raise TypeError(
                "pyforger.run cannot AST-compile a lambda. Pass a `def` "
                "function — lambdas can't express lazy for/if."
            )
        from pyforger.graph.compiler import compile_function
        try:
            resolved = compile_function(pipeline)
        except _COMPILE_SUPPORT_ERRORS as e:
            logger.warning(
                "pyforger.graph.compile_function fell back to eager for %s: %s",
                getattr(pipeline, "__name__", "<fn>"), e,
            )
            resolved = pipeline()
    else:
        resolved = pipeline

    dataset: DataSpec = resolved
    if cache is None:
        cache = {}
    return _evaluate(dataset, cache)


def _is_lambda(fn) -> bool:
    return getattr(fn, "__name__", "") == "<lambda>"


# --- evaluator ----------------------------------------------------------------

def _evaluate(node: DataSpec, cache: dict[int, Any], persistent: bool = True) -> Any:
    """Recursively evaluate a node; handles all NodeTypes.

    Caching rules:
      - ``cache`` dedupes within a single run() call.
      - ``persistent=True`` also writes ``node._cached_result`` for reuse.
      - ``persistent=False`` (inside foreach body) NEVER reads or writes
        ``node._cached_result`` — the sub_cache is fresh per row.
    """
    node_id = id(node)
    nt = node.node_type

    if nt == NodeType.ITEM:
        return node.execute_fn()

    if nt == NodeType.EXPR:
        child_vals = [_evaluate(dep, cache, persistent=persistent) for dep in node.inputs]
        return node.execute_fn(*child_vals)

    if nt == NodeType.COND:
        pred = node._cond_pred
        pred_val = (_evaluate(pred, cache, persistent=persistent)
                    if isinstance(pred, DataSpec) else pred)
        branch = node._cond_then if pred_val else node._cond_else
        return _evaluate(branch, cache, persistent=persistent)

    if nt == NodeType.FOREACH:
        if persistent and node_id in cache:
            return cache[node_id]
        source_rows = _evaluate(node._foreach_source, cache, persistent=persistent)
        if source_rows is None:
            source_rows = []
        uid = node._foreach_item_uid
        body = node._foreach_body
        out: list[Any] = []
        for row in source_rows:
            token = _push_frame({uid: row})
            try:
                sub_cache: dict[int, Any] = {}
                per_row = _evaluate(body, sub_cache, persistent=False)
            finally:
                _pop_frame(token)
            if isinstance(per_row, list):
                out.extend(per_row)
            elif per_row is not None:
                out.append(per_row)
        cache[node_id] = out
        if persistent:
            node._cached_result = out
            node._status = ExecutionStatus.COMPLETED
        return out

    # Regular nodes.
    if node_id in cache:
        return cache[node_id]
    if persistent and node.is_completed:
        cache[node_id] = node.result
        return node.result

    input_vals = [_evaluate(dep, cache, persistent=persistent) for dep in node.inputs]
    if persistent:
        node._status = ExecutionStatus.RUNNING
    try:
        result = node.execute_fn(*input_vals)
    except NotImplementedError:
        if persistent:
            node._status = ExecutionStatus.FAILED
        raise
    except Exception as e:
        if persistent:
            node._status = ExecutionStatus.FAILED
        raise RuntimeError(f"Error executing '{node.name}': {e}") from e

    cache[node_id] = result
    if persistent:
        node._cached_result = result
        node._status = ExecutionStatus.COMPLETED
    return result
