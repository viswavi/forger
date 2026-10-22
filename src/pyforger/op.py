"""Content-addressed disk cache + tracing for Forger ops.

Wrap a plain Python function with ``@op`` and it gains two things: its
output is memoized on disk keyed by a hash of its inputs, and every call
is recorded in a trace that can be inspected post-hoc.

.. warning::
    This module uses ``pickle`` for serialization. Only use this with a cache
    directory that is trusted and secured (e.g., restricted to the current user).
    Loading malicious pickle data can lead to arbitrary code execution.

Used by approach #6 (pipeline-as-function): pipelines are plain Python,
control flow is plain Python, but the expensive boundaries (LLM calls,
filtering passes) get cached at the @op layer.
"""

from __future__ import annotations

import difflib
import hashlib
import inspect
import os
import threading
import time
import types
from collections import deque
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from pyforger import _blobs, _tracking
from pyforger.utils import (
    canonical,
    code_names,
    function_key,
    get_formatted_logger,
    global_names,
    keyed_globals,
)

logger = get_formatted_logger("forger.op")

T = TypeVar("T")

# Part of every cache key. Bump it when what goes into keys changes, so entries
# written under the old layout are never read as hits. 2: plain-data globals
# are hashed by value; functions no longer carry their __module__ or docstrings.
# 3: keys are _KeyEncoder bytes, not a pickle of canonical(key).
_CACHE_KEY_VERSION = 3


def _cache_dir() -> Path:
    """Resolve the on-disk cache root, honoring ``$FORGER_CACHE`` per call.

    Read at call time (not import time) so test fixtures can redirect the
    cache via ``monkeypatch.setenv`` without reloading the module.
    """
    return Path(os.environ.get("FORGER_CACHE", Path.home() / ".cache" / "forger"))


class CallRecord(BaseModel):
    """CallRecord contains metadata about each @op call, for tracing and debugging."""

    op_name: str
    input_hash: str
    output_hash: str | None
    cache_hit: bool
    duration_s: float


_TRACE_DEFAULT_MAX = 10_000


def _trace_max() -> int:
    """We cap the number of trace records to observe -- 0 disables tracing."""
    # This environment variable must be set before importing Forger to take effect.
    raw = os.environ.get("PYFORGER_TRACE_MAX")
    if raw is None:
        return _TRACE_DEFAULT_MAX
    try:
        return max(0, int(raw))
    except ValueError:
        return _TRACE_DEFAULT_MAX


_TRACE: deque[CallRecord] = deque(maxlen=_trace_max())
_TRACE_LOCK = threading.Lock()


def trace() -> list[CallRecord]:
    """Return a copy of the in-memory trace of @op calls."""
    with _TRACE_LOCK:
        return list(_TRACE)


def clear_trace() -> None:
    """Drop all trace records. Does not affect the disk cache."""
    with _TRACE_LOCK:
        _TRACE.clear()


# Large str / bytes are hashed by a memoized digest, so time, not size, is what
# matters: 75 MiB of rows with 47K-char traces hash in about 0.1s.
_HASH_WARN_BYTES = int(os.environ.get("PYFORGER_HASH_WARN_BYTES", 1024 * 1024 * 1024))
_HASH_WARN_SECONDS = float(os.environ.get("PYFORGER_HASH_WARN_SECONDS", 1.0))
_HASH_WARNED: set[str] = set()
_HASH_WARN_LOCK = threading.Lock()


def _hash(obj: Any) -> tuple[str, int]:
    """Hash obj by value; equal values always hash equally.

    Returns the sha256 hex digest and the number of bytes hashed. The bytes are
    ``_KeyEncoder``'s encoding of obj, which does not depend on which equal
    strings happen to be the same object (a field name typed in the source and
    the same name read back from a cached result hash equally).
    """
    enc = _KeyEncoder()
    enc.feed(obj)
    payload = b"".join(enc.out)
    return hashlib.sha256(payload).hexdigest(), len(payload) + enc.digested


class _KeyEncoder:
    """Encode a value for its cache key as type-tagged, length-prefixed bytes.

    The same distinctions as ``canonical``, without building a canonical copy
    and pickling it: dict items and set elements are sorted by their encoding,
    so equal values encode equally whatever their order, and lists, tuples,
    sets, and frozensets have their own tags. A str or bytes of 4K or more is
    encoded as its sha256 (``b"S"`` + 32 bytes), so a 47K-char thinking trace
    is read once by sha256 and then memoized for later keys. A container seen
    earlier in the same key is copied from its first encoding. Other objects
    are encoded by ``canonical`` (e.g. a ``Model`` by its ``cache_key()``).
    """

    def __init__(self) -> None:
        self.out: list[bytes] = []
        self.digested = 0  # bytes of the large str / bytes hashed by digest
        # id(container) -> its slice of out. Ids are unique while the key is
        # encoded: every object is reachable from it or held in _keep.
        self._seen: dict[int, tuple[int, int]] = {}
        self._keep: list[Any] = []

    def feed(self, obj: Any) -> None:
        out = self.out
        tp = type(obj)
        # Exact types first: rows are dicts of str, int, and lists of those.
        if tp is str:
            if len(obj) >= _blobs.DIGEST_MIN_SIZE:
                self.digested += len(obj)
                out.append(b"S" + _blobs.digest(obj))
            else:
                data = obj.encode("utf-8", "surrogatepass")
                out.append(b"s%d:" % len(data))
                out.append(data)
        elif tp is dict or tp is list or tp is tuple:
            self._feed_container(obj, tp)
        elif tp is int:
            out.append(b"i%d;" % obj)
        elif obj is None:
            out.append(b"N")
        elif tp is bool:
            out.append(b"T" if obj else b"F")
        elif tp is float:
            out.append(b"f%s;" % float.__repr__(obj).encode())
        # Subclasses are keyed like their base type, as in canonical, e.g. a
        # StrEnum member like its str value and an OrderedDict like a dict.
        elif isinstance(obj, str):
            self.feed(str.__str__(obj))
        elif isinstance(obj, bytes):
            if len(obj) >= _blobs.DIGEST_MIN_SIZE:
                self.digested += len(obj)
                out.append(b"B" + _blobs.digest(obj))
            else:
                out.append(b"b%d:" % len(obj))
                out.append(bytes(obj))
        elif isinstance(obj, int):
            out.append(b"i%d;" % obj)
        elif isinstance(obj, float):
            out.append(b"f%s;" % float.__repr__(obj).encode())
        elif isinstance(obj, complex):
            out.append(b"c%s;" % complex.__repr__(obj).encode())
        elif isinstance(obj, dict):
            self._feed_container(obj, dict)
        elif isinstance(obj, list):
            self._feed_container(obj, list)
        elif isinstance(obj, tuple):
            self._feed_container(obj, tuple)
        elif isinstance(obj, (set, frozenset)):
            self._feed_container(obj, frozenset if isinstance(obj, frozenset) else set)
        else:
            canon = canonical(obj)
            self._keep.append(canon)
            out.append(b"o")
            self.feed(canon)

    def _feed_container(self, obj: Any, tp: type) -> None:
        out = self.out
        seen = self._seen.get(id(obj))
        if seen is not None:
            out.extend(out[seen[0] : seen[1]])
            return
        start = len(out)
        if tp is dict:
            out.append(b"d%d:" % len(obj))
            for k in obj:
                if type(k) is not str:
                    break
            else:
                # E.g. a row: str keys sort as themselves and are written inline.
                for k in sorted(obj):
                    if len(k) < _blobs.DIGEST_MIN_SIZE:
                        data = k.encode("utf-8", "surrogatepass")
                        out.append(b"s%d:" % len(data))
                        out.append(data)
                    else:
                        self.feed(k)
                    self.feed(obj[k])
                self._seen[id(obj)] = (start, len(out))
                return
            for k_bytes, v in sorted(
                ((self._alone(k), v) for k, v in obj.items()), key=lambda kv: kv[0]
            ):
                out.append(k_bytes)
                self.feed(v)
        elif tp is list or tp is tuple:
            out.append(b"%s%d:" % (b"l" if tp is list else b"t", len(obj)))
            for x in obj:
                self.feed(x)
        else:
            out.append(b"%s%d:" % (b"z" if tp is frozenset else b"e", len(obj)))
            out.extend(sorted(self._alone(x) for x in obj))
        self._seen[id(obj)] = (start, len(out))

    def _alone(self, obj: Any) -> bytes:
        """Encode obj on its own, for sorting dict keys and set elements."""
        sub = _KeyEncoder()
        sub._keep = self._keep
        sub.feed(obj)
        self.digested += sub.digested
        return b"".join(sub.out)


# True while an op called with ``_no_cache=True`` runs, so the ops it calls
# also skip the cache (e.g. by_llm_score's inner judge-score op).
_NO_CACHE_SCOPE: ContextVar[bool] = ContextVar("forger_no_cache_scope", default=False)


class _CallState:
    """Mutable per-call flags: ``skip_write`` keeps the result out of the cache.

    ``from_thread`` is set when the flag came from an op in a worker thread
    that did not copy the context (see ``_skip_running_calls``).
    """

    __slots__ = ("skip_write", "from_thread")

    def __init__(self) -> None:
        self.skip_write = False
        self.from_thread = False


# The innermost running op call (including a @step's ``step.*`` wrapper).
_CALL_STATE: ContextVar[_CallState | None] = ContextVar(
    "forger_call_state", default=None
)
# Every op call running in the process, in any thread.
_RUNNING_CALLS: set[_CallState] = set()
_RUNNING_CALLS_LOCK = threading.Lock()


def _skip_running_calls(origin: _CallState) -> None:
    """Keep every other running op and step call out of the cache.

    For an op that skipped its write with no enclosing call in its context: in
    a worker thread started without ``contextvars.copy_context().run``, the op
    or step that started the thread cannot be told apart from unrelated ones
    running at the same time, so none of them is cached.
    """
    with _RUNNING_CALLS_LOCK:
        for state in _RUNNING_CALLS:
            if state is not origin:
                state.skip_write = True
                state.from_thread = True


def _lost_context() -> bool:
    """True in a worker thread that did not copy the context (plain ``pool.submit``).

    Only there can an op with no enclosing call in its context have one. The
    main thread, a pipeline, and a step (one run with ``.submit`` too) keep
    their context, so an op there with no enclosing call has none.
    """
    if threading.current_thread() is threading.main_thread():
        return False
    return not _tracking.is_reporting()


def skip_cache_write() -> None:
    """Do not cache the result of the running op or of any op or step enclosing it.

    For an op that returns a result it should not replay, e.g. one with ``None``
    slots for prompts the provider failed on: the result is still returned, and
    the next call runs the op again. A no-op outside ops. In a worker thread
    that did not copy the context (plain ``pool.submit``), the enclosing calls
    cannot be found, so every op and step running in the process at that time
    is left uncached instead.
    """
    state = _CALL_STATE.get()
    if state is not None:
        state.skip_write = True


_STEP_META_SUFFIX = ".step.json"
# A filter's per-row scores, so a cache hit still shows its histogram.
_SCORES_SUFFIX = ".scores.json"
# What the call added to the pipeline's retrieval check (see _tracking.ReachLog).
_REACH_SUFFIX = ".reach.json"

# Set on every @op / tracked wrapper, e.g. {"name": "filter.by_length", "kind":
# "op", "cache": True}; scripts/gen_op_docs.py reads it to mark cached ops.
OP_INFO_ATTR = "__forger_op__"


class _RequestScope:
    """The @op call a Model's requests come from, for request-cache keys.

    ``key`` hashes the op name, its body, and every argument except the input
    rows (and the op's ``samples_arg``), so another op, or the same op with
    another template, never shares samples. ``rows_key`` hashes the whole rows
    argument, for prompts that name no row of their own. ``take_slots`` numbers
    completions in call order, so a prompt asked again in the same op call
    (a duplicate row, a retry, a later batch_size chunk) gets new slots.
    Both hashes are computed on first use, so ops that call no model pay
    nothing.
    """

    def __init__(self, op_name: str, parts: Any, rows: Any) -> None:
        """`parts`: what identifies the call besides the rows; `rows` or None."""
        self.op_name = op_name
        self._parts = parts
        self._rows = rows
        self._key: str | None = None
        self._rows_key: str | None = None
        self._lock = threading.Lock()
        self._next_slot: dict[str, int] = {}
        self.reused = 0
        self.total = 0

    @property
    def key(self) -> str:
        """Hash of the op name, body, and non-row arguments."""
        with self._lock:
            if self._key is None:
                self._key, _ = _hash(self._parts)
            return self._key

    @property
    def rows_key(self) -> str:
        """Hash of the whole rows argument; "(no rows)" for ops without one."""
        with self._lock:
            if self._rows_key is None:
                self._rows_key = (
                    "(no rows)" if self._rows is None else _hash(self._rows)[0]
                )
            return self._rows_key

    def take_slots(self, request_key: str, n: int) -> int:
        """First of the next `n` slots for `request_key` in this op call."""
        with self._lock:
            first = self._next_slot.get(request_key, 0)
            self._next_slot[request_key] = first + n
            return first

    def count(self, reused: int, total: int) -> None:
        """Add one generate_batch call's reused and requested completions."""
        with self._lock:
            self.reused += reused
            self.total += total


# The op call running in this context, read by Model.generate_batch.
_REQUEST_SCOPE: ContextVar[_RequestScope | None] = ContextVar(
    "forger_request_scope", default=None
)


def _scope_value(value: Any) -> Any:
    """An argument as the request scope sees it.

    A Model contributes only what can change its output
    (``request_scope_key()``), so changing its timeout, retries, or
    max_concurrency keeps the request cache.
    """
    scope_key = getattr(value, "request_scope_key", None)
    if callable(scope_key) and not isinstance(value, type):
        return ["__request_scope__", scope_key()]
    return value


def _request_scope(
    op_name: str,
    body_key: Any,
    normalized: dict[str, Any],
    args: tuple[Any, ...],
    samples_arg: str | None,
    bound: bool,
) -> _RequestScope:
    """The request scope of one op call; the rows are the first list argument.

    E.g. ``generate_from_context(rows, "Solve {q}", model)`` has scope
    ``[op name, body, {"prompt_fn": "Solve {q}", "model": ..., ...}]`` and rows
    ``rows``. Acquire ops take no rows, so all their arguments are in scope.
    """
    first = args[0] if args else None
    has_rows = isinstance(first, list) and _tracking.category_of(op_name) != "acquire"
    if not bound:  # no signature, e.g. a built-in: {"args": [...], "kwargs": {...}}
        positional = list(args[1:] if has_rows else args)
        named: dict[str, Any] = {
            "args": [_scope_value(v) for v in positional],
            "kwargs": {k: _scope_value(v) for k, v in normalized["kwargs"].items()},
        }
    else:
        names = list(normalized)
        rows_name = names[0] if has_rows and names else None
        named = {
            name: _scope_value(value)
            for name, value in normalized.items()
            if name not in (rows_name, samples_arg)
        }
    return _RequestScope(
        op_name, [op_name, body_key, named], first if has_rows else None
    )


def _maybe_warn_hash_cost(op_name: str, duration: float, size: int) -> None:
    if size <= _HASH_WARN_BYTES and duration <= _HASH_WARN_SECONDS:
        return
    with _HASH_WARN_LOCK:
        if op_name in _HASH_WARNED:
            return
        _HASH_WARNED.add(op_name)
    logger.warning(
        "@op(%s) cache-key hash took %.2fs over %d MiB of input; drop row fields"
        " that later steps do not need (e.g. a thinking trace), or pass large"
        " objects that define cache_key(), for a cheaper key.",
        op_name,
        duration,
        size // (1024 * 1024),
    )


# Globals of these types need no warning: loggers and typing objects (e.g. a
# TypeVar) do not change what an op returns.
_STABLE_TYPE_MODULES = ("logging", "typing", "typing_extensions")
# Keyed by name and code, like today: modules, functions, methods, and classes.
_NAMED_GLOBAL_TYPES = (
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    type,
)


def _warn_unstable_refs(fn: Callable[..., Any], op_name: str) -> None:
    """Warn if fn reads globals the cache key misses, env vars, or closure variables.

    We look for referents that can quietly change without changing the cache
    key, which leads to stale cached results. Plain-data globals (e.g. a prompt
    string or a config dict) and objects with ``cache_key()`` (e.g. a ``Model``)
    are hashed by value, and modules, functions, and classes by name, so they do
    not warn; a client, or a callable object without ``cache_key()``, does.
    Globals read in nested lambdas, comprehensions, and inner functions count.
    """
    code = fn.__code__
    g = fn.__globals__
    keyed = keyed_globals(fn)
    unhashed_globals = sorted(
        n
        for n in global_names(code)
        if n in g
        and n not in keyed
        and not (n.startswith("__") and n.endswith("__"))
        and not isinstance(g[n], _NAMED_GLOBAL_TYPES)
        and type(g[n]).__module__.split(".")[0] not in _STABLE_TYPE_MODULES
    )
    # Environment variables, e.g. os.environ["KEY"] or os.getenv("KEY").
    env_used = bool({"environ", "getenv"} & code_names(code))
    # We look for closure variables, e.g. `config["multiplier"]`` in
    # def new_op(x):
    #     return x * config["multiplier"]
    closure_vars = sorted(code.co_freevars) if fn.__closure__ else []
    if not (unhashed_globals or env_used or closure_vars):
        return
    parts = []
    if unhashed_globals:
        parts.append(f"reads globals {unhashed_globals}")
    if env_used:
        parts.append("reads os.environ/os.getenv")
    if closure_vars:
        parts.append(f"closes over {closure_vars}")
    logger.warning(
        "@op(%s) %s; changes to these may not invalidate the cache key, "
        "so cached results may be stale.",
        op_name,
        " and ".join(parts),
    )


def op(
    _fn: Callable[..., T] | None = None,
    *,
    name: str | None = None,
    cache: bool = True,
    samples_arg: str | None = None,
    unkeyed: tuple[str, ...] = (),
    category: str | None = None,
    smoke: tuple[str, ...] = (),
) -> Any:
    """Decorator: cache output on disk, content-addressed by inputs.

    ``samples_arg`` names the argument that only sets how many samples to
    draw (e.g. ``"n"``). It is left out of the request-cache scope, so raising
    it reuses the samples already drawn and requests only the extra ones.

    Usage::

        @op
        def by_rouge_l(items, *, field, threshold=0.7): ...

        @op(name="custom", cache=False)
        def debug_op(x): ...

        @op(unkeyed=("show_progress",))
        def score_all(items, *, show_progress=True): ...

    ``unkeyed`` names parameters left out of the cache key. Use it only for
    knobs that cannot change what the op returns, e.g. a progress bar, so
    ``score_all(rows, show_progress=False)`` is a cache hit after
    ``score_all(rows)``. A worker count can change results when the op has
    wall-clock timeouts (more workers, more contention, more timeouts), so
    Forger keys its own worker counts.

    Pass ``_no_cache=True`` at call time to bypass the cache for one call,
    including the ops that call makes. Pass ``_cache_only=True`` to return only
    an existing cached result, raising ``FileNotFoundError`` on a miss without
    executing the function.

    Pass ``min_yield=`` / ``max_yield=`` (fractions in [0, 1]) to check the
    ratio of output rows to input rows, e.g. ``filter.by_length(rows, ...,
    min_yield=0.5)``. A violation is a warning in the ``@step`` report (or a
    logged warning outside steps). Neither keyword is part of the cache key.
    ``acquire.*`` ops take no input rows and reject both keywords.

    ``category`` ("acquire", "alter", "filter", "inspect", or "other") is what
    ``@step`` / ``@pipeline`` checks see, e.g. "no quality filter ran". Ops
    defined in ``pyforger.<category>`` default to that category; any other op
    defaults to "other", whatever its name.

    In a limited run (``@pipeline(limit=3)`` or ``FORGER_LIMIT=3``) the
    cache key also holds the row limit, so limited results never mix with a
    full run's. ``smoke`` names count parameters capped to the limit: an int
    above it is lowered (``n=100`` → ``n=3``), a list is cut to its first
    rows, e.g. ``@op(name="acquire.sample_from_LLM", smoke=("n",))``. The
    request cache is not keyed on the limit, so the full run reuses the
    completions the limited run got for the same rows.
    """

    def decorate(fn: Callable[..., T]) -> Callable[..., T]:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            # Built-in functions and some callables don't have inspectable signatures.
            logger.debug("inspect.signature failed for %s: %s", fn, exc)
            if unkeyed:
                raise TypeError("unkeyed needs a function with a signature") from exc
            sig = None
        else:
            # These names are reserved for per-call cache controls and yield checks.
            assert sig is not None  # mypy hint
            for reserved in ("_no_cache", "_cache_only", "min_yield", "max_yield"):
                if reserved in sig.parameters:
                    raise TypeError(
                        "Forger functions decorated with @op cannot use the reserved "
                        + f"argument name '{reserved}'."
                    )
            for param in unkeyed:
                if param not in sig.parameters:
                    raise TypeError(f"unkeyed names {param!r}, not a parameter")
            _check_smoke_params(sig, smoke)

        op_name = name or f"{fn.__module__}.{fn.__qualname__}"
        op_category = _tracking.register_op(op_name, fn.__module__, category)
        # Sanitize op_name to be a safe directory name
        safe_op_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in op_name)
        # Code and closure are fixed here; plain-data globals are read per call.
        body_key = function_key(fn)
        if cache:
            _warn_unstable_refs(fn, op_name)

        @wraps(fn)
        def wrapper(
            *args: Any,
            _no_cache: bool = False,
            _cache_only: bool = False,
            min_yield: float | None = None,
            max_yield: float | None = None,
            **kwargs: Any,
        ) -> T:
            if _cache_only and (_no_cache or not cache):
                raise ValueError("_cache_only requires an enabled cache")
            if not _cache_only and _NO_CACHE_SCOPE.get():
                _no_cache = True
            _check_yield_bounds(min_yield, max_yield)
            if (
                min_yield is not None or max_yield is not None
            ) and op_category == "acquire":
                raise ValueError(
                    f"{op_name} takes no input rows, so min_yield / max_yield do "
                    "not apply; put them on the alter or filter op that follows"
                )
            args, kwargs = _smoke_capped(sig, smoke, args, kwargs)
            token = _tracking.enter_op(op_name)
            scope_token = _NO_CACHE_SCOPE.set(True) if _no_cache else None
            state = _CallState()
            parent_state = _CALL_STATE.get()
            state_token = _CALL_STATE.set(state)
            with _RUNNING_CALLS_LOCK:
                _RUNNING_CALLS.add(state)
            call: _tracking.OpCall | None = None
            rows: tuple[Any, Any] = (args[0] if args else None, None)
            start = time.perf_counter()
            try:
                result, cache_hit, first_arg, duration = _call(
                    args, kwargs, _no_cache, _cache_only, state
                )
                rows = (first_arg, result)
                call = _op_call(
                    op_name,
                    op_category,
                    first_arg,
                    result,
                    cache_hit=cache_hit,
                    duration=duration,
                    usage=token.usage_totals(),
                    scores=_tracking.current_scores(),
                    min_yield=min_yield,
                    max_yield=max_yield,
                    body_warnings=_tracking.current_warnings(),
                )
            except Exception as exc:
                call = _failed_call(op_name, args, exc, token, start)
                raise
            finally:
                _CALL_STATE.reset(state_token)
                with _RUNNING_CALLS_LOCK:
                    _RUNNING_CALLS.discard(state)
                # A result the op kept out of the cache makes every enclosing
                # op's (and step's) result stale too.
                if state.skip_write and parent_state is not None:
                    parent_state.skip_write = True
                elif state.skip_write and not state.from_thread and _lost_context():
                    _skip_running_calls(state)
                if scope_token is not None:
                    _NO_CACHE_SCOPE.reset(scope_token)
                _tracking.exit_op(token, call, rows)
            return result

        def _call(
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            _no_cache: bool,
            _cache_only: bool,
            state: _CallState,
        ) -> tuple[Any, bool, Any, float]:
            """Run or load the op; return (result, cache_hit, first arg, seconds)."""
            # perf_counter, like the step and pipeline times: time.time() also
            # counts a laptop's sleep, so an op could outlast its step.
            start = time.perf_counter()
            if sig is not None:
                try:
                    # Attempt to normalize the op function's arguments before hashing.
                    bound = sig.bind(*args, **kwargs)
                except TypeError as exc:
                    # fn would raise anyway; name the op and the likely fix. The
                    # "_" is stripped since e.g. topic_based_clustering passes its
                    # extra keywords on to the inner op _topic_based_clustering.
                    display = fn.__name__.lstrip("_")
                    raise _bind_error(display, sig, args, kwargs, exc) from None
                bound.apply_defaults()
                # Normalize the different ways of stating the same arguments.
                args = bound.args
                kwargs = bound.kwargs
                normalized = dict(bound.arguments)
                for param in unkeyed:
                    normalized.pop(param, None)
            else:
                normalized = {"args": list(args), "kwargs": dict(kwargs)}
            # E.g. a PROMPT constant edited since the last run changes the key.
            globals_key = keyed_globals(fn)
            key = [_CACHE_KEY_VERSION, op_name, body_key, globals_key, normalized]
            smoke_rows = _tracking.smoke_rows()
            if smoke_rows is not None:  # never read or overwrite a full run's entry
                key.append(["smoke", smoke_rows])
            t0 = time.perf_counter()
            input_hash, hash_size = _hash(key)
            _maybe_warn_hash_cost(op_name, time.perf_counter() - t0, hash_size)

            cache_root = _cache_dir()
            cache_path = (cache_root / safe_op_name / f"{input_hash}.pkl").resolve()

            # Ensure the cache path is within the cache root. This prevents users from
            # supplying cache paths that point outside Forger's directories.
            if cache_root.resolve() not in cache_path.parents:
                raise ValueError(f"Invalid cache path provided: {cache_path}")

            use_cache = cache and not _no_cache
            cache_hit = False
            is_step = op_name.startswith(_tracking.STEP_OP_PREFIX)
            meta_path = cache_path.with_name(input_hash + _STEP_META_SUFFIX)
            scores_path = cache_path.with_name(input_hash + _SCORES_SUFFIX)
            reach_path = cache_path.with_name(input_hash + _REACH_SUFFIX)

            if use_cache and cache_path.exists():
                try:
                    # Security: verify file ownership matches current process owner
                    if (
                        hasattr(os, "getuid")
                        and cache_path.stat().st_uid != os.getuid()
                    ):
                        raise PermissionError(
                            f"Cache file {cache_path} is owned by another user."
                        )

                    result = _read_result(cache_path, cache_root)
                    cache_hit = True
                    if is_step and meta_path.exists():
                        _tracking.restore_step_meta(meta_path.read_text())
                    elif not is_step and scores_path.exists():
                        _tracking.restore_scores(scores_path.read_text())
                except _blobs.MissingBlobError as exc:
                    # E.g. blobs/ was deleted but the entry was not: a miss.
                    logger.debug("cache entry %s is incomplete: %s", cache_path, exc)
                except Exception as exc:
                    if _cache_only:
                        raise RuntimeError(
                            f"Cannot read cached result for {op_name}"
                        ) from exc
                    # Cache file is unreadable or incompatible with the current
                    # code version; fall through to executing the op directly.
                    # The cache will be overwritten with a new result.
                    logger.debug(
                        "cache read failed for %s: %s: %s",
                        cache_path,
                        type(exc).__name__,
                        exc,
                    )
            if cache_hit:
                _replay_reach(reach_path)

            if not cache_hit:
                if _cache_only:
                    raise FileNotFoundError(
                        f"No cached result for {op_name} with these arguments"
                    )
                scope = _request_scope(
                    op_name, body_key, normalized, args, samples_arg, sig is not None
                )
                scope_token = _REQUEST_SCOPE.set(scope)
                try:
                    with _tracking.reach_scope() as reach:
                        result = fn(*args, **kwargs)
                finally:
                    _REQUEST_SCOPE.reset(scope_token)
                    # Also "0 of N": a rerun that reuses nothing usually means
                    # a key changed, e.g. a row holding id(row) or a timestamp.
                    if scope.total:
                        logger.info(
                            "request cache: %d of %d completions reused in %s",
                            scope.reused,
                            scope.total,
                            op_name,
                        )
                meta = None
                if is_step:
                    # Raises before caching when a @step breaks its contract.
                    meta = _tracking.check_step_result(result)

                if use_cache and state.skip_write:
                    logger.debug("not caching %s %s", op_name, input_hash[:8])
                    step_rec = _tracking.active_step() if is_step else None
                    if state.from_thread and step_rec is not None:
                        step_rec.warnings.append(
                            "not cached: an op in a worker thread kept its result "
                            "out of the cache (e.g. provider errors)"
                        )
                elif use_cache:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    if meta is not None:
                        # Written first, so a cached step always has its ops.
                        _atomic_write(meta_path, meta.encode())
                    if not is_step:
                        scores = _tracking.scores_meta(_tracking.current_scores())
                        if scores is not None:
                            _atomic_write(scores_path, scores.encode())
                        else:  # no stale scores from an earlier result
                            scores_path.unlink(missing_ok=True)
                    if reach is not None:
                        # Written even when empty: a missing file means the
                        # result was cached outside a pipeline or by an older
                        # Forger, so its prompts are unknown (see _replay_reach).
                        _atomic_write(reach_path, reach.dumps().encode())
                    else:  # no log from an earlier run describing another result
                        reach_path.unlink(missing_ok=True)
                    _write_result(cache_path, result, cache_root)

            duration = time.perf_counter() - start
            # If tracing is enabled, record this call on the in-memory trace.
            if _TRACE.maxlen is not None and _TRACE.maxlen > 0:
                output_hash = None if cache_hit else _hash(result)[0]
                with _TRACE_LOCK:
                    _TRACE.append(
                        CallRecord(
                            op_name=op_name,
                            input_hash=input_hash,
                            output_hash=output_hash,
                            cache_hit=cache_hit,
                            duration_s=duration,
                        )
                    )

            if cache_hit:
                logger.debug("cache HIT %s %s", op_name, input_hash[:8])
            else:
                logger.debug(
                    "cache MISS %s %s (%.2fs)",
                    op_name,
                    input_hash[:8],
                    duration,
                )

            return result, cache_hit, (args[0] if args else None), duration

        setattr(wrapper, OP_INFO_ATTR, {"name": op_name, "kind": "op", "cache": cache})
        return wrapper

    if _fn is not None:
        return decorate(_fn)
    return decorate


# Keywords every @op call accepts on top of the function's own parameters.
_CALL_KEYWORDS = ("min_yield", "max_yield", "_no_cache", "_cache_only")


def _suggest(key: str, candidates: list[str]) -> list[str]:
    """Return likely intended names, e.g. 'max_workers' -> ['verifier_workers']."""
    close = difflib.get_close_matches(key, candidates, n=3)
    # Same last word, which difflib scores low for long prefixes.
    last = key.rsplit("_", 1)[-1]
    if len(last) >= 4:
        close += [c for c in candidates if c.rsplit("_", 1)[-1] == last]
    return list(dict.fromkeys(close))[:3]


def _bind_error(
    display: str,
    sig: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    exc: TypeError,
) -> TypeError:
    """Explain a call that does not match the op's signature.

    E.g. ``by_pass_at_k() got an unexpected keyword argument 'max_workers'; did
    you mean 'verifier_workers'? Accepted keywords: rows, model, ...``.
    """
    params = list(sig.parameters.values())
    keywords = [
        p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    accepted = (
        f" Accepted keywords: {', '.join(keywords) or '(none)'} "
        f"(plus {', '.join(_CALL_KEYWORDS)})."
    )
    candidates = keywords + list(_CALL_KEYWORDS)
    takes_any = any(p.kind == p.VAR_KEYWORD for p in params)
    unknown = [] if takes_any else [k for k in kwargs if k not in candidates]
    if len(unknown) == 1:
        message = f"{display}() got an unexpected keyword argument {unknown[0]!r}"
        close = _suggest(unknown[0], candidates)
        if close:
            message += f"; did you mean {' or '.join(map(repr, close))}?"
        else:
            message += "."
        return TypeError(message + accepted)
    if unknown:
        message = (
            f"{display}() got unexpected keyword arguments "
            f"{', '.join(map(repr, unknown))}."
        )
        for key in unknown:
            close = _suggest(key, candidates)
            if close:
                message += (
                    f" For {key!r}, did you mean {' or '.join(map(repr, close))}?"
                )
        return TypeError(message + accepted)
    try:
        bound = sig.bind_partial(*args, **kwargs)
    except TypeError:
        return TypeError(f"{display}(): {exc}." + accepted)
    missing = [
        p
        for p in params
        if p.name not in bound.arguments
        and p.default is p.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    if not missing:
        return TypeError(f"{display}(): {exc}." + accepted)
    names = ", ".join(
        repr(p.name) + (" (keyword-only)" if p.kind == p.KEYWORD_ONLY else "")
        for p in missing
    )
    noun = "argument" if len(missing) == 1 else "arguments"
    return TypeError(f"{display}() missing required {noun}: {names}." + accepted)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so concurrent readers never see a partial file."""
    tmp_path = path.parent / f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)


def _write_result(cache_path: Path, result: Any, cache_root: Path) -> None:
    """Pickle result to cache_path, its large values to ``cache_root/blobs``."""
    _atomic_write(cache_path, _blobs.dumps(result, cache_root / "blobs"))


def _replay_reach(path: Path) -> None:
    """Replay what a cached call added to the pipeline's retrieval check."""
    if not _tracking.reach_active():
        return
    try:
        text = path.read_text()
    except OSError:
        # Cached outside a pipeline or before this check existed: its prompts
        # are unknown, so the pipeline must not claim retrieved text went unused.
        _tracking.mark_reach_blind()
        return
    _tracking.replay_reach(text)


def _read_result(cache_path: Path, cache_root: Path) -> Any:
    """Load what ``_write_result`` wrote, or an entry pickled without blobs."""
    return _blobs.loads(cache_path.read_bytes(), cache_root / "blobs")


def tracked(
    op_name: str, *, smoke: tuple[str, ...] = ()
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Report calls in ``@step`` reports like an op, without caching or hashing.

    For functions whose arguments cannot be hashed into a cache key, e.g.
    ``acquire.retrieve`` (a retrieval source holds a client and credentials),
    and for cheap ops not worth caching, e.g. ``filter.by_predicate``. Calls
    take ``min_yield=`` / ``max_yield=`` like ``@op`` calls; ``_no_cache=True``
    is accepted (nothing is cached) and ``_cache_only=True`` raises.
    ``smoke`` names count parameters capped in a smoke run, as for ``@op``.
    """

    def decorate(fn: Callable[..., T]) -> Callable[..., T]:
        sig = inspect.signature(fn)
        for reserved in ("_no_cache", "_cache_only", "min_yield", "max_yield"):
            if reserved in sig.parameters:
                raise TypeError(
                    "Forger functions decorated with @tracked cannot use the "
                    f"reserved argument name '{reserved}'."
                )
        _check_smoke_params(sig, smoke)
        op_category = _tracking.register_op(op_name, fn.__module__)

        @wraps(fn)
        def wrapper(
            *args: Any,
            _no_cache: bool = False,
            _cache_only: bool = False,
            min_yield: float | None = None,
            max_yield: float | None = None,
            **kwargs: Any,
        ) -> T:
            if _cache_only:
                raise ValueError(f"_cache_only: {op_name} is never cached")
            _check_yield_bounds(min_yield, max_yield)
            if (min_yield is not None or max_yield is not None) and (
                _tracking.category_of(op_name) == "acquire"
            ):
                raise ValueError(
                    f"{op_name} takes no input rows, so min_yield / max_yield do "
                    "not apply; put them on the alter or filter op that follows"
                )
            args, kwargs = _smoke_capped(sig, smoke, args, kwargs)
            # The rows, also when passed by keyword: by_predicate(items=rows, ...).
            try:
                first_arg = next(iter(sig.bind(*args, **kwargs).arguments.values()))
            except (TypeError, StopIteration):
                first_arg = args[0] if args else None
            token = _tracking.enter_op(op_name)
            start = time.perf_counter()
            call: _tracking.OpCall | None = None
            rows: tuple[Any, Any] = (first_arg, None)
            try:
                result = fn(*args, **kwargs)
                rows = (rows[0], result)
                call = _op_call(
                    op_name,
                    op_category,
                    first_arg,
                    result,
                    cache_hit=False,
                    duration=time.perf_counter() - start,
                    usage=token.usage_totals(),
                    scores=_tracking.current_scores(),
                    min_yield=min_yield,
                    max_yield=max_yield,
                    body_warnings=_tracking.current_warnings(),
                )
            except Exception as exc:
                call = _failed_call(op_name, args, exc, token, start)
                raise
            finally:
                _tracking.exit_op(token, call, rows)
            return result

        setattr(wrapper, OP_INFO_ATTR, {"name": op_name, "kind": "tracked"})
        return wrapper

    return decorate


def _check_smoke_params(sig: inspect.Signature | None, smoke: tuple[str, ...]) -> None:
    """Raise TypeError when ``smoke`` names a parameter the function lacks."""
    missing = [p for p in smoke if sig is None or p not in sig.parameters]
    if missing:
        raise TypeError(f"smoke= names parameters the function lacks: {missing}")


def _smoke_capped(
    sig: inspect.Signature | None,
    smoke: tuple[str, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Cap the ``smoke`` parameters to the smoke run's row limit, if one is active.

    An int above the limit becomes the limit (``n=100`` → ``n=3``); a list is
    cut to its first rows. Calls whose arguments do not bind pass unchanged.
    """
    if not smoke or sig is None:
        return args, kwargs
    limit = _tracking.smoke_rows()
    if limit is None:
        return args, kwargs
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError:
        return args, kwargs  # the call itself raises
    bound.apply_defaults()
    for param in smoke:
        value = bound.arguments.get(param)
        if isinstance(value, int) and not isinstance(value, bool) and value > limit:
            bound.arguments[param] = limit
        elif isinstance(value, list) and len(value) > limit:
            bound.arguments[param] = value[:limit]
    return bound.args, bound.kwargs


def _failed_call(
    op_name: str,
    args: tuple[Any, ...],
    exc: Exception,
    token: _tracking.OpToken,
    start: float,
) -> _tracking.OpCall | None:
    """The step-report entry for an op that raised, carrying its usage so far.

    A step that catches the error then still shows what the op spent.
    """
    if token.is_step:
        return None
    category = _tracking.category_of(op_name)
    first = args[0] if args else None
    usage = token.usage_totals()
    error = f"{type(exc).__name__}: {exc}"
    if len(error) > 120:
        error = error[:119] + "…"
    return _tracking.OpCall(
        op_name=op_name,
        category=category,
        n_in=len(first) if isinstance(first, list) and category != "acquire" else None,
        n_out=None,
        yield_ratio=None,
        cache_hit=False,
        duration_s=time.perf_counter() - start,
        prompt_tokens=usage[0],
        completion_tokens=usage[1],
        cost_usd=usage[2],
        token_units=usage[3],
        error=error,
        warnings=[f"{op_name} raised {error}"],
    )


def _check_yield_bounds(min_yield: float | None, max_yield: float | None) -> None:
    """Raise ValueError unless 0 <= min_yield <= max_yield <= 1 (either optional)."""
    for label, bound in (("min_yield", min_yield), ("max_yield", max_yield)):
        if bound is not None and not 0.0 <= bound <= 1.0:
            raise ValueError(f"{label} must be in [0, 1], got {bound}")
    if min_yield is not None and max_yield is not None and min_yield > max_yield:
        raise ValueError(
            f"min_yield ({min_yield}) must not exceed max_yield ({max_yield})"
        )


# Input rows under which a limited run's yield misses are only suggestions.
_SMALL_SAMPLE = 20


def _op_call(
    op_name: str,
    category: str,
    first_arg: Any,
    result: Any,
    *,
    cache_hit: bool,
    duration: float,
    usage: tuple[int, int, float, float],
    scores: list[_tracking.ScoreRecord],
    min_yield: float | None,
    max_yield: float | None,
    body_warnings: list[str] | None = None,
) -> _tracking.OpCall:
    """Build the step-report entry for one op call and apply its yield bounds.

    ``body_warnings`` come from ``_tracking.add_op_warning`` in the op's body;
    the op logged them already, so they are only added to the report.
    """
    # An acquire op's first argument is a prompt or query, not input rows.
    n_in = (
        len(first_arg)
        if isinstance(first_arg, list) and category != "acquire"
        else None
    )
    n_out = len(result) if isinstance(result, list) else None
    yield_ratio = n_out / n_in if n_in and n_out is not None else None
    call = _tracking.OpCall(
        op_name=op_name,
        category=category,
        n_in=n_in,
        n_out=n_out,
        yield_ratio=yield_ratio,
        min_yield=min_yield,
        max_yield=max_yield,
        cache_hit=cache_hit,
        duration_s=duration,
        prompt_tokens=usage[0],
        completion_tokens=usage[1],
        cost_usd=usage[2],
        token_units=usage[3],
        scores=scores[0] if scores else None,
    )
    # A limited run's yield on a few rows (limit=3) says little, so a miss
    # there is only a suggestion; on 20 or more rows (limit=50) it counts.
    small = _tracking.smoke_rows() is not None and (n_in or 0) < _SMALL_SAMPLE
    missed = call.suggestions if small else call.warnings
    prefix = "limited run: " if small else ""
    if (min_yield is not None or max_yield is not None) and yield_ratio is None:
        if n_in == 0 and n_out is not None:
            if min_yield:
                missed.append(
                    f"{prefix}{op_name}: no input rows, so yield is 0 and below "
                    f"min_yield {min_yield:.2f}"
                )
        else:
            call.warnings.append(
                f"{op_name}: yield unchecked; min_yield / max_yield need a list of "
                "rows in and a list out"
            )
    if yield_ratio is not None:
        if min_yield is not None and yield_ratio < min_yield:
            missed.append(
                f"{prefix}{op_name}: yield {yield_ratio:.2f} below min_yield "
                f"{min_yield:.2f}"
            )
        if max_yield is not None and yield_ratio > max_yield:
            missed.append(
                f"{prefix}{op_name}: yield {yield_ratio:.2f} above max_yield "
                f"{max_yield:.2f}"
            )
    # Completions cut off at max_tokens, from any model call the op made.
    call.warnings.extend(_tracking.truncation_warnings(op_name))
    if not _tracking.is_reporting():
        for warning in call.warnings:
            logger.warning(warning)
    call.warnings.extend(body_warnings or [])
    return call
