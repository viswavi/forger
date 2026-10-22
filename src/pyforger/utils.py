"""This module contains helper functions used across Forger."""

from __future__ import annotations

import dis
import functools
import inspect
import logging
import os
import pickle
import sys
import threading
import types
from typing import Any


class _ForgerStreamHandler(logging.StreamHandler):
    """Forger's stderr handler; it stays quiet when another handler prints the record.

    Forger loggers propagate, so a handler on the root logger (from
    ``logging.basicConfig()``, or one a library adds) gets every Forger record too.
    If that handler writes to the terminal, it prints the line in its own format and
    this handler skips it, so the line appears once. Handlers that write elsewhere
    (a log file, pytest's ``caplog``) do not count: this handler still prints.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Print the record unless another terminal handler will."""
        if not self._printed_elsewhere(record):
            super().emit(record)

    def _printed_elsewhere(self, record: logging.LogRecord) -> bool:
        streams = (self.stream, sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
        terminals = {id(s) for s in streams if s is not None}
        logger: logging.Logger | None = logging.getLogger(record.name)
        # the handlers Logger.callHandlers reaches: up the parents while they propagate
        while logger is not None:
            for handler in logger.handlers:
                if isinstance(handler, _ForgerStreamHandler):
                    continue
                if not isinstance(handler, logging.StreamHandler):
                    continue
                if record.levelno < handler.level:
                    continue
                if id(handler.stream) in terminals or _isatty(handler.stream):
                    return True
            logger = logger.parent if logger.propagate else None
        return False


def _isatty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        # no isatty(), or a closed file
        return False


def get_formatted_logger(logger_name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a formatted logger.

    The logger prints to stderr through its own handler and also propagates, so
    handlers the application adds to the root logger receive its records. When such
    a handler already prints to the terminal, Forger's handler skips the record, so
    each line appears once, in the application's format.

    Args:
        logger_name: The name of the logger, usually the name
            of the component that uses the logger.
        level: The logging level (by default, INFO).

    Returns:
        A logger object.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # Check if the logger already has a StreamHandler to prevent adding another one.
    if not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
        handler = _ForgerStreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_field(item: dict, field: str, caller: str) -> Any:
    """Retrieve a field from a dict item, raising a clear error if missing.

    Args:
        item: The dictionary to look up.
        field: The key to retrieve.
        caller: Name of the calling function, used in the error message.

    Raises:
        KeyError: If the field is not present in the item.
    """
    try:
        return item[field]
    except KeyError:
        raise KeyError(
            f"{caller}: item is missing required field '{field}'. "
            f"Available fields: {list(item.keys())}"
        )


_CODE_T = type((lambda: None).__code__)

# Python 3.14+ flags code whose co_consts[0] is its docstring.
_CO_HAS_DOCSTRING = getattr(inspect, "CO_HAS_DOCSTRING", None)
# Their co_consts[0] is an ordinary constant, e.g. "pre" in ("pre" + s for s in x).
_NO_DOCSTRING_NAMES = ("<genexpr>", "<listcomp>", "<setcomp>", "<dictcomp>")


def _has_docstring(code: Any) -> bool:
    """True when ``code.co_consts[0]`` is the function's docstring.

    In 3.12 and 3.13 the compiler puts a function's docstring, or None when it
    has none, first in co_consts, so ``def f(): return "x"`` has consts
    ``(None, "x")``. Class bodies (no CO_OPTIMIZED) and generator expressions
    do not reserve that slot.
    """
    if not code.co_consts:
        return False
    if _CO_HAS_DOCSTRING is not None:
        return bool(code.co_flags & _CO_HAS_DOCSTRING)
    return (
        bool(code.co_flags & inspect.CO_OPTIMIZED)
        and code.co_name not in _NO_DOCSTRING_NAMES
        and isinstance(code.co_consts[0], str)
    )


def _code_key(code: Any) -> list[Any]:
    """Behavior-only key for a code object (to avoid repeating code execution).

    Docstrings are left out (keyed as None, like a function without one), so
    fixing a typo in a step's description keeps its cache.
    """
    consts = [
        _code_key(c) if isinstance(c, _CODE_T) else canonical(c) for c in code.co_consts
    ]
    if _has_docstring(code):
        consts[0] = None
    return [
        code.co_code,
        consts,
        list(code.co_names),
        list(code.co_varnames),
        list(code.co_freevars),
        code.co_argcount,
    ]


# LOAD_NAME and LOAD_FROM_DICT_OR_GLOBALS read globals from class bodies.
_GLOBAL_LOADS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"})


@functools.lru_cache(maxsize=4096)
def global_names(code: Any) -> tuple[str, ...]:
    """Sorted names that a code object and the code nested in it load as globals.

    Nested code covers lambdas, comprehensions, and inner functions. Only
    global loads count: ``r.split()`` does not read a global named ``split``.
    """
    names = {
        ins.argval for ins in dis.get_instructions(code) if ins.opname in _GLOBAL_LOADS
    }
    for const in code.co_consts:
        if isinstance(const, _CODE_T):
            names.update(global_names(const))
    return tuple(sorted(names))


@functools.lru_cache(maxsize=4096)
def global_attrs(code: Any) -> tuple[tuple[str, str], ...]:
    """Sorted ``(global, attribute)`` pairs read as ``name.attr`` in code.

    Nested code counts, as in ``global_names``.

    E.g. ``harness.make_row(r)`` gives ``("harness", "make_row")``.
    """
    pairs: set[tuple[str, str]] = set()
    previous = None
    for ins in dis.get_instructions(code):
        if ins.opname == "LOAD_ATTR" and previous is not None:
            pairs.add((previous, ins.argval))
        previous = ins.argval if ins.opname in _GLOBAL_LOADS else None
    for const in code.co_consts:
        if isinstance(const, _CODE_T):
            pairs.update(global_attrs(const))
    return tuple(sorted(pairs))


@functools.lru_cache(maxsize=4096)
def code_names(code: Any) -> frozenset[str]:
    """All of ``co_names`` (globals and attributes) in code and its nested code."""
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, _CODE_T):
            names.update(code_names(const))
    return frozenset(names)


_PLAIN_SCALARS = (type(None), bool, int, float, complex, str, bytes)


def is_plain_data(obj: Any, _seen: set[int] | None = None) -> bool:
    """True for None, bool, numbers, str, bytes, and containers of those.

    E.g. ``"Answer briefly."``, ``{"easy": [1, 2]}``, ``frozenset({"a"})``; not
    a logger, a model, or a list holding a function. A cyclic list is not plain.
    """
    if isinstance(obj, _PLAIN_SCALARS):
        return True
    if not isinstance(obj, (list, tuple, set, frozenset, dict)):
        return False
    seen = set() if _seen is None else _seen
    if id(obj) in seen:
        return False
    seen.add(id(obj))
    items = [*obj.keys(), *obj.values()] if isinstance(obj, dict) else obj
    plain = all(is_plain_data(x, seen) for x in items)
    seen.discard(id(obj))
    return plain


def has_cache_key(obj: Any) -> bool:
    """True for an instance with a ``cache_key()`` method, e.g. a ``Model``."""
    return not isinstance(obj, (type, types.ModuleType)) and callable(
        getattr(obj, "cache_key", None)
    )


def keyed_globals(fn: Any) -> dict[str, Any]:
    """The module globals ``fn``'s code reads that go into its cache key, by name.

    These are plain data (``PROMPT = "..."``), objects with ``cache_key()``
    (``MODEL = Model(...)``), and your own functions that ``fn`` calls by name
    (``tag(row)``), keyed by their code and in turn by the globals they read.
    Dunder names (``__name__``, ``__file__``) are skipped: they differ between
    ``python pipeline.py`` and ``import pipeline`` without changing behavior.
    """
    code = getattr(fn, "__code__", None)
    g = getattr(fn, "__globals__", None)
    if code is None or not isinstance(g, dict):
        return {}
    keyed: dict[str, Any] = {}
    mine: bool | None = None  # is fn your code? Forger's ops key no helpers
    for n in global_names(code):
        if n not in g or (n.startswith("__") and n.endswith("__")):
            continue
        if is_plain_data(g[n]) or has_cache_key(g[n]):
            keyed[n] = g[n]
        elif isinstance(g[n], (types.FunctionType, types.ModuleType)):
            if mine is None:
                mine = is_user_function(fn)
            if not mine:
                continue
            if is_user_function(g[n]):
                keyed[n] = _helper_key(inspect.unwrap(g[n]))
            elif _is_user_module(g[n]):  # harness.make_row(r), harness.PROMPT
                for name, attr in global_attrs(code):
                    value = getattr(g[n], attr, None) if name == n else None
                    if value is not None and is_user_function(value):
                        keyed[f"{n}.{attr}"] = _helper_key(inspect.unwrap(value))
                    elif value is not None and (
                        is_plain_data(value) or has_cache_key(value)
                    ):
                        keyed[f"{n}.{attr}"] = value
    return keyed


# Python file paths of the standard library and installed packages: their
# functions are keyed by name, yours by code (see ``is_user_function``).
_LIBRARY_PREFIXES = tuple(
    {os.path.realpath(p) + os.sep for p in (sys.prefix, sys.base_prefix)}
)
# ids of the helpers being keyed in this thread, so ``def fib(n): ... fib(n-1)``
# or two helpers calling each other key the inner reference by name.
_KEYING = threading.local()
_WARNED_HELPERS: set[str] = set()


def is_user_function(obj: Any) -> bool:
    """True for a Python function you wrote, in your pipeline file or another.

    E.g. ``tag`` in ``rows = [tag(r) for r in rows]``. False for functions of
    the standard library, installed packages, and Forger itself, whose code
    does not change between your runs. ``@op`` and ``@step`` functions count,
    by the function they wrap.
    """
    if not isinstance(obj, types.FunctionType):
        return False
    fn = inspect.unwrap(obj)
    path = getattr(getattr(fn, "__code__", None), "co_filename", "")
    return _is_user_code(getattr(fn, "__module__", None) or "", path)


def _is_user_module(obj: Any) -> bool:
    """True for a module you wrote, e.g. ``harness`` in ``import harness``."""
    return isinstance(obj, types.ModuleType) and _is_user_code(
        obj.__name__, getattr(obj, "__file__", None) or ""
    )


def _is_user_code(module: str, path: str) -> bool:
    """True unless ``path`` is in the standard library, a package, or Forger."""
    if module == "pyforger" or module.startswith("pyforger."):
        return False
    if not path or path.startswith("<frozen"):
        return False
    if path.startswith("<"):  # e.g. <stdin> or a notebook cell
        return True
    real = os.path.realpath(path)
    return not real.startswith(_LIBRARY_PREFIXES) and not any(
        part in ("site-packages", "dist-packages") for part in real.split(os.sep)
    )


def _helper_key(fn: Any) -> Any:
    """The key of a helper ``fn``: its code, closure, and keyed globals.

    A helper already being keyed (recursion) is keyed by name. A helper whose
    closure cannot be keyed is keyed by name too, with a warning once, as
    before helpers were keyed by code.
    """
    active = getattr(_KEYING, "ids", None)
    if active is None:
        active = _KEYING.ids = set()
    name = getattr(fn, "__qualname__", repr(fn))
    if id(fn) in active:
        return ["__recursive__", name]
    active.add(id(fn))
    try:
        return [*function_key(fn), canonical(keyed_globals(fn))]
    except TypeError as exc:
        if name not in _WARNED_HELPERS:
            _WARNED_HELPERS.add(name)
            logging.getLogger("forger.op").warning(
                "@op cannot key helper %s by its code (%s); changes to its body "
                "will not invalidate cached results.",
                name,
                exc,
            )
        return ["__callable__", name]
    finally:
        active.discard(id(fn))


def function_key(fn: Any) -> list[Any]:
    """Key a Python function by qualname, code, and closure values, not globals.

    No ``__module__``: a step run as ``__main__`` and the same step imported
    from pipeline.py share keys. ``canonical`` adds ``keyed_globals(fn)``, so
    two same-named functions reading different PROMPT strings still differ.
    """
    closure = getattr(fn, "__closure__", None) or ()
    return [
        "__callable__",
        getattr(fn, "__qualname__", repr(fn)),
        _code_key(fn.__code__),
        [canonical(cell.cell_contents) for cell in closure],
    ]


_WARNED_INSTANCE_TYPES: set[str] = set()


def _callable_instance_key(obj: Any) -> list[Any]:
    """Key an object with ``__call__`` by its type, ``__call__`` code, and state.

    E.g. ``AtLeast(3)`` and ``AtLeast(7)`` differ by ``{"t": 3}`` vs ``{"t": 7}``.
    State that is not plain data is pickled, with a warning once per type.
    """
    cls = type(obj)
    call = getattr(cls, "__call__", None)
    state = getattr(obj, "__dict__", None)
    if is_plain_data(state):
        state_key = canonical(state)
    else:
        name = f"{cls.__module__}.{cls.__qualname__}"
        if name not in _WARNED_INSTANCE_TYPES:
            _WARNED_INSTANCE_TYPES.add(name)
            logging.getLogger("forger.op").warning(
                "@op keys callable %s by pickling its state, which is not plain "
                "data; add a cache_key() method to it for a stable key.",
                name,
            )
        state_key = _pickle_key(obj)
    return [
        "__instance__",
        cls.__qualname__,
        canonical(call) if getattr(call, "__code__", None) is not None else None,
        state_key,
    ]


def _pickle_key(obj: Any) -> bytes:
    """Pickle obj for its key, or raise a TypeError that suggests cache_key()."""
    try:
        return pickle.dumps(obj, protocol=4)
    except Exception as e:
        raise TypeError(
            f"@op cannot derive a stable cache key for {type(obj).__name__}: "
            f"{e}. Add a cache_key() method to this type."
        ) from e


def canonical(obj: Any) -> Any:
    """Convert obj to a stable, picklable structure for hashing.

    Tuples, dicts, and sets are wrapped as ``(type_tag, items)`` so different
    container types never collide.
    """
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    if isinstance(obj, list):
        return [canonical(x) for x in obj]
    if isinstance(obj, tuple):
        return ("__tuple__", [canonical(x) for x in obj])
    if isinstance(obj, dict):
        items = [(canonical(k), canonical(v)) for k, v in obj.items()]
        items.sort(key=lambda item: repr(item[0]))
        return ("__dict__", items)
    if isinstance(obj, (set, frozenset)):
        items = [canonical(x) for x in obj]
        items.sort(key=repr)
        return ("__frozenset__" if isinstance(obj, frozenset) else "__set__", items)
    if isinstance(obj, type):
        return f"<type {obj.__module__}.{obj.__qualname__}>"
    if hasattr(obj, "cache_key") and callable(obj.cache_key):
        return ["__cache_key__", canonical(obj.cache_key())]
    if callable(obj):
        if getattr(obj, "__code__", None) is not None:
            return [*function_key(obj), canonical(keyed_globals(obj))]
        if isinstance(obj, functools.partial):
            # E.g. partial(at_least, t=5): same function and arguments, same key.
            return [
                "__partial__",
                canonical(obj.func),
                canonical(obj.args),
                canonical(obj.keywords),
            ]
        qualname = getattr(obj, "__qualname__", None)
        if isinstance(qualname, str):
            # Builtins have no code to tell them apart, e.g. math.sqrt, cmath.sqrt.
            return ["__callable__", f"{getattr(obj, '__module__', '?')}.{qualname}"]
        return _callable_instance_key(obj)
    return _pickle_key(obj)
