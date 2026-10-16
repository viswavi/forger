"""This module contains helper functions used across Forger."""

from __future__ import annotations

import functools
import logging
import pickle
from typing import Any


def get_formatted_logger(logger_name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a formatted logger.

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
        handler = logging.StreamHandler()
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


def _code_key(code: Any) -> list[Any]:
    """Behavior-only key for a code object (to avoid repeating code execution)."""
    consts = [
        _code_key(c) if isinstance(c, _CODE_T) else canonical(c) for c in code.co_consts
    ]
    return [
        code.co_code,
        consts,
        list(code.co_names),
        list(code.co_varnames),
        list(code.co_freevars),
        code.co_argcount,
    ]


@functools.lru_cache(maxsize=4096)
def global_names(code: Any) -> tuple[str, ...]:
    """Sorted names that a code object and the code nested in it may read as globals.

    Nested code covers lambdas, comprehensions, and inner functions.
    ``co_names`` also holds attribute names (``x.split`` adds ``split``), so
    callers look each name up in the function's globals and skip misses.
    """
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, _CODE_T):
            names.update(global_names(const))
    return tuple(sorted(names))


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


def plain_globals(fn: Any) -> dict[str, Any]:
    """The plain-data module globals ``fn``'s code reads, by name.

    Dunder names (``__name__``, ``__file__``) are skipped: they differ between
    ``python pipeline.py`` and ``import pipeline`` without changing behavior.
    """
    code = getattr(fn, "__code__", None)
    g = getattr(fn, "__globals__", None)
    if code is None or not isinstance(g, dict):
        return {}
    return {
        n: g[n]
        for n in global_names(code)
        if n in g
        and not (n.startswith("__") and n.endswith("__"))
        and is_plain_data(g[n])
    }


def function_key(fn: Any) -> list[Any]:
    """Key a Python function by qualname, code, and closure values, not globals.

    No ``__module__``: a step run as ``__main__`` and the same step imported
    from pipeline.py share keys. ``canonical`` adds ``plain_globals(fn)``, so
    two same-named functions reading different PROMPT strings still differ.
    """
    closure = getattr(fn, "__closure__", None) or ()
    return [
        "__callable__",
        getattr(fn, "__qualname__", repr(fn)),
        _code_key(fn.__code__),
        [canonical(cell.cell_contents) for cell in closure],
    ]


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
            return [*function_key(obj), canonical(plain_globals(obj))]
        # Builtins have no code to tell them apart, e.g. math.sqrt and cmath.sqrt.
        qualname = getattr(obj, "__qualname__", repr(obj))
        return ["__callable__", f"{getattr(obj, '__module__', '?')}.{qualname}"]
    try:
        return pickle.dumps(obj, protocol=4)
    except Exception as e:
        raise TypeError(
            f"@op cannot derive a stable cache key for {type(obj).__name__}: "
            f"{e}. Add a cache_key() method to this type."
        ) from e
