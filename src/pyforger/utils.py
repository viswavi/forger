"""This module contains helper functions used across Forger."""

from __future__ import annotations

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


def canonical(obj: Any) -> Any:
    """Convert obj to a stable, picklable structure for hashing."""
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    if isinstance(obj, (list, tuple)):
        # We treat input lists and input tuples with the same content as identical.
        return [canonical(x) for x in obj]
    if isinstance(obj, dict):
        items = [(canonical(k), canonical(v)) for k, v in obj.items()]
        items.sort(key=lambda item: repr(item[0]))
        return items
    if isinstance(obj, (set, frozenset)):
        items = [canonical(x) for x in obj]
        items.sort(key=repr)
        return ["__frozenset__" if isinstance(obj, frozenset) else "__set__", items]
    if isinstance(obj, type):
        return f"<type {obj.__module__}.{obj.__qualname__}>"
    if hasattr(obj, "cache_key") and callable(obj.cache_key):
        return ["__cache_key__", canonical(obj.cache_key())]
    if callable(obj):
        ident = (
            f"{getattr(obj, '__module__', '?')}."
            f"{getattr(obj, '__qualname__', repr(obj))}"
        )
        code = getattr(obj, "__code__", None)
        if code is not None:
            closure = getattr(obj, "__closure__", None) or ()
            closure_vals = [canonical(cell.cell_contents) for cell in closure]
            return ["__callable__", ident, _code_key(code), closure_vals]
        return ["__callable__", ident]
    try:
        return pickle.dumps(obj, protocol=4)
    except Exception as e:
        raise TypeError(
            f"@op cannot derive a stable cache key for {type(obj).__name__}: "
            f"{e}. Add a cache_key() method to this type."
        ) from e
