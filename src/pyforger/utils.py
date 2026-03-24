"""This module contains helper functions used across Forger."""

from __future__ import annotations

import logging
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
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
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
