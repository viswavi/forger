"""This module contains helper functions used across Forger."""

from __future__ import annotations

import logging


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
