"""This module contains helper functions used across Forger."""

from __future__ import annotations

import logging


<<<<<<< HEAD
def get_formatted_logger(logger_name: str, level: int = logging.INFO) -> logging.Logger:
=======
def get_formatted_logger(logger_name: str):
>>>>>>> 34bce2c (Create logger class)
    """Create a formatted logger.

    Args:
        logger_name: The name of the logger, usually the name
            of the component that uses the logger.
<<<<<<< HEAD
        level: The logging level (by default, INFO).
=======
>>>>>>> 34bce2c (Create logger class)

    Returns:
        A logger object.
    """
    logger = logging.getLogger(logger_name)
<<<<<<< HEAD
    logger.setLevel(level)
=======
>>>>>>> 34bce2c (Create logger class)
    # Check if the logger already has a StreamHandler to prevent adding another one.
    if not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
<<<<<<< HEAD
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
=======
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
>>>>>>> 34bce2c (Create logger class)
    return logger
