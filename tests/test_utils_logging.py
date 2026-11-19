"""Tests for get_formatted_logger: each Forger log line is printed once."""

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import textwrap

import pytest

import pyforger
from pyforger.utils import get_formatted_logger


class _Tty(io.StringIO):
    """A terminal-like stream: isatty() is true."""

    def isatty(self) -> bool:
        return True


@pytest.fixture
def forger_logger(request):
    """A fresh Forger logger whose own handler writes to a buffer."""
    logger = get_formatted_logger(f"forger.test_logging.{request.node.name}")
    buf = io.StringIO()
    (handler,) = logger.handlers
    handler.setStream(buf)
    yield logger, buf
    logger.handlers.clear()


@pytest.fixture
def root_handler():
    """Add a handler to the root logger for one test, then remove it."""
    added = []

    def add(handler: logging.Handler) -> logging.Handler:
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logging.getLogger().addHandler(handler)
        added.append(handler)
        return handler

    yield add
    for handler in added:
        logging.getLogger().removeHandler(handler)


def test_prints_through_own_handler_when_root_has_none(forger_logger):
    """With nothing on the root logger, Forger's handler prints, in its format."""
    logger, buf = forger_logger
    logger.info("hello")
    assert buf.getvalue().count("hello") == 1
    assert " - INFO - hello" in buf.getvalue()


def test_root_terminal_handler_prints_the_line_instead(forger_logger, root_handler):
    """A root handler on a terminal (basicConfig, absl) prints it; Forger's skips."""
    logger, buf = forger_logger
    tty = _Tty()
    root_handler(logging.StreamHandler(tty))
    logger.warning("hello")
    assert buf.getvalue() == ""
    assert tty.getvalue() == f"WARNING:{logger.name}:hello\n"


def test_root_handler_on_the_same_stream_counts_as_terminal(
    forger_logger, root_handler
):
    """A root handler writing to Forger's own stream would print the line twice."""
    logger, buf = forger_logger
    shared = root_handler(logging.StreamHandler(buf))
    logger.info("hello")
    assert buf.getvalue().count("hello") == 1
    assert shared.stream is buf


def test_root_handler_to_a_file_does_not_silence_forger(
    forger_logger, root_handler, tmp_path
):
    """A log file gets the line and the terminal still does too."""
    logger, buf = forger_logger
    file_handler = root_handler(logging.FileHandler(tmp_path / "run.log"))
    logger.info("hello")
    file_handler.flush()
    assert buf.getvalue().count("hello") == 1
    assert (tmp_path / "run.log").read_text() == f"INFO:{logger.name}:hello\n"
    file_handler.close()


def test_root_terminal_handler_above_record_level_does_not_count(
    forger_logger, root_handler
):
    """A WARNING-level root handler drops INFO lines, so Forger prints them."""
    logger, buf = forger_logger
    tty = _Tty()
    root_handler(logging.StreamHandler(tty)).setLevel(logging.WARNING)
    logger.info("hello")
    assert buf.getvalue().count("hello") == 1
    assert tty.getvalue() == ""


def test_handler_past_a_non_propagating_parent_does_not_count(
    forger_logger, root_handler
):
    """The root handler never sees records stopped by a parent's propagate=False."""
    logger, buf = forger_logger
    tty = _Tty()
    root_handler(logging.StreamHandler(tty))
    parent = logging.getLogger("forger.test_logging")
    parent.propagate = False
    try:
        logger.info("hello")
    finally:
        parent.propagate = True
    assert buf.getvalue().count("hello") == 1
    assert tty.getvalue() == ""


def test_caplog_still_captures_forger_records(forger_logger, caplog):
    """Records still reach caplog's root handler, and Forger still prints them."""
    logger, buf = forger_logger
    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("hello")
    assert [r.getMessage() for r in caplog.records] == ["hello"]
    assert buf.getvalue().count("hello") == 1


def _stderr_of(script: str) -> str:
    # run with this checkout's src/ on the path
    src = os.path.dirname(os.path.dirname(pyforger.__file__))
    env = {**os.environ, "PYTHONPATH": src}
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stderr


def test_basic_config_prints_each_line_once():
    """In a real process: Forger's format before basicConfig(), the root's after."""
    stderr = _stderr_of(
        """
        import logging
        from pyforger.utils import get_formatted_logger
        log = get_formatted_logger("forger.report")
        log.warning("before")
        logging.basicConfig()
        log.warning("after")
        """
    )
    assert stderr.count("before") == 1
    assert " - forger.report - WARNING - before" in stderr
    assert stderr.count("after") == 1
    assert "WARNING:forger.report:after" in stderr
