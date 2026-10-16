"""Test ordered mapping using real spawned processes and filesystem barriers."""
from __future__ import annotations

import multiprocessing
import os
import signal
import time
from functools import partial
from pathlib import Path
from threading import Thread

import pytest

from pyforger.pipeline import map_pipeline


def _work(item, prefix=""):
    """Return picklable values or exercise exception and crash paths."""
    if item == "fail":
        raise ValueError("credential-must-not-leak")
    if item == "crash":
        os._exit(7)
    if item == "unpicklable":
        return lambda: None
    if item == "interrupt":
        raise KeyboardInterrupt
    delay, value = item
    time.sleep(delay)
    return prefix, value, os.getpid()


def _hang(directory):
    """Preserve a partial artifact then hang despite a termination signal."""
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    path = Path(directory)
    (path / "pid").write_text(str(os.getpid()))
    (path / "partial").write_text("saved")
    time.sleep(60)
    (path / "late").touch()


def _barrier(item):
    """Require exactly two concurrent children before releasing each pair."""
    directory, index = item
    path = Path(directory)
    (path / f"start-{index}").touch()
    partner = index ^ 1
    deadline = time.monotonic() + 10
    while not (path / f"start-{partner}").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("missing partner")
        time.sleep(0.01)
    if index < 2:
        assert not (path / "start-2").exists()
        assert not (path / "start-3").exists()
    (path / f"checked-{index}").touch()
    while not (path / f"checked-{partner}").exists():
        time.sleep(0.01)
    return index


def test_order_partial_and_large_result():
    """Keep input order, call partials, and drain large outputs before join."""
    values = ["slow", "x" * 2_000_000, Path("artifact")]
    rows = map_pipeline(
        partial(_work, prefix="ok"),
        [(0.2, values[0]), (0, values[1]), (0, values[2])],
        concurrency=2,
        timeout_s=30,
    )
    assert [row["status"] for row in rows] == ["completed"] * 3
    assert [row["value"][:2] for row in rows] == [("ok", v) for v in values]
    assert all(row["value"][2] != os.getpid() for row in rows)


def test_failure_crash_and_serialization():
    """Isolate failures and avoid including exception messages in records."""
    rows = map_pipeline(
        _work,
        ["fail", "crash", "unpicklable", "interrupt", (0, "good")],
        timeout_s=30,
    )
    assert [row["status"] for row in rows] == ["failed"] * 4 + ["completed"]
    assert rows[0]["error"] == "ValueError"
    assert rows[-1]["value"][1] == "good"
    assert "credential-must-not-leak" not in repr(rows)


def test_timeout_kills_worker_and_preserves_partial(tmp_path):
    """Kill the exact child and keep artifacts already written by the callable."""
    before = {p.pid for p in multiprocessing.active_children()}
    started = time.monotonic()
    assert map_pipeline(_hang, [tmp_path], timeout_s=30) == [
        {"status": "timed_out", "error": "TimeoutError"}
    ]
    assert time.monotonic() - started < 40
    assert (tmp_path / "partial").read_text() == "saved"
    assert not (tmp_path / "late").exists()
    assert {p.pid for p in multiprocessing.active_children()} == before
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "pid").read_text()), 0)


def test_bounded_concurrency(tmp_path):
    """Run two workers together without starting the next pair prematurely."""
    rows = map_pipeline(
        _barrier, [(tmp_path, i) for i in range(4)], concurrency=2, timeout_s=30
    )
    assert rows == [{"status": "completed", "value": i} for i in range(4)]


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5, "2", None])
def test_invalid_concurrency(concurrency):
    """Reject invalid worker counts even for an empty iterable."""
    with pytest.raises(ValueError, match="concurrency"):
        map_pipeline(_work, [], concurrency=concurrency)


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan"), "1"])
def test_invalid_timeout(timeout):
    """Reject nonpositive, nonfinite, and nonnumeric time budgets."""
    with pytest.raises(ValueError, match="timeout_s"):
        map_pipeline(_work, [], timeout_s=timeout)


def test_empty_generator():
    """Accept empty iterables without spawning any children."""
    assert map_pipeline(_work, iter([])) == []


def _wait_for_file(path):
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"worker did not create {path.name}")
        time.sleep(0.01)


def _return_before_exit(directory):
    Thread(target=time.sleep, args=(60,)).start()
    (directory / "pid").write_text(str(os.getpid()))
    return "ready"


def _cancel_mapping(directory, mode):
    def items():
        yield directory
        _wait_for_file(directory / "pid")
        if mode == "input_interrupt":
            os.kill(os.getpid(), signal.SIGINT)
        raise RuntimeError("input iteration failed")

    expected = RuntimeError if mode == "input_error" else KeyboardInterrupt
    try:
        if mode in {"wait_interrupt", "join_interrupt"}:
            worker = _return_before_exit if mode == "join_interrupt" else _hang
            map_pipeline(worker, [directory], concurrency=2)
        else:
            map_pipeline(_hang, items(), concurrency=2)
    except expected:
        assert not multiprocessing.active_children()
        (directory / "cancelled").touch()


@pytest.fixture
def fork_transport(monkeypatch):
    """Use real fork workers to avoid repeated heavyweight package imports."""
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires fork transport")
    context = multiprocessing.get_context("fork")
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)
    return context


@pytest.mark.parametrize(
    "mode", ["input_error", "input_interrupt", "wait_interrupt", "join_interrupt"]
)
def test_caller_cancellation_reaps_hanging_child(tmp_path, fork_transport, mode):
    """Propagate iteration errors and caller SIGINT without waiting on a hang."""
    runner = fork_transport.Process(target=_cancel_mapping, args=(tmp_path, mode))
    runner.start()
    try:
        if mode in {"wait_interrupt", "join_interrupt"}:
            _wait_for_file(tmp_path / "pid")
            time.sleep(0.2)
            os.kill(runner.pid, signal.SIGINT)
        runner.join(10)
        assert not runner.is_alive(), "map_pipeline hung during cancellation"
        assert runner.exitcode == 0
        assert (tmp_path / "cancelled").exists()
        if mode != "join_interrupt":
            assert (tmp_path / "partial").read_text() == "saved"
        assert not (tmp_path / "late").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "pid").read_text()), 0)
    finally:
        if runner.is_alive():
            runner.kill()
            runner.join()
            if (tmp_path / "pid").exists():
                try:
                    os.kill(int((tmp_path / "pid").read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        runner.close()


def _mark_consumed(item):
    directory, index = item
    (directory / f"finished-{index}").touch()
    return index


def test_generator_consumption_is_bounded(tmp_path, fork_transport):
    """Refill only after a worker completes, without eagerly exhausting input."""

    def items():
        for index in range(12):
            if index >= 2:
                assert len(list(tmp_path.glob("finished-*"))) >= index - 1
            yield tmp_path, index

    rows = map_pipeline(_mark_consumed, items(), concurrency=2, timeout_s=5)
    assert rows == [{"status": "completed", "value": i} for i in range(12)]
