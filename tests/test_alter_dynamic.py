"""Tests for dynamic pipeline operations."""

import importlib
import threading
import time
from types import SimpleNamespace

import pytest

from pyforger.alter import dynamic_step, dynamic_step_source
from pyforger.op import clear_trace, trace

dynamic = importlib.import_module("pyforger.alter.dynamic")


@pytest.fixture(autouse=True)
def isolated_op_cache(tmp_path, monkeypatch):
    """Isolated op cache."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "op-cache"))


def test_source_cache_order_and_overlap(tmp_path, monkeypatch):
    """Test source cache order and overlap."""
    calls = []
    executions = []

    def generate(goal, item, *args):
        calls.append(item)
        return "def run(item):\n    return item.upper()"

    def execute(code, item, timeout):
        executions.append(item)
        return item.upper()

    monkeypatch.setattr(dynamic, "_generate", generate)
    monkeypatch.setattr(dynamic, "_execute", execute)
    model = SimpleNamespace(_model_id="test")
    assert dynamic_step(["a", "b", "a"], "uppercase", model) == ["A", "B", "A"]
    assert dynamic_step(["b", "c"], "uppercase", model) == ["B", "C"]
    assert sorted(calls) == ["a", "b", "c"]
    assert sorted(executions) == ["a", "b", "b", "c"]
    assert (
        len(list((tmp_path / "op-cache" / "alter.dynamic_step_source").glob("*.pkl")))
        == 3
    )


def test_parallel_limit(tmp_path, monkeypatch):
    """Test parallel limit."""
    lock = threading.Lock()
    active = peak = 0

    def generate(*args):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return "code"

    monkeypatch.setattr(dynamic, "_generate", generate)
    monkeypatch.setattr(dynamic, "_execute", lambda code, item, timeout: item)
    assert dynamic_step(
        list("abcd"), "goal", SimpleNamespace(_model_id="test"), max_parallel=2
    ) == list("abcd")
    assert peak == 2


def test_worker_restrictions_and_timeout():
    """Test worker restrictions and timeout."""
    assert (
        dynamic._execute("def run(item):\n    return item.upper()", "hello", 5)
        == "HELLO"
    )
    with pytest.raises(ValueError, match="not allowed"):
        dynamic._execute("import os\ndef run(item):\n    return os.environ", "hello", 5)
    with pytest.raises(ValueError, match="string"):
        dynamic._execute("def run(item):\n    return 3", "hello", 5)
    from subprocess import TimeoutExpired

    with pytest.raises(TimeoutExpired):
        dynamic._execute("def run(item):\n    while True:\n        pass", "hello", 0.5)


def test_empty_and_validation(tmp_path):
    """Test empty and validation."""
    assert dynamic_step([], "goal", SimpleNamespace(_model_id="test")) == []
    with pytest.raises(ValueError):
        dynamic_step([], "goal", None, max_parallel=0)


def test_failed_generation_is_not_cached(tmp_path, monkeypatch):
    """Test failed generation is not cached."""

    def fail(*args):
        raise RuntimeError("no submission")

    monkeypatch.setattr(dynamic, "_generate", fail)
    with pytest.raises(RuntimeError, match="no submission"):
        dynamic_step(["a"], "goal", SimpleNamespace(_model_id="test"))
    assert not list((tmp_path / "op-cache" / "alter.dynamic_step_source").glob("*.pkl"))


def test_source_is_returned_with_standard_op_cache_hit(tmp_path, monkeypatch):
    """Test source is returned with standard op cache hit."""
    monkeypatch.setattr(
        dynamic, "_generate", lambda *args: "def run(item):\n    return item"
    )
    model = SimpleNamespace(_model_id="test")
    clear_trace()
    first = dynamic_step_source("a", "goal", model)
    second = dynamic_step_source("a", "goal", model)
    assert first == second == "def run(item):\n    return item"
    hits = [
        record.cache_hit
        for record in trace()
        if record.op_name == "alter.dynamic_step_source"
    ]
    assert hits == [False, True]


def test_source_inspection_never_generates_when_disabled(tmp_path, monkeypatch):
    """Test source inspection never generates when disabled."""
    monkeypatch.setattr(
        dynamic, "_generate", lambda *args: pytest.fail("agent launched")
    )
    with pytest.raises(FileNotFoundError, match="No cached result"):
        dynamic_step_source(
            "a", "goal", SimpleNamespace(_model_id="test"), _cache_only=True
        )
