"""Tests for pyforger.op."""

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

import pyforger
from pyforger.op import CallRecord, clear_trace, op, trace

# The attribute pyforger.op is the decorator, so fetch the module itself.
op_module = importlib.import_module("pyforger.op")

_SOME_CONFIG = "v1"


def test_op_rejects_reserved_cache_arg_name() -> None:
    """Test that @op raises if the decorated function has a reserved argument name."""
    with pytest.raises(TypeError, match=r"reserved argument name '_no_cache'"):

        @op
        def _sample(x: int, *, _no_cache: bool = False) -> int:
            return x


@pytest.mark.parametrize("reserved", ["_cache_only", "min_yield", "max_yield"])
def test_op_rejects_other_reserved_arg_names(reserved: str) -> None:
    """Cache controls and yield bounds are reserved call-time keywords."""
    namespace: dict = {}
    exec(f"def _sample(x, *, {reserved}=None):\n    return x", namespace)
    with pytest.raises(TypeError, match=f"reserved argument name '{reserved}'"):
        op(namespace["_sample"])


def test_cache_only_never_executes_on_miss(tmp_path, monkeypatch) -> None:
    """Test cache only never executes on miss."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path))
    calls = []

    @op
    def sample(x: int) -> int:
        calls.append(x)
        return x * 2

    with pytest.raises(FileNotFoundError, match="No cached result"):
        sample(3, _cache_only=True)
    assert calls == []
    assert sample(3) == 6
    assert sample(3, _cache_only=True) == 6
    assert calls == [3]


def test_cache_only_reports_invalid_call_and_corrupt_entry(
    tmp_path, monkeypatch
) -> None:
    """Cache inspection reports binding and read failures accurately."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path))

    @op
    def sample(x: int) -> int:
        return x * 2

    with pytest.raises(TypeError):
        sample(_cache_only=True)
    assert sample(3) == 6
    entry = next(tmp_path.rglob("*.pkl"))
    entry.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Cannot read cached result"):
        sample(3, _cache_only=True)


def test_op_warns_on_unstable_refs(caplog: pytest.LogCaptureFixture) -> None:
    """@op warns when the body reads module globals or env vars (not in cache key)."""
    with caplog.at_level(logging.WARNING, logger="forger.op"):

        @op
        def reads_global(x: int) -> int:
            return x + len(_SOME_CONFIG)

        @op
        def reads_env(x: int) -> str:
            return f"{x}-{os.environ.get('FOO', '')}"

        @op
        def reads_without_global_or_env(x: int) -> int:
            return x + 1

    # Check aggregate logs for key words in warning messages from the three Op calls.
    messages = [r.message for r in caplog.records]
    assert any("reads_global" in m and "_SOME_CONFIG" in m for m in messages)
    assert any("reads_env" in m and "os.environ" in m for m in messages)
    assert not any("reads_without_global_or_env" in m for m in messages)


def test_op_warns_on_closure(caplog: pytest.LogCaptureFixture) -> None:
    """@op warns when the body closes over enclosing-scope variables."""
    with caplog.at_level(logging.WARNING, logger="forger.op"):

        def factory(config: dict) -> object:
            @op
            def closure_op(x: int) -> int:
                return x * config["multiplier"]

            return closure_op

        factory({"multiplier": 2})

    messages = [r.message for r in caplog.records]
    assert any(
        "closure_op" in m and "closes over" in m and "config" in m for m in messages
    )


@pytest.mark.parametrize(
    "a, b",
    [
        ({}, []),
        ({"k": 1}, [("k", 1)]),
        ({"k": 1}, [["k", 1]]),
        (set(), []),
        ({1}, ["__set__", [1]]),
        ((), []),
        ((1, 2), [1, 2]),
        (("__tuple__", [1]), [1]),
        ((1,), ["__tuple__", [1]]),
    ],
)
def test_op_distinguishes_container_types(a, b) -> None:
    """Containers of different types get separate cache keys."""

    @op
    def type_name(x: object) -> str:
        return type(x).__name__

    assert type_name(a) == type(a).__name__
    assert type_name(b) == type(b).__name__


def test_op_raises_for_unpicklable_arg_without_cache_key() -> None:
    """Unpicklable args with no cache_key() get a clear TypeError, not a silent repr."""

    class UnpicklableObject:
        def __init__(self) -> None:
            self._lock = threading.Lock()  # makes the object unpicklable

    @op
    def takes_anything(x: UnpicklableObject) -> int:
        return 1

    with pytest.raises(TypeError, match=r"cannot derive a stable cache key.*cache_key"):
        takes_anything(UnpicklableObject())


def test_op_accepts_unpicklable_arg_with_cache_key() -> None:
    """Unpicklable args with a cache_key() method are accepted."""

    class UnpicklableObject:
        def __init__(self) -> None:
            self._lock = threading.Lock()  # makes the object unpicklable

        def cache_key(self) -> str:
            return "stable-key"

    calls = 0

    @op
    def takes_unpicklable(x: UnpicklableObject) -> int:
        nonlocal calls
        calls += 1
        return 1

    assert takes_unpicklable(UnpicklableObject()) == 1
    assert takes_unpicklable(UnpicklableObject()) == 1  # cache hit on second call
    # Confirm that the op function only executes once, proving there was a cache hit.
    assert calls == 1


def test_trace_records_are_serializable_models(tmp_path, monkeypatch):
    """Real operation traces validate and roundtrip through Pydantic JSON."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path))
    clear_trace()
    try:

        @op
        def double(value):
            return value * 2

        assert double(3) == double(3) == 6
        records = trace()
        assert [record.cache_hit for record in records] == [False, True]
        assert records[0].output_hash is not None
        assert records[1].output_hash is None
        for record in records:
            assert isinstance(record, BaseModel)
            assert CallRecord.model_validate_json(record.model_dump_json()) == record
            assert set(record.model_dump()) == {
                "op_name",
                "input_hash",
                "output_hash",
                "cache_hit",
                "duration_s",
            }
    finally:
        clear_trace()


@pytest.mark.parametrize(
    "field,value",
    [("op_name", []), ("cache_hit", "invalid"), ("duration_s", "invalid")],
)
def test_call_record_validates_fields(field, value):
    """Malformed trace metadata fails model validation."""
    data = dict(
        op_name="test",
        input_hash="input",
        output_hash=None,
        cache_hit=False,
        duration_s=0.1,
    )
    data[field] = value
    with pytest.raises(ValidationError):
        CallRecord(**data)


def test_no_cache_applies_to_nested_ops() -> None:
    """_no_cache=True on an op also skips the cache of the ops it calls."""
    calls = []

    @op(name="test.inner")
    def inner(x):
        calls.append(x)
        return x

    @op(name="test.outer")
    def outer(x):
        return inner(x)

    outer(1)
    outer(1, _no_cache=True)
    assert calls == [1, 1]
    outer(1)
    assert calls == [1, 1]


def test_cache_written_under_the_old_key_is_still_found(monkeypatch) -> None:
    """Entries cached under the memoized-pickle key are adopted, not recomputed."""
    calls = []

    @op(name="test.legacy")
    def double(rows):
        calls.append(1)
        return [r * 2 for r in rows]

    key = ["text", {"text": 1}]
    assert op_module._hash(key)[0] != op_module._legacy_hash(key)
    new_hash = op_module._hash
    monkeypatch.setattr(
        op_module, "_hash", lambda obj: (op_module._legacy_hash(obj), 0)
    )
    double([1, 2])
    monkeypatch.setattr(op_module, "_hash", new_hash)
    assert double([1, 2]) == [2, 4] and calls == [1]
    assert trace()[-1].cache_hit


_CROSS_PROCESS_SCRIPT = """
import json, sys
from pyforger import filter, op
from pyforger.op import trace
from pyforger.steps import step


class Judge:
    calls = 0

    def cache_key(self):
        return ("Judge",)

    def generate_batch(self, prompts, *, n=1, temperature=None, max_tokens=None):
        Judge.calls += len(prompts)
        return [[json.dumps({"score": 4})] * n for _ in prompts]


@op(name="alter.make")
def make(k):
    return [{"instruction": f"q{i}", "response": f"a{i}"} for i in range(k)]


@step
def first(rows):
    return filter.by_length(rows, instruction_field="instruction", min_len=1)


@step
def second(rows):
    return filter.by_exact_match(rows, field="instruction")


rows = make(3)
second(first(rows))
filter.by_llm_score(rows, model=Judge(), input_field="instruction",
                    output_field="response", score_threshold=int(sys.argv[1]))
hits = {r.op_name: r.cache_hit for r in trace() if r.op_name.startswith("step.")}
print(json.dumps({"hits": hits, "judge_calls": Judge.calls}))
"""


def test_cache_keys_match_across_processes(tmp_path) -> None:
    """A rerun in a new process hits every step, and re-thresholding reuses scores.

    Rows read back from the cache must hash like the rows that were cached,
    whatever the identity of their strings.
    """
    script = tmp_path / "rerun.py"
    script.write_text(_CROSS_PROCESS_SCRIPT)
    src = str(Path(pyforger.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "FORGER_CACHE": str(tmp_path / "cache"),
        "FORGER_STEP_REPORT": "0",
        "PYTHONPATH": src,
    }

    def run(threshold: int) -> dict:
        out = subprocess.run(
            [sys.executable, str(script), str(threshold)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(out.stdout.strip().splitlines()[-1])

    first_run = run(3)
    assert first_run["hits"] == {"step.first": False, "step.second": False}
    assert first_run["judge_calls"] == 3
    second_run = run(4)
    assert second_run["hits"] == {"step.first": True, "step.second": True}
    assert second_run["judge_calls"] == 0
