"""Tests for pyforger.op."""

from __future__ import annotations

import inspect
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from pyforger.op import CallRecord, clear_trace, op, trace

_SOME_CONFIG = "v1"


def test_op_rejects_reserved_cache_arg_name() -> None:
    """Test that @op raises if the decorated function has a reserved argument name."""
    with pytest.raises(TypeError, match=r"reserved argument name '_no_cache'"):

        @op
        def _sample(x: int, *, _no_cache: bool = False) -> int:
            return x


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


@pytest.mark.parametrize("decorator", ["@op", '@op(name="typed")'])
def test_op_preserves_static_signature(tmp_path, monkeypatch, decorator) -> None:
    """Check real mypy diagnostics without executing intentionally invalid calls."""
    valid = f"""from typing import TypeVar, assert_type
from pyforger.op import op

{decorator}
def sample(value: int, /, *, label: str, scale: int = 2) -> str:
    return label * (value * scale)

assert_type(sample(3, label="x"), str)
assert_type(sample(3, label="x", scale=4), str)
assert_type(sample.no_cache(3, label="x"), str)
assert_type(sample.cache_only(3, label="x"), str)

T = TypeVar("T")

{decorator}
def identity(value: T) -> T:
    return value

assert_type(identity(3), int)
assert_type(identity("x"), str)
assert_type(identity.no_cache(3), int)
assert_type(identity.cache_only("x"), str)
"""
    fixture = tmp_path / "op_typing.py"
    fixture.write_text(valid)
    monkeypatch.setenv("MYPYPATH", str(Path(__file__).resolve().parents[1] / "src"))
    mypy_args = [
        "--strict",
        "--follow-imports=silent",
        "--show-error-codes",
        "--cache-dir",
        str(tmp_path / "mypy-cache"),
        str(Path(__file__).resolve().parents[1] / "src/pyforger/op.py"),
        str(fixture),
    ]
    result = subprocess.run(
        [sys.executable, "-m", "mypy", *mypy_args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    invalid = [
        'sample("wrong", label="x")',
        "sample(3, label=1)",
        "sample(3)",
        'sample(3, label="x", unknown=True)',
        'sample(3, label="x", _no_cache=True)',
        'sample(3, label="x", _cache_only=True)',
        'sample.no_cache("wrong", label="x")',
        "sample.no_cache(3)",
        'sample.no_cache(3, label="x", unknown=True)',
        'sample.cache_only("wrong", label="x")',
        "sample.cache_only(3)",
        'sample.cache_only(3, label="x", unknown=True)',
    ]
    fixture.write_text(valid + "\n".join(invalid) + "\n")
    result = subprocess.run(
        [sys.executable, "-m", "mypy", *mypy_args],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    first_invalid_line = len(valid.splitlines()) + 1
    for line in range(first_invalid_line, first_invalid_line + len(invalid)):
        assert f"{fixture}:{line}: error:" in output, output
    assert output.count(": error:") == len(invalid), output


@pytest.mark.parametrize("configured", [False, True])
def test_typed_cache_controls(configured) -> None:
    """Typed methods and legacy kwargs share cache entries without extra executions."""
    calls = []

    def sample(value: int, /, *, label: str, scale: int = 2) -> str:
        calls.append(value)
        return label * (value * scale)

    wrapped = op(name="controls")(sample) if configured else op(sample)
    with pytest.raises(FileNotFoundError, match="No cached result"):
        wrapped.cache_only(3, label="x")
    assert calls == []
    assert wrapped(3, label="x") == "xxxxxx"
    assert wrapped.cache_only(3, label="x", scale=2) == "xxxxxx"
    assert wrapped(3, label="x", _cache_only=True) == "xxxxxx"
    assert calls == [3]
    assert wrapped.no_cache(3, label="x") == "xxxxxx"
    assert wrapped(3, label="x", _no_cache=True) == "xxxxxx"
    assert calls == [3, 3, 3]
    assert wrapped.no_cache(4, label="x") == "xxxxxxxx"
    with pytest.raises(FileNotFoundError):
        wrapped.cache_only(4, label="x")
    assert calls == [3, 3, 3, 4]


def test_typed_cache_only_with_disabled_cache() -> None:
    """Typed cache controls respect disabled caching."""

    @op(cache=False)
    def sample(value: int) -> int:
        return value

    assert sample.no_cache(1) == 1
    with pytest.raises(ValueError, match="enabled cache"):
        sample.cache_only(1)


def test_op_preserves_runtime_signature_and_metadata() -> None:
    """Wrapping retains introspection metadata."""

    def sample(value: int, /, *, label: str = "x") -> str:
        """Sample operation."""
        return label * value

    wrapped = op(sample)
    assert inspect.signature(wrapped) == inspect.signature(sample)
    assert inspect.unwrap(wrapped) is sample
    assert wrapped.__name__ == sample.__name__
    assert wrapped.__doc__ == sample.__doc__


def test_op_preserves_method_binding() -> None:
    """Operations bind instances like ordinary methods."""

    class Sample:
        def cache_key(self) -> str:
            return "sample"

        @op
        def repeat(self, value: int) -> int:
            return value * 2

    assert Sample().repeat(3) == 6
    assert Sample.repeat(Sample(), 3) == 6
