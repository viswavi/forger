"""Tests for @op cache keys over large values and blob storage of cached results."""

from __future__ import annotations

import importlib
import logging
import os
import pickle
import subprocess
import sys
from collections import OrderedDict, deque
from pathlib import Path

import pytest

import pyforger
from pyforger import op
from pyforger.op import trace
from pyforger.steps import pipeline, step

op_module = importlib.import_module("pyforger.op")

TRACE = "think " * 10_000  # 60K chars: a blob
TESTS = [{"input": "x" * 5_000, "expected": str(i)} for i in range(10)]  # 50K: a blob


def _rows(n: int = 3) -> list[dict]:
    return [
        {"id": i, "trace": f"{i}:{TRACE}", "tests": [dict(t) for t in TESTS]}
        for i in range(n)
    ]


def _blobs(cache: Path) -> list[Path]:
    return sorted((cache / "blobs").iterdir()) if (cache / "blobs").exists() else []


def _entry(cache: Path, op_name: str) -> Path:
    (path,) = (cache / op_name).glob("*.pkl")
    return path


def test_large_values_round_trip_through_blobs(tmp_path) -> None:
    """A cache hit returns what the op returned; large values live in blobs/."""
    calls = []
    payload = {
        "rows": _rows(),
        "raw": b"\x00\xff" * 10_000,
        "pair": ("a" * 20_000, [1.5, None, True]),
        "surrogate": "\ud800" + "y" * 20_000,
        "small": "kept inline",
    }

    @op(name="alter.payload")
    def make(n: int) -> dict:
        calls.append(n)
        return payload

    assert make(1) == payload
    hit = make(1)
    assert calls == [1]
    assert hit == payload
    assert type(hit["pair"]) is tuple and type(hit["raw"]) is bytes
    entry = _entry(tmp_path, "alter.payload")
    assert entry.stat().st_size < 10_000
    assert _blobs(tmp_path)


def test_steps_carrying_a_payload_store_it_once(tmp_path) -> None:
    """Six entries (three steps and their ops) share one blob per large value."""

    @op(name="alter.tag")
    def tag(rows, *, field):
        return [{**r, field: True} for r in rows]

    @step
    def one(rows):
        return tag(rows, field="a")

    @step
    def two(rows):
        return tag(rows, field="b")

    @step
    def three(rows):
        return tag(rows, field="c")

    @pipeline(prod=True)
    def run(rows):
        return three(two(one(rows)))

    rows = _rows()
    out = run(rows)
    # Three traces and one tests list (equal in every row), for six entries.
    assert len(_blobs(tmp_path)) == 4
    assert sum(p.stat().st_size for p in tmp_path.rglob("*.pkl")) < 20_000
    assert run(rows) == out
    assert all(r.cache_hit for r in trace()[-3:])
    assert len(_blobs(tmp_path)) == 4


def test_entries_pickled_without_blobs_still_load(tmp_path) -> None:
    """An entry written as a plain pickle (older Forger) is a cache hit."""
    calls = []

    @op(name="alter.old")
    def make(n: int) -> list:
        calls.append(n)
        return _rows()

    make(1)
    entry = _entry(tmp_path, "alter.old")
    old_value = [{"id": 0, "trace": TRACE}]
    entry.write_bytes(pickle.dumps(old_value, protocol=4))
    assert make(1) == old_value
    assert calls == [1]


def test_a_missing_blob_is_a_cache_miss(tmp_path) -> None:
    """Deleting blobs/ recomputes the result, and _cache_only reports a miss."""
    calls = []

    @op(name="alter.gone")
    def make(n: int) -> list:
        calls.append(n)
        return _rows()

    expected = make(1)
    for blob in _blobs(tmp_path):
        blob.unlink()
    with pytest.raises(FileNotFoundError, match="No cached result"):
        make(1, _cache_only=True)
    assert make(1) == expected
    assert calls == [1, 1]
    # The recomputed result wrote its blobs again.
    assert make(1, _cache_only=True) == expected
    assert calls == [1, 1]


def test_a_list_that_contains_itself_round_trips(monkeypatch) -> None:
    """A cyclic list large enough to be a blob is pickled inside its own blob."""
    # The trace hashes each result, which a cyclic value cannot be.
    monkeypatch.setattr(op_module, "_TRACE", deque(maxlen=0))

    @op(name="alter.cycle")
    def make(n: int) -> list:
        inner: list = ["z" * 1_000 for _ in range(20)]
        inner.append(inner)
        return [inner]

    make(1)
    (inner,) = make(1)
    assert inner[-1] is inner
    assert inner[0] == "z" * 1_000


def test_large_rows_hash_without_a_cost_warning(caplog) -> None:
    """60 MiB of row payloads hash quickly, so they log no cost warning."""
    rows = [{"id": i, "trace": f"{i}" + "x" * 2**20} for i in range(60)]

    @op(name="alter.count")
    def count(rows: list) -> int:
        return len(rows)

    with caplog.at_level(logging.WARNING, logger="forger.op"):
        assert count(rows) == 60
    assert not [r for r in caplog.records if "cache-key hash took" in r.message]


def _key(value) -> str:
    return op_module._hash(value)[0]


def test_large_strings_are_keyed_by_their_whole_content() -> None:
    """A one-character change anywhere in a large value changes its key."""
    big = "a" * 100_000
    assert _key(big) == _key("".join(["a"] * 100_000))
    assert _key(big) != _key(big[:-1] + "b")
    assert _key(big) != _key(big.encode())
    assert _key([big]) != _key((big,))
    assert _key({"t": big}) == _key({"t": "a" * 100_000})


@pytest.mark.parametrize(
    "a, b",
    [
        (1, 1.0),
        (1, True),
        ("1", 1),
        ("a", b"a"),
        ({1}, frozenset({1})),
        ([1, [2]], [[1], 2]),
        (["ab"], ["a", "b"]),
        ({"a": "bc"}, {"ab": "c"}),
        ({1: "x"}, {"1": "x"}),
    ],
)
def test_keys_tell_values_apart(a, b) -> None:
    """Values of different types or split differently get different keys."""
    assert _key(a) != _key(b)


def test_equal_values_get_equal_keys() -> None:
    """Dict and set order, subclasses, and repeated objects do not change keys."""
    shared = ["x" * 5_000]
    assert _key({"a": 1, "b": 2}) == _key({"b": 2, "a": 1})
    assert _key({(1, 2): "p", 3: "q"}) == _key({3: "q", (1, 2): "p"})
    assert _key({"b", "a", "c"}) == _key({"c", "a", "b"})
    assert _key(OrderedDict(a=1)) == _key({"a": 1})
    assert _key([shared, shared]) == _key([list(shared), list(shared)])
    # Rows read back from a cache entry have new string objects.
    rows = _rows()
    assert _key(rows) == _key(pickle.loads(pickle.dumps(rows)))


_KEY_VALUE = {"rows": [{"t": "think " * 10_000, "s": {"b", "a"}, "n": (1, 2.5, None)}]}
_KEY_SCRIPT = f"""
import importlib
op_module = importlib.import_module("pyforger.op")
print(op_module._hash({_KEY_VALUE!r})[0])
"""


def test_keys_are_stable_across_processes(tmp_path) -> None:
    """Another process (with another str hash seed) keys the same value equally."""
    script = tmp_path / "key.py"
    script.write_text(_KEY_SCRIPT)
    src = str(Path(pyforger.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src, "PYTHONHASHSEED": "12345"}
    out = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == _key(_KEY_VALUE)
