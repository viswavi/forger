"""Tests for pyforger.alter.execute_code."""
from __future__ import annotations

import pytest

from pyforger.alter import execute_code
from pyforger.sandbox import Sandbox


class TestHappyPath:
    """The op runs each item's code and writes the return value to result."""

    def test_writes_result_to_output_field(self):
        """The function's return value lands in the result field."""
        items = [{"code": "def f(x):\n    return x * 2", "value": 21}]
        out = execute_code(items, code_field="code", arg_field="value")
        assert out[0]["result"] == 42
        assert out[0]["error"] is None

    def test_preserves_other_fields(self):
        """Fields not named by code_field or arg_field pass through unchanged."""
        items = [{"code": "def f(x):\n    return x", "value": "hi", "id": 7}]
        out = execute_code(items, code_field="code", arg_field="value")
        assert out[0]["id"] == 7
        assert out[0]["value"] == "hi"

    def test_input_not_mutated(self):
        """The op returns new dicts and does not mutate the originals."""
        original = {"code": "def f(x):\n    return x", "value": 1}
        items = [original]
        execute_code(items, code_field="code", arg_field="value")
        assert original == {"code": "def f(x):\n    return x", "value": 1}

    def test_custom_output_and_error_fields(self):
        """output_field and error_field are honored when overridden."""
        items = [{"code": "def f(x):\n    return x + 1", "value": 1}]
        out = execute_code(
            items,
            code_field="code",
            arg_field="value",
            output_field="answer",
            error_field="why_not",
        )
        assert out[0]["answer"] == 2
        assert out[0]["why_not"] is None


class TestCompileCacheReuse:
    """Repeated code strings compile once even across many items."""

    def test_same_code_compiles_once(self):
        """100 items with one shared snippet leave one entry in the compile cache."""
        sandbox = Sandbox()
        items = [
            {"code": "def f(x):\n    return x + 1", "value": i} for i in range(100)
        ]
        out = execute_code(items, code_field="code", arg_field="value", sandbox=sandbox)
        assert [row["result"] for row in out] == list(range(1, 101))
        assert len(sandbox._compile_cache) == 1

    def test_different_code_compiles_separately(self):
        """Distinct snippets each get their own compile cache entry."""
        sandbox = Sandbox()
        items = [
            {"code": "def f(x):\n    return x + 1", "value": 1},
            {"code": "def f(x):\n    return x * 2", "value": 3},
            {"code": "def f(x):\n    return x + 1", "value": 10},
        ]
        out = execute_code(items, code_field="code", arg_field="value", sandbox=sandbox)
        assert [row["result"] for row in out] == [2, 6, 11]
        assert len(sandbox._compile_cache) == 2


class TestErrors:
    """User-code and compile failures are captured, not raised."""

    def test_runtime_error_lands_in_error_field(self):
        """A raised exception in user code is stringified into error_field."""
        items = [{"code": "def f(x):\n    raise ValueError('bad')", "value": None}]
        out = execute_code(items, code_field="code", arg_field="value")
        assert out[0]["result"] is None
        assert "ValueError: bad" in out[0]["error"]

    def test_syntax_error_lands_in_error_field(self):
        """A malformed snippet surfaces its compile failure on the row."""
        items = [{"code": "def f(:\n    pass", "value": None}]
        out = execute_code(items, code_field="code", arg_field="value")
        assert out[0]["result"] is None
        assert "compile" in out[0]["error"].lower()

    def test_error_field_can_be_suppressed(self):
        """Setting error_field=None omits the error key entirely."""
        items = [{"code": "def f(x):\n    raise RuntimeError('x')", "value": None}]
        out = execute_code(
            items, code_field="code", arg_field="value", error_field=None
        )
        assert out[0]["result"] is None
        assert "error" not in out[0]

    def test_one_failure_does_not_block_others(self):
        """A failure on one item does not stop processing of subsequent items."""
        items = [
            {"code": "def f(x):\n    return x + 1", "value": 1},
            {"code": "def f(x):\n    raise ValueError('bad')", "value": 2},
            {"code": "def f(x):\n    return x * 10", "value": 3},
        ]
        out = execute_code(items, code_field="code", arg_field="value")
        assert out[0]["result"] == 2
        assert out[1]["result"] is None and "bad" in out[1]["error"]
        assert out[2]["result"] == 30


class TestMissingFields:
    """Missing required keys raise KeyError with caller context."""

    def test_missing_code_field_raises(self):
        """An item without code_field surfaces a KeyError naming the field."""
        items = [{"value": 1}]
        with pytest.raises(KeyError, match="code"):
            execute_code(items, code_field="code", arg_field="value")

    def test_missing_arg_field_raises(self):
        """An item without arg_field surfaces a KeyError naming the field."""
        items = [{"code": "def f(x):\n    return x"}]
        with pytest.raises(KeyError, match="value"):
            execute_code(items, code_field="code", arg_field="value")


class TestEmpty:
    """No items in, no items out."""

    def test_empty_input_returns_empty(self):
        """Empty input produces empty output without touching the Sandbox."""
        assert execute_code([], code_field="code", arg_field="value") == []
