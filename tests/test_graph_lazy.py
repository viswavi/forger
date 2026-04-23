"""Integration tests for the fully-lazy graph infrastructure.

Covers:
- Compilation produces zero execute_fn calls (laziness proof)
- Basic for/if routing, filter-after-foreach, method-call predicates
- Recompile idempotence
- Reserved-word collision on ItemSpec
- Thread safety of the frame stack (ContextVar)
- Lambda rejection
- Empty source, symbolic-if without else surfacing
"""
from __future__ import annotations

import threading

import pytest

import pyforger
from pyforger.acquire import from_list
from pyforger.core import DataSpec, NodeType
from pyforger.graph.compiler import compile_function, TracingScopeError
from pyforger.graph.symbolic import ItemSpec, TracingError


class Counter:
    def __init__(self):
        self.n = 0


def tracked_alter(source: DataSpec, tag: str, counter: Counter) -> DataSpec:
    """Per-row / batch compatible ALTER that bumps a counter each execution."""
    def _exec(input_):
        counter.n += 1
        if isinstance(input_, dict):
            return [{**input_, "tag": tag}]
        return [{**r, "tag": tag} for r in input_]
    return DataSpec(NodeType.ALTER, [source], _exec, f"tracked_{tag}")


# --- laziness ----------------------------------------------------------------

def test_compile_is_lazy():
    counter = Counter()

    def pipeline():
        data = from_list([{"n": 1}, {"n": 2}, {"n": 3}], name="data")
        out = []
        for row in data:
            out.append(tracked_alter(row, "x", counter))
        return pyforger.alter.concat(out)

    spec = compile_function(pipeline)
    assert counter.n == 0
    result = pyforger.run(spec)
    assert counter.n >= 3
    assert len(result) == 3


def test_if_else_routing():
    counter = Counter()

    def pipeline():
        data = from_list([{"n": 1}, {"n": 4}], name="data")
        out = []
        for row in data:
            if row["n"] >= 3:
                out.append(tracked_alter(row, "big", counter))
            else:
                out.append(tracked_alter(row, "small", counter))
        return pyforger.alter.concat(out)

    spec = compile_function(pipeline)
    assert counter.n == 0
    result = pyforger.run(spec)
    tags = [r["tag"] for r in result]
    assert tags == ["small", "big"]


def test_method_call_predicate():
    counter = Counter()

    def pipeline():
        data = from_list([{"s": "Apple"}, {"s": "banana"}], name="data")
        out = []
        for row in data:
            if row["s"].startswith("A") == True:  # noqa: E712
                out.append(tracked_alter(row, "upper_a", counter))
            else:
                out.append(tracked_alter(row, "other", counter))
        return pyforger.alter.concat(out)

    result = pyforger.run(compile_function(pipeline))
    assert [r["tag"] for r in result] == ["upper_a", "other"]


def test_filter_after_foreach():
    rows = [{"n": 0}, {"n": 1}, {"n": 2}, {"n": 3}, {"n": 4}]

    def pipeline():
        data = from_list(rows, name="data")
        out = []
        for row in data:
            out.append(tracked_alter(row, "x", Counter()))
        return pyforger.filter.by_length(
            pyforger.alter.concat(out),
            instruction_field="tag", min_len=1, max_len=10,
        )

    result = pyforger.run(compile_function(pipeline))
    assert len(result) == 5


def test_recompile_idempotence():
    counter = Counter()

    def pipeline():
        data = from_list([{"n": 1}, {"n": 2}], name="data")
        out = []
        for row in data:
            out.append(tracked_alter(row, "x", counter))
        return pyforger.alter.concat(out)

    spec_a = compile_function(pipeline)
    spec_b = compile_function(pipeline)
    assert spec_a is not spec_b
    pyforger.run(spec_a)
    a = counter.n
    counter.n = 0
    pyforger.run(spec_b)
    assert counter.n == a


# --- error paths -------------------------------------------------------------

def test_lambda_rejected():
    with pytest.raises(TypeError, match="lambda"):
        pyforger.run(lambda: from_list([{"a": 1}]))


def test_symbolic_if_without_else():
    def pipeline():
        data = from_list([{"n": 1}], name="data")
        out = []
        for row in data:
            if row["n"] == 1:
                out.append(tracked_alter(row, "x", Counter()))
        return pyforger.alter.concat(out)

    with pytest.raises(NotImplementedError, match="without `else`"):
        compile_function(pipeline)


def test_top_level_if_on_concrete_value_compiles_fine():
    def pipeline():
        data = from_list([{"n": 1}], name="data")
        typ = pyforger.alter.generate_from_context
        if typ:
            pass
        out = []
        for row in data:
            out.append(tracked_alter(row, "x", Counter()))
        return pyforger.alter.concat(out)

    compile_function(pipeline)


def test_tracingscopeerror_on_topleveL_if_does_not_silently_eager():
    """Top-level `if dataspec:` must surface the compile error, not fall back.

    Plain Python `if dataspec:` is truthy-by-default, so eager fallback
    would produce the wrong branch silently. This must raise.
    """
    def pipeline():
        data = from_list([{"n": 1}], name="data")
        if data:
            return data
        return data

    with pytest.raises(TracingScopeError, match="Top-level `if`"):
        pyforger.run(pipeline)


def test_unsupported_ast_with_logged_warning(caplog):
    """NotImplementedError compile errors eager-fall-back WITH a warning."""
    import logging

    def pipeline():
        try:
            data = from_list([{"n": 1}], name="data")
            return data
        except Exception:
            return from_list([])

    caplog.set_level(logging.WARNING, logger="Runner")
    result = pyforger.run(pipeline)
    assert isinstance(result, list)
    assert any("fell back to eager" in rec.getMessage() for rec in caplog.records), \
        "expected a WARNING log when eager fallback fires"


# --- reserved-word collision -------------------------------------------------

def test_itemspec_reserved_words_return_dataspec_field():
    """`item.name`/`item.inputs` return the DataSpec attribute, not a
    symbolic field access — row fields with reserved names need `item["name"]`."""
    item = ItemSpec()
    assert isinstance(item.name, str)
    assert isinstance(item.inputs, tuple)


def test_itemspec_field_access_returns_expr():
    item = ItemSpec()
    got = item["custom_field"]
    assert got is not None


# --- empty source ------------------------------------------------------------

def test_empty_source():
    def pipeline():
        data = from_list([], name="empty")
        out = []
        for row in data:
            out.append(tracked_alter(row, "x", Counter()))
        return pyforger.alter.concat(out)

    result = pyforger.run(compile_function(pipeline))
    assert result == []


# --- thread safety of frame stack --------------------------------------------

def _shared_pipeline_factory(tag: str, rows: list[dict]):
    """All threads call the same factory — ItemSpec uids may collide."""
    def _tag_row(row):
        return DataSpec(
            NodeType.ALTER, [row],
            lambda r: [{**r, "tag_times_n": f"{tag}_{r['n']}"}],
            f"tag_{tag}",
        )

    def pipeline():
        data = from_list(rows, name=f"shared_data")
        out = []
        for row in data:
            out.append(_tag_row(row))
        return pyforger.alter.concat(out)

    pipeline.__name__ = f"pipeline_{tag}"
    return pipeline


def test_frame_stack_thread_safety():
    """Heavy thread-contention test: 8 threads × 50 rows, shared factory, per-row identity check."""
    rows = [{"n": i} for i in range(50)]
    results: dict[str, list] = {}
    errors: list = []

    def run_one(tag: str):
        try:
            spec = compile_function(_shared_pipeline_factory(tag, rows))
            results[tag] = pyforger.run(spec)
        except Exception as e:  # noqa: BLE001
            errors.append((tag, e))

    threads = [threading.Thread(target=run_one, args=(f"T{i}",)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, errors
    for tag, rs in results.items():
        assert len(rs) == 50
        for i, r in enumerate(rs):
            assert r["tag_times_n"] == f"{tag}_{i}", (tag, i, r)


# --- regression guards ------------------------------------------------------

def test_fallback_exec_does_not_crash_on_exprspec_in_scope():
    """Assigning a symbolic expression into scope must not trip
    ExprSpec.__bool__ when the compiler later inspects scope values."""
    def pipeline():
        data = from_list([{"n": 1}], name="data")
        out = []
        for row in data:
            v = row["n"]  # noqa: F841
            out.append(tracked_alter(row, "x", Counter()))
        return pyforger.alter.concat(out)

    result = pyforger.run(compile_function(pipeline))
    assert len(result) == 1
