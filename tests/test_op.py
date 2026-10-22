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
from pyforger.utils import canonical

# The attribute pyforger.op is the decorator, so fetch the module itself.
op_module = importlib.import_module("pyforger.op")


class _Client:
    """A stand-in for a model or API client held in a module global."""


_SOME_CLIENT = _Client()


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
    """@op warns when the body reads non-data globals or env vars (not in cache key)."""
    with caplog.at_level(logging.WARNING, logger="forger.op"):

        @op
        def reads_global(x: int) -> int:
            return x + id(_SOME_CLIENT)

        @op
        def reads_env(x: int) -> str:
            return f"{x}-{os.environ.get('FOO', '')}"

        @op
        def reads_without_global_or_env(x: int) -> int:
            return x + 1

    # Check aggregate logs for key words in warning messages from the three Op calls.
    messages = [r.message for r in caplog.records]
    assert any("reads_global" in m and "_SOME_CLIENT" in m for m in messages)
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


def test_unkeyed_args_are_left_out_of_the_cache_key() -> None:
    """Changing an unkeyed argument, given or defaulted, is a cache hit."""
    calls = []

    @op(name="test.unkeyed", unkeyed=("workers",))
    def double(xs, *, workers=1):
        calls.append(workers)
        return [x * 2 for x in xs]

    assert double([1, 2]) == [2, 4]
    assert double([1, 2], workers=8) == [2, 4]
    assert double([1, 2], workers=8, _cache_only=True) == [2, 4]
    assert double([3]) == [6]
    assert calls == [1, 1]


def test_unkeyed_must_name_a_parameter() -> None:
    """A misspelled unkeyed name fails at decoration time."""
    with pytest.raises(TypeError, match="'max_wokers', not a parameter"):

        @op(unkeyed=("max_wokers",))
        def _sample(x: int, *, max_workers: int = 1) -> int:
            return x


def _module_fn(source: str, module: str = "pipeline", **globals_: object):
    """Run ``source`` as module ``module`` with ``globals_``; return its ``f``.

    The namespace is the function's ``__globals__``, so tests can edit a global
    after decoration like a user editing a constant and rerunning.
    """
    namespace: dict = {"__name__": module, **globals_}
    exec(source, namespace)
    return namespace["f"], namespace


def _hit_after_edit(source: str, name: str, edit) -> bool:
    """Call the op, apply ``edit`` to its globals, call again; True on a cache hit."""
    fn, namespace = _module_fn(source, PROMPT="Answer briefly.", CONFIG={"k": [1]})
    wrapped = op(fn, name=name)
    wrapped(["q"])
    edit(namespace)
    wrapped(["q"])
    return trace()[-1].cache_hit


_READS_PROMPT = {
    "direct": "def f(rows):\n    return [PROMPT] + rows",
    "lambda": (
        "def f(rows):\n"
        "    fmt = lambda r: PROMPT + r\n"
        "    return [fmt(r) for r in rows]"
    ),
    "comprehension": "def f(rows):\n    return [PROMPT + r for r in rows]",
    "inner function": (
        "def f(rows):\n"
        "    def fmt(r):\n"
        "        return PROMPT + r\n"
        "    return list(map(fmt, rows))"
    ),
}


@pytest.mark.parametrize("where", list(_READS_PROMPT))
def test_editing_a_global_prompt_invalidates_the_key(where: str) -> None:
    """A prompt constant read anywhere in the body is hashed by value."""
    source = _READS_PROMPT[where]

    def same(ns: dict) -> None:
        ns["PROMPT"] = "Answer" + " briefly."  # equal value, new object

    def edited(ns: dict) -> None:
        ns["PROMPT"] = "Answer in detail."

    assert _hit_after_edit(source, f"test.same.{where}", same)
    assert not _hit_after_edit(source, f"test.edit.{where}", edited)


def test_editing_a_container_global_invalidates_the_key() -> None:
    """Dict, list, set, and tuple globals are hashed by value, nested or not."""
    source = "def f(rows):\n    return [r * len(CONFIG['k']) for r in rows]"

    def append(ns: dict) -> None:
        ns["CONFIG"]["k"].append(2)  # mutated in place, between calls

    def replace(ns: dict) -> None:
        ns["CONFIG"] = {"k": [1], "extra": (frozenset({"a"}), {None, 1.5})}

    assert not _hit_after_edit(source, "test.container.append", append)
    assert not _hit_after_edit(source, "test.container.replace", replace)


def test_non_data_globals_warn_even_in_nested_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A client read only inside a lambda warns; loggers, modules, typing do not."""
    with caplog.at_level(logging.WARNING, logger="forger.op"):
        fn, _ = _module_fn(
            "def f(rows):\n    return list(map(lambda r: (CLIENT, r), rows))",
            CLIENT=_Client(),
        )
        op(fn, name="test.client_in_lambda")
        fn, _ = _module_fn(
            "def f(rows):\n"
            "    log.info('x')\n"
            "    return [(json.dumps(r), T, PROMPT) for r in rows]",
            log=logging.getLogger("x"),
            json=json,
            T=__import__("typing").TypeVar("T"),
            PROMPT="hi",
        )
        op(fn, name="test.stable_globals")
    messages = [r.message for r in caplog.records]
    assert any("test.client_in_lambda" in m and "CLIENT" in m for m in messages)
    assert not any("test.stable_globals" in m for m in messages)


def test_main_and_imported_function_share_a_key() -> None:
    """The same step run as ``__main__`` and imported from pipeline.py hits the cache.

    Globals are hashed by value, so equal values share a key and different
    values do not, whichever module the function is in.
    """
    source = "def f(rows):\n    return [PROMPT + r for r in rows]"
    as_main, _ = _module_fn(source, module="__main__", PROMPT="a")
    imported, _ = _module_fn(source, module="pipeline", PROMPT="a")
    other, _ = _module_fn(source, module="other", PROMPT="b")
    assert as_main.__module__ != imported.__module__
    assert canonical(as_main) == canonical(imported) != canonical(other)

    op(as_main, name="step.shared")(["q"])
    op(imported, name="step.shared")(["q"])
    assert trace()[-1].cache_hit
    op(other, name="step.shared")(["q"])
    assert not trace()[-1].cache_hit


def test_docstrings_are_not_in_the_key() -> None:
    """Editing a step's description keeps its cache; its code still counts."""
    documented, _ = _module_fn(
        'def f(rows):\n    """Keep the short rows."""\n    return rows'
    )
    retyped, _ = _module_fn(
        'def f(rows):\n    """Keep the short rows (fixed typo)."""\n    return rows'
    )
    undocumented, _ = _module_fn("def f(rows):\n    return rows")
    assert canonical(documented) == canonical(retyped) == canonical(undocumented)

    op(documented, name="step.described")(["q"])
    op(retyped, name="step.described")(["q"])
    assert trace()[-1].cache_hit


def test_a_first_string_constant_is_not_taken_for_a_docstring() -> None:
    """Without a docstring, a string the body starts with is still keyed."""
    first, _ = _module_fn('def f(rows):\n    "Answer: " + rows[0]\n    return rows')
    second, _ = _module_fn('def f(rows):\n    "Reply: " + rows[0]\n    return rows')
    assert first.__doc__ is None
    assert canonical(first) != canonical(second)
    genexpr = "def f(rows):\n    return list({!r} + r for r in rows)"
    assert canonical(_module_fn(genexpr.format("a"))[0]) != canonical(
        _module_fn(genexpr.format("b"))[0]
    )


def test_nested_docstrings_are_not_in_the_key() -> None:
    """An inner function's docstring is ignored; its string constants are not."""
    source = (
        "def f(rows):\n"
        "    def fmt(r):\n"
        '        """{doc}"""\n'
        "        return {prefix!r} + r\n"
        "    return [fmt(r) for r in rows]"
    )

    def key(doc: str, prefix: str) -> object:
        return canonical(_module_fn(source.format(doc=doc, prefix=prefix))[0])

    assert key("Prefix a row.", "Q: ") == key("Prefix one row.", "Q: ")
    assert key("Prefix a row.", "Q: ") != key("Prefix a row.", "A: ")


def test_builtins_keep_their_module_in_the_key() -> None:
    """Code-less callables are told apart by module, e.g. math.sqrt and cmath.sqrt."""
    import cmath
    import math

    assert canonical(math.sqrt) != canonical(cmath.sqrt)


class _Keyed:
    """A global like ``MODEL = Model(...)``: not plain data, but has cache_key()."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._lock = threading.Lock()  # unpicklable, so only cache_key() can key it

    def cache_key(self) -> tuple:
        return ("Keyed", self.model_id)


def test_globals_with_cache_key_are_in_the_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Swapping a global model for another changes the key, with no warning."""
    from pyforger.acquire.model import Model

    source = "def f(rows):\n    return [f'{r} via {MODEL.model_id}' for r in rows]"
    with caplog.at_level(logging.WARNING, logger="forger.op"):
        fn, namespace = _module_fn(source, MODEL=_Keyed("gpt-4o"))
        which_model = op(fn, name="test.which_model")
    assert not any("test.which_model" in r.message for r in caplog.records)
    assert which_model(["q1"]) == ["q1 via gpt-4o"]
    namespace["MODEL"] = _Keyed("gpt-4o-mini")
    assert which_model(["q1"]) == ["q1 via gpt-4o-mini"]
    namespace["MODEL"] = _Keyed("gpt-4o")
    which_model(["q1"])
    assert trace()[-1].cache_hit

    a, _ = _module_fn(source, MODEL=Model("gpt-4o", show_progress=False))
    b, _ = _module_fn(source, MODEL=Model("gpt-4o-mini", show_progress=False))
    assert canonical(a) != canonical(b)


def test_forger_objects_without_cache_key_still_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only loggers and typing objects are exempt, not every pyforger object."""
    record = CallRecord(
        op_name="x", input_hash="h", output_hash=None, cache_hit=False, duration_s=0
    )
    with caplog.at_level(logging.WARNING, logger="forger.op"):
        fn, _ = _module_fn("def f(rows):\n    return [RECORD] + rows", RECORD=record)
        op(fn, name="test.forger_global")
    assert any(
        "test.forger_global" in r.message and "RECORD" in r.message
        for r in caplog.records
    )


class _AtLeast:
    """A callable object whose state, not its address, should be keyed."""

    def __init__(self, t: int) -> None:
        self.t = t

    def __call__(self, x: int) -> bool:
        return x >= self.t


def _at_least(x: int, t: int) -> bool:
    return x >= t


def test_callable_objects_and_partials_are_keyed_by_state() -> None:
    """AtLeast(3) and AtLeast(7) differ; equal partials share a key across runs."""
    import functools

    @op(name="test.keep")
    def keep(rows, pred):
        return [r for r in rows if pred(r)]

    rows = list(range(10))
    assert keep(rows, _AtLeast(3)) == [3, 4, 5, 6, 7, 8, 9]
    assert keep(rows, _AtLeast(7)) == [7, 8, 9]
    keep(rows, _AtLeast(7))
    assert trace()[-1].cache_hit

    five = functools.partial(_at_least, t=5)
    assert canonical(five) == canonical(functools.partial(_at_least, t=5))
    assert canonical(five) != canonical(functools.partial(_at_least, t=9))
    for key in (canonical(_AtLeast(3)), canonical(five)):
        assert " at 0x" not in repr(key)  # no memory address


class _Holds:
    """A callable object holding another object, so its state is not plain data."""

    def __init__(self, inner: object) -> None:
        self.inner = inner

    def __call__(self, x: int) -> int:
        return x


def test_callable_objects_with_unplain_state_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-plain state is pickled with a warning; unpicklable state needs cache_key."""
    with caplog.at_level(logging.WARNING, logger="forger.op"):
        assert canonical(_Holds(_AtLeast(1))) != canonical(_Holds(_AtLeast(2)))
    assert any(
        "_Holds" in r.message and "cache_key" in r.message for r in caplog.records
    )
    with pytest.raises(TypeError, match="cache_key"):
        canonical(_Holds(threading.Lock()))


def test_attribute_names_are_not_read_as_globals() -> None:
    """``r.split()`` does not make a global named ``split`` part of the key."""
    from pyforger.utils import keyed_globals

    fn, namespace = _module_fn(
        "def f(rows):\n    return [r for r in rows if r['text'].split()]",
        split="train",
    )
    before = canonical(fn)
    namespace["split"] = "test"
    assert keyed_globals(fn) == {}
    assert canonical(fn) == before

    reads, _ = _module_fn("def f(rows):\n    return [split] + rows", split="train")
    assert keyed_globals(reads) == {"split": "train"}


_HELPER_SOURCE = """
def tag(row):
    return row + SUFFIX

def f(rows):
    return [tag(r) for r in rows]
"""


def test_editing_a_helper_misses_the_cache() -> None:
    """A helper's body and the globals it reads are part of the caller's key."""
    fn, namespace = _module_fn(_HELPER_SOURCE, SUFFIX="-v1")
    wrapped = op(fn, name="test.helper_edit")
    assert wrapped(["a"]) == ["a-v1"]
    assert wrapped(["a"]) == ["a-v1"] and trace()[-1].cache_hit

    exec("def tag(row):\n    return row.upper() + SUFFIX", namespace)
    assert wrapped(["a"]) == ["A-v1"] and not trace()[-1].cache_hit

    namespace["SUFFIX"] = "-v2"  # read by the helper, not by f
    assert wrapped(["a"]) == ["A-v2"] and not trace()[-1].cache_hit


def test_recursive_and_wrapped_helpers_are_keyed() -> None:
    """Recursion keys the inner reference by name; @op helpers by the code they wrap."""
    from pyforger.utils import keyed_globals

    source = (
        "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"
        "def even(n):\n    return n == 0 or odd(n - 1)\n"
        "def odd(n):\n    return n != 0 and even(n - 1)\n"
        "def f(rows):\n    return [fib(r) + even(r) for r in rows]\n"
    )
    fn, namespace = _module_fn(source)
    assert set(keyed_globals(fn)) == {"fib", "even"}
    before = canonical(fn)
    exec("def odd(n):\n    return n % 2 == 1", namespace)
    assert canonical(fn) != before

    fn, namespace = _module_fn(_HELPER_SOURCE, SUFFIX="")
    namespace["tag"] = op(namespace["tag"], name="test.inner_helper")
    before = canonical(fn)
    exec("def tag(row):\n    return row * 2", namespace)
    namespace["tag"] = op(namespace["tag"], name="test.inner_helper")
    assert canonical(fn) != before


def test_helpers_in_your_modules_are_keyed(tmp_path: Path, monkeypatch) -> None:
    """``harness.make_row(r)`` keys make_row's code and globals; os.path does not."""
    import importlib

    from pyforger.utils import keyed_globals

    (tmp_path / "harness_t.py").write_text(
        "PROMPT = 'v1'\ndef make_row(r):\n    return r + PROMPT\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    harness = importlib.import_module("harness_t")
    fn, _ = _module_fn(
        "def f(rows):\n    return [harness_t.make_row(os.path.join(r)) for r in rows]",
        harness_t=harness,
        os=os,
    )
    assert set(keyed_globals(fn)) == {"harness_t.make_row"}
    before = canonical(fn)
    setattr(harness, "PROMPT", "v2")  # read by make_row
    assert canonical(fn) != before


def test_library_functions_are_keyed_by_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Standard-library and Forger functions stay out; unkeyable helpers warn."""
    from pyforger.utils import keyed_globals

    fn, _ = _module_fn(
        "def f(rows):\n    return [join(r, 'x') for r in get_field(rows)]",
        join=os.path.join,
        get_field=pyforger.utils.get_field,
    )
    assert keyed_globals(fn) == {}

    lock = threading.Lock()
    fn, namespace = _module_fn(
        "def make(lock):\n"
        "    def tag(row):\n        return lock and row\n    return tag\n"
        "def f(rows):\n    return [tag(r) for r in rows]\n"
    )
    namespace["tag"] = namespace["make"](lock)
    with caplog.at_level(logging.WARNING, logger="forger.op"):
        assert keyed_globals(fn) == {"tag": ["__callable__", "make.<locals>.tag"]}
    assert any("cannot key helper" in r.message for r in caplog.records)


def test_importing_pyforger_logs_no_unstable_ref_warnings() -> None:
    """Library ops read only data, loggers, and helpers, so import is quiet."""
    script = (
        "import importlib, logging, pkgutil\n"
        "records = []\n"
        "class Keep(logging.Handler):\n"
        "    def emit(self, record):\n"
        "        records.append(record.getMessage())\n"
        "logging.getLogger().addHandler(Keep())\n"
        "import pyforger\n"
        "for m in pkgutil.walk_packages(pyforger.__path__, 'pyforger.'):\n"
        "    try:\n"
        "        importlib.import_module(m.name)\n"
        "    except ImportError:\n"
        "        pass\n"
        "print([r for r in records if 'cache key' in r])\n"
    )
    src = str(Path(pyforger.__file__).resolve().parents[1])
    out = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": src},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stderr
    assert "may not invalidate the cache key" not in out.stderr


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


@op(name="alter.make", category="alter")
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


def _pass_at_k_op(tmp_path, monkeypatch):
    """An op shaped like filter.by_pass_at_k, recording the calls that ran."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path))
    calls = []

    @op
    def by_pass_at_k(rows, *, model, k=4, verifier_workers=4):
        calls.append(rows)
        return rows

    return by_pass_at_k, calls


def test_unknown_keyword_names_op_and_suggests_close_match(
    tmp_path, monkeypatch
) -> None:
    """A misspelled keyword names the op, suggests the real one and lists all."""
    by_pass_at_k, calls = _pass_at_k_op(tmp_path, monkeypatch)
    with pytest.raises(TypeError) as excinfo:
        by_pass_at_k([1], model="m", max_workers=8)
    message = str(excinfo.value)
    assert message.startswith(
        "by_pass_at_k() got an unexpected keyword argument 'max_workers'; "
        "did you mean 'verifier_workers'?"
    )
    assert (
        "Accepted keywords: rows, model, k, verifier_workers (plus min_yield" in message
    )
    assert excinfo.value.__cause__ is None
    assert calls == []


def test_unknown_keyword_without_close_match_lists_accepted(
    tmp_path, monkeypatch
) -> None:
    """With no close match, every unknown keyword is named and no guess is made."""
    by_pass_at_k, _ = _pass_at_k_op(tmp_path, monkeypatch)
    with pytest.raises(TypeError) as excinfo:
        by_pass_at_k([1], model="m", zzz=1, qqq=2)
    message = str(excinfo.value)
    assert message.startswith(
        "by_pass_at_k() got unexpected keyword arguments 'zzz', 'qqq'."
    )
    assert "did you mean" not in message
    assert "Accepted keywords: rows, model, k, verifier_workers" in message


def test_misspelled_call_keyword_suggests_reserved_name(tmp_path, monkeypatch) -> None:
    """Per-call keywords like min_yield are suggestion candidates too."""
    by_pass_at_k, _ = _pass_at_k_op(tmp_path, monkeypatch)
    with pytest.raises(TypeError, match=r"did you mean 'min_yield'"):
        by_pass_at_k([1], model="m", min_yeild=0.5)
    with pytest.raises(TypeError, match=r"For 'k_', did you mean 'k'\?"):
        by_pass_at_k([1], model="m", k_=2, zzz=1)


def test_missing_keyword_only_argument_names_op(tmp_path, monkeypatch) -> None:
    """A missing required keyword-only argument names the op and the argument."""
    by_pass_at_k, calls = _pass_at_k_op(tmp_path, monkeypatch)
    with pytest.raises(
        TypeError,
        match=r"^by_pass_at_k\(\) missing required argument: 'model' "
        r"\(keyword-only\)\. Accepted keywords: rows, model",
    ):
        by_pass_at_k([1])
    with pytest.raises(
        TypeError, match=r"missing required arguments: 'rows', 'model' \(keyword-only\)"
    ):
        by_pass_at_k()
    assert calls == []


def test_other_bind_errors_name_the_op(tmp_path, monkeypatch) -> None:
    """Other bind failures keep Python's message, prefixed with the op name."""
    by_pass_at_k, _ = _pass_at_k_op(tmp_path, monkeypatch)
    with pytest.raises(TypeError, match=r"^by_pass_at_k\(\): too many positional"):
        by_pass_at_k([1], "m", model="m")


def test_valid_calls_and_var_keyword_ops_are_unchanged(tmp_path, monkeypatch) -> None:
    """Valid calls, **kwargs ops and reserved keywords still work as before."""
    by_pass_at_k, calls = _pass_at_k_op(tmp_path, monkeypatch)
    assert by_pass_at_k([1], model="m", verifier_workers=8) == [1]
    assert by_pass_at_k([1], model="m", verifier_workers=8, _no_cache=True) == [1]
    assert by_pass_at_k(
        [1], model="m", verifier_workers=8, min_yield=0.5, _cache_only=True
    ) == [1]
    assert len(calls) == 2

    @op(name="test.loose")
    def loose(rows, **options):
        return sorted(options)

    assert loose([1], anything=1, _no_cache=True) == ["anything"]
