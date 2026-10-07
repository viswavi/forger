"""RestrictedPython sandbox vendored from pyforger/sandbox.py for rubric scoring.

The rubric harness further restricts imports and runs a disposable worker
(see rubric_worker.py). This in-process sandbox alone is not a security boundary.

Allowlisted imports/builtins and guarded attribute reads/writes support Python
functions, classes, and scripts. Underscored attributes and nonlocal are blocked;
sys exposes only restricted I/O and utility functions.

compile_function/call/execute reuse a compiled function; call_method creates
a fresh instance per call; run_program re-executes a script with supplied stdin."""
from __future__ import annotations

import ast
import builtins
import io
import operator
import random
import signal
import sys
import threading
import types
from collections.abc import Callable
from typing import Any

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.transformer import (
    IOPERATOR_TO_STR,
    RestrictingNodeTransformer,
    copy_locations,
)

DEFAULT_ALLOWED_IMPORTS = frozenset(
    {
        "json",
        "re",
        "math",
        "string",
        "collections",
        "itertools",
        "functools",
        "statistics",
        "datetime",
        "decimal",
        "numpy",
        "urllib.parse",
        "html",
        "unicodedata",
        "hashlib",
        "sys",
        "heapq",
        "bisect",
        "typing",
        "array",
        "operator",
        "copy",
        "random",
    }
)

_SAFE_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hex",
    "input",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
)

_INPLACE_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "+=": lambda a, b: a + b,
    "-=": lambda a, b: a - b,
    "*=": lambda a, b: a * b,
    "/=": lambda a, b: a / b,
    "//=": lambda a, b: a // b,
    "%=": lambda a, b: a % b,
    "**=": lambda a, b: a**b,
    "<<=": lambda a, b: a << b,
    ">>=": lambda a, b: a >> b,
    "&=": lambda a, b: a & b,
    "^=": lambda a, b: a ^ b,
    "|=": lambda a, b: a | b,
}


def _inplacevar(op: str, a: Any, b: Any) -> Any:
    try:
        return _INPLACE_OPS[op](a, b)
    except KeyError:
        raise NotImplementedError(f"sandbox: in-place op {op!r} not supported")


# Temporary variable names for the AugAssign rewrite. Sandboxed code can't reference
# these since RestrictedPython rejects names starting with "_".
_AUG_OBJ = "_aug_obj"
_AUG_KEY = "_aug_key"


def _write_call(value: ast.expr) -> ast.Call:
    return ast.Call(func=ast.Name("_write_", ast.Load()), args=[value], keywords=[])


def _slice_to_expr(node: ast.expr) -> ast.expr:
    """Turn a subscript slice (``a:b``, ``i, a:b``) into a plain expression."""
    if isinstance(node, ast.Slice):
        parts = [node.lower, node.upper, node.step]
        return ast.Call(
            func=ast.Name("slice", ast.Load()),
            args=[p if p is not None else ast.Constant(None) for p in parts],
            keywords=[],
        )
    if isinstance(node, ast.Tuple):
        return ast.Tuple([_slice_to_expr(e) for e in node.elts], ast.Load())
    return node


class _SandboxPolicy(RestrictingNodeTransformer):
    """Allow augmented attribute/subscript assignment through write/in-place guards."""

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt | list[ast.stmt]:
        node = self.node_contents_visit(node)
        target = node.target
        op_str = ast.Constant(IOPERATOR_TO_STR[type(node.op)])

        if isinstance(target, ast.Name):
            new_node = ast.Assign(
                targets=[target],
                value=ast.Call(
                    func=ast.Name("_inplacevar_", ast.Load()),
                    args=[op_str, ast.Name(target.id, ast.Load()), node.value],
                    keywords=[],
                ),
            )
            copy_locations(new_node, node)
            return new_node

        # Store the augmented-assignment object/key once to preserve side effects.
        # The visited target already wraps its object in _write_().
        if not isinstance(target, (ast.Subscript, ast.Attribute)):  # pragma: no cover
            raise NotImplementedError(
                f"sandbox: unknown AugAssign target type: {type(target)}"
            )
        wrapped = target.value
        assert isinstance(wrapped, ast.Call)
        obj = ast.Name(_AUG_OBJ, ast.Load())
        stmts: list[ast.stmt] = [
            ast.Assign([ast.Name(_AUG_OBJ, ast.Store())], wrapped.args[0])
        ]
        if isinstance(target, ast.Subscript):
            stmts.append(
                ast.Assign(
                    [ast.Name(_AUG_KEY, ast.Store())], _slice_to_expr(target.slice)
                )
            )
            key = ast.Name(_AUG_KEY, ast.Load())
            read_target: ast.expr = ast.Subscript(obj, key, ast.Load())
            write_target: ast.expr = ast.Subscript(_write_call(obj), key, ast.Store())
        else:
            read_target = ast.Attribute(obj, target.attr, ast.Load())
            write_target = ast.Attribute(_write_call(obj), target.attr, ast.Store())
        stmts.append(
            ast.Assign(
                targets=[write_target],
                value=ast.Call(
                    func=ast.Name("_inplacevar_", ast.Load()),
                    args=[op_str, read_target, node.value],
                    keywords=[],
                ),
            )
        )
        temps = [_AUG_OBJ]
        if isinstance(target, ast.Subscript):
            temps.append(_AUG_KEY)
        stmts.append(ast.Delete([ast.Name(t, ast.Del()) for t in temps]))
        for stmt in stmts:
            copy_locations(stmt, node)
        return stmts


# `operator` minus attrgetter/methodcaller, whose attribute lookups enable
# bypassing _getattr_ (e.g. `attrgetter("__class__")(1)`), which is not good.
_SAFE_OPERATOR = types.SimpleNamespace(
    **{
        k: v
        for k, v in vars(operator).items()
        if not k.startswith("_") and k not in ("attrgetter", "methodcaller")
    }
)


def _make_random() -> types.SimpleNamespace:
    """`random` backed by its own Random(), so seeding doesn't touch the host's."""
    rng = random.Random()
    return types.SimpleNamespace(
        **{k: getattr(rng, k, getattr(random, k)) for k in random.__all__}
    )


class _GuardedStream:
    """Wraps a stream to expose only reading, writing, and flushing."""

    _ALLOWED = frozenset(
        {"read", "readline", "readlines", "write", "writelines", "flush", "buffer"}
    )

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __getattr__(self, name: str) -> Any:
        if name not in self._ALLOWED:
            raise AttributeError(f"sandbox: stream.{name} is not allowed")
        value = getattr(self._stream, name)
        # Also guard sys.stdin.buffer.
        return _GuardedStream(value) if isinstance(value, io.IOBase) else value

    def __iter__(self) -> Any:
        yield from self._stream


class _SandboxSys:
    """The subset of ``sys`` that sandboxed code gets from ``import sys``."""

    maxsize = sys.maxsize

    def __init__(self, stdin: Any = None, stdout: Any = None) -> None:
        self._stdin = stdin
        self._stdout = stdout

    @property
    def stdin(self) -> Any:
        return _GuardedStream(self._stdin if self._stdin is not None else sys.stdin)

    @property
    def stdout(self) -> Any:
        return _GuardedStream(self._stdout if self._stdout is not None else sys.stdout)

    @property
    def stderr(self) -> Any:
        return _GuardedStream(sys.stderr)

    @staticmethod
    def exit(code: Any = None) -> None:
        raise SystemExit(code)

    @staticmethod
    def getrecursionlimit() -> int:
        return sys.getrecursionlimit()

    @staticmethod
    def setrecursionlimit(limit: int) -> None:
        sys.setrecursionlimit(limit)


class _SandboxPrinter:
    """Allow print() in RestrictedPython, writing to the sandbox's stdout."""

    def __init__(self, _getattr_: Any, sys_: _SandboxSys) -> None:
        self._getattr_: Any = _getattr_
        self._sys = sys_

    def write(self, text: str) -> None:
        self._sys.stdout.write(text)

    def flush(self) -> None:
        self._sys.stdout.flush()

    def __call__(self) -> str:
        return ""

    def _call_print(self, *objects: Any, **kwargs: Any) -> None:
        if kwargs.get("file") is None:
            kwargs["file"] = self
        else:
            self._getattr_(kwargs["file"], "write")
        print(*objects, **kwargs)


# Sandboxed code runs under "__main__" by default.
_SANDBOX_MODULES = ("sandbox", "__main__")


def _write_(ob: Any) -> Any:
    """Allow ordinary object writes while rejecting writes to modules and their members."""
    if (
        isinstance(ob, (types.ModuleType, type, types.FunctionType))
        and getattr(ob, "__module__", None) not in _SANDBOX_MODULES
    ) or isinstance(ob, (_SandboxSys, types.SimpleNamespace)):
        raise TypeError("sandbox: assigning to module attributes is not allowed")
    return ob


def _build_globals(
    allowed_imports: frozenset[str], stdin: Any = None, stdout: Any = None
) -> dict[str, Any]:
    sys_ = _SandboxSys(stdin, stdout)

    def _input(prompt: Any = "") -> str:
        if prompt:
            sys_.stdout.write(str(prompt))
        line = sys_.stdin.readline()
        if not line:
            raise EOFError("EOF when reading a line")
        return line.rstrip("\n")

    def _restricted_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0:
            raise ImportError("sandbox: relative imports are disallowed")
        root = name.split(".")[0]
        if root not in allowed_imports:
            raise ImportError(f"sandbox: import of {name!r} is not allowed")
        mod = __import__(name, globals_, locals_, fromlist, level)
        # Check the names of imported modules to make sure that sandboxed code
        # cannot indirectly import protected modules.
        for attr in fromlist or ():
            value = getattr(mod, attr, None)
            if (
                isinstance(value, types.ModuleType)
                and _check_module(value) is not value
            ):
                raise ImportError(f"sandbox: import of {name}.{attr} is not allowed")
        return _check_module(mod)

    # We define specific modules where the sandboxed code can only access a restricted
    # stand-in instead of the real thing.
    stand_ins = {sys: sys_, operator: _SAFE_OPERATOR, random: _make_random()}

    def _check_module(mod: types.ModuleType) -> Any:
        if mod.__name__.split(".")[0] not in allowed_imports:
            raise ImportError(
                f"sandbox: access to module {mod.__name__!r} is not allowed"
            )
        return stand_ins.get(mod, mod)

    def _getattr_(obj: Any, name: str, default: Any = None) -> Any:
        # Allowed modules expose others as attributes (e.g. `typing.sys` is
        # the real sys), so check any module reached this way too.
        value = safer_getattr(obj, name, default)
        if isinstance(value, types.ModuleType):
            return _check_module(value)
        return value

    # Build on safe_globals so common exception classes (ValueError, ...) exist.
    safe_builtins = dict(safe_globals["__builtins__"])
    for n in _SAFE_BUILTIN_NAMES:
        if n not in safe_builtins and hasattr(builtins, n):
            safe_builtins[n] = getattr(builtins, n)
    safe_builtins["__import__"] = _restricted_import
    safe_builtins["input"] = _input

    g = dict(safe_globals)
    g["__builtins__"] = safe_builtins
    g["__name__"] = "sandbox"
    # Required by RestrictedPython's compiled class statements.
    g["__metaclass__"] = type
    g["_getattr_"] = _getattr_
    g["_getitem_"] = lambda obj, key: obj[key]
    g["_getiter_"] = iter
    g["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
    g["_unpack_sequence_"] = guarded_unpack_sequence
    g["_inplacevar_"] = _inplacevar
    g["_write_"] = _write_
    g["_print_"] = lambda _getattr_=None: _SandboxPrinter(_getattr_, sys_)
    return g


def _unwrap_main_guard(code: str) -> str:
    """Inline a trailing __main__ guard so RestrictedPython never sees __name__.

    Return unparseable code unchanged."""
    try:
        tree = ast.parse(code)
        if not tree.body:
            return code
        last = tree.body[-1]
        if (
            isinstance(last, ast.If)
            and ast.unparse(last.test).strip() == "__name__ == '__main__'"
        ):
            new_body = tree.body[:-1] + last.body
            return ast.unparse(ast.Module(body=new_body, type_ignores=[]))
    except Exception:
        pass
    return code


class SandboxTimeout(TimeoutError):
    """Raised when a sandboxed call exceeds its wall-clock budget."""


class SandboxExit(RuntimeError):
    """Raised in place of SystemExit (e.g. sys.exit()) from sandboxed code."""

    def __init__(self, code: Any) -> None:
        """Keep the exit code on `.code`, like SystemExit."""
        super().__init__(f"sandbox: program exited with code {code!r}")
        self.code = code


# SIGALRM is a process-wide timer, so track which thread owns the active one.
_timer_thread: int | None = None


def _on_alarm(signum: int, frame: Any) -> None:
    raise SandboxTimeout("sandbox execution timed out")


class Sandbox:
    """Compile once and call in-process at native speed, without process isolation."""

    def __init__(
        self,
        *,
        allowed_imports: frozenset[str] | tuple[str, ...] = DEFAULT_ALLOWED_IMPORTS,
        default_timeout_s: float = 10.0,
        cache_results: bool = False,
    ) -> None:
        """Configure imports, call/execute timeout, and optional execute-result caching.

        Caching applies only to hashable positional arguments without kwargs."""
        self._allowed_imports = frozenset(allowed_imports)
        self._globals = _build_globals(self._allowed_imports)
        self._default_timeout_s = default_timeout_s
        self._bytecode_cache: dict[str, Any] = {}
        self._compile_cache: dict[str, Callable[..., Any]] = {}
        self._namespace_cache: dict[str, dict[str, Any]] = {}
        self._result_cache: dict[tuple, Any] | None = {} if cache_results else None

    def cache_key(self) -> tuple:
        """Stable hash key for @op caching (excludes the mutable compile cache)."""
        return (
            type(self).__name__,
            tuple(sorted(self._allowed_imports)),
            self._default_timeout_s,
            self._result_cache is not None,
        )

    def _compile_restricted(self, code: str) -> Any:
        """``compile_restricted``, cached by source string."""
        cached = self._bytecode_cache.get(code)
        if cached is not None:
            return cached
        try:
            bytecode = compile_restricted(
                code, "<sandbox>", "exec", policy=_SandboxPolicy
            )
        except SyntaxError as exc:
            raise ValueError(f"sandbox: failed to compile code: {exc}") from exc
        self._bytecode_cache[code] = bytecode
        return bytecode

    def _exec_bytecode(
        self, bytecode: Any, ns: dict[str, Any], timeout_s: float | None = None
    ) -> None:
        """``exec(bytecode, ns)`` under the same SIGALRM budget as ``call``."""
        self.call(lambda: exec(bytecode, ns), timeout_s=timeout_s)

    def _exec_namespace(self, code: str) -> dict[str, Any]:
        """Execute code once per source and cache its namespace across calls."""
        cached = self._namespace_cache.get(code)
        if cached is not None:
            return cached
        bytecode = self._compile_restricted(code)
        ns: dict[str, Any] = dict(self._globals)
        self._exec_bytecode(bytecode, ns)
        self._namespace_cache[code] = ns
        return ns

    def compile_function(self, code: str) -> Callable[..., Any]:
        """Compile ``code`` and return the last function defined; cached by source."""
        cached = self._compile_cache.get(code)
        if cached is not None:
            return cached
        bytecode = self._compile_restricted(code)

        # Single dict for globals+locals so helper functions in the snippet
        # can see each other via __globals__.
        ns: dict[str, Any] = dict(self._globals)
        baseline = set(self._globals)
        self._exec_bytecode(bytecode, ns)
        for name in reversed(list(ns)):
            if name in baseline:
                continue
            obj = ns[name]
            if callable(obj) and hasattr(obj, "__code__"):
                self._compile_cache[code] = obj
                return obj
        raise ValueError("sandbox: code did not define any function to call")

    def call_method(
        self,
        code: str,
        class_name: str,
        method_name: str,
        *args: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Compile code, create a fresh no-argument instance, and call its named method."""
        if method_name.startswith("_"):
            raise ValueError(f"sandbox: method name {method_name!r} starts with '_'")

        def run() -> Any:
            ns = self._exec_namespace(code)
            cls = ns.get(class_name)
            if not isinstance(cls, type):
                raise ValueError(
                    f"sandbox: {class_name!r} is not a class defined in the given code"
                )
            return getattr(cls(), method_name)(*args, **kwargs)

        # One timed call, so top-level code, __init__ and the method share a budget.
        return self.call(run, timeout_s=timeout_s)

    def run_program(
        self, code: str, *, stdin: str = "", timeout_s: float | None = None
    ) -> str:
        """Run code with fresh globals on every call and return captured stdout."""
        bytecode = self._compile_restricted(_unwrap_main_guard(code))
        # TextIOWrappers over BytesIO (not bare StringIOs) so `sys.stdin.buffer`
        # and `sys.stdout.buffer` are present.
        stdin_stream = io.TextIOWrapper(
            io.BytesIO(stdin.encode("utf-8")), encoding="utf-8"
        )
        out = io.BytesIO()
        captured = io.TextIOWrapper(out, encoding="utf-8", write_through=True)
        ns = _build_globals(self._allowed_imports, stdin_stream, captured)
        ns["__name__"] = "__main__"
        try:
            self._exec_bytecode(bytecode, ns, timeout_s=timeout_s)
        except SandboxExit as exc:
            if exc.code not in (None, 0):
                raise
        return out.getvalue().decode("utf-8", errors="replace")

    def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call ``fn`` with a SIGALRM wall-clock timeout. Main thread only."""
        global _timer_thread
        if _timer_thread == threading.get_ident():  # Nested call: use active timer.
            return fn(*args, **kwargs)
        budget = timeout_s if timeout_s is not None else self._default_timeout_s
        sigalrm = getattr(signal, "SIGALRM", None)
        itimer_real = getattr(signal, "ITIMER_REAL", None)
        if sigalrm is None or itimer_real is None or not hasattr(signal, "setitimer"):
            raise NotImplementedError(
                "sandbox: timeouts require Unix signal.setitimer/SIGALRM"
            )
        try:
            old_handler = signal.signal(sigalrm, _on_alarm)
        except ValueError as exc:
            raise RuntimeError(
                "sandbox: timeouts require running on the main thread"
            ) from exc
        old_timer = signal.getitimer(itimer_real)
        # Unset any sys.setrecursionlimit() from the sandboxed code.
        old_recursion_limit = sys.getrecursionlimit()
        signal.setitimer(itimer_real, budget)
        _timer_thread = threading.get_ident()
        try:
            return fn(*args, **kwargs)
        except SystemExit as exc:
            raise SandboxExit(exc.code) from None
        finally:
            sys.setrecursionlimit(old_recursion_limit)
            _timer_thread = None
            signal.setitimer(itimer_real, old_timer[0], old_timer[1])
            signal.signal(sigalrm, old_handler)

    def execute(
        self,
        code: str,
        *args: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Compile (or cache-hit) then call, all within one timeout."""

        def run() -> Any:
            return self.compile_function(code)(*args, **kwargs)

        cache = self._result_cache
        if cache is None:
            return self.call(run, timeout_s=timeout_s)

        cache_key: tuple | None = None
        if not kwargs:
            try:
                candidate = (code, args)
                hash(candidate)
                cache_key = candidate
            except TypeError:
                pass  # unhashable arg; fall through without caching
        if cache_key is not None and cache_key in cache:
            return cache[cache_key]
        result = self.call(run, timeout_s=timeout_s)
        if cache_key is not None:
            cache[cache_key] = result
        return result
