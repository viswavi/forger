"""In-process sandboxed code execution via RestrictedPython.

Allows most of Python: expressions, control flow, def/class, comprehensions,
attribute reads via ``safer_getattr``, attribute/item assignment via a
permissive ``_write_`` guard. Some things not allowed include:

* ``import`` only succeeds for modules whose root is in ``allowed_imports``
* Builtins are limited to ``_SAFE_BUILTIN_NAMES`` plus stated exceptions.
  No open, eval, exec, compile, getattr/setattr, globals, or locals.
* Underscored attribute access (e.g. x.__class__, self._cache) are not allowed.
* ``nonlocal`` is rejected at compile time (``global`` is fine).
* ``import sys`` returns a restricted copy of the module (stdin/stdout/stderr, exit,
  maxsize, get/setrecursionlimit), not the full thing.

Three execution models are supported through the sandbox:

* compile_function/call/execute -- compile once, extract the last top-level
  function, and call it. Good if your solution is a single function.
* call_method -- compile once, instantiate the named class fresh each call, and
  call the named method. E.g. `Solution` is loaded once and `solve` called per
  test case.
* run_program -- run the code as a script with the given stdin and return what
  it printed. Good if your solution reads stdin and writes stdout.

Note that this is not a proper security boundary against adversarial input, but
rather a guardrail against accidental damage from machine-generated code.
"""
from __future__ import annotations

import ast
import builtins
import io
import operator
import signal
import sys
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
    """Expanded policy to allow augmented assignment to subscripts/attributes.

    RestrictedPython's default policy rejects ALL augmented assignment to
    subscripts and attributes (d[k] += v, self.x += v) outright -- unlike
    most of its other restrictions, there's no globals hook to opt back in.
    This override allows these the same way the base policy already allows
    standard augmented assignment (`n += v -> n` = _inplacevar_("+=", n, v)`),
    routed through the same _write_/_inplacevar_ guards.
    Note that d[k] += v  and self.x += v are very common in ordinary Python
    (DP tables, counters, memoized instance state), so this one hard-coded
    restriction would otherwise pose a significant hindrance.
    """

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

        # Concrete example of how this works: we rewrite `cnt[q.popleft()] += 1` as:
        #     _aug_obj = cnt
        #     _aug_key = q.popleft()
        #     _write_(_aug_obj)[_aug_key] = _inplacevar_("+=", _aug_obj[_aug_key], 1)
        #     del _aug_obj, _aug_key
        # so q.popleft() runs once, not twice (once to read, once to write).
        # target.value.args[0] is `cnt`: the visit above wrapped it in _write_().
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


class _SandboxSys:
    """The subset of ``sys`` that sandboxed code gets from ``import sys``."""

    maxsize = sys.maxsize

    def __init__(self, stdin: Any = None, stdout: Any = None) -> None:
        self._stdin = stdin
        self._stdout = stdout

    @property
    def stdin(self) -> Any:
        return self._stdin if self._stdin is not None else sys.stdin

    @property
    def stdout(self) -> Any:
        return self._stdout if self._stdout is not None else sys.stdout

    @property
    def stderr(self) -> Any:
        return sys.stderr

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
    """Make the RestrictedPython guard allow attribute/item assignment.

    Without this, RestrictedPython's own `full_write_guard` refuses
    `obj.attr = value` on any plain object that doesn't define
    `__guarded_setattr__` -- which is every ordinary class. This would make
    typical ``class Solution:``-style code with instance state uncompilable.
    Certain modules (e.g. libraries) can be accessed, but writes to those
    modules (or their classes and functions) are never allowed.
    """
    if (
        isinstance(ob, (types.ModuleType, type, types.FunctionType))
        and getattr(ob, "__module__", None) not in _SANDBOX_MODULES
    ) or (isinstance(ob, _SandboxSys) or ob is _SAFE_OPERATOR):
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
    stand_ins = {sys: sys_, operator: _SAFE_OPERATOR}

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
    """Strip a trailing ``if __name__ == "__main__": ...``.

    If the last top-level statement is exactly that, then replace the whole
    snippet with everything before it plus that guard's body, unindented, so
    the sandbox never needs to see the identifier ``__name__`` at all.
    RestrictedPython rejects *any* bare name starting with ``_``, dunders
    included, so this is necessary. Falls back to the code unchanged if it
    doesn't parse.
    """
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


def _on_alarm(signum: int, frame: Any) -> None:
    raise SandboxTimeout("sandbox execution timed out")


class Sandbox:
    """In-process sandbox. Compile a snippet once, call the function many times.

    Calls run at native Python speed (no subprocess), so there is no process-level
    isolation; this is shallow protection, not a security boundary against adversarial
    input.
    """

    def __init__(
        self,
        *,
        allowed_imports: frozenset[str] | tuple[str, ...] = DEFAULT_ALLOWED_IMPORTS,
        default_timeout_s: float = 10.0,
        cache_results: bool = False,
    ) -> None:
        """Build a Sandbox.

        Args:
            allowed_imports: Module names the snippet may import.
            default_timeout_s: Wall-clock cap for ``call`` / ``execute``.
            cache_results: When True, ``execute(code, *args)`` memoizes the
                return value keyed by ``(code, args)``. Only kicks in when
                all args are hashable (immutable) and no kwargs are passed;
                unhashable args fall through to a normal call.
        """
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
        """Exec ``code`` once (cached by source) into a fresh namespace, and return it.

        Namespace is seeded from the sandbox's globals. Module-level
        definitions (classes, helper functions, precomputed tables) are
        shared across calls -- matches how a real grader compiles a
        submission once and calls its method per test case.
        """
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
        """Instantiate `class_name` from `code` and call `method_name` on it.

        Compiles `code` (which must define `class_name` at the module level),
        instantiates it fresh (no constructor args) and calls `method_name`
        on that instance with ``*args``/``**kwargs``.
        """
        if method_name.startswith("_"):
            raise ValueError(f"sandbox: method name {method_name!r} starts with '_'")
        ns = self._exec_namespace(code)
        cls: Any = ns.get(class_name)
        if not isinstance(cls, type):
            raise ValueError(
                f"sandbox: {class_name!r} is not a class defined in the given code"
            )
        # Construct inside the timed call so a slow __init__ is also bounded.
        return self.call(
            lambda: getattr(cls(), method_name)(*args, **kwargs), timeout_s=timeout_s
        )

    def run_program(
        self, code: str, *, stdin: str = "", timeout_s: float | None = None
    ) -> str:
        """Execute ``code`` as a script; return everything it wrote to stdout.

        Unlike `compile_function`'s "define once, call many times" model,
        the whole snippet re-execs from scratch every call (fresh globals).
        """
        bytecode = self._compile_restricted(_unwrap_main_guard(code))
        # A TextIOWrapper over BytesIO (not a bare StringIO) so
        # `sys.stdin.buffer` is present.
        stdin_stream = io.TextIOWrapper(
            io.BytesIO(stdin.encode("utf-8")), encoding="utf-8"
        )
        captured = io.StringIO()
        ns = _build_globals(self._allowed_imports, stdin_stream, captured)
        ns["__name__"] = "__main__"
        try:
            self._exec_bytecode(bytecode, ns, timeout_s=timeout_s)
        except SandboxExit as exc:
            if exc.code not in (None, 0):
                raise
        return captured.getvalue()

    def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call ``fn`` with a SIGALRM wall-clock timeout. Main thread only."""
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
        try:
            return fn(*args, **kwargs)
        except SystemExit as exc:
            raise SandboxExit(exc.code) from None
        finally:
            sys.setrecursionlimit(old_recursion_limit)
            signal.setitimer(itimer_real, old_timer[0], old_timer[1])
            signal.signal(sigalrm, old_handler)

    def execute(
        self,
        code: str,
        *args: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Compile (or cache-hit) then call."""
        cache = self._result_cache
        if cache is None:
            return self.call(
                self.compile_function(code), *args, timeout_s=timeout_s, **kwargs
            )

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
        result = self.call(
            self.compile_function(code), *args, timeout_s=timeout_s, **kwargs
        )
        if cache_key is not None:
            cache[cache_key] = result
        return result
