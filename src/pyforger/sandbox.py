"""In-process sandboxed code execution via RestrictedPython.

Snippets get most of Python: expressions, control flow, def/class,
comprehensions, attribute reads via ``safer_getattr``, attribute/item
assignment (``self.x = ...``, ``d[k] = ...``) via a permissive ``_write_``
guard. The deviations:

* ``import`` only succeeds for modules whose dotted root is in
  ``allowed_imports`` (``DEFAULT_ALLOWED_IMPORTS`` by default);
  relative imports are rejected.
* Builtins are limited to ``_SAFE_BUILTIN_NAMES`` plus exception
  classes. No ``open``, ``eval``, ``exec``, ``compile``,
  ``getattr``/``setattr``, ``globals``/``locals``.
* Underscore-prefixed attribute access (``x.__class__``,
  ``().__class__.__bases__``, even single-underscore names like
  ``self._cache``) is rejected at compile time by RestrictedPython
  itself -- there is no way to opt back into it per-snippet.
* ``nonlocal`` is rejected at compile time (``global`` is fine).

Two execution models, for the two shapes LiveCodeBench-style problems come
in:

* ``compile_function``/``call``/``execute`` -- compile once, extract the
  last top-level function defined, call it directly with positional args.
  For plain function-style problems.
* ``call_method`` -- compile once (cached, so module-level state like a
  helper class or precomputed table is shared across calls, matching how a
  real grader compiles a submission once and calls its method per test
  case), instantiate the named class fresh each call, call the named
  method. For ``class Solution:``-style (LeetCode/"functional") problems.
* ``run_program`` -- exec the snippet as a standalone script each call,
  with ``sys.stdin``/``sys.stdout`` redirected and ``__name__`` set to
  ``"__main__"`` so an ``if __name__ == "__main__":`` guard actually runs.
  For stdin/stdout-style (Codeforces/AtCoder) problems.

Not a security boundary against adversarial input, just a guardrail
against accidental damage from machine-generated code.
"""
from __future__ import annotations

import ast
import builtins
import copy
import io
import signal
import sys
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


class _SandboxPolicy(RestrictingNodeTransformer):
    """Compiler policy that also permits augmented assignment to subscripts/attributes.

    RestrictedPython's default policy rejects ALL augmented assignment to
    subscripts and attributes (``d[k] += v``, ``self.x += v``) outright --
    unlike most of its other restrictions, there's no globals hook to opt
    back in, only overriding the compiler policy itself. This override
    lowers them the same way the base policy already lowers plain-name
    augmented assignment (``n += v`` -> ``n = _inplacevar_("+=", n, v)``),
    routed through the same ``_write_``/``_inplacevar_`` guards. ``d[k] += v``
    and ``self.x += v`` are extremely common in ordinary Python (DP tables,
    counters, memoized instance state), so this one hard-coded restriction
    would otherwise block a large fraction of correct, non-adversarial
    generated code.
    """

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        node = self.node_contents_visit(node)
        target = node.target
        op_str = ast.Constant(IOPERATOR_TO_STR[type(node.op)])

        if isinstance(target, ast.Name):
            read_target: ast.expr = ast.Name(target.id, ast.Load())
        elif isinstance(target, ast.Subscript):
            read_target = ast.Subscript(
                value=copy.deepcopy(target.value),
                slice=copy.deepcopy(target.slice),
                ctx=ast.Load(),
            )
        elif isinstance(target, ast.Attribute):
            read_target = ast.Attribute(
                value=copy.deepcopy(target.value), attr=target.attr, ctx=ast.Load()
            )
        else:  # pragma: no cover -- ast.c only ever produces these three target types
            raise NotImplementedError(
                f"sandbox: unknown AugAssign target type: {type(target)}"
            )

        new_node = ast.Assign(
            targets=[target],
            value=ast.Call(
                func=ast.Name("_inplacevar_", ast.Load()),
                args=[op_str, read_target, node.value],
                keywords=[],
            ),
        )
        copy_locations(new_node, node)
        return new_node


class _SandboxPrinter:
    """``_print_`` factory RestrictedPython requires for any ``print(...)`` call.

    It rewrites ``print(...)`` to ``_print._call_print(...)`` where
    ``_print = _print_(_getattr_)`` is injected at the top of whichever
    function/module scope uses it (see RestrictedPython's own
    ``PrintCollector``, which this mirrors). Unlike ``PrintCollector``
    (which buffers into a list disconnected from ``sys.stdout``), this
    forwards straight to the live ``sys.stdout`` so ``print(...)`` output
    interleaves correctly with anything written via ``sys.stdout.write()``
    directly in the same snippet.
    """

    def __init__(self, _getattr_: Any = None) -> None:
        self._getattr_: Any = _getattr_

    def write(self, text: str) -> None:
        sys.stdout.write(text)

    def __call__(self) -> str:
        return ""

    def _call_print(self, *objects: Any, **kwargs: Any) -> None:
        if kwargs.get("file") is None:
            kwargs["file"] = self
        else:
            self._getattr_(kwargs["file"], "write")
        print(*objects, **kwargs)


def _write_(ob: Any) -> Any:
    """Permissive attribute/item-assignment guard.

    RestrictedPython compiles ``obj.attr = value`` / ``obj[key] = value`` to
    ``_write_(obj).attr = value`` / ``_write_(obj)[key] = value`` and requires
    this name in globals. By the time it runs, the attribute name is already
    known-safe (any underscore-prefixed name was rejected at compile time --
    see module docstring), so this is a plain passthrough: normal attribute
    and item assignment (``self.memo = {}``, ``dp[i] = x``, ``seen[x] = True``)
    works exactly like native Python. Without this, RestrictedPython's own
    ``full_write_guard`` refuses `obj.attr = value` on any plain object that
    doesn't define ``__guarded_setattr__`` -- which is every ordinary class,
    making ``class Solution:``-style code with instance state uncompilable.
    """
    return ob


def _build_globals(allowed_imports: frozenset[str]) -> dict[str, Any]:
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
        return __import__(name, globals_, locals_, fromlist, level)

    # Build on safe_globals so common exception classes (ValueError, ...) exist.
    safe_builtins = dict(safe_globals["__builtins__"])
    for n in _SAFE_BUILTIN_NAMES:
        if n not in safe_builtins and hasattr(builtins, n):
            safe_builtins[n] = getattr(builtins, n)
    safe_builtins["__import__"] = _restricted_import

    g = dict(safe_globals)
    g["__builtins__"] = safe_builtins
    g["__name__"] = "sandbox"
    # Required by RestrictedPython's compiled class statements.
    g["__metaclass__"] = type
    g["_getattr_"] = safer_getattr
    g["_getitem_"] = lambda obj, key: obj[key]
    g["_getiter_"] = iter
    g["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
    g["_unpack_sequence_"] = guarded_unpack_sequence
    g["_inplacevar_"] = _inplacevar
    g["_write_"] = _write_
    g["_print_"] = _SandboxPrinter
    return g


def _unwrap_main_guard(code: str) -> str:
    """Strip a trailing ``if __name__ == "__main__": ...`` guard, unindented.

    If the last top-level statement is exactly that guard, replace the whole
    snippet with everything before it plus that guard's body, unindented -- so
    the sandbox never needs to see the identifier ``__name__`` at all.
    RestrictedPython rejects *any* bare name starting with ``_``, dunders
    included, so faking ``__name__`` in globals doesn't help; unwrapping the
    guard (like LiveCodeBench's own ``clean_if_name`` does for the same
    reason) does. Falls back to the code unchanged if it doesn't parse, or
    doesn't end in exactly that guard.
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


def _on_alarm(signum: int, frame: Any) -> None:
    raise SandboxTimeout("sandbox execution timed out")


class Sandbox:
    """In-process sandbox. Compile a snippet once, call the function many times.

    Calls run at native Python speed (no subprocess), so there is no
    process-level isolation; this is not a security boundary against
    adversarial input. Timeouts use ``signal.SIGALRM`` + ``signal.setitimer``
    (Unix only), so calls must be on the main thread.
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
        """Instantiate ``class_name`` from ``code`` and call ``method_name`` on it.

        Compiles ``code`` (must define ``class_name`` at module level),
        instantiates it fresh (no constructor args) and calls ``method_name``
        on that instance with ``*args``/``**kwargs``, under the same SIGALRM
        timeout as ``call``. A fresh instance every call means per-instance
        state (e.g. ``self.memo``) never leaks between calls. For
        LiveCodeBench-style ``class Solution:`` problems, where
        ``compile_function``'s "find the last top-level function" convention
        doesn't apply.
        """
        ns = self._exec_namespace(code)
        cls = ns.get(class_name)
        if not isinstance(cls, type):
            raise ValueError(
                f"sandbox: {class_name!r} is not a class defined in the given code"
            )
        instance = cls()
        method = getattr(instance, method_name)
        return self.call(method, *args, timeout_s=timeout_s, **kwargs)

    def run_program(
        self, code: str, *, stdin: str = "", timeout_s: float | None = None
    ) -> str:
        """Execute ``code`` as a script; return everything it wrote to stdout.

        Unlike ``compile_function``'s "define once, call many times" model,
        the whole snippet re-execs from scratch every call (fresh globals),
        since a stdin/stdout script is meant to run exactly once per input
        and may hold module-level mutable state. ``sys.stdin``/``sys.stdout``
        are redirected for the duration of the call so bare ``input()``/
        ``print()`` -- or ``sys.stdin.readline()`` after an explicit
        ``import sys`` -- behave like a real script invocation, and
        ``__name__`` is set to ``"__main__"`` so an
        ``if __name__ == "__main__":`` guard actually runs.

        A bare ``sys.exit()``/``SystemExit`` with code ``None`` or ``0`` is
        treated as a normal, successful exit (competitive-programming
        solutions commonly call it to end early); this sandbox runs
        in-process, so letting ``SystemExit`` propagate would kill the
        *host* process, not just this call. A nonzero exit code raises
        ``RuntimeError``.
        """
        bytecode = self._compile_restricted(_unwrap_main_guard(code))
        ns: dict[str, Any] = dict(self._globals)
        ns["__name__"] = "__main__"

        old_stdin, old_stdout = sys.stdin, sys.stdout
        # A TextIOWrapper over BytesIO (not a bare StringIO) so `.buffer` is
        # present -- competitive-programming solutions commonly read via
        # `sys.stdin.buffer.read()` for speed, and safer_getattr silently
        # returns None for a genuinely missing attribute (no AttributeError),
        # which then surfaces confusingly downstream as "NoneType is not
        # callable" on the following `.read()`.
        sys.stdin = io.TextIOWrapper(
            io.BytesIO(stdin.encode("utf-8")), encoding="utf-8"
        )
        captured = io.StringIO()
        sys.stdout = captured
        try:
            try:
                self._exec_bytecode(bytecode, ns, timeout_s=timeout_s)
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise RuntimeError(
                        f"sandbox: program exited with code {exc.code!r}"
                    ) from None
        finally:
            sys.stdin, sys.stdout = old_stdin, old_stdout
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
        signal.setitimer(itimer_real, budget)
        try:
            return fn(*args, **kwargs)
        finally:
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
