"""In-process sandboxed code execution via RestrictedPython."""
from __future__ import annotations

import builtins
import signal
from collections.abc import Callable
from typing import Any

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr

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
    g["_getattr_"] = safer_getattr
    g["_getitem_"] = lambda obj, key: obj[key]
    g["_getiter_"] = iter
    g["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
    g["_inplacevar_"] = _inplacevar
    return g


class SandboxTimeout(TimeoutError):
    """Raised when a sandboxed call exceeds its wall-clock budget."""


def _on_alarm(signum: int, frame: Any) -> None:
    raise SandboxTimeout("sandbox execution timed out")


class Sandbox:
    """In-process sandbox. Compile a snippet once, call the function many times.

    Calls run at native Python speed (no subprocess), so there is no
    process-level isolation; this is not a security boundary against
    adversarial input. Timeouts use ``signal.SIGALRM``, so calls must be
    on the main thread.
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
        self._compile_cache: dict[str, Callable[..., Any]] = {}
        self._result_cache: dict[tuple, Any] | None = {} if cache_results else None

    def cache_key(self) -> tuple:
        """Stable hash key for @op caching (excludes the mutable compile cache)."""
        return (
            type(self).__name__,
            tuple(sorted(self._allowed_imports)),
            self._default_timeout_s,
        )

    def compile_function(self, code: str) -> Callable[..., Any]:
        """Compile ``code`` and return the last function defined; cached by source."""
        cached = self._compile_cache.get(code)
        if cached is not None:
            return cached
        try:
            bytecode = compile_restricted(code, "<sandbox>", "exec")
        except SyntaxError as exc:
            raise ValueError(f"sandbox: failed to compile code: {exc}") from exc

        # Single dict for globals+locals so helper functions in the snippet
        # can see each other via __globals__.
        ns: dict[str, Any] = dict(self._globals)
        baseline = set(self._globals)
        exec(bytecode, ns)

        for name in reversed(list(ns)):
            if name in baseline:
                continue
            obj = ns[name]
            if callable(obj):
                self._compile_cache[code] = obj
                return obj
        raise ValueError("sandbox: code did not define any function to call")

    def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call ``fn`` with a SIGALRM wall-clock timeout. Main thread only."""
        budget = timeout_s if timeout_s is not None else self._default_timeout_s
        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, budget)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

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
