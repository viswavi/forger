"""Tests for pyforger.sandbox."""
from __future__ import annotations

import signal

import pytest

from pyforger.sandbox import Sandbox, SandboxTimeout


class TestCompileAndCall:
    """Compile a snippet, then call the function it produced."""

    def test_call_returns_value(self):
        """A compiled function is called and its return value surfaces."""
        sb = Sandbox()
        fn = sb.compile_function("def f(x):\n    return x * 2")
        assert sb.call(fn, 21) == 42

    def test_compile_cache_returns_same_function_object(self):
        """Two compile_function calls with the same source return the same function."""
        sb = Sandbox()
        src = "def f(x):\n    return x"
        assert sb.compile_function(src) is sb.compile_function(src)

    def test_last_defined_function_is_returned(self):
        """Snippets with helpers expose the last function and can call earlier ones."""
        sb = Sandbox()
        fn = sb.compile_function(
            "def helper():\n"
            "    return 1\n"
            "def main(x):\n"
            "    return x + helper()\n"
        )
        assert sb.call(fn, 10) == 11

    def test_execute_compiles_and_calls(self):
        """execute() is the compile-then-call convenience."""
        sb = Sandbox()
        assert sb.execute("def f(x):\n    return x + 1", 41) == 42


class TestTimeouts:
    """SIGALRM-based wall-clock budgets per call."""

    def test_default_timeout_fires(self):
        """Without per-call override, default_timeout_s ends a runaway call."""
        sb = Sandbox(default_timeout_s=0.2)
        fn = sb.compile_function("def f(x):\n    while True:\n        pass")
        with pytest.raises(SandboxTimeout):
            sb.call(fn, None)

    def test_per_call_timeout_overrides_default(self):
        """A timeout_s passed to call() takes precedence over default_timeout_s."""
        sb = Sandbox(default_timeout_s=10.0)
        fn = sb.compile_function("def f(x):\n    while True:\n        pass")
        with pytest.raises(SandboxTimeout):
            sb.call(fn, None, timeout_s=0.2)

    def test_signal_handler_restored_after_call(self):
        """The caller's prior SIGALRM handler is restored after call() returns."""
        sentinel = lambda *_: None  # noqa: E731
        prior = signal.signal(signal.SIGALRM, sentinel)
        try:
            sb = Sandbox()
            sb.execute("def f(x):\n    return x", 1)
            assert signal.getsignal(signal.SIGALRM) is sentinel
        finally:
            signal.signal(signal.SIGALRM, prior)


class TestRestrictions:
    """The sandbox blocks the obvious escape hatches."""

    def test_open_is_blocked(self):
        """The open() builtin is absent, so a call to it hits NameError."""
        sb = Sandbox()
        fn = sb.compile_function("def f(x):\n    return open('/etc/passwd').read()")
        with pytest.raises(NameError):
            sb.call(fn, None)

    def test_import_of_os_is_blocked(self):
        """The os module is outside the default allowlist and import raises."""
        sb = Sandbox()
        fn = sb.compile_function("def f(x):\n    import os\n    return os.environ\n")
        with pytest.raises(ImportError, match="os"):
            sb.call(fn, None)

    def test_dunder_attribute_access_is_rejected_at_compile_time(self):
        """Underscore-prefixed attribute access is rejected at compile time."""
        # Closes the classic ().__class__.__bases__ escape route before exec.
        sb = Sandbox()
        with pytest.raises(ValueError, match="__class__"):
            sb.compile_function("def f(x):\n    return x.__class__")

    def test_allowlisted_import_works(self):
        """Imports of modules on the allowlist (e.g. json) succeed."""
        sb = Sandbox()
        fn = sb.compile_function(
            "def f(x):\n    import json\n    return json.dumps({'x': x})\n"
        )
        assert sb.call(fn, 1) == '{"x": 1}'

    def test_custom_allowed_imports(self):
        """A caller can narrow the allowlist to exclude modules from the default set."""
        sb = Sandbox(allowed_imports=("re",))
        fn = sb.compile_function(
            "def f(x):\n    import json\n    return json.dumps(x)\n"
        )
        with pytest.raises(ImportError, match="json"):
            sb.call(fn, 1)


class TestResultCache:
    """Opt-in (code, args) memoization on execute()."""

    def test_disabled_by_default(self):
        """cache_results=False (default) means no result cache is allocated."""
        sb = Sandbox()
        sb.execute("def f(x):\n    return x * 2", 21)
        assert sb._result_cache is None

    def test_hashable_args_are_cached(self):
        """With cache_results=True a hashable arg lands in the result cache."""
        sb = Sandbox(cache_results=True)
        code = "def f(x):\n    return x * 2"
        assert sb.execute(code, 21) == 42
        assert sb._result_cache[(code, (21,))] == 42

    def test_cache_hit_skips_recomputation(self):
        """A cached entry is returned verbatim on the next matching execute()."""
        # Prove the cache is consulted by mutating it to a sentinel and observing it.
        sb = Sandbox(cache_results=True)
        code = "def f(x):\n    return x * 2"
        sb.execute(code, 21)
        sb._result_cache[(code, (21,))] = "from-cache"
        assert sb.execute(code, 21) == "from-cache"

    def test_unhashable_arg_falls_through(self):
        """Unhashable args (e.g. lists) don't crash and aren't cached."""
        sb = Sandbox(cache_results=True)
        code = "def f(x):\n    return len(x)"
        assert sb.execute(code, [1, 2, 3]) == 3
        assert sb._result_cache == {}

    def test_kwargs_skip_cache(self):
        """Calls with kwargs bypass the cache to avoid dict-hashing complexity."""
        sb = Sandbox(cache_results=True)
        code = "def f(x):\n    return x"
        sb.execute(code, x=5)
        assert sb._result_cache == {}


class TestErrors:
    """Compile errors raise ValueError; runtime errors propagate verbatim."""

    def test_syntax_error_raises_value_error(self):
        """A malformed snippet's SyntaxError is wrapped as ValueError on compile."""
        sb = Sandbox()
        with pytest.raises(ValueError, match="failed to compile"):
            sb.compile_function("def bad(:\n    pass\n")

    def test_no_function_defined_raises(self):
        """A snippet that defines no callable is rejected on compile."""
        sb = Sandbox()
        with pytest.raises(ValueError, match="did not define"):
            sb.compile_function("x = 1\ny = 2\n")

    def test_user_exception_propagates(self):
        """Exceptions raised by the user's code reach the caller unchanged."""
        sb = Sandbox()
        fn = sb.compile_function("def f(x):\n    raise ValueError('boom')\n")
        with pytest.raises(ValueError, match="boom"):
            sb.call(fn, None)
