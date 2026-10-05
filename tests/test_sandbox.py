"""Tests for pyforger.sandbox."""
from __future__ import annotations

import signal
import threading
import time

import pytest

from pyforger.sandbox import Sandbox, SandboxExit, SandboxTimeout


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

    def test_nested_call_stays_within_outer_budget(self):
        """An inner call with a longer timeout still stops at the outer one."""
        sb = Sandbox(default_timeout_s=60)

        def spin():
            while True:
                pass

        start = time.monotonic()
        with pytest.raises(SandboxTimeout):
            sb.call(lambda: sb.call(spin), timeout_s=0.2)
        assert time.monotonic() - start < 5

    def test_nested_call_on_another_sandbox_keeps_outer_budget(self):
        """The outer timeout holds even if the inner call is a different Sandbox."""
        outer, inner = Sandbox(), Sandbox(default_timeout_s=60)
        start = time.monotonic()
        with pytest.raises(SandboxTimeout):
            outer.call(lambda: inner.call(time.sleep, 5), timeout_s=0.2)
        assert time.monotonic() - start < 1

    def test_other_thread_cannot_ride_on_active_timer(self):
        """A worker thread calling in mid-call is rejected, not left unbounded."""
        sb = Sandbox()
        errors = []

        def worker():
            try:
                sb.call(lambda: None)
            except RuntimeError as exc:
                errors.append(exc)

        def outer():
            t = threading.Thread(target=worker)
            t.start()
            t.join()

        sb.call(outer)
        assert len(errors) == 1 and "main thread" in str(errors[0])

    def test_nested_call_time_counts_against_outer_budget(self):
        """Time spent in an inner call isn't given back to the outer call."""
        sb = Sandbox()

        def phases():
            sb.call(time.sleep, 0.5)
            while True:
                pass

        start = time.monotonic()
        with pytest.raises(SandboxTimeout):
            sb.call(phases, timeout_s=0.6)
        assert time.monotonic() - start < 0.9

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


class TestWriteGuardAndClasses:
    """Attribute/item assignment and class definitions (self.x = ..., d[k] = ...)."""

    def test_dict_item_assignment(self):
        """d[key] = value works without a NameError on the _write_ guard."""
        sb = Sandbox()
        assert (
            sb.execute("def f(x):\n    d = {}\n    d[x] = x + 1\n    return d[x]\n", 5)
            == 6
        )

    def test_class_definition_compiles_and_runs(self):
        """A bare class statement no longer needs __metaclass__ from the caller."""
        sb = Sandbox()
        code = (
            "class Solution:\n"
            "    def f(self, x):\n"
            "        self.memo = {}\n"
            "        self.memo[x] = x * 2\n"
            "        return self.memo[x]\n"
        )
        assert sb.call_method(code, "Solution", "f", 5) == 10

    def test_call_method_uses_a_fresh_instance_per_call(self):
        """Instance state set in one call_method call doesn't leak into the next."""
        sb = Sandbox()
        code = (
            "class Solution:\n"
            "    def __init__(self):\n"
            "        self.seen = []\n"
            "    def add(self, x):\n"
            "        self.seen.append(x)\n"
            "        return len(self.seen)\n"
        )
        assert sb.call_method(code, "Solution", "add", 1) == 1
        assert sb.call_method(code, "Solution", "add", 99) == 1

    def test_call_method_missing_class_raises(self):
        """Asking for a class name the code doesn't define is a clear ValueError."""
        sb = Sandbox()
        with pytest.raises(ValueError, match="not a class"):
            sb.call_method("class Missing:\n    pass\n", "Solution", "f")

    def test_heapq_bisect_typing_imports_work(self):
        """Default allowed_imports covers competitive-programming staples."""
        sb = Sandbox()
        code = (
            "import heapq\n"
            "from typing import List\n"
            "class Solution:\n"
            "    def kth(self, nums: List[int], k: int) -> int:\n"
            "        h = []\n"
            "        for n in nums:\n"
            "            heapq.heappush(h, n)\n"
            "        return heapq.nsmallest(k, h)[-1]\n"
        )
        assert sb.call_method(code, "Solution", "kth", [5, 3, 8, 1], 2) == 3


class TestRunProgram:
    """Standalone-script execution with real stdin/stdout."""

    def test_print_flush(self):
        """print(..., flush=True) works like regular print."""
        sb = Sandbox()
        assert sb.run_program("print('x', flush=True)\n") == "x\n"

    def test_stdin_and_print_round_trip(self):
        """input()/print() read from and write to the given stdin/captured stdout."""
        sb = Sandbox()
        code = "n = int(input())\nprint(n * 2)\n"
        assert sb.run_program(code, stdin="21\n") == "42\n"

    def test_main_guard_runs(self):
        """A trailing `if __name__ == "__main__":` guard is unwrapped and runs."""
        sb = Sandbox()
        code = (
            "def main():\n"
            "    print('hi')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        assert sb.run_program(code) == "hi\n"

    def test_sys_stdin_readline_and_exit_zero(self):
        """`sys.stdin`/`sys.exit(0)` work; a clean exit doesn't kill the host."""
        sb = Sandbox()
        code = (
            "import sys\n"
            "line = sys.stdin.readline()\n"
            "sys.stdout.write('got:' + line)\n"
            "sys.exit(0)\n"
        )
        assert sb.run_program(code, stdin="ok\n") == "got:ok\n"

    def test_nonzero_exit_raises(self):
        """A nonzero sys.exit() code raises RuntimeError, not a silent success."""
        sb = Sandbox()
        with pytest.raises(RuntimeError, match="1"):
            sb.run_program("import sys\nsys.exit(1)\n")

    def test_fresh_globals_each_call(self):
        """Module-level state from one run_program call doesn't leak into the next."""
        sb = Sandbox()
        code = "count = 0\ncount += 1\nprint(count)\n"
        assert sb.run_program(code) == "1\n"
        assert sb.run_program(code) == "1\n"

    def test_host_stdout_is_not_swapped(self, capsys):
        """run_program leaves the host's sys.stdout alone."""
        import sys

        host_stdout = sys.stdout
        sb = Sandbox()
        code = "import sys\nprint(sys.stdout is not None)\n"
        assert sb.run_program(code) == "True\n"
        assert sys.stdout is host_stdout
        assert capsys.readouterr().out == ""


class TestSysFacade:
    """`import sys` gives a restricted facade, not the host module."""

    def test_sys_modules_unavailable(self):
        """The facade doesn't expose sys.modules (a route to open/os)."""
        sb = Sandbox()
        # safer_getattr returns None for missing attributes.
        assert sb.execute("import sys\ndef f():\n    return sys.modules\n") is None

    def test_sys_via_other_module_is_the_facade(self):
        """typing.sys gives the facade, not the host sys."""
        sb = Sandbox()
        code = "import typing\ndef f():\n    return typing.sys.modules\n"
        assert sb.execute(code) is None

    def test_disallowed_module_via_attribute_raises(self):
        """re.enum isn't reachable since enum isn't an allowed import."""
        sb = Sandbox()
        with pytest.raises(ImportError, match="enum"):
            sb.execute("import re\ndef f():\n    return re.enum\n")

    def test_from_import_of_host_module_raises(self):
        """`from typing import sys` can't sneak the host sys in."""
        sb = Sandbox()
        with pytest.raises(ImportError):
            sb.execute("from typing import sys\ndef f():\n    return sys\n")

    def test_operator_attrgetter_removed(self):
        """The attrgetter function would bypass _getattr_, so it isn't exposed."""
        sb = Sandbox()
        with pytest.raises(ImportError):
            sb.execute("from operator import attrgetter\ndef f():\n    pass\n")
        code = "import typing\ndef f():\n    return typing.operator.attrgetter\n"
        assert sb.execute(code) is None

    def test_operator_basics_work(self):
        """Ordinary operator functions are still available."""
        sb = Sandbox()
        code = (
            "import functools, operator\n"
            "def f(xs):\n"
            "    return functools.reduce(operator.mul, xs)\n"
        )
        assert sb.execute(code, (2, 3, 4)) == 24

    def test_exit_outside_run_program_is_contained(self):
        """sys.exit() in execute raises SandboxExit, not SystemExit."""
        sb = Sandbox()
        with pytest.raises(SandboxExit):
            sb.execute("import sys\ndef f():\n    sys.exit(0)\n")

    def test_module_attribute_assignment_refused(self):
        """Snippets can't patch shared modules or the operator stand-in."""
        sb = Sandbox()
        for code in (
            "import json\ndef f():\n    json.dumps = None\n",
            "import operator\ndef f():\n    operator.mul = None\n",
        ):
            with pytest.raises(TypeError, match="module attributes"):
                sb.execute(code)

    def test_imported_class_and_function_assignment_refused(self):
        """Classes/functions from allowed modules are shared, so also read-only."""
        sb = Sandbox()
        for code in (
            "import json\ndef f():\n    json.JSONDecoder.x = 1\n",
            "import json\ndef f():\n    json.dumps.x = 1\n",
        ):
            with pytest.raises(TypeError, match="module attributes"):
                sb.execute(code)

    def test_sandbox_class_and_function_assignment_allowed(self):
        """Classes and functions defined in the snippet stay writable."""
        sb = Sandbox()
        code = (
            "class Node:\n"
            "    count = 0\n"
            "def g():\n"
            "    pass\n"
            "def f():\n"
            "    Node.count += 1\n"
            "    g.memo = 2\n"
            "    return Node.count + g.memo\n"
        )
        assert sb.execute(code) == 3
        assert sb.run_program(code + "print(f())\n") == "3\n"

    def test_recursion_limit_restored(self):
        """sys.setrecursionlimit() inside the sandbox doesn't leak out."""
        import sys

        before = sys.getrecursionlimit()
        sb = Sandbox()
        sb.execute("import sys\ndef f():\n    sys.setrecursionlimit(50)\n")
        assert sys.getrecursionlimit() == before

    def test_host_streams_cannot_be_closed(self):
        """Only read/write/flush can be used on the underlying host streams."""
        sb = Sandbox()
        for code in (
            "import sys\ndef f():\n    sys.stdout.close()\n",
            "import sys\ndef f():\n    sys.stdout.buffer.close()\n",
            "import sys\ndef f():\n    sys.stdin.seek(0)\n",
            "import sys\ndef f():\n    sys.stdout.reconfigure()\n",
        ):
            with pytest.raises(TypeError):  # safer_getattr turns it into None
                sb.execute(code)

    def test_stdin_buffer_reads_work(self):
        """sys.stdin.buffer.read() (common in fast-IO solutions) still works."""
        sb = Sandbox()
        code = "import sys\nprint(sys.stdin.buffer.read().split())\n"
        assert sb.run_program(code, stdin="1 2\n") == "[b'1', b'2']\n"

    def test_stdout_buffer_write(self):
        """sys.stdout.buffer.write() works and stays in order with print()."""
        sb = Sandbox()
        code = "import sys\nprint('a')\nsys.stdout.buffer.write(b'b\\n')\nprint('c')\n"
        assert sb.run_program(code) == "a\nb\nc\n"

    def test_random_seed_does_not_touch_host_rng(self):
        """random.seed() in the sandbox leaves the host's global RNG alone."""
        import random

        random.seed(0)
        expected = random.random()
        random.seed(0)
        sb = Sandbox()
        code = "import random\nrandom.seed(1)\nprint(random.randint(1, 6))\n"
        assert sb.run_program(code) == sb.run_program(code)
        assert random.random() == expected

    def test_random_class_still_available(self):
        """random.Random(seed) still works for code that makes its own RNG."""
        sb = Sandbox()
        code = "import random\ndef f():\n    return random.Random(3).random()\n"
        import random

        assert sb.execute(code) == random.Random(3).random()

    def test_iterating_stdin_works(self):
        """`for line in sys.stdin` still works through the stream wrapper."""
        sb = Sandbox()
        code = "import sys\nfor line in sys.stdin:\n    print(line.strip())\n"
        assert sb.run_program(code, stdin="a\nb\n") == "a\nb\n"

    def test_timeout_covers_top_level_code(self):
        """timeout_s also bounds the first exec of the snippet's top level."""
        sb = Sandbox(default_timeout_s=60)
        loop = "while True:\n    pass\n"
        with pytest.raises(SandboxTimeout):
            sb.call_method(loop + "class S:\n    pass\n", "S", "f", timeout_s=0.2)
        with pytest.raises(SandboxTimeout):
            sb.execute(loop + "def f():\n    pass\n", timeout_s=0.2)

    def test_call_method_rejects_underscore_names(self):
        """call_method won't call dunders like __class__."""
        sb = Sandbox()
        with pytest.raises(ValueError, match="starts with"):
            sb.call_method("class Solution:\n    pass\n", "Solution", "__class__")

    def test_from_sys_import_works(self):
        """`from sys import stdin` resolves against the facade."""
        sb = Sandbox()
        code = "from sys import stdin\nprint(stdin.readline().strip())\n"
        assert sb.run_program(code, stdin="hello\n") == "hello\n"


class TestAugAssignTargets:
    """d[k] += v and obj.x += v evaluate their target once, as in Python."""

    def test_subscript_key_evaluated_once(self):
        """d[keys.pop()] += 5 pops once."""
        sb = Sandbox()
        code = (
            "def f():\n"
            "    keys = [0, 1]\n"
            "    d = [10, 20]\n"
            "    d[keys.pop()] += 5\n"
            "    return d, keys\n"
        )
        assert sb.execute(code) == ([10, 25], [0])

    def test_attribute_object_evaluated_once(self):
        """boxes.pop().x += 1 pops once."""
        sb = Sandbox()
        code = (
            "class Box:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "def f():\n"
            "    boxes = [Box(), Box()]\n"
            "    boxes.pop().x += 1\n"
            "    return len(boxes), boxes[0].x\n"
        )
        assert sb.execute(code) == (1, 1)

    def test_slice_target(self):
        """a[1:] += [4] works."""
        sb = Sandbox()
        code = "def f():\n    a = [1, 2, 3]\n    a[1:] += [4]\n    return a\n"
        assert sb.execute(code) == [1, 2, 3, 4]


class TestCallMethodTimeout:
    """call_method timeouts."""

    def test_constructor_is_bounded_by_timeout(self):
        """A non-terminating __init__ hits the timeout too."""
        sb = Sandbox()
        code = (
            "class Solution:\n"
            "    def __init__(self):\n"
            "        while True:\n"
            "            pass\n"
            "    def f(self):\n"
            "        return 1\n"
        )
        with pytest.raises(SandboxTimeout):
            sb.call_method(code, "Solution", "f", timeout_s=0.2)


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
