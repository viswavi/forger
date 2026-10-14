"""One disposable RestrictedPython execution, with no API credentials in its env."""
import contextlib
import json
import math
import resource
import signal
import sys
import time
from pathlib import Path

# -I ignores PYTHONPATH and the current directory. Import only our sibling clone.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rubric_sandbox import (  # noqa: E402 - isolated worker/module bootstrap
    DEFAULT_ALLOWED_IMPORTS,
    Sandbox,
)

GRADE_TIMEOUT_S = 20  # whole criterion, including every run below
RUN_TIMEOUT_S = 4  # one run of the response's code


def _budgeted(fn, seconds):
    """Run fn with its own wall-clock budget inside the criterion's remaining one.

    Errors, timeouts, and exits of the response's code become (False, error type)
    instead of failing the criterion.
    """
    remaining = signal.getitimer(signal.ITIMER_REAL)[0]
    started = time.monotonic()
    signal.setitimer(
        signal.ITIMER_REAL, min(seconds, remaining) if remaining else seconds
    )
    try:
        return True, fn()
    except Exception as exc:
        return False, type(exc).__name__
    finally:
        left = remaining - (time.monotonic() - started)
        signal.setitimer(signal.ITIMER_REAL, max(left, 0.001) if remaining else 0)


def solution_helpers():
    """Functions grade() can call to execute code from the response.

    The response's code runs in its own RestrictedPython namespace with captured
    stdin/stdout and competitive-programming imports, so it cannot reach this
    worker's output or the grader's names. Names starting with "_" do not compile.
    """
    solutions = Sandbox(allowed_imports=DEFAULT_ALLOWED_IMPORTS)

    def run_program(code, stdin=""):
        """Run code as a script on stdin; return (True, stdout) or (False, error)."""
        if not isinstance(code, str) or not isinstance(stdin, str):
            return False, "TypeError"
        return _budgeted(
            lambda: solutions.run_program(code, stdin=stdin), RUN_TIMEOUT_S
        )

    def call_method(code, class_name, method_name, *args):
        """Return (True, class_name().method_name(*args)) or (False, error type).

        Results are JSON-normalized (tuples become lists), so a response cannot
        return objects with custom comparisons.
        """
        if not all(isinstance(v, str) for v in (code, class_name, method_name)):
            return False, "TypeError"

        def run():
            result = solutions.call_method(code, class_name, method_name, *args)
            return json.loads(json.dumps(result))

        return _budgeted(run, RUN_TIMEOUT_S)

    return {"run_program": run_program, "call_method": call_method}


def main():
    """Run the command-line utility described in the module documentation."""
    # Darwin cannot set this address-space limit; Linux training retains it.
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (GRADE_TIMEOUT_S + 5, GRADE_TIMEOUT_S + 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    request = json.load(sys.stdin)
    sandbox = Sandbox(
        allowed_imports=(
            "json",
            "re",
            "math",
            "string",
            "collections",
            "itertools",
            "functools",
            "statistics",
        ),
        default_timeout_s=GRADE_TIMEOUT_S,
    )
    sandbox._globals.update(solution_helpers())

    class Discard:
        def write(self, value):
            return len(value)

        def flush(self):
            pass

    try:
        with contextlib.redirect_stdout(Discard()), contextlib.redirect_stderr(
            Discard()
        ):
            # Select the named entry point, even if helper functions follow it.
            def execute():
                namespace = sandbox._exec_namespace(request["code"])
                return namespace["grade"](
                    request["prompt"], request["response"], request["ground_truth"]
                )

            value = sandbox.call(execute)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("score must be a finite number in [0,1]")
        print(json.dumps({"score": float(value)}, allow_nan=False))
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__}))


if __name__ == "__main__":
    main()
