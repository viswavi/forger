"""Run one solution on its tests exactly as grade()'s run_program/call_method do.

Reads {"code": ..., "tests": [...]} as JSON on stdin. A test is {"input": stdin}
for a script, or {"call": [class_name, method_name, arg, ...]} for a class-based
solution. Prints {"results": [{"ok": true, "value": stdout or return value} or
{"ok": false, "value": error type}, ...]}, one per test.

Same sandbox, imports, 4-second budget per run, and resource limits as
rubric_worker.py, so a solution that passes here passes in an RL reward.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

# -I ignores PYTHONPATH and the current directory. Import only our sibling clone.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rubric_sandbox import (  # noqa: E402 - isolated worker/module bootstrap
    Sandbox,
    _unwrap_main_guard,
)
from rubric_worker import RUN_TIMEOUT_S, apply_limits, solution_helpers  # noqa: E402


def main():
    """Run the command-line utility described in the module documentation."""
    request = json.load(sys.stdin)
    tests = request["tests"]
    apply_limits(cpu_seconds=RUN_TIMEOUT_S * len(tests) + 5)
    helpers = solution_helpers()
    # Sandbox.call installs the SIGALRM handler the helpers' budgets rely on.
    timer = Sandbox(default_timeout_s=RUN_TIMEOUT_S + 1)
    results = []
    # As in rubric_worker: nothing the solution does reaches our stdout.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        for test in tests:
            if "call" in test:
                run = lambda test=test: helpers["call_method"](  # noqa: E731
                    request["code"], *test["call"]
                )
            else:
                run = lambda test=test: helpers["run_program"](  # noqa: E731
                    request["code"], test.get("input") or ""
                )
            try:
                ok, value = timer.call(run)
            except Exception as exc:  # a timeout outside the run's own budget
                ok, value = False, type(exc).__name__
            result = {"ok": ok, "value": value}
            if not ok and value == "ValueError":
                # Say why, e.g. 'Line 2: "__slots__" is an invalid variable name'.
                source = request["code"]
                if "call" not in test:
                    source = _unwrap_main_guard(source)
                message = _compile_error(source)
                if message:
                    result["message"] = message
            results.append(result)
    print(json.dumps({"results": results}))


def _compile_error(source):
    """The sandbox's compile error for source, or None when it compiles."""
    try:
        Sandbox()._compile_restricted(source)
    except ValueError as exc:
        return str(exc)[:1000]
    return None


if __name__ == "__main__":
    main()
