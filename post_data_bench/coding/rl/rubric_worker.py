"""One disposable RestrictedPython execution, with no API credentials in its env.

rubric_rm runs this with ``--judge-fds R,W``: grade()'s judge_cpp sends its
requests to the parent over those pipes (cpp_judge.JudgeChannel), since these
limits cannot compile or run C++ here. Run without arguments, as in
``python rl/rubric_worker.py < criterion.json``, it starts itself that way, as
the RL reward does, and prints the result.
"""
import contextlib
import importlib
import json
import math
import os
import resource
import signal
import sys
import time
from pathlib import Path

# -I ignores PYTHONPATH and the current directory. Import only our sibling clone.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpp_judge import NOT_JSON, encode_request  # noqa: E402 - see above
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


class JudgeClient:
    """grade()'s end of the judge channel: one JSON line out, one back."""

    def __init__(self, requests_fd, replies_fd):
        """Wrap the two pipe ends rubric_rm passed to this worker."""
        self.requests = os.fdopen(requests_fd, "wb")
        self.replies = os.fdopen(replies_fd, "rb")
        self.broken = False

    def ask(self, line):
        """Send one encoded request and return (score, statuses)."""
        self.requests.write(line)
        self.requests.flush()
        reply = json.loads(self.replies.readline())
        return float(reply["score"]), [str(s) for s in reply["statuses"]]


def solution_helpers(judge=None):
    """Functions grade() can call to execute code from the response.

    The response's code runs in its own RestrictedPython namespace with captured
    stdin/stdout and competitive-programming imports, so it cannot reach this
    worker's output or the grader's names. Names starting with "_" do not compile.
    ``judge`` (a JudgeClient) serves judge_cpp; without one it scores 0.
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

    def judge_cpp(
        code, tests, checker=None, interactor=None, time_limit=1.0, memory_mb=256
    ):
        """Compile and judge a C++ program on tests: (score in [0, 1], statuses).

        Judged outside this worker (cpp_judge.judge_tests) within the
        criterion's remaining time. Errors score 0 with one status saying why.
        """
        if judge is None or judge.broken:
            return 0.0, ["Judge Unavailable: judge_cpp runs only in the RL grader"]
        remaining = signal.getitimer(signal.ITIMER_REAL)[0]
        line = encode_request(
            code,
            tests,
            checker,
            interactor,
            time_limit,
            memory_mb,
            budget_s=remaining or GRADE_TIMEOUT_S,
        )
        if line is None:
            return 0.0, [NOT_JSON]
        ok, value = _budgeted(lambda: judge.ask(line), GRADE_TIMEOUT_S)
        if not ok:
            # A late reply must never be read as the answer to a later call.
            judge.broken = True
            return 0.0, [f"Judge Unavailable: {value}"]
        return value

    return {
        "run_program": run_program,
        "call_method": call_method,
        "judge_cpp": judge_cpp,
    }


def apply_limits(cpu_seconds=GRADE_TIMEOUT_S + 5):
    """The worker's resource limits (solution_worker.py applies the same ones)."""
    # Darwin cannot set this address-space limit; Linux training retains it.
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))


def _run_as_reward():
    """Score stdin's criterion through rubric_rm, exactly as an RL reward does."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    rubric_rm = importlib.import_module(f"{here.name}.rubric_rm")
    result = rubric_rm._run_worker(sys.stdin.read())
    if result is None:
        message = f"the criterion ran past {rubric_rm.WORKER_TIMEOUT_S} seconds"
        print(json.dumps({"error": "TimeoutExpired", "message": message}))
    elif getattr(result, "judge_error", None):
        print(json.dumps({"error": "JudgeError", "message": result.judge_error}))
    elif result.returncode:
        message = f"worker exited with {result.returncode}: {result.stderr[-1000:]}"
        print(json.dumps({"error": "WorkerFailed", "message": message}))
    else:
        sys.stdout.write(result.stdout)


def main():
    """Run the command-line utility described in the module documentation."""
    if sys.argv[1:2] != ["--judge-fds"]:
        return _run_as_reward()
    judge = JudgeClient(*map(int, sys.argv[2].split(",")))
    apply_limits()
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
    sandbox._globals.update(solution_helpers(judge))

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
        # The message makes a broken grade() debuggable instead of a silent 0,
        # e.g. 'Line 4: "_final" is an invalid variable name ...'.
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)[:2000]}))


if __name__ == "__main__":
    main()
