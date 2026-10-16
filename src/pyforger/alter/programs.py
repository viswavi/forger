r"""Run stdin/stdout programs against tests in subprocesses.

``run_programs`` runs each row's program on each of its tests and records what
happened; ``filter.by_tests`` keeps rows whose program passes, and
``filter.program_verifier`` grades ``filter.by_pass_at_k`` samples the same way.

A test is a dict: ``{"input": "3\n1 2 3\n", "output": "6"}``. ``"output"`` is
optional (the run is recorded but not graded, e.g. to collect a brute-force
solution's answers), as are ``"args"`` (command-line arguments, e.g.
``["7", "small"]`` for a test generator) and ``"timeout_s"`` (overrides the
op's ``timeout_s`` for that test).

Each run is its own process, started with an empty environment except
``PATH`` and the locale (so no API keys), in a fresh temporary directory,
with a wall-clock timeout and limits on memory (not on macOS) and output size.
Runs use threads and no signals, so the ops work from any thread.

This is NOT a security sandbox. The program runs as your user, with your
files and network. Run untrusted code in a container or VM.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pyforger.op import op
from pyforger.utils import get_field, get_formatted_logger

# (actual stdout, expected output) -> passed
Checker = Callable[[str, str], bool]

# The child sets its own limits and then execs the program, so no preexec_fn
# (unsafe with threads) is needed. Limits survive exec. A limit of 0 is not
# set, and one the system rejects is skipped (macOS rejects RLIMIT_AS, so
# run_program passes 0 there).
_BOOTSTRAP = """\
import os, sys
try:
    import resource
except ImportError:
    resource = None
mem, fsize = int(sys.argv[1]), int(sys.argv[2])
if resource is not None:
    for limit, value in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_FSIZE, fsize)):
        if value > 0:
            try:
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                pass
os.execv(sys.argv[3], sys.argv[3:])
"""

# Only what a program needs to start; nothing from the parent's environment.
_CHILD_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}
_MAC_MEMORY_WARNED = threading.Event()


def _logger() -> Any:
    # Looked up per call: a module-level logger read inside an @op body would
    # trigger @op's "reads globals" warning.
    return get_formatted_logger("Programs")


def outputs_match(actual: str, expected: str) -> bool:
    r"""Default checker: equal up to whitespace.

    E.g. ``"1 2\n3\n"`` matches ``"1 2 3"``, but ``"12"`` does not match ``"1 2"``.
    """
    return actual.split() == str(expected).split()


_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)


def last_code_block(text: str) -> str | None:
    r"""The last fenced code block tagged python, py, python3, or untagged.

    E.g. ``"Idea...\n```python\nprint(1)\n```"`` gives ``"print(1)\n"``.
    None when there is no such block.
    """
    blocks = [
        code
        for info, code in _FENCE.findall(text or "")
        if info.strip().lower() in ("", "python", "py", "python3")
    ]
    return blocks[-1] if blocks else None


def run_program(
    code: str,
    stdin: str = "",
    *,
    args: list[str] | tuple[str, ...] = (),
    timeout_s: float = 10.0,
    memory_mb: int | None = 2048,
    max_output_mb: int = 64,
    command: list[str] | None = None,
    suffix: str = ".py",
) -> dict[str, Any]:
    """Run ``code`` once as a script with ``stdin``; return what happened.

    Not an op: no caching, and safe to call from many threads at once.

    Returns:
        ``{"status", "exit_code", "timed_out", "stdout", "stderr", "seconds"}``.
        ``status`` is ``"ok"`` (exit code 0), ``"runtime_error"`` (non-zero
        exit or killed by a signal; exceeding ``memory_mb`` usually shows up
        here as a ``MemoryError``), ``"timeout"``, or ``"output_limit"``
        (stdout reached ``max_output_mb``). ``exit_code`` is None after a
        timeout and negative when a signal killed the program.
    """
    if os.name != "posix":
        raise OSError("run_program needs a POSIX system (Linux or macOS)")
    memory_bytes = (memory_mb or 0) << 20
    if memory_bytes and sys.platform == "darwin":
        # macOS has no working RLIMIT_AS (setrlimit fails), so runs are not
        # memory-limited there; warn once per process.
        memory_bytes = 0
        if not _MAC_MEMORY_WARNED.is_set():
            _MAC_MEMORY_WARNED.set()
            _logger().warning("memory_mb is not enforced on macOS (no RLIMIT_AS)")
    workdir = tempfile.mkdtemp(prefix="forger_run_")
    try:
        program = os.path.join(workdir, "main" + suffix)
        with open(program, "w", encoding="utf-8") as f:
            f.write(code)
        target = _target_command(command, program, args)
        output_limit = max_output_mb << 20
        launch = [
            sys.executable,
            "-I",
            "-c",
            _BOOTSTRAP,
            str(memory_bytes),
            str(output_limit),
            *target,
        ]
        out_path = os.path.join(workdir, "stdout")
        err_path = os.path.join(workdir, "stderr")
        with (
            open(os.path.join(workdir, "stdin"), "w+", encoding="utf-8") as fin,
            open(out_path, "wb") as fout,
            open(err_path, "wb") as ferr,
        ):
            fin.write(stdin)
            fin.seek(0)
            start = time.perf_counter()
            proc = subprocess.Popen(
                launch,
                stdin=fin,
                stdout=fout,
                stderr=ferr,
                cwd=workdir,
                env=dict(_CHILD_ENV),
                start_new_session=True,  # its own process group, killed below
            )
            timed_out = False
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
            # Kill the whole group: the program, and anything it started.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
            seconds = time.perf_counter() - start
        stdout_bytes = os.path.getsize(out_path)
        if timed_out:
            status, exit_code = "timeout", None
        elif stdout_bytes >= output_limit:
            status, exit_code = "output_limit", proc.returncode
        else:
            exit_code = proc.returncode
            status = "ok" if exit_code == 0 else "runtime_error"
        return {
            "status": status,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout": _read(out_path),
            "stderr": _read(err_path),
            "seconds": round(seconds, 3),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _target_command(
    command: list[str] | None, program: str, args: list[str] | tuple[str, ...]
) -> list[str]:
    """The argv that runs ``program``: Python by default, else ``command``.

    ``command`` names the file as ``{file}``, e.g. ``["node", "{file}"]``.
    """
    if command is None:
        return [sys.executable, "-I", program, *map(str, args)]
    if not command:
        raise ValueError("command must not be empty")
    executable = shutil.which(command[0]) or command[0]
    rest = [part.replace("{file}", program) for part in command[1:]]
    return [executable, *rest, *map(str, args)]


def _read(path: str) -> str:
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def run_tests(
    code: str,
    tests: list[dict],
    *,
    checker: Checker = outputs_match,
    stop_on_failure: bool = False,
    max_chars: int | None = 10_000,
    **run_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run ``code`` on each test in order; one result per test that ran.

    Each result is ``run_program``'s dict plus ``"passed"``: True or False
    when the test has an ``"output"``, else None. A run that exits 0 with the
    wrong output gets status ``"wrong_answer"``. ``stop_on_failure`` stops
    after the first test that does not pass. ``max_chars`` cuts each kept
    stdout and stderr (None keeps them whole). ``run_kwargs`` go to
    ``run_program``.
    """
    results = []
    for test in tests:
        result = _run_test(code, test, checker, max_chars, run_kwargs)
        results.append(result)
        if stop_on_failure and result["passed"] is not True:
            break
    return results


def _run_test(
    code: str,
    test: dict,
    checker: Checker,
    max_chars: int | None,
    run_kwargs: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(test, dict):
        raise TypeError(f"a test must be a dict with an 'input' key, got {test!r}")
    kwargs = dict(run_kwargs)
    if test.get("timeout_s") is not None:
        kwargs["timeout_s"] = float(test["timeout_s"])
    result = run_program(
        code, str(test.get("input", "")), args=test.get("args") or (), **kwargs
    )
    expected = test.get("output")
    if expected is None:
        result["passed"] = None
    elif result["status"] != "ok":
        result["passed"] = False
    else:
        result["passed"] = bool(checker(result["stdout"], str(expected)))
        if not result["passed"]:
            result["status"] = "wrong_answer"
    if max_chars is not None:
        result["stdout"] = result["stdout"][:max_chars]
        # The end of stderr, where a traceback's error is.
        result["stderr"] = result["stderr"][-max_chars:] if max_chars else ""
    return result


def _code_of(
    item: dict, code_field: str, extract_code: Callable[[str], str | None] | None
) -> str | None:
    code = get_field(item, code_field, "run_programs")
    if code is None:
        return None
    code = str(code)
    if extract_code is not None:
        code = extract_code(code)
    return code if code and code.strip() else None


def _default_workers() -> int:
    return min(8, os.cpu_count() or 1)


@op(name="alter.run_programs", unkeyed=("max_workers",))
def run_programs(
    items: list[dict],
    *,
    code_field: str,
    tests_field: str = "tests",
    result_field: str = "test_results",
    extract_code: Callable[[str], str | None] | None = None,
    checker: Checker = outputs_match,
    timeout_s: float = 10.0,
    memory_mb: int | None = 2048,
    max_output_mb: int = 64,
    max_chars: int | None = 10_000,
    command: list[str] | None = None,
    suffix: str = ".py",
    max_workers: int | None = None,
) -> list[dict]:
    r"""Run each row's stdin/stdout program on each of its tests.

    Every row gets ``result_field``: one result per test, in test order, e.g.
    ``{"status": "wrong_answer", "passed": False, "exit_code": 0, "timed_out":
    False, "stdout": "5\n", "stderr": "", "seconds": 0.03}``. ``status`` is
    ``"ok"``, ``"wrong_answer"``, ``"runtime_error"``, ``"timeout"``, or
    ``"output_limit"``; ``passed`` is None for a test without ``"output"``.
    A row whose code is missing or empty gets ``None``.

    Args:
        items: Input rows.
        code_field: Key holding the program source.
        tests_field: Key holding the row's tests, a list of ``{"input",
            "output"?, "args"?, "timeout_s"?}`` dicts.
        result_field: Key the results are written to.
        extract_code: Optional ``str -> str | None`` applied to the field
            first, e.g. ``last_code_block`` to take the code out of a model's
            answer. None uses the field as is.
        checker: ``(stdout, expected) -> bool``; default ``outputs_match``
            (equal up to whitespace).
        timeout_s: Wall-clock limit per run, in seconds.
        memory_mb: Address-space limit per run (None for none). Not enforced
            on macOS, which has no working ``RLIMIT_AS``.
        max_output_mb: Stdout size at which a run stops with ``output_limit``.
        max_chars: Characters of stdout (first) and stderr (last) kept per
            result. None keeps them whole, e.g. to use a reference program's
            outputs as expected outputs.
        command: Interpreter command for non-Python programs, with ``{file}``
            for the program file, e.g. ``["node", "{file}"]`` and
            ``suffix=".js"``. None runs Python (this interpreter, ``-I``).
        suffix: File name suffix for the program.
        max_workers: Runs at a time across all rows (default
            ``min(8, cpu count)``).

    Returns:
        New rows (inputs are not mutated), in input order.
    """
    workers = max_workers or _default_workers()
    if workers < 1:
        raise ValueError("max_workers must be at least 1")
    run_kwargs = dict(
        timeout_s=timeout_s,
        memory_mb=memory_mb,
        max_output_mb=max_output_mb,
        command=command,
        suffix=suffix,
    )
    codes = [_code_of(item, code_field, extract_code) for item in items]
    tests = [
        list(get_field(item, tests_field, "run_programs") or [])
        if code is not None
        else []
        for item, code in zip(items, codes)
    ]
    jobs = [(i, j) for i, row_tests in enumerate(tests) for j in range(len(row_tests))]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _run_test, codes[i] or "", tests[i][j], checker, max_chars, run_kwargs
            )
            for i, j in jobs
        ]
        try:
            done = [f.result() for f in futures]
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    results: list[list[dict] | None] = [
        [] if code is not None else None for code in codes
    ]
    for (i, _), result in zip(jobs, done):
        results[i].append(result)  # type: ignore[union-attr]
    return [{**item, result_field: res} for item, res in zip(items, results)]
