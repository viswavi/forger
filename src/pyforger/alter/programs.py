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
with a wall-clock timeout and limits on memory (not on macOS) and on the
size of each file it writes. A timeout kills the program's process group, so
a process that leaves the group (``setsid``, a daemon) is not killed.
Runs use threads and no signals, so the ops work from any thread.

This is NOT a security sandbox. The program runs as your user, with your
files and network. Run untrusted code in a container or VM.

``language="cpp"`` compiles each distinct program once (``g++ -O2 -pipe
-std=gnu++17`` by default) and runs the binary on every test; a program that
does not compile gets status ``"compile_error"`` on every test, with the
compiler's message in ``stderr``.

A checker grades a test from the program's stdout and the expected output. It
returns a bool, or a score in [0, 1] for partial credit; a test passes when it
scores 1. A checker with a third parameter also gets the test, e.g. to read
its input: ``testlib_checker(source)`` runs a testlib checker program as
``checker input output answer`` and scores its verdict.

``runner=`` replaces the subprocess with your own executor, e.g. the sandbox
an RL reward will run the code in, so a solution that passes here also passes
there. A runner takes ``(code, tests)`` and returns one result dict per test
(see ``Runner``); it runs once per row, several rows at a time.
"""
from __future__ import annotations

import contextvars
import inspect
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union

from pyforger.op import op
from pyforger.utils import get_field, get_formatted_logger

# (actual stdout, expected output[, test]) -> passed, or a score in [0, 1]
Checker = Callable[..., Union[bool, float]]
# (code, tests) -> one result per test, in order. Each result is a dict with
# "status" ("ok", "runtime_error", "timeout", ...) and "stdout"; others
# ("stderr", "seconds", "exit_code", "timed_out") are filled in when missing. A
# runner that grades a test itself (e.g. comparing a returned value) sets
# "passed" or "score"; otherwise the checker grades "stdout" against the
# test's "output".
Runner = Callable[[str, list[dict]], list[dict]]

# Compiled languages: the compiler command, with {file} for the source and
# {out} for the binary, and the source file's suffix.
_COMPILERS: dict[str, tuple[list[str], str]] = {
    "cpp": (["g++", "-O2", "-pipe", "-std=gnu++17", "{file}", "-o", "{out}"], ".cpp"),
}
LANGUAGES = ("python", *_COMPILERS)
# Fence tags of each language's code blocks ("" is an untagged block).
_FENCE_TAGS = {
    "python": ("", "python", "py", "python3"),
    "cpp": ("", "cpp", "c++", "cc", "cxx"),
}

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


def last_code_block(text: str, language: str = "python") -> str | None:
    r"""The last fenced code block in ``language``, or untagged.

    Python blocks are tagged python, py, or python3; C++ blocks cpp, c++, cc,
    or cxx. E.g. ``"Idea...\n```python\nprint(1)\n```"`` gives
    ``"print(1)\n"``. None when there is no such block.
    """
    tags = _FENCE_TAGS[_checked_language(language)]
    blocks = [
        code
        for info, code in _FENCE.findall(text or "")
        if info.strip().lower() in tags
    ]
    return blocks[-1] if blocks else None


def _checked_language(language: str) -> str:
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}, got {language!r}")
    return language


def compile_program(
    code: str,
    workdir: str,
    language: str = "cpp",
    *,
    compile_command: list[str] | None = None,
    timeout_s: float = 60.0,
) -> tuple[str | None, str]:
    """Compile ``code`` into ``workdir``; return ``(binary, "")`` or ``(None, error)``.

    ``compile_command`` replaces the default (``g++ -O2 -pipe -std=gnu++17
    {file} -o {out}`` for C++), e.g. ``["clang++", "-std=c++20", "-O2",
    "{file}", "-o", "{out}"]``. The compiler runs with the same bare
    environment as programs. The error is the compiler's output, or a note
    that it timed out. Not an op; safe to call from many threads at once.
    """
    default, suffix = _COMPILERS[_checked_language(language)]
    command = compile_command or default
    if not command:
        raise ValueError("compile_command must not be empty")
    source = "main" + suffix
    binary = os.path.join(workdir, "main")
    with open(os.path.join(workdir, source), "w", encoding="utf-8") as f:
        f.write(code)
    # Relative names, so errors read "main.cpp:1:11: error: ...".
    argv = [part.replace("{file}", source).replace("{out}", "main") for part in command]
    argv[0] = shutil.which(argv[0]) or argv[0]
    try:
        proc = subprocess.run(
            argv,
            cwd=workdir,
            env=dict(_CHILD_ENV),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, f"compilation timed out after {timeout_s:g} s"
    except FileNotFoundError:
        raise OSError(f"compiler not found: {command[0]!r}") from None
    output = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0 or not os.path.exists(binary):
        return None, output or f"compiler exited with status {proc.returncode}"
    return binary, ""


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
        here as a ``MemoryError``), ``"timeout"``, or ``"output_limit"``.
        ``exit_code`` is None after a timeout and negative when a signal
        killed the program.

        ``max_output_mb`` caps every file the program writes (``RLIMIT_FSIZE``:
        stdout, stderr, and scratch files); a write past it fails. The status
        is ``"output_limit"`` when stdout reached the cap (its output is cut
        short), or when the run failed after stderr or a file in its working
        directory reached it. 0 sets no cap.
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
        stdin_path = os.path.join(workdir, "stdin")  # ours, so not size-checked
        with (
            open(stdin_path, "w+", encoding="utf-8") as fin,
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
            # Kill the program's process group: the program and the processes
            # it started, except any that left the group (setsid, daemons).
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
            seconds = time.perf_counter() - start
        exit_code = proc.returncode
        if timed_out:
            status, exit_code = "timeout", None
        elif output_limit and os.path.getsize(out_path) >= output_limit:
            status = "output_limit"
        elif exit_code == 0:
            status = "ok"
        elif output_limit and _wrote_past(workdir, output_limit, {stdin_path, program}):
            status = "output_limit"  # e.g. a write to stderr failed with EFBIG
        else:
            status = "runtime_error"
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


def _wrote_past(workdir: str, limit: int, skip: set[str]) -> bool:
    """Whether a file under ``workdir``, except the paths in ``skip``, hit ``limit``.

    E.g. stderr, or a scratch file the program wrote, at ``max_output_mb``.
    """
    for root, _, files in os.walk(workdir):
        for path in (os.path.join(root, name) for name in files):
            try:
                if path not in skip and os.path.getsize(path) >= limit:
                    return True
            except OSError:  # removed while we looked, or not a regular file
                continue
    return False


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
    runner: Runner | None = None,
    language: str = "python",
    compile_command: list[str] | None = None,
    **run_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run ``code`` on each test in order; one result per test that ran.

    Each result is ``run_program``'s dict plus ``"passed"`` and ``"score"``:
    when the test has an ``"output"``, the checker's score in [0, 1] (a bool
    counts as 1 or 0) and whether it is 1; else both None. A run that exits 0
    and scores 0 gets status ``"wrong_answer"``, one that scores between 0 and
    1 ``"partial"``. ``stop_on_failure`` stops after the first test that does
    not pass. ``max_chars`` cuts each kept stdout and stderr (None keeps them
    whole). ``run_kwargs`` go to ``run_program``. ``language="cpp"`` compiles
    ``code`` once first (see ``compile_program``). ``runner`` runs every test
    in one call instead (see ``Runner``); ``stop_on_failure`` then cuts the
    results after the first failure.
    """
    if runner is not None:
        results = _run_with(runner, code, tests, checker, max_chars)
        if stop_on_failure:
            failed = [i for i, r in enumerate(results) if r["passed"] is not True]
            results = results[: failed[0] + 1] if failed else results
        return results
    if _checked_language(language) == "python":
        return _run_each(code, tests, checker, stop_on_failure, max_chars, run_kwargs)
    _check_compiled_kwargs(run_kwargs)
    with tempfile.TemporaryDirectory(prefix="forger_build_") as build:
        binary, error = compile_program(
            code, build, language, compile_command=compile_command
        )
        if binary is None:
            failures = [_compile_failure(t, error, checker, max_chars) for t in tests]
            return failures[:1] if stop_on_failure else failures
        kwargs = {**run_kwargs, "command": [binary]}
        return _run_each("", tests, checker, stop_on_failure, max_chars, kwargs)


def _run_each(
    code: str,
    tests: list[dict],
    checker: Checker,
    stop_on_failure: bool,
    max_chars: int | None,
    run_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for test in tests:
        result = _run_test(code, test, checker, max_chars, run_kwargs)
        results.append(result)
        if stop_on_failure and result["passed"] is not True:
            break
    return results


def _check_compiled_kwargs(run_kwargs: dict[str, Any]) -> None:
    """A compiled program runs as its binary: ``command`` and ``suffix`` are unused."""
    if (
        run_kwargs.get("command") is not None
        or run_kwargs.get("suffix", ".py") != ".py"
    ):
        raise ValueError(
            "command and suffix are for interpreted programs; with a compiled "
            "language, use compile_command"
        )


def _compile_failure(
    test: dict, error: str, checker: Checker, max_chars: int | None
) -> dict[str, Any]:
    """The result of a test whose program did not compile."""
    _check_test(test)
    return _graded(
        {"status": "compile_error", "stderr": error}, test, checker, max_chars
    )


def _run_test(
    code: str,
    test: dict,
    checker: Checker,
    max_chars: int | None,
    run_kwargs: dict[str, Any],
) -> dict[str, Any]:
    _check_test(test)
    kwargs = dict(run_kwargs)
    if test.get("timeout_s") is not None:
        kwargs["timeout_s"] = float(test["timeout_s"])
    result = run_program(
        code, _stdin_text(test.get("input")), args=test.get("args") or (), **kwargs
    )
    return _graded(result, test, checker, max_chars)


def _check_test(test: Any) -> None:
    if not isinstance(test, dict):
        raise TypeError(f"a test must be a dict with an 'input' key, got {test!r}")


def _run_with(
    runner: Runner,
    code: str,
    tests: list[dict],
    checker: Checker,
    max_chars: int | None,
) -> list[dict[str, Any]]:
    """Run all of one row's tests with ``runner``, then grade each result."""
    for test in tests:
        _check_test(test)
    if not tests:
        return []
    raw = list(runner(code, tests))
    if len(raw) != len(tests):
        raise ValueError(
            f"runner {runner!r} returned {len(raw)} results for {len(tests)} tests"
        )
    return [_graded(dict(r), t, checker, max_chars) for r, t in zip(raw, tests)]


def _graded(
    result: dict[str, Any],
    test: dict,
    checker: Checker,
    max_chars: int | None,
) -> dict[str, Any]:
    """Fill in a run's missing keys and its ``"passed"`` / ``"wrong_answer"``."""
    result = {
        "exit_code": None,
        "timed_out": result.get("status") == "timeout",
        "stdout": "",
        "stderr": "",
        "seconds": 0.0,
        **result,
    }
    result["stdout"] = str(result["stdout"])
    result["stderr"] = str(result["stderr"])
    expected = test.get("output")
    if expected is None:
        score = None
    elif result["status"] != "ok":
        score = 0.0
    elif result.get("score") is not None:  # the runner scored it
        score = _score(result["score"])
    elif result.get("passed") is not None:  # the runner graded it
        score = float(bool(result["passed"]))
    else:
        score = _score(_check(checker, result["stdout"], str(expected), test))
    result["score"] = score
    result["passed"] = None if score is None else score >= 1.0
    if score is not None and score < 1.0 and result["status"] == "ok":
        result["status"] = "partial" if score > 0.0 else "wrong_answer"
    if max_chars is not None:
        result["stdout"] = result["stdout"][:max_chars]
        # The end of stderr, where a traceback's error is.
        result["stderr"] = result["stderr"][-max_chars:] if max_chars else ""
    return result


def _check(checker: Checker, stdout: str, expected: str, test: dict) -> Any:
    """Call ``checker``, passing the test too when it takes a third argument."""
    if _takes_test(checker):
        return checker(stdout, expected, test)
    return checker(stdout, expected)


def _takes_test(checker: Checker) -> bool:
    try:
        params = inspect.signature(checker).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 3 or any(p.kind is p.VAR_POSITIONAL for p in params)


def _score(value: Any) -> float:
    """A checker's verdict as a score in [0, 1]: a bool is 1 or 0."""
    if isinstance(value, bool) or value is None:
        return float(bool(value))
    score = float(value)
    if math.isnan(score):
        return 0.0
    return min(1.0, max(0.0, score))


def _stdin_text(value: Any) -> str:
    """Stdin for a test's ``"input"``: missing or None is empty, not ``"None"``."""
    return "" if value is None else str(value)


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


@op(name="alter.run_programs")
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
    runner: Runner | None = None,
    language: str = "python",
    compile_command: list[str] | None = None,
) -> list[dict]:
    r"""Run each row's stdin/stdout program on each of its tests.

    Every row gets ``result_field``: one result per test, in test order, e.g.
    ``{"status": "wrong_answer", "passed": False, "score": 0.0, "exit_code":
    0, "timed_out": False, "stdout": "5\n", "stderr": "", "seconds": 0.03}``.
    ``status`` is ``"ok"``, ``"wrong_answer"``, ``"partial"`` (a score
    between 0 and 1), ``"runtime_error"``, ``"timeout"``, ``"output_limit"``,
    or ``"compile_error"``; ``passed`` and ``score`` are None for a test
    without ``"output"``. A row whose code is missing or empty gets ``None``.

    Args:
        items: Input rows.
        code_field: Key holding the program source.
        tests_field: Key holding the row's tests, a list of ``{"input",
            "output"?, "args"?, "timeout_s"?}`` dicts.
        result_field: Key the results are written to.
        extract_code: Optional ``str -> str | None`` applied to the field
            first, e.g. ``last_code_block`` to take the code out of a model's
            answer. None uses the field as is.
        checker: ``(stdout, expected) -> bool | float``, or ``(stdout,
            expected, test)``; a float is a partial score in [0, 1]. Default
            ``outputs_match`` (equal up to whitespace); ``testlib_checker``
            runs a testlib checker program.
        timeout_s: Wall-clock limit per run, in seconds.
        memory_mb: Address-space limit per run (None for none). Not enforced
            on macOS, which has no working ``RLIMIT_AS``.
        max_output_mb: Size cap on every file a run writes (stdout, stderr,
            scratch files); see ``run_program`` for when it gives
            ``output_limit``. 0 for none.
        max_chars: Characters of stdout (first) and stderr (last) kept per
            result. None keeps them whole, e.g. to use a reference program's
            outputs as expected outputs.
        command: Interpreter command for non-Python programs, with ``{file}``
            for the program file, e.g. ``["node", "{file}"]`` and
            ``suffix=".js"``. None runs Python (this interpreter, ``-I``).
        suffix: File name suffix for the program.
        max_workers: Runs at a time across all rows (default
            ``min(8, cpu count)``). Part of the cache key: timeouts are
            wall-clock, so fewer workers can turn timeouts into passes.
        runner: Your own executor, called once per row with ``(code, tests)``
            (see ``Runner``), ``max_workers`` rows at a time; the limits above
            and ``command`` then do not apply. E.g. run solutions in the
            sandbox that will score them, so what passes here passes there.
            Part of the cache key: give it a ``cache_key()`` method.
        language: ``"python"`` (default) or ``"cpp"``, which compiles each
            distinct program once and runs the binary on every test.
        compile_command: The compiler command for ``language="cpp"``, with
            ``{file}`` and ``{out}``; default ``g++ -O2 -pipe -std=gnu++17``.

    Returns:
        New rows (inputs are not mutated), in input order.
    """
    _checked_language(language)
    workers = max_workers or _default_workers()
    if workers < 1:
        raise ValueError("max_workers must be at least 1")
    if runner is not None:
        return _run_rows_with(
            items,
            runner,
            code_field,
            tests_field,
            result_field,
            extract_code,
            checker,
            max_chars,
            workers,
        )
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
    if language == "python":
        results = _run_jobs(codes, tests, checker, max_chars, run_kwargs, workers)
    else:
        _check_compiled_kwargs(run_kwargs)
        with tempfile.TemporaryDirectory(prefix="forger_build_") as build:
            built = _compile_all(codes, build, language, compile_command, workers)
            results = _run_jobs(
                codes, tests, checker, max_chars, run_kwargs, workers, built
            )
    return [{**item, result_field: res} for item, res in zip(items, results)]


def _compile_all(
    codes: list[str | None],
    build: str,
    language: str,
    compile_command: list[str] | None,
    workers: int,
) -> dict[str, tuple[str | None, str]]:
    """Compile each distinct program once, in its own directory under ``build``."""
    distinct = list(dict.fromkeys(c for c in codes if c is not None))
    dirs = [os.path.join(build, str(k)) for k in range(len(distinct))]
    for path in dirs:
        os.mkdir(path)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        compiled = list(
            pool.map(
                lambda job: compile_program(
                    job[0], job[1], language, compile_command=compile_command
                ),
                zip(distinct, dirs),
            )
        )
    return dict(zip(distinct, compiled))


def _run_jobs(
    codes: list[str | None],
    tests: list[list[dict]],
    checker: Checker,
    max_chars: int | None,
    run_kwargs: dict[str, Any],
    workers: int,
    built: dict[str, tuple[str | None, str]] | None = None,
) -> list[list[dict] | None]:
    """Every test of every row with code, ``workers`` runs at a time.

    With ``built`` (compiled programs by source), each row runs its binary,
    and a row whose program did not compile gets a compile error per test.
    """

    def run(i: int, j: int) -> dict[str, Any]:
        code = codes[i] or ""
        if built is None:
            return _run_test(code, tests[i][j], checker, max_chars, run_kwargs)
        binary, error = built[code]
        if binary is None:
            return _compile_failure(tests[i][j], error, checker, max_chars)
        kwargs = {**run_kwargs, "command": [binary]}
        return _run_test("", tests[i][j], checker, max_chars, kwargs)

    jobs = [(i, j) for i, row_tests in enumerate(tests) for j in range(len(row_tests))]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, i, j) for i, j in jobs]
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
    return results


# testlib's exit codes for a verdict that scores: _ok, and _points (quitp).
_TESTLIB_SCORING_EXITS = (0, 7)
_RATIO_RE = re.compile(r"Ratio: ([\d.]+)")


def testlib_checker(
    source: str,
    *,
    include_dirs: list[str] | tuple[str, ...] = (),
    compile_command: list[str] | None = None,
    timeout_s: float = 10.0,
    memory_mb: int | None = 256,
) -> TestlibChecker:
    """A checker that runs a testlib checker program on each test.

    ``source`` is the checker's C++ source (``#include "testlib.h"``), found
    through ``include_dirs``; it is compiled once, on first use. Each test runs
    it as ``checker input output answer``, with the test's ``"input"``, the
    program's stdout, and the test's ``"output"`` as files. The score is the
    ``Ratio: x`` the checker prints when it exits with testlib's ``_ok`` (0)
    or ``_points`` (7), else 1 for those exits; any other exit (``_wa``,
    ``_pe``, ``_fail``, a crash, a timeout) scores 0. E.g.
    ``run_programs(rows, code_field="solution", language="cpp",
    checker=testlib_checker(chk, include_dirs=["rl/testlib"]))``. A checker
    that does not compile raises, naming the compiler's error.
    """
    return TestlibChecker(
        source, tuple(include_dirs), compile_command, timeout_s, memory_mb
    )


# Their names start with "test"; keep pytest from collecting them where imported.
testlib_checker.__test__ = False  # type: ignore[attr-defined]


class TestlibChecker:
    """``(stdout, expected, test) -> score``: see ``testlib_checker``."""

    __test__ = False

    def __init__(
        self,
        source: str,
        include_dirs: tuple[str, ...],
        compile_command: list[str] | None,
        timeout_s: float,
        memory_mb: int | None,
    ) -> None:
        """Keep the settings; the checker compiles on first use."""
        self.source = source
        self.include_dirs = include_dirs
        self.compile_command = compile_command
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self._lock = threading.Lock()
        self._binary: str | None = None

    def cache_key(self) -> tuple:
        """Settings only, so equal checkers share cached results."""
        return (
            "testlib_checker",
            self.source,
            self.include_dirs,
            self.compile_command,
            self.timeout_s,
            self.memory_mb,
        )

    def __repr__(self) -> str:
        """Short name for error messages and reports."""
        return "testlib_checker(...)"

    def _compiled(self) -> str:
        with self._lock:
            if self._binary is None:
                build = tempfile.mkdtemp(prefix="forger_checker_")
                command = self.compile_command or [
                    *_COMPILERS["cpp"][0][:-3],
                    *(f"-I{os.path.abspath(d)}" for d in self.include_dirs),
                    "{file}",
                    "-o",
                    "{out}",
                ]
                binary, error = compile_program(
                    self.source, build, "cpp", compile_command=command
                )
                if binary is None:
                    shutil.rmtree(build, ignore_errors=True)
                    raise RuntimeError(f"testlib checker does not compile:\n{error}")
                weakref.finalize(self, shutil.rmtree, build, True)
                self._binary = binary
            return self._binary

    def __call__(self, stdout: str, expected: str, test: dict) -> float:
        """Score one test's output with the checker."""
        binary = self._compiled()
        with tempfile.TemporaryDirectory(prefix="forger_check_") as work:
            paths = []
            for name, text in (
                ("input.txt", _stdin_text(test.get("input"))),
                ("output.txt", stdout),
                ("answer.txt", expected),
            ):
                path = os.path.join(work, name)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                paths.append(path)
            run = run_program(
                "",
                args=paths,
                command=[binary],
                timeout_s=self.timeout_s,
                memory_mb=self.memory_mb,
            )
        if run["timed_out"] or run["exit_code"] not in _TESTLIB_SCORING_EXITS:
            return 0.0
        match = _RATIO_RE.search(run["stdout"] or run["stderr"])
        return float(match.group(1)) if match else 1.0


def _context_run() -> Callable[..., Any]:
    """A copy of this context's run, typed for Executor.submit (as in model.py)."""
    run: Callable[..., Any] = contextvars.copy_context().run
    return run


def _run_rows_with(
    items: list[dict],
    runner: Runner,
    code_field: str,
    tests_field: str,
    result_field: str,
    extract_code: Callable[[str], str | None] | None,
    checker: Checker,
    max_chars: int | None,
    workers: int,
) -> list[dict]:
    """``run_programs`` with a runner: one runner call per row with code."""
    codes = [_code_of(item, code_field, extract_code) for item in items]
    tests = [list(get_field(item, tests_field, "run_programs") or []) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # A copied context per call, so ops a runner calls stay attributed.
        futures = {
            i: pool.submit(
                _context_run(),
                _run_with,
                runner,
                code,
                tests[i],
                checker,
                max_chars,
            )
            for i, code in enumerate(codes)
            if code is not None
        }
        try:
            done = {i: f.result() for i, f in futures.items()}
        except BaseException:
            for f in futures.values():
                f.cancel()
            raise
    return [{**item, result_field: done.get(i)} for i, item in enumerate(items)]
