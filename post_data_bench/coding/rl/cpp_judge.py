"""Grade one C++ solution on one problem with testlib checkers or interactors.

This follows a go-judge-based judge engine, running programs as local processes
under rlimits instead of go-judge sandboxes:

- The solution compiles with ``g++ main.cpp -O2 -pipe -std=gnu++17`` (10 s CPU).
- Each test runs it with the test's time limit as CPU limit and twice that as wall
  limit, its memory limit as address-space and stack limit, and the test input on
  stdin. A run that is not clean (time, memory, signal, nonzero exit) scores 0.
- A checker runs as ``chk in.txt out.txt ans.txt`` (10 s CPU, 256 MiB). For an
  interactive problem the interactor runs as ``interactor in.txt tout.txt ans.txt``
  with its stdout piped to the solution's stdin and back (4x the time and memory).
- A test's ratio is the ``Ratio: x`` that the checker or interactor prints when it
  exits 0 or 7 (testlib's ``_ok`` and ``points``), else 1 for an accepted test and 0
  otherwise. The problem score is the mean ratio over every test file, times 100.

Only the checker's or interactor's own output is parsed, never the solution's.

    python -m rl.cpp_judge --problem-dir problems/1 --testlib-dir testlib sol.cpp
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml  # type: ignore[import]

MIB = 1 << 20
TESTLIB_OK_EXIT_CODE = 0
TESTLIB_POINTS_EXIT_CODE = 7
COMPILE_CPU_S = 10.0
COMPILE_MEMORY = 512 * MIB
CHECKER_COMPILE_CPU_S = 30.0
CHECKER_CPU_S = 10.0
CHECKER_MEMORY = 256 * MIB
STDOUT_MAX = 128 * MIB
STDERR_MAX = 1 * MIB
# go-judge counts resident memory for runs without an address-space limit; a
# process gets this much address space so it cannot take down the node.
SAFETY_ADDRESS_SPACE = 16 << 30


@dataclasses.dataclass
class Case:
    """One test file: input and answer names under testdata/ and its limits."""

    input: str
    output: str
    time_s: float
    memory: int


@dataclasses.dataclass
class Problem:
    """A problem directory's judging setup, as problem_manager.js loads it."""

    pdir: Path
    type: str
    checker: str | None
    interactor: str | None
    cases: list[Case]


@dataclasses.dataclass
class Run:
    """The outcome of one process, in go-judge's status vocabulary."""

    status: str  # Accepted, Time Limit Exceeded, Memory Limit Exceeded,
    # Output Limit Exceeded, Signalled, Nonzero Exit Status
    exit_status: int
    cpu_s: float
    memory: int
    stdout: bytes = b""
    stderr: bytes = b""


def to_seconds(value) -> float:
    """``toNs`` in utils.js, in seconds: '1s', '500ms', or a number of seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    m = re.match(r"^([\d.]+)\s*(ms|s)?$", str(value), re.IGNORECASE)
    number = float(m.group(1)) if m else 0.0
    unit = (m.group(2) if m and m.group(2) else "s").lower()
    return number / 1000 if unit == "ms" else number


def to_bytes(value) -> int:
    """``toBytes`` in utils.js: '256m', '1g', '512k', or a number of bytes."""
    if isinstance(value, (int, float)):
        return int(value)
    m = re.match(r"^([\d.]+)\s*(k|m|g|)$|^([\d.]+)$", str(value), re.IGNORECASE)
    number = float((m.group(1) or m.group(3)) if m else 0)
    unit = ((m.group(2) or "") if m else "").lower()
    return round(number * {"g": 1 << 30, "m": MIB, "k": 1 << 10}.get(unit, 1))


def _discovered_cases(testdata: Path) -> list[tuple[str, str]]:
    """``findTestCases``: every .in file with a matching .ans or .out, by name."""
    files = os.listdir(testdata)
    inputs = {f[:-3]: f for f in files if f.endswith(".in")}
    outputs: dict[str, str] = {}
    for f in files:
        if f.endswith(".ans") or f.endswith(".out"):
            outputs[f[:-4]] = f  # the later of .ans/.out in listing order, as in JS
    pairs = [(base, inputs[base], outputs[base]) for base in inputs if base in outputs]
    pairs.sort(key=lambda p: p[0])
    return [(i, o) for _, i, o in pairs]


def load_problem(pdir: str | Path) -> Problem:
    """Load config.yaml's subtasks plus every other complete test pair in testdata/."""
    pdir = Path(pdir)
    cfg = yaml.safe_load((pdir / "config.yaml").read_text())
    if not isinstance(cfg.get("subtasks"), list):
        raise ValueError(f"{pdir}: config.yaml must define subtasks")
    kind = cfg.get("type") or "default"
    if kind == "leetcode":
        raise ValueError(f"{pdir}: LeetCode problems are not supported")
    global_time = cfg.get("time_limit") or cfg.get("time") or "1s"
    global_memory = cfg.get("memory_limit") or cfg.get("memory") or "256m"
    in_prefix = cfg.get("input_prefix") or ""
    out_prefix = cfg.get("output_prefix") or ""
    in_suffix = cfg.get("input_suffix") or ".in"
    out_suffix = cfg.get("output_suffix") or ".ans"
    cases: list[Case] = []
    current = 1
    for si, st in enumerate(cfg["subtasks"]):
        st_time = st.get("time_limit") or st.get("time") or global_time
        st_memory = st.get("memory_limit") or st.get("memory") or global_memory
        if st.get("n_cases") is not None:
            for i in range(st["n_cases"]):
                cases.append(
                    Case(
                        f"{in_prefix}{current + i}{in_suffix}",
                        f"{out_prefix}{current + i}{out_suffix}",
                        to_seconds(st_time),
                        to_bytes(st_memory),
                    )
                )
            current += st["n_cases"]
        elif isinstance(st.get("cases"), list):
            for c in st["cases"]:
                cases.append(
                    Case(
                        c["input"],
                        c["output"],
                        to_seconds(c.get("time") or st_time),
                        to_bytes(c.get("memory") or st_memory),
                    )
                )
        else:
            raise ValueError(f"{pdir}: subtask {si} needs n_cases or cases")
    # Released packages carry former-internal tests beyond n_cases: judge them too,
    # with the last subtask's limits.
    last = cfg["subtasks"][-1] if cfg["subtasks"] else {}
    fallback_time = to_seconds(
        last.get("time_limit") or last.get("time") or global_time
    )
    fallback_memory = to_bytes(
        last.get("memory_limit") or last.get("memory") or global_memory
    )
    configured = {c.input for c in cases}
    for inp, out in _discovered_cases(pdir / "testdata"):
        if inp not in configured:
            cases.append(Case(inp, out, fallback_time, fallback_memory))
    interactive = kind == "interactive"
    return Problem(
        pdir=pdir,
        type=kind,
        checker=None if interactive else (cfg.get("checker") or "chk.cc"),
        interactor=(cfg.get("interactor") or "interactor.cc") if interactive else None,
        cases=cases,
    )


def _answer_path(pdir: Path, case: Case) -> Path:
    """The .ans twin of a .out answer when it exists, as judgeCase reads it."""
    ans = pdir / "testdata" / re.sub(r"\.out$", ".ans", case.output)
    return ans if ans.exists() else pdir / "testdata" / case.output


# The launcher sets the rlimits and reports the program's own CPU time and peak
# resident memory. Measuring from Python instead would be wrong: Linux carries a
# process's peak RSS across exec, so a child forked from a large Python worker
# would inherit the worker's peak and look like it used that much memory.
_LAUNCHER_SOURCE = r"""
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
/* fcs_run REPORT CPU_S AS STACK FSIZE PROG ARGS... */
static void lim(int what, long long v) {
    if (v <= 0) return;
    struct rlimit r = {(rlim_t)v, (rlim_t)v};
    setrlimit(what, &r);
}
int main(int argc, char** argv) {
    if (argc < 7) return 2;
    pid_t pid = fork();
    if (pid < 0) return 3;
    if (pid == 0) {
        lim(RLIMIT_CPU, atoll(argv[2]));
        struct rlimit core = {0, 0};
        setrlimit(RLIMIT_CORE, &core);
        lim(RLIMIT_AS, atoll(argv[3]));
        lim(RLIMIT_STACK, atoll(argv[4]));
        lim(RLIMIT_FSIZE, atoll(argv[5]));
        execvp(argv[6], argv + 6);
        _exit(127);
    }
    int st = 0;
    struct rusage ru;
    while (wait4(pid, &st, 0, &ru) < 0)
        if (errno != EINTR) return 4;
    FILE* f = fopen(argv[1], "w");
    if (!f) return 5;
    fprintf(f, "%.6f %ld %d %d\n",
            ru.ru_utime.tv_sec + ru.ru_stime.tv_sec
                + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e6,
            ru.ru_maxrss, WIFEXITED(st) ? WEXITSTATUS(st) : 0,
            WIFSIGNALED(st) ? WTERMSIG(st) : 0);
    fclose(f);
    return 0;
}
"""
_launchers: dict[Path, Path] = {}
_launcher_lock = threading.Lock()


def launcher(cache_dir: Path) -> Path:
    """The compiled process launcher in ``cache_dir`` (built once per source)."""
    cache_dir = Path(cache_dir)
    with _launcher_lock:
        if cache_dir in _launchers:
            return _launchers[cache_dir]
        digest = hashlib.sha256(_LAUNCHER_SOURCE.encode()).hexdigest()[:16]
        binary = cache_dir / f"fcs_run-{digest}"
        if not binary.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".build-", dir=cache_dir) as tmp:
                src = Path(tmp) / "fcs_run.c"
                src.write_text(_LAUNCHER_SOURCE)
                subprocess.run(
                    ["gcc", "-O2", "-o", str(Path(tmp) / "bin"), str(src)],
                    check=True,
                    capture_output=True,
                )
                os.replace(Path(tmp) / "bin", binary)
        _launchers[cache_dir] = binary
        return binary


def _drain(stream, limit: int, sink: list, over: threading.Event) -> None:
    """Read a pipe to EOF, keeping at most ``limit`` bytes and flagging overflow."""
    kept = 0
    while True:
        chunk = stream.read(1 << 16)
        if not chunk:
            break
        if kept < limit:
            sink.append(chunk[: limit - kept])
        kept += len(chunk)
        if kept > limit:
            over.set()
    stream.close()


class _Process:
    """A started process whose rusage we collect with wait4."""

    def __init__(
        self,
        args,
        *,
        launch: Path,
        cwd,
        stdin,
        stdout,
        cpu_s,
        wall_s,
        memory,
        address_space: bool,
        stdout_max=STDOUT_MAX,
    ):
        self.cpu_limit = cpu_s
        self.memory = memory
        self.killed_for_wall = False
        self.report = tempfile.NamedTemporaryFile(prefix="fcs-run-", delete=False)
        self.report.close()
        self.proc = subprocess.Popen(
            [
                str(launch),
                self.report.name,
                str(math.ceil(cpu_s) + 1),  # hard stop just above the limit
                str(memory if address_space else SAFETY_ADDRESS_SPACE),
                str(memory),
                str(STDOUT_MAX * 2),
                *args,
            ],
            cwd=cwd,
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.PIPE,
            env={"PATH": "/usr/bin:/bin"},
            start_new_session=True,
        )
        self.stdout_chunks: list[bytes] = []
        self.stderr_chunks: list[bytes] = []
        self.stdout_over = threading.Event()
        self.stderr_over = threading.Event()
        self.readers = [
            threading.Thread(
                target=_drain,
                args=(
                    self.proc.stderr,
                    STDERR_MAX,
                    self.stderr_chunks,
                    self.stderr_over,
                ),
                daemon=True,
            )
        ]
        if stdout == subprocess.PIPE:
            self.readers.append(
                threading.Thread(
                    target=_drain,
                    args=(
                        self.proc.stdout,
                        stdout_max,
                        self.stdout_chunks,
                        self.stdout_over,
                    ),
                    daemon=True,
                )
            )
        for reader in self.readers:
            reader.start()
        self.timer = threading.Timer(wall_s, self._kill_for_wall)
        self.timer.start()

    def _kill_for_wall(self) -> None:
        self.killed_for_wall = True
        self.kill()

    def kill(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def wait(self) -> Run:
        self.proc.wait()
        self.timer.cancel()
        self.kill()  # reap any children left in the group
        for reader in self.readers:
            reader.join()
        report = Path(self.report.name)
        fields = report.read_text().split() if report.exists() else []
        report.unlink(missing_ok=True)
        if len(fields) == 4:
            cpu, memory = float(fields[0]), int(fields[1]) * 1024
            code = -int(fields[3]) if int(fields[3]) else int(fields[2])
        else:  # killed before the launcher could report (wall clock)
            cpu, memory, code = self.cpu_limit, 0, -signal.SIGKILL
        if (
            cpu > self.cpu_limit
            or self.killed_for_wall
            or (code < 0 and -code == signal.SIGXCPU)
        ):
            status = "Time Limit Exceeded"
        elif memory > self.memory:
            status = "Memory Limit Exceeded"
        elif self.stdout_over.is_set():
            status = "Output Limit Exceeded"
        elif code < 0:
            status = "Signalled"
        elif code > 0:
            status = "Nonzero Exit Status"
        else:
            status = "Accepted"
        return Run(
            status=status,
            exit_status=code,
            cpu_s=cpu,
            memory=memory,
            stdout=b"".join(self.stdout_chunks),
            stderr=b"".join(self.stderr_chunks),
        )


def compile_cpp(
    source: Path,
    binary: Path,
    include: Path | None = None,
    cpu_s: float = COMPILE_CPU_S,
    *,
    launch: Path,
) -> tuple[bool, str]:
    """``g++ src -O2 -pipe -std=gnu++17 [-I include] -o binary``; (ok, message)."""
    args = ["g++", source.name, "-O2", "-pipe", "-std=gnu++17"]
    if include is not None:
        args += ["-I", str(include)]
    args += ["-o", str(binary)]
    # The compiler itself gets a generous wall clock: the CPU limit is the rule.
    proc = _Process(
        args,
        launch=launch,
        cwd=source.parent,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        cpu_s=cpu_s,
        wall_s=cpu_s * 6,
        memory=COMPILE_MEMORY * 8,  # cc1plus maps far more than it touches
        address_space=False,
    )
    run = proc.wait()
    message = (run.stdout + run.stderr).decode(errors="replace")[-4000:]
    return run.status == "Accepted" and binary.exists(), message


_judge_cache_lock = threading.Lock()


def compiled_judge_program(
    problem: Problem, testlib_dir: Path, cache_dir: Path
) -> Path:
    """The problem's checker or interactor binary, compiled once per source."""
    name = problem.interactor or problem.checker
    if name is None:
        raise ValueError(f"{problem.pdir}: no checker or interactor")
    source = problem.pdir / name
    digest = hashlib.sha256(
        source.read_bytes() + (testlib_dir / "testlib.h").read_bytes()
    ).hexdigest()[:16]
    binary = cache_dir / f"{problem.pdir.name}-{Path(name).stem}-{digest}"
    with _judge_cache_lock:
        if not binary.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Build next to the cache so the final rename stays on one filesystem.
            with tempfile.TemporaryDirectory(prefix=".build-", dir=cache_dir) as work:
                src = Path(work) / source.name
                shutil.copy(source, src)
                ok, message = compile_cpp(
                    src,
                    Path(work) / "bin",
                    include=testlib_dir,
                    cpu_s=CHECKER_COMPILE_CPU_S,
                    launch=launcher(cache_dir),
                )
                if not ok:
                    raise RuntimeError(f"{source} does not compile:\n{message}")
                os.replace(Path(work) / "bin", binary)
    return binary


def _parse_ratio(message: str, name: str) -> float | None:
    """The first ``{name}: x`` in a judge message, if x is a finite number."""
    m = re.search(name + r": ([\d.]+)", message)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _is_scoring_verdict(run: Run) -> bool:
    """The checker/interactor finished on its own with testlib's _ok or points."""
    return run.status in ("Accepted", "Nonzero Exit Status") and run.exit_status in (
        TESTLIB_OK_EXIT_CODE,
        TESTLIB_POINTS_EXIT_CODE,
    )


def _case_result(ok: bool, status: str, run: Run, msg: str, judge_msg: str) -> dict:
    """``applyScoreRatio``: the bounded and unbounded ratios from the judge message."""
    bounded = _parse_ratio(judge_msg, "Ratio")
    unbounded = _parse_ratio(judge_msg, "RatioUnbounded")
    ratio = bounded if bounded is not None else (1.0 if ok else 0.0)
    return {
        "ok": ok,
        "status": status,
        "time": round(run.cpu_s, 3),
        "memory": run.memory,
        "ratio": ratio,
        "ratio_unbounded": unbounded if unbounded is not None else ratio,
        "msg": msg[-500:],
    }


def judge_case(
    problem: Problem, case: Case, binary: Path, checker: Path, launch: Path
) -> dict:
    """judgeCase: run the solution on the input, then the checker on its output."""
    inp = problem.pdir / "testdata" / case.input
    with tempfile.TemporaryDirectory(prefix="fcs-case-") as tmp:
        work = Path(tmp)
        with open(inp, "rb") as stdin:
            run = _Process(
                [str(binary)],
                launch=launch,
                cwd=work,
                stdin=stdin,
                stdout=subprocess.PIPE,
                cpu_s=case.time_s,
                wall_s=case.time_s * 2,
                memory=case.memory,
                address_space=True,
            ).wait()
        if run.status != "Accepted":
            extra = f" (signal={-run.exit_status})" if run.status == "Signalled" else ""
            msg = run.stderr.decode(errors="replace") + extra
            return _case_result(False, run.status, run, msg, "")
        (work / "out.txt").write_bytes(run.stdout)
        shutil.copy(inp, work / "in.txt")
        shutil.copy(_answer_path(problem.pdir, case), work / "ans.txt")
        chk = _Process(
            [str(checker), "in.txt", "out.txt", "ans.txt"],
            launch=launch,
            cwd=work,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            cpu_s=CHECKER_CPU_S,
            wall_s=CHECKER_CPU_S * 2,
            memory=CHECKER_MEMORY,
            address_space=False,
            stdout_max=STDERR_MAX,
        ).wait()
    ok = chk.status == "Accepted" and chk.exit_status == 0
    msg = (chk.stdout or chk.stderr).decode(errors="replace")
    return _case_result(
        ok,
        "Accepted" if ok else "Wrong Answer",
        run,
        msg,
        msg if _is_scoring_verdict(chk) else "",
    )


def judge_interactive_case(
    problem: Problem, case: Case, binary: Path, interactor: Path, launch: Path
) -> dict:
    """judgeInteractiveCase: the solution and interactor talk through two pipes."""
    with tempfile.TemporaryDirectory(prefix="fcs-int-") as tmp:
        work = Path(tmp)
        shutil.copy(problem.pdir / "testdata" / case.input, work / "in.txt")
        shutil.copy(_answer_path(problem.pdir, case), work / "ans.txt")
        to_interactor_r, to_interactor_w = os.pipe()
        to_solution_r, to_solution_w = os.pipe()
        sol_dir = work / "sol"
        sol_dir.mkdir()
        try:
            solution = _Process(
                [str(binary)],
                launch=launch,
                cwd=sol_dir,
                stdin=to_solution_r,
                stdout=to_interactor_w,
                cpu_s=case.time_s,
                wall_s=case.time_s * 2,
                memory=case.memory,
                address_space=False,
            )
            judge = _Process(
                [str(interactor), "in.txt", "tout.txt", "ans.txt"],
                launch=launch,
                cwd=work,
                stdin=to_interactor_r,
                stdout=to_solution_w,
                cpu_s=case.time_s * 4,
                wall_s=case.time_s * 8,
                memory=case.memory * 4,
                address_space=False,
            )
        finally:
            for fd in (to_interactor_r, to_interactor_w, to_solution_r, to_solution_w):
                os.close(fd)
        sub = solution.wait()
        inter = judge.wait()
    # The interactor's stdout is the pipe, so its message is its stderr.
    inter_msg = inter.stderr.decode(errors="replace")
    inter_judge_msg = inter_msg if _is_scoring_verdict(inter) else ""
    if (
        sub.status == "Accepted"
        and inter.status == "Accepted"
        and inter.exit_status == 0
        and sub.exit_status == 0
    ):
        return _case_result(True, "Accepted", sub, inter_msg, inter_judge_msg)
    if sub.status != "Accepted":
        # The run itself failed: no credit, whatever either side printed.
        extra = f" (signal={-sub.exit_status})" if sub.status == "Signalled" else ""
        msg = sub.stderr.decode(errors="replace") + extra
        result = _case_result(False, sub.status, sub, msg, "")
        result[
            "interactor"
        ] = f"{inter.status} exit={inter.exit_status}: {inter_msg[-300:]}"
        return result
    if inter.status != "Accepted":
        return _case_result(False, inter.status, sub, inter_msg, inter_judge_msg)
    return _case_result(False, "Wrong Answer", sub, inter_msg, inter_judge_msg)


def judge(
    problem_dir: str | Path,
    code: str,
    *,
    testlib_dir: str | Path,
    cache_dir: str | Path | None = None,
    workers: int = 8,
) -> dict:
    """Score ``code`` on every test of the problem: ``score`` is 0-100.

    Returns ``score``, ``score_unbounded``, ``passed`` (every ratio is 1),
    ``status`` ("done" or "compile_error"), ``cases`` (one dict per test), and
    the compiler ``message`` on a compile error (which scores 0).
    """
    problem = load_problem(problem_dir)
    testlib_dir = Path(testlib_dir)
    cache_dir = Path(cache_dir or Path(tempfile.gettempdir()) / "fcs-judge-cache")
    judge_bin = compiled_judge_program(problem, testlib_dir, cache_dir)
    launch = launcher(cache_dir)
    with tempfile.TemporaryDirectory(prefix="fcs-sol-") as tmp:
        work = Path(tmp)
        source = work / "main.cpp"
        source.write_text(code)
        binary = work / "main"
        ok, message = compile_cpp(source, binary, launch=launch)
        if not ok:
            return {
                "status": "compile_error",
                "score": 0.0,
                "score_unbounded": 0.0,
                "passed": False,
                "cases": [],
                "message": message,
            }
        run = judge_interactive_case if problem.interactor else judge_case
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            cases = list(
                pool.map(
                    lambda c: run(problem, c, binary, judge_bin, launch), problem.cases
                )
            )
    n = len(cases)
    return {
        "status": "done",
        "score": 100 * sum(c["ratio"] for c in cases) / n if n else 0.0,
        "score_unbounded": 100 * sum(c["ratio_unbounded"] for c in cases) / n
        if n
        else 0.0,
        "passed": all(c["ratio"] == 1.0 for c in cases),
        "cases": cases,
    }


def main() -> None:
    """Judge one solution file and print the result as JSON."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("solution", type=Path)
    p.add_argument("--problem-dir", type=Path, required=True)
    p.add_argument("--testlib-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    start = time.monotonic()
    result = judge(
        args.problem_dir,
        args.solution.read_text(),
        testlib_dir=args.testlib_dir,
        cache_dir=args.cache_dir,
        workers=args.workers,
    )
    result["seconds"] = round(time.monotonic() - start, 2)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
