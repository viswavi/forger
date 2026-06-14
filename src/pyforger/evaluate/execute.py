"""Hardened subprocess executor for HumanEval pass@1 scoring.

Runs ``prompt + body + test + check(entry_point)`` (the standard HumanEval
contract) in a fresh ``python`` subprocess with a wall-clock timeout, an
RLIMIT_AS memory cap (best-effort; macOS sometimes refuses), and a scrubbed
env (``HF_*``, ``OPENAI_*``, etc. dropped). RLIMIT_CPU is intentionally
unset because macOS process-time accounting fires it before the wall-clock
kill and masks the cleaner ``timeout`` outcome.

This is hardening, not a true sandbox: only use it on trusted machines.
Callers must pass ``allow_code_execution=True`` so the unsafe path is never
the silent default of importing :mod:`pyforger.evaluate`.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

_PASS = "pass"
_FAIL = "fail"
_TIMEOUT = "timeout"
_ERROR = "error"


@dataclass(frozen=True)
class ScoreOutcome:
    """Result of running one candidate program against its HumanEval test.

    Attributes:
        outcome: ``"pass"`` (assertions held), ``"fail"`` (assertion raised),
            ``"timeout"`` (wall-clock kill), or ``"error"`` (anything else,
            e.g. ``SyntaxError`` / ``ImportError``).
        returncode: Subprocess exit code; ``None`` for ``timeout``.
        stdout: Captured stdout, truncated.
        stderr: Captured stderr, truncated.
        duration_s: Wall-clock seconds spent in the subprocess.
    """

    outcome: str
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float

    @property
    def passed(self) -> bool:
        return self.outcome == _PASS


_OUTPUT_TRUNCATE_BYTES = 4096


def _truncate(s: str) -> str:
    if len(s) <= _OUTPUT_TRUNCATE_BYTES:
        return s
    head = s[: _OUTPUT_TRUNCATE_BYTES]
    return head + f"\n...[truncated {len(s) - _OUTPUT_TRUNCATE_BYTES} bytes]"


def _build_program(prompt: str, body: str, test: str, entry_point: str) -> str:
    parts = [prompt, body, "\n", test, "\n", f"check({entry_point})\n"]
    return "".join(parts)


def _make_preexec(memory_limit_mb: int):
    """Return a ``preexec_fn`` that caps RLIMIT_AS (best-effort; ``None`` on Windows)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Unix-only project
        return None

    mem_limit = max(1, memory_limit_mb) * 1024 * 1024

    def _preexec() -> None:  # pragma: no cover - runs in subprocess
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
        except (ValueError, OSError):
            pass

    return _preexec


def _safe_env() -> dict[str, str]:
    """Minimal env (keeps PATH/HOME/locale/PYTHONPATH; drops HF/OpenAI/WANDB secrets)."""
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH")
    env: dict[str, str] = {}
    for key in keep:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def score_program(
    prompt: str,
    body: str,
    test: str,
    entry_point: str,
    *,
    timeout_s: int = 10,
    memory_limit_mb: int = 1024,
    allow_code_execution: bool = False,
) -> ScoreOutcome:
    """Execute the assembled HumanEval program and report whether it passed.

    Args:
        prompt: HumanEval ``prompt`` field (signature + docstring).
        body: candidate function body, typically from :func:`extract_code`.
        test: HumanEval ``test`` field defining ``check(candidate)``.
        entry_point: function name passed to ``check``.
        timeout_s: wall-clock limit (over ``timeout_s`` reports ``"timeout"``).
        memory_limit_mb: RLIMIT_AS cap in MiB (best-effort).
        allow_code_execution: must be ``True`` to opt in.

    Raises:
        RuntimeError: if ``allow_code_execution`` is not ``True``.
    """
    if not allow_code_execution:
        raise RuntimeError(
            "score_program runs model-generated code in a subprocess and is "
            "disabled by default. Pass allow_code_execution=True to opt in. "
            "Only enable on machines you control."
        )

    program = _build_program(prompt, body, test, entry_point)

    with tempfile.TemporaryDirectory(prefix="pyforger_eval_") as tmp:
        program_path = os.path.join(tmp, "candidate.py")
        with open(program_path, "w", encoding="utf-8") as f:
            f.write(program)

        import time

        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "-I", program_path],
                cwd=tmp,
                env=_safe_env(),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_make_preexec(memory_limit_mb),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - start
            return ScoreOutcome(
                outcome=_TIMEOUT,
                returncode=None,
                stdout=_truncate(exc.stdout.decode("utf-8", "replace") if exc.stdout else ""),
                stderr=_truncate(exc.stderr.decode("utf-8", "replace") if exc.stderr else ""),
                duration_s=duration,
            )
        duration = time.time() - start

    if result.returncode == 0:
        return ScoreOutcome(
            outcome=_PASS,
            returncode=0,
            stdout=_truncate(result.stdout),
            stderr=_truncate(result.stderr),
            duration_s=duration,
        )

    stderr = result.stderr or ""
    is_assertion = "AssertionError" in stderr
    return ScoreOutcome(
        outcome=_FAIL if is_assertion else _ERROR,
        returncode=result.returncode,
        stdout=_truncate(result.stdout),
        stderr=_truncate(stderr),
        duration_s=duration,
    )
