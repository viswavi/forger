"""Retain neutral task execution evidence without imposing admission policy."""

import json
import re
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from pyforger.agent_backend import AgentBackend, AgentSpec, WorkspaceTask
from pyforger.agent_tasks import TaskSpec, run_task
from pyforger.utils import redact, run_concurrently


def _fresh_directory(output: Path) -> Path:
    if output.is_symlink():
        raise ValueError("Evidence output must not be a symlink")
    output = output.parent.resolve() / output.name
    output.mkdir(parents=True, exist_ok=False)
    return output


def _validate_task(task: TaskSpec, timeout_s: float) -> None:
    WorkspaceTask(**task.__dict__, timeout_s=timeout_s)
    names = [*task.outputs, "agent/trajectory.json", "run-report.json"]
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if (
                name == other
                or name.startswith(other + "/")
                or other.startswith(name + "/")
            ):
                raise ValueError("Declared outputs collide with evidence paths")


def _json_value(value: Any) -> Any:
    return json.loads(redact(json.dumps(value, default=str, allow_nan=False)))


def _workers(concurrency: int) -> int:
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    return concurrency


def run_recorded_task(
    task: TaskSpec,
    *,
    agent: AgentSpec,
    backend: AgentBackend | None = None,
    output: Path,
    timeout_s: float = 1800,
) -> dict[str, Any]:
    """Execute once in a fresh evidence directory and record normalized results.

    Configuration/path errors raise before launch. Execution and malformed or
    missing artifacts after a verifier pass are unscored infrastructure errors.
    Missing submissions after failed verification remain scored false.
    Native execution and budget failures retain their original attribution.
    """
    _validate_task(task, timeout_s)
    output = Path(output)
    output = _fresh_directory(output)
    trajectory: dict[str, Any] | None = None
    diagnostics: Any = {}
    try:
        result = run_task(task, agent=agent, backend=backend, timeout_s=timeout_s)
        diagnostics = _json_value(result["diagnostics"])
        retained = result["trajectory"]
        if retained is not None and not isinstance(retained, dict):
            raise ValueError("Malformed trajectory: expected an object or null")
        trajectory = _json_value(retained)
        for name in task.outputs:
            if name not in result["files"]:
                continue
            content = result["files"][name]
            if isinstance(content, str):
                content = content.encode("utf-8")
            if not isinstance(content, bytes):
                raise TypeError(f"Malformed output artifact: {name}")
            text = content.decode("utf-8", errors="replace")
            if redact(text) != text:
                raise ValueError(f"Credential detected in output artifact: {name}")
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
        missing = [name for name in task.outputs if name not in result["files"]]
        if missing and result["passed"] is True:
            raise ValueError(f"Missing declared outputs: {missing}")
        report = {
            "status": result["status"],
            "passed": result["passed"],
            "failure_kind": result["failure_kind"],
            "diagnostics": diagnostics,
        }
    except Exception as error:
        report = {
            "status": "infrastructure_error",
            "passed": None,
            "failure_kind": "infrastructure",
            "diagnostics": diagnostics,
            "error": redact(f"{type(error).__name__}: {error}"),
        }
    report["trial_dir"] = str(output)
    normalized: dict[str, Any] = _json_value(report)
    (output / "agent").mkdir(exist_ok=True)
    (output / "agent/trajectory.json").write_text(
        json.dumps(trajectory, indent=2), encoding="utf-8"
    )
    (output / "run-report.json").write_text(
        json.dumps(normalized, indent=2), encoding="utf-8"
    )
    return normalized


def run_controls(
    task: TaskSpec,
    agents: Mapping[str, AgentSpec],
    *,
    backend: AgentBackend | None,
    output: Path,
    timeout_s: float = 1800,
    concurrency: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Record controls in parallel; None starts every named control at once."""
    workers = _workers(max(1, len(agents)) if concurrency is None else concurrency)
    _validate_task(task, timeout_s)
    controls = list(agents.items())
    for name, _ in controls:
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", name
        ):
            raise ValueError("Control names must be safe IDs")
    output = Path(output)
    output = _fresh_directory(output)

    def invoke(control: tuple[str, AgentSpec]) -> tuple[str, dict[str, Any]]:
        name, agent = control
        return name, run_recorded_task(
            task,
            agent=agent,
            backend=backend,
            output=output / name,
            timeout_s=timeout_s,
        )

    return dict(
        run_concurrently(
            (partial(invoke, control) for control in controls), concurrency=workers
        )
    )
