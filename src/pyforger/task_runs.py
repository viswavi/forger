"""Retain neutral task execution evidence without imposing admission policy."""

import json
import os
import re
import tempfile
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from pyforger.agent_backend import AgentBackend, AgentSpec, WorkspaceTask
from pyforger.agent_diagnostics import (
    ArtifactSafetyError,
    sanitizer,
    validate_artifact_bytes,
)
from pyforger.agent_tasks import TaskSpec, run_task
from pyforger.artifacts import FileInput
from pyforger.evidence_journal import materialize_journals
from pyforger.usage import usage_record
from pyforger.utils import redact, run_concurrently


def _atomic_json(path: Path, value: Any) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".manifest-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _fresh_directory(output: Path) -> Path:
    if output.is_symlink():
        raise ValueError("Evidence output must not be a symlink")
    output = output.parent.resolve() / output.name
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    return output


def _validate_task(task: TaskSpec, timeout_s: float) -> None:
    WorkspaceTask(**task.__dict__, timeout_s=timeout_s)
    names = [*task.outputs, "agent", "run-report.json"]
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
    usage = usage_record(zero=agent.kind in {"oracle", "nop"})
    trajectory: dict[str, Any] | None = None
    diagnostics: Any = {}
    try:
        result = run_task(task, agent=agent, backend=backend, timeout_s=timeout_s)
        usage = result["usage"]
        journal_memo: dict = {}
        diagnostics = _json_value(
            materialize_journals(
                sanitizer([agent])(result["diagnostics"]), output, [agent], journal_memo
            )
        )
        native_result = diagnostics.get("native_evidence", {}).get("result.json", {})
        if "evidence_file" in native_result:
            diagnostics["harbor_result"] = str(
                output / native_result["evidence_file"]["path"]
            )
        retained = result["trajectory"]
        if retained is not None and not isinstance(retained, dict):
            raise ValueError("Malformed trajectory: expected an object or null")
        trajectory = _json_value(
            materialize_journals(
                sanitizer([agent])(retained), output, [agent], journal_memo
            )
        )
        validate_artifact_bytes(
            {
                name: result["files"][name]
                for name in task.outputs
                if name in result["files"]
            },
            [agent],
        )
        for name in task.outputs:
            if name not in result["files"]:
                continue
            content = result["files"][name]
            if isinstance(content, str):
                content = content.encode("utf-8")
            if not isinstance(content, (bytes, FileInput)):
                raise TypeError(f"Malformed output artifact: {name}")
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, FileInput):
                content.copy_to(path)
            else:
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
        evidence = getattr(error, "diagnostics", None)
        if evidence and evidence.get("usage"):
            usage = evidence["usage"]
        if evidence and not diagnostics:
            try:
                diagnostics = _json_value(
                    materialize_journals(sanitizer([agent])(evidence), output, [agent])
                )
                trajectory = diagnostics.get(
                    "current_trajectory", diagnostics.get("trajectory")
                )
            except (ValueError, OSError, TypeError) as evidence_error:
                diagnostics = {
                    "evidence_error": sanitizer([agent])(str(evidence_error))
                }
        report = {
            "status": "artifact_safety_error"
            if isinstance(error, ArtifactSafetyError)
            else "infrastructure_error",
            "passed": None,
            "failure_kind": "artifact_safety"
            if isinstance(error, ArtifactSafetyError)
            else "infrastructure",
            "diagnostics": diagnostics,
            "error": sanitizer([agent])(f"{type(error).__name__}: {error}"),
        }
    report["usage"] = usage
    report["trial_dir"] = str(output)
    normalized: dict[str, Any] = _json_value(report)
    (output / "agent").mkdir(exist_ok=True)
    _atomic_json(output / "agent/trajectory.json", trajectory)
    _atomic_json(output / "run-report.json", normalized)
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
