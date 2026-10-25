"""Task and command execution as agent functions over any backend.

A :class:`TaskSpec` is a packaged unit of work (instruction, files,
optional verifier, declared outputs). Running it means: run the agent,
run the verifier in the same workspace, collect outputs. Command execution
runs a caller-supplied shell command without a model call. Neither function
names a sandbox.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyforger.agent_backend import AgentBackend, AgentSpec, Network, WorkspaceTask
from pyforger.agents import _default
from pyforger.artifacts import FileInput, write_input


@dataclass(frozen=True)
class TaskSpec:
    """Packaged work for an agent: instruction, files, verifier, outputs."""

    instruction: str
    files: Mapping[str, str | bytes | FileInput] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    verifier: str | None = None
    verifier_files: Mapping[str, str | bytes | FileInput] = field(default_factory=dict)
    solution_files: Mapping[str, str | bytes | FileInput] = field(default_factory=dict)
    packages: tuple[str, ...] = ()
    network: Network = Network.OFFLINE

    def __post_init__(self) -> None:
        """Validate payload paths before any adapter can materialize them."""
        WorkspaceTask(**self.__dict__)

    def save(self, path: str | Path) -> Path:
        """Save a neutral task into a new directory; never overwrite existing data."""
        WorkspaceTask(**self.__dict__)
        root = Path(path)
        if root.is_symlink():
            raise ValueError("Task root must not be a symlink")
        root = root.parent.resolve() / root.name
        _reject_links(root)
        root.mkdir()
        (root / "instruction.md").write_text(self.instruction, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "outputs": list(self.outputs),
            "packages": list(self.packages),
            "network": self.network.value,
            "verifier": self.verifier,
        }
        for directory, payload in (
            ("solver", self.files),
            ("verifier", self.verifier_files),
            ("solution", self.solution_files),
        ):
            (root / directory).mkdir()
            for name, content in payload.items():
                entry = root / directory / name
                entry.parent.mkdir(parents=True, exist_ok=True)
                write_input(entry, content)
        (root / "task.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return root

    @classmethod
    def load(cls, path: str | Path) -> TaskSpec:
        """Load only explicit solver inputs; reject links anywhere in the bundle."""
        root = Path(path)
        if root.is_symlink():
            raise ValueError("Task root must not be a symlink")
        root = root.parent.resolve() / root.name
        _reject_links(root)
        manifest = json.loads((root / "task.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("Unsupported neutral task schema")
        if set(manifest) != {
            "schema_version",
            "outputs",
            "packages",
            "network",
            "verifier",
        }:
            raise ValueError("Invalid neutral task manifest fields")
        for key in ("outputs", "packages"):
            if not isinstance(manifest[key], list) or any(
                not isinstance(value, str) for value in manifest[key]
            ):
                raise ValueError(f"{key} must be a list of strings")
        if manifest["verifier"] is not None and not isinstance(
            manifest["verifier"], str
        ):
            raise ValueError("verifier must be a string or null")

        def payload(directory: str) -> dict[str, bytes]:
            base = root / directory
            if not base.is_dir():
                raise ValueError(f"Missing task payload directory: {directory}")
            return {
                entry.relative_to(base).as_posix(): entry.read_bytes()
                for entry in sorted(base.rglob("*"))
                if entry.is_file()
            }

        return cls(
            instruction=(root / "instruction.md").read_text(encoding="utf-8"),
            files=payload("solver"),
            verifier_files=payload("verifier"),
            solution_files=payload("solution"),
            outputs=tuple(manifest["outputs"]),
            packages=tuple(manifest["packages"]),
            network=Network(manifest["network"]),
            verifier=manifest["verifier"],
        )


def _reject_links(root: Path) -> None:
    for ancestor in (root, *root.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Task paths must not contain symlinks: {ancestor}")
    if root.exists():
        for entry in root.rglob("*"):
            if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
                raise ValueError(f"Unsafe task entry: {entry}")


def run_task(
    task: TaskSpec,
    *,
    agent: AgentSpec,
    backend: AgentBackend | None = None,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Run an agent on a task and its verifier; return a normalized outcome.

    Failed verification is a normal scored result, not an execution failure.
    Exceptions raised by executors propagate and must not be scored as rejection.
    """
    workspace = WorkspaceTask(
        instruction=task.instruction,
        files=dict(task.files),
        outputs=tuple(task.outputs),
        packages=tuple(task.packages),
        verifier=task.verifier,
        verifier_files=dict(task.verifier_files),
        solution_files=dict(task.solution_files),
        network=task.network,
        timeout_s=timeout_s,
    )
    result = _default(backend).run_agent(workspace, agent)
    retained_trajectory = result.trajectory
    if retained_trajectory is None and agent.kind in {"oracle", "nop"}:
        retained_trajectory = {"steps": [{"kind": agent.kind, "status": result.status}]}
    return {
        "trajectory": retained_trajectory,
        "failure_kind": result.failure_kind,
        "status": result.status,
        "passed": result.passed,
        "files": result.files,
        "diagnostics": result.diagnostics,
    }


def run_command(
    task: TaskSpec,
    command: str,
    *,
    backend: AgentBackend | None = None,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Run a shell command in the task workspace without a model call.

    The command is passed unchanged to an oracle agent on the selected backend.
    Task verification and declared output collection follow normal task execution.
    """
    agent = AgentSpec(kind="oracle", solution=command)
    return run_task(task, agent=agent, backend=backend, timeout_s=timeout_s)
