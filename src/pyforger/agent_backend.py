"""Backend-agnostic agent specification and execution backend contract.

Two independent axes:

- Agent functionality (what an agent does) is expressed by an
  :class:`AgentSpec` plus the functions in :mod:`pyforger.agents` and
  :mod:`pyforger.agent_tasks`. Those functions never name a sandbox.
- Execution (where an agent runs) is expressed by an :class:`AgentBackend`.
  Backends own no model and no prompts; they materialize a
  :class:`WorkspaceTask`, run the specified agent in it, optionally run a
  verifier, collect declared outputs, and replay retained programs.

Every agent function accepts any ``AgentSpec`` kind on any backend.
Deterministic kinds (``oracle``, ``nop``) need no model, which lets the
full function-by-backend matrix be tested without LLM calls.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

from pyforger.acquire.model import Model
from pyforger.structured import GenerationAttempt

AGENT_KINDS = ("openhands-sdk", "oracle", "nop")


class ProgramValidationError(ValueError):
    """Retained source failed replay validation, not backend infrastructure.

    ValueError ancestry preserves existing callers that reject invalid programs.
    """


class Network(Enum):
    """Network access granted to an agent run (program replay is offline)."""

    OFFLINE = "offline"
    MODEL = "model"
    PUBLIC = "public"


@dataclass(frozen=True)
class AgentSpec:
    """What an agent is, independent of where it runs.

    ``model`` is required for ``openhands-sdk`` and ignored by deterministic
    kinds. ``solution`` is the workspace script an ``oracle`` agent executes.
    Credentials are never stored here; ``api_key_env`` names the host
    environment variable a backend may forward.
    """

    model: Model | None = None
    kind: str = "openhands-sdk"
    max_iterations: int = 20
    api_key_env: str | None = None
    solution: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the agent kind/model combination."""
        if self.kind not in AGENT_KINDS:
            raise ValueError(f"agent kind must be one of {AGENT_KINDS}")
        if self.kind == "openhands-sdk" and self.model is None:
            raise ValueError("openhands-sdk agents require a Model")
        if self.kind == "oracle" and not self.solution:
            raise ValueError("oracle agents require a solution script")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")

    def cache_key(self) -> object:
        """Identify functional agent settings, never credentials."""
        return (
            "agent-spec-v1",
            self.kind,
            self.model.cache_key() if self.model is not None else None,
            self.model.config.api_base if self.model is not None else None,
            self.max_iterations,
            self.solution,
            json.dumps(dict(self.kwargs), sort_keys=True, default=str),
        )


@dataclass(frozen=True)
class WorkspaceTask:
    """One agent run: instruction, workspace files, declared outputs.

    File keys are normalized relative paths materialized in the workspace.
    ``outputs`` are workspace-relative paths collected after the run.
    ``verifier`` is a shell script run in the workspace after the agent; it
    signals success by writing ``1`` to ``reward.txt`` in the workspace root.
    ``verifier_files`` are materialized under ``verifier/`` only for that
    phase, never visible to the agent. ``solution_files`` are materialized
    only for ``oracle`` agents, before their solution script runs.
    """

    instruction: str
    files: Mapping[str, str | bytes] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    packages: tuple[str, ...] = ()
    verifier: str | None = None
    verifier_files: Mapping[str, str | bytes] = field(default_factory=dict)
    solution_files: Mapping[str, str | bytes] = field(default_factory=dict)
    network: Network = Network.MODEL
    timeout_s: float = 600.0

    def __post_init__(self) -> None:
        """Validate instruction, file paths, and output names."""
        if not self.instruction.strip():
            raise ValueError("instruction must be nonempty")
        for path in [
            *self.files,
            *self.verifier_files,
            *self.solution_files,
            *self.outputs,
        ]:
            validate_workspace_path(path)
        if not (0 < self.timeout_s):
            raise ValueError("timeout_s must be positive")


def validate_workspace_path(path: str) -> None:
    """Accept only canonical relative POSIX paths on every host platform."""
    if (
        not isinstance(path, str)
        or not path
        or "\\" in path
        or "\0" in path
        or ":" in path
        or PurePosixPath(path).is_absolute()
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError(f"workspace paths must be canonical and relative: {path!r}")


@dataclass
class AgentRunResult:
    """Normalized outcome of a backend agent run."""

    status: str
    files: dict[str, bytes] = field(default_factory=dict)
    passed: bool | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    validation_attempts: tuple[GenerationAttempt[str, dict[str, bytes]], ...] = ()
    trajectory: dict[str, Any] | None = None
    failure_kind: str | None = None

    def __post_init__(self) -> None:
        """Classify unsuccessful execution when the backend has no finer cause."""
        if self.failure_kind is None and self.status != "completed":
            self.failure_kind = (
                "budget"
                if self.status == "timed_out"
                else (
                    "execution"
                    if self.diagnostics.get("exit_code") not in (None, 0, "0")
                    else "infrastructure"
                )
            )


@runtime_checkable
class AgentBackend(Protocol):
    """Concrete execution of agents and retained programs.

    Backends receive all agent identity through :class:`AgentSpec`; their
    constructors take only execution settings (environment, images, hosts).
    """

    name: str
    supports_inner_loop_validation: bool

    def cache_key(self) -> object:
        """Identify backend and its functional configuration."""
        ...

    def run_agent(self, task: WorkspaceTask, agent: AgentSpec) -> AgentRunResult:
        """Run the specified agent on the task and collect declared outputs."""
        ...

    def execute_program(
        self, code: str, item: dict[str, Any], timeout_s: float
    ) -> list[str]:
        """Replay source; raise ProgramValidationError for rejected programs.

        Legacy ValueError/TypeError validation failures remain supported.
        Infrastructure and transport failures propagate separately.
        """
        ...


def reward_passed(value: bytes) -> bool:
    """Accept numeric unit rewards, including legacy ``1.0`` text."""
    try:
        return float(value.strip()) == 1.0
    except (ValueError, OverflowError):
        return False
