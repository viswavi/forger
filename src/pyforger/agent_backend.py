"""Backend-agnostic agent specification and execution backend contract.

Two independent axes:

- Agent functionality (what an agent does) is expressed by an
  :class:`AgentSpec` plus the backend-agnostic functions in
  :mod:`pyforger.agents`. Those functions never name a sandbox.
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
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

from pyforger.acquire.model import Model
from pyforger.artifacts import FileInput
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
class LearnerRuntime:
    """Opt-in SDK tools/history, separate from iteration and wall-clock budgets.

    ``condenser_max_tokens`` uses SDK token estimates and the agent model for
    summaries. Local/Sail preserve system/task and target half this threshold
    before inserting a summary; this is not a gateway-counted hard context cap.
    """

    tools: str = "backend-default"
    reasoning_replay: bool = False
    condenser_max_tokens: int | None = None

    def __post_init__(self) -> None:
        """Reject unknown profiles and ambiguous threshold values."""
        if self.tools not in {"backend-default", "science-rl"}:
            raise ValueError("Unknown learner tool profile")
        if type(self.reasoning_replay) is not bool:
            raise ValueError("reasoning_replay must be a boolean")
        if self.condenser_max_tokens is not None and (
            type(self.condenser_max_tokens) is not int or self.condenser_max_tokens < 1
        ):
            raise ValueError("condenser_max_tokens must be a positive integer")

    def cache_key(self) -> object:
        """Identify requested behavior without endpoint credentials."""
        return (self.tools, self.reasoning_replay, self.condenser_max_tokens)


@dataclass(frozen=True)
class AgentSpec:
    """What an agent is, independent of where it runs.

    ``model`` is required for ``openhands-sdk`` and ignored by deterministic
    kinds. ``solution`` is the workspace script an ``oracle`` agent executes.
    ``max_iterations=None`` disables the step cap, not the task wall deadline.
    Credentials are never stored here; ``api_key_env`` names the host
    environment variable a backend may forward.
    """

    model: Model | None = None
    kind: str = "openhands-sdk"
    max_iterations: int | None = 20
    api_key_env: str | None = None
    solution: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    runtime: LearnerRuntime | None = None

    def __post_init__(self) -> None:
        """Validate the agent kind/model combination."""
        if self.runtime is not None and not isinstance(self.runtime, LearnerRuntime):
            raise TypeError("runtime must be a LearnerRuntime")
        if self.runtime is not None and self.kind != "openhands-sdk":
            raise ValueError("Only SDK agents accept learner runtime settings")
        if self.kind not in AGENT_KINDS:
            raise ValueError(f"agent kind must be one of {AGENT_KINDS}")
        if self.kind == "openhands-sdk" and self.model is None:
            raise ValueError("openhands-sdk agents require a Model")
        if self.kind == "oracle" and not self.solution:
            raise ValueError("oracle agents require a solution script")
        if self.max_iterations is not None and (
            type(self.max_iterations) is not int or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be positive or None")

    def cache_key(self) -> object:
        """Identify functional agent settings, never credentials."""
        return (
            "agent-spec-v2",
            self.kind,
            self.model.cache_key() if self.model is not None else None,
            self.model.config.api_base if self.model is not None else None,
            self.max_iterations,
            self.solution,
            json.dumps(dict(self.kwargs), sort_keys=True, default=str),
            self.runtime.cache_key() if self.runtime is not None else None,
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
    files: Mapping[str, str | bytes | FileInput] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    packages: tuple[str, ...] = ()
    verifier: str | None = None
    verifier_files: Mapping[str, str | bytes | FileInput] = field(default_factory=dict)
    solution_files: Mapping[str, str | bytes | FileInput] = field(default_factory=dict)
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
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")


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
    files: dict[str, bytes | FileInput] = field(default_factory=dict)
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
