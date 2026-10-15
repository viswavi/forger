"""Runner-neutral execution of fresh agent samples with verified records."""

from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from pyforger.artifacts import ArtifactRef

NonEmptyStr = Annotated[str, StringConstraints(pattern=r"\S")]


def _stored_path(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    return path.parent.resolve() / path.name


class TrialRequest(BaseModel):
    """Identity of one independent sample against a frozen task artifact.

    Configuration contains public execution parameters, not credential values.
    A different ``trial_id`` requests a fresh sample rather than a cached repeat.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    trial_id: NonEmptyStr
    task: ArtifactRef
    model: NonEmptyStr
    configuration: dict[str, JsonValue] = Field(default_factory=dict)

    def cache_key(self) -> tuple[str, str, str, str]:
        """Include sample identity and all public execution configuration."""
        return (
            self.trial_id,
            self.task.cache_key(),
            self.model,
            json.dumps(self.configuration, sort_keys=True, allow_nan=False),
        )


class RolloutResult(BaseModel):
    """Execution status and full artifacts, independent of scientific grading."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    status: Literal[
        "completed", "failed", "timed_out", "infrastructure_error", "incomplete"
    ]
    artifacts: ArtifactRef | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    usage: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_usage(self) -> RolloutResult:
        """Check the known accounting keys an open usage dict cannot type."""
        if "cost_usd" in self.usage:
            value = self.usage["cost_usd"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ValueError("cost_usd must be finite and nonnegative")
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cached_tokens",
        ):
            if key in self.usage:
                value = self.usage[key]
                if type(value) is not int or value < 0:
                    raise ValueError(f"{key} must be a nonnegative integer")
        return self


class RolloutRecord(BaseModel):
    """Execution identity and retained rollout, without grading or replay policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: TrialRequest
    rollout: RolloutResult

    @model_validator(mode="after")
    def validate_evidence(self) -> RolloutRecord:
        """Verify referenced files whenever a saved execution is validated."""
        self.verify()
        return self

    def verify(self) -> None:
        """Verify task and retained execution evidence before reuse."""
        self.request.task.verify()
        if self.rollout.artifacts is not None:
            self.rollout.artifacts.verify()


class AgentRunner(Protocol):
    """Runtime adapter; output is a reserved, initially absent child path."""

    def run(
        self, request: TrialRequest, output: Path, *, environment: Any = None
    ) -> RolloutResult:
        """Run a fresh sample; external runtimes may own their environment."""
        ...


class EnvironmentFactory(Protocol):
    """Open an adapter-owned environment from public execution settings."""

    def open(self, spec: dict[str, Any]) -> AbstractContextManager[Any]:
        """Create a fresh environment with a managed lifecycle."""
        ...


def execute_agent(
    request: TrialRequest,
    runner: AgentRunner,
    output: Path,
    *,
    environment_factory: EnvironmentFactory | None = None,
) -> RolloutResult:
    """Execute one fresh sample and persist ``output/rollout.json``.

    The runner receives the initially absent ``output/rollout`` directory.
    ``configuration["environment"]`` supplies factory-owned settings as a dict.
    Omitting the factory lets an external runtime own its environment. Adapter
    exceptions propagate and partial evidence is not overwritten on retry.
    """
    identity = request.cache_key()
    output = Path(output)
    environment_settings: dict[str, Any] | None = None
    if environment_factory is not None:
        environment = request.configuration.get("environment")
        if not isinstance(environment, dict):
            raise ValueError(
                "environment_factory requires "
                "request.configuration['environment'] to be an object of "
                "factory-specific settings"
            )
        environment_settings = deepcopy(environment)
    output.mkdir(parents=True, exist_ok=False)
    context: AbstractContextManager[Any] = nullcontext(None)
    if environment_factory is not None:
        context = environment_factory.open(environment_settings or {})
    with context as environment:
        rollout = runner.run(request, output / "rollout", environment=environment)
    if not isinstance(rollout, RolloutResult):
        raise TypeError("Runner must return RolloutResult")
    if request.cache_key() != identity:
        raise ValueError("Runner changed trial request identity")
    if rollout.artifacts is not None and rollout.artifacts.root != _stored_path(
        output / "rollout"
    ):
        raise ValueError("Runner artifacts must be captured from output/rollout")
    record = RolloutRecord(request=request, rollout=rollout)
    (output / "rollout.json").write_text(
        json.dumps(
            record.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    record.verify()
    return rollout


def load_rollout(record: ArtifactRef, relative: str = "rollout.json") -> RolloutRecord:
    """Load execution metadata from an independently retained verified artifact.

    The reference is the trust anchor: never recapture an edited record before
    loading it. A durable RunLedger can supply this enclosing artifact reference.
    """
    value = json.loads(record.path(relative).read_text(encoding="utf-8"))
    result = RolloutRecord.model_validate(value)
    record.verify()
    return result
