"""Runner-neutral execution of fresh agent samples with verified records."""

from __future__ import annotations

import json
import math
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyforger.artifacts import ArtifactRef

_ROLLOUT_STATUSES = {
    "completed",
    "failed",
    "timed_out",
    "infrastructure_error",
    "incomplete",
}


def _json_object(value: dict[str, Any], name: str) -> dict[str, Any]:
    def validate(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError(f"{name} keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, list):
            for child in item:
                validate(child)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{name} must contain finite numbers")
        elif item is not None and type(item) not in (str, bool, int, float):
            raise ValueError(f"{name} must contain JSON values")

    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    validate(value)
    return deepcopy(value)


class TrialRequest(BaseModel):
    """Identity of one independent sample against a frozen task artifact.

    Configuration contains public execution parameters, not credential values.
    A different ``trial_id`` requests a fresh sample rather than a cached repeat.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_id: str
    task: ArtifactRef
    model: str
    configuration: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_record(self) -> TrialRequest:
        """Validate sample identity and snapshot JSON-compatible configuration."""
        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise ValueError("trial_id must be nonempty")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be nonempty")
        if not isinstance(self.task, ArtifactRef):
            raise TypeError("task must be an ArtifactRef")
        object.__setattr__(
            self, "configuration", _json_object(self.configuration, "configuration")
        )
        return self

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

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    artifacts: ArtifactRef | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_record(self) -> RolloutResult:
        """Validate status, snapshot metadata, and check usage quantities."""
        if self.status not in _ROLLOUT_STATUSES:
            raise ValueError(f"Unknown rollout status: {self.status}")
        if self.artifacts is not None and not isinstance(self.artifacts, ArtifactRef):
            raise TypeError("artifacts must be an ArtifactRef or None")
        object.__setattr__(self, "metadata", _json_object(self.metadata, "metadata"))
        object.__setattr__(self, "usage", _json_object(self.usage, "usage"))
        for key in (
            "cost_usd",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
        ):
            if key in self.usage:
                value = self.usage[key]
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{key} must be finite and nonnegative")
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
    output.mkdir(parents=True, exist_ok=False)
    context: AbstractContextManager[Any] = nullcontext(None)
    if environment_factory is not None:
        environment = request.configuration.get("environment")
        if not isinstance(environment, dict):
            raise ValueError(
                "environment_factory requires "
                "request.configuration['environment'] to be an object of "
                "factory-specific settings"
            )
        context = environment_factory.open(deepcopy(environment))
    with context as environment:
        rollout = runner.run(request, output / "rollout", environment=environment)
    if not isinstance(rollout, RolloutResult):
        raise TypeError("Runner must return RolloutResult")
    if request.cache_key() != identity:
        raise ValueError("Runner changed trial request identity")
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
