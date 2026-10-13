"""Runner-neutral agent trials with explicit evaluation and replay evidence."""
from __future__ import annotations

import json
import math
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

from pyforger.artifacts import ArtifactRef
from pyforger.attempt_outcome import AttemptOutcome

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

    model_config = ConfigDict(frozen=True)

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

    def cache_key(self) -> tuple:
        """Include sample identity and all public execution configuration."""
        return (
            self.trial_id,
            self.task.cache_key(),
            self.model,
            json.dumps(self.configuration, sort_keys=True, allow_nan=False),
        )


class RolloutResult(BaseModel):
    """Execution status and full artifacts, independent of scientific grading."""

    model_config = ConfigDict(frozen=True)

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


class EvaluationResult(BaseModel):
    """Generic grading; domain-specific integrity flags belong in metadata."""

    model_config = ConfigDict(frozen=True)

    score: StrictFloat | StrictInt | None = None
    passed: StrictBool | None = None
    evidence: ArtifactRef | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    attribution: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_record(self) -> EvaluationResult:
        """Validate optional outcomes and snapshot grading metadata."""
        if self.score is not None and (
            type(self.score) not in (int, float) or not math.isfinite(self.score)
        ):
            raise ValueError("score must be finite or None")
        if self.passed is not None and type(self.passed) is not bool:
            raise TypeError("passed must be a boolean or None")
        if self.evidence is not None and not isinstance(self.evidence, ArtifactRef):
            raise TypeError("evidence must be an ArtifactRef or None")
        if self.attribution is not None:
            if self.reason is None:
                raise ValueError("Attributed evaluation requires a reason")
            AttemptOutcome(
                passed=self.passed, attribution=self.attribution, reason=self.reason
            )
        object.__setattr__(self, "metadata", _json_object(self.metadata, "metadata"))
        return self


class TrialResult(BaseModel):
    """One execution with optional grading, without admission policy."""

    model_config = ConfigDict(frozen=True)

    request: TrialRequest
    rollout: RolloutResult
    evaluation: EvaluationResult | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> TrialResult:
        """Verify referenced files whenever a trial result is validated."""
        self.verify()
        return self

    def verify(self) -> None:
        """Verify task and retained rollout/evaluation evidence before reuse."""
        self.request.task.verify()
        for artifact in (
            self.rollout.artifacts,
            self.evaluation.evidence if self.evaluation is not None else None,
        ):
            if artifact is not None:
                artifact.verify()


AgentRequest = TrialRequest


class AgentRunner(Protocol):
    """Runtime adapter; output is a reserved, initially absent child path."""

    def run(
        self, request: TrialRequest, output: Path, *, environment: Any = None
    ) -> RolloutResult:
        """Run a fresh sample; external runtimes may own their environment."""
        ...


class TrialEvaluator(Protocol):
    """Domain-specific grading independent of agent execution."""

    def evaluate(
        self, request: TrialRequest, rollout: RolloutResult, output: Path
    ) -> EvaluationResult:
        """Evaluate retained artifacts without modifying them."""
        ...


class ReplayRunner(Protocol):
    """Execute a retained program without sampling a new agent solution."""

    def replay(
        self,
        request: TrialRequest,
        rollout: RolloutResult,
        output: Path,
        *,
        environment: Any = None,
    ) -> RolloutResult:
        """Replay saved work in a newly opened environment."""
        ...


def _persist(result: TrialResult, output: Path) -> None:
    result.verify()
    (output / "trial.json").write_text(
        json.dumps(
            result.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _execute(request, runner, output, environment_factory, source=None):
    identity = request.cache_key()
    source_snapshot = source.model_dump(mode="json") if source is not None else None
    request.task.verify()
    if source is not None:
        TrialResult(request=request, rollout=source).verify()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = nullcontext(None)
    if environment_factory is not None:
        environment = request.configuration.get("environment")
        if not isinstance(environment, dict):
            raise ValueError(
                "environment_factory requires "
                "request.configuration['environment'] to be an object of "
                "factory-specific settings"
            )
        context = environment_factory.open(dict(environment))
    with context as environment:
        if source is None:
            rollout = runner.run(request, output / "rollout", environment=environment)
        else:
            rollout = runner.replay(
                request, source, output / "rollout", environment=environment
            )
    if not isinstance(rollout, RolloutResult):
        raise TypeError("Runner must return RolloutResult")
    if request.cache_key() != identity:
        raise ValueError("Runner changed trial request identity")
    if source is not None:
        if source.model_dump(mode="json") != source_snapshot:
            raise ValueError("Replay runner changed retained rollout metadata")
        TrialResult(request=request, rollout=source).verify()
    _persist(TrialResult(request=request, rollout=rollout), output)
    return rollout


def execute_agent(
    request: TrialRequest,
    runner: AgentRunner,
    output: Path,
    *,
    environment_factory=None,
) -> RolloutResult:
    """Execute and retain one fresh sample, without requiring an evaluator.

    ``configuration["environment"]`` supplies factory-owned settings as a dict.
    Omitting the factory lets an external runtime own its environment. Adapter
    exceptions propagate and partial evidence is not overwritten on retry.
    """
    return _execute(request, runner, output, environment_factory)


def evaluate_agent(
    request: TrialRequest,
    rollout: RolloutResult,
    evaluator: TrialEvaluator,
    output: Path,
) -> EvaluationResult:
    """Grade a retained rollout separately, checking identity and evidence."""
    identity = request.cache_key()
    snapshot = rollout.model_dump(mode="json")
    TrialResult(request=request, rollout=rollout).verify()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evaluation = evaluator.evaluate(request, rollout, output / "evaluation")
    if not isinstance(evaluation, EvaluationResult):
        raise TypeError("TrialEvaluator.evaluate must return EvaluationResult")
    if request.cache_key() != identity or rollout.model_dump(mode="json") != snapshot:
        raise ValueError("Evaluator changed trial request identity or rollout metadata")
    _persist(
        TrialResult(request=request, rollout=rollout, evaluation=evaluation), output
    )
    return evaluation


def replay_agent(
    request: TrialRequest,
    rollout: RolloutResult,
    runner: ReplayRunner,
    output: Path,
    *,
    environment_factory=None,
) -> RolloutResult:
    """Replay retained work without universal grading.

    The factory opens a fresh environment when supplied; otherwise the adapter
    must own a fresh provider lifecycle, just as for standalone execution.
    """
    return _execute(request, runner, output, environment_factory, source=rollout)


def run_trial(
    request: TrialRequest,
    runner: AgentRunner,
    evaluator: TrialEvaluator,
    output: Path,
    *,
    environment_factory=None,
) -> TrialResult:
    """Convenience composition of execution and completed-rollout evaluation."""
    output = Path(output)
    rollout = execute_agent(
        request, runner, output, environment_factory=environment_factory
    )
    evaluation = None
    if rollout.status == "completed":
        evaluation = evaluate_agent(request, rollout, evaluator, output / "grading")
    result = TrialResult(request=request, rollout=rollout, evaluation=evaluation)
    _persist(result, output)
    return result


def load_trial(record: ArtifactRef, relative: str = "trial.json") -> TrialResult:
    """Load grading metadata from an independently retained verified artifact.

    The reference is the trust anchor: never recapture an edited record before
    loading it. A durable RunLedger can supply this enclosing artifact reference.
    """
    value = json.loads(record.path(relative).read_text(encoding="utf-8"))
    result = TrialResult.model_validate(value)
    record.verify()
    return result
