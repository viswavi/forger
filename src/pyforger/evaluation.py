"""Generic artifact evaluation, independent of agent execution and task policy."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

from pyforger.artifacts import ArtifactRef

FiniteScore = Annotated[StrictFloat, Field(allow_inf_nan=False)]


def _stored_path(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    return path.parent.resolve() / path.name


class EvaluationResult(BaseModel):
    """Optional outcomes, attribution, metadata, and independently saved evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    score: FiniteScore | StrictInt | None = None
    passed: StrictBool | None = None
    evidence: ArtifactRef | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    attribution: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def verify_evidence(self) -> EvaluationResult:
        """Verify retained evidence whenever an outcome is validated."""
        self.verify()
        return self

    def verify(self) -> None:
        """Verify retained evaluation evidence before reuse."""
        if self.evidence is not None:
            self.evidence.verify()


class Evaluator(Protocol):
    """Evaluate any verified artifact without modifying it."""

    def evaluate(self, artifact: ArtifactRef, output: Path) -> EvaluationResult:
        """Return an outcome and optional evidence in an initially absent path."""
        ...


def evaluate(
    artifact: ArtifactRef, evaluator: Evaluator, output: Path
) -> EvaluationResult:
    """Evaluate an artifact and persist ``output/evaluation.json`` independently.

    The evaluator receives the initially absent ``output/evidence`` directory.
    Exceptions propagate and an existing output is never overwritten. Callers
    choose which artifacts and execution statuses to evaluate.
    """
    artifact.verify()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = evaluator.evaluate(artifact, output / "evidence")
    if not isinstance(result, EvaluationResult):
        raise TypeError("Evaluator.evaluate must return EvaluationResult")
    if result.evidence is not None and result.evidence.root != _stored_path(
        output / "evidence"
    ):
        raise ValueError("Evaluator evidence must be captured from output/evidence")
    # The evaluator had write access to its evidence directory and read access
    # to the artifact; verify both once before persisting the result.
    artifact.verify()
    result.verify()
    (output / "evaluation.json").write_text(
        json.dumps(
            result.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def load_evaluation(
    record: ArtifactRef, relative: str = "evaluation.json"
) -> EvaluationResult:
    """Load an outcome using an enclosing artifact as its integrity anchor."""
    value = json.loads(record.path(relative).read_text(encoding="utf-8"))
    result = EvaluationResult.model_validate(value)
    record.verify()
    return result
