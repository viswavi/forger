"""Generic artifact evaluation, independent of agent execution and task policy."""

from __future__ import annotations

import json
import math
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


class EvaluationResult(BaseModel):
    """Optional outcomes, attribution, metadata, and independently saved evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: StrictFloat | StrictInt | None = None
    passed: StrictBool | None = None
    evidence: ArtifactRef | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    attribution: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_record(self) -> EvaluationResult:
        """Require lossless JSON metadata and finite scores; verify evidence."""
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("score must be finite or None")

        def validate(value: Any) -> None:
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError("metadata keys must be strings")
                for child in value.values():
                    validate(child)
            elif isinstance(value, list):
                for child in value:
                    validate(child)
            elif isinstance(value, float) and not math.isfinite(value):
                raise ValueError("metadata must contain finite numbers")
            elif value is not None and type(value) not in (str, bool, int, float):
                raise ValueError("metadata must contain JSON values")

        validate(self.metadata)
        object.__setattr__(self, "metadata", deepcopy(self.metadata))
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
    artifact.verify()
    result.verify()
    (output / "evaluation.json").write_text(
        json.dumps(
            result.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    artifact.verify()
    result.verify()
    return result


def load_evaluation(
    record: ArtifactRef, relative: str = "evaluation.json"
) -> EvaluationResult:
    """Load an outcome using an enclosing artifact as its integrity anchor."""
    value = json.loads(record.path(relative).read_text(encoding="utf-8"))
    result = EvaluationResult.model_validate(value)
    record.verify()
    return result
