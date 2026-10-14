"""Standalone evaluation of real artifacts without agent requests or rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pyforger.artifacts import ArtifactRef
from pyforger.evaluation import EvaluationResult, evaluate, load_evaluation


class ArithmeticEvaluator:
    def evaluate(self, artifact: ArtifactRef, output: Path) -> EvaluationResult:
        assert not output.exists()
        answer = json.loads(artifact.path("answer.json").read_text())
        expected = sum(json.loads(artifact.path("input.json").read_text()))
        passed = answer == expected
        output.mkdir()
        (output / "grade.json").write_text(json.dumps({"passed": passed}))
        return EvaluationResult(
            passed=passed,
            score=float(passed),
            metadata={"domain_flag": not passed},
            evidence=ArtifactRef.capture(output),
        )


def artifact(tmp_path, answer=6):
    """Artifact."""
    root = tmp_path / "data"
    root.mkdir()
    (root / "input.json").write_text("[1, 2, 3]")
    (root / "answer.json").write_text(json.dumps(answer))
    return ArtifactRef.capture(root)


@pytest.mark.parametrize("answer,passed", [(6, True), (7, False)])
def test_evaluate_arbitrary_artifact_and_restore(tmp_path, answer, passed):
    """Evaluate arbitrary artifact and restore."""
    source = artifact(tmp_path, answer)
    result = evaluate(source, ArithmeticEvaluator(), tmp_path / "grade")
    assert result.passed is passed
    assert result.score == float(passed)
    assert result.metadata == {"domain_flag": not passed}
    assert json.loads(result.evidence.path("grade.json").read_text()) == {
        "passed": passed
    }
    assert load_evaluation(ArtifactRef.capture(tmp_path / "grade")) == result
    assert json.loads(
        (tmp_path / "grade" / "evaluation.json").read_text()
    ) == result.model_dump(mode="json")
    assert not (tmp_path / "grade" / "rollout.json").exists()
    source.verify()


def test_evaluation_can_omit_outcomes_and_evidence(tmp_path):
    """Evaluation can omit outcomes and evidence."""
    class DescriptiveEvaluator:
        def evaluate(self, artifact, output):
            return EvaluationResult(
                metadata={"answer": artifact.path("answer.json").read_text()}
            )

    result = evaluate(artifact(tmp_path), DescriptiveEvaluator(), tmp_path / "grade")
    assert result.passed is result.score is result.evidence is None
    assert not (tmp_path / "grade" / "evidence").exists()
    assert load_evaluation(ArtifactRef.capture(tmp_path / "grade")) == result


def test_existing_output_is_not_overwritten(tmp_path):
    """Existing output is not overwritten."""
    source = artifact(tmp_path)
    output = tmp_path / "grade"
    evaluate(source, ArithmeticEvaluator(), output)
    saved = ArtifactRef.capture(output)
    with pytest.raises(FileExistsError):
        evaluate(source, ArithmeticEvaluator(), output)
    saved.verify()


def test_evaluations_are_fresh_and_independent(tmp_path):
    """Evaluations are fresh and independent."""
    source = artifact(tmp_path)
    first = evaluate(source, ArithmeticEvaluator(), tmp_path / "first")
    record = ArtifactRef.capture(tmp_path / "first")
    second = evaluate(source, ArithmeticEvaluator(), tmp_path / "second")
    assert first.evidence.root != second.evidence.root
    assert load_evaluation(record) == first
    record.verify()


def test_changed_input_fails_before_evaluator(tmp_path):
    """Changed input fails before evaluator."""
    source = artifact(tmp_path)
    (source.root / "answer.json").write_text("99")
    with pytest.raises(ValueError, match="tampered"):
        evaluate(source, ArithmeticEvaluator(), tmp_path / "grade")
    assert not (tmp_path / "grade").exists()


def test_evaluator_cannot_mutate_input(tmp_path):
    """Evaluator cannot mutate input."""
    class MutatingEvaluator(ArithmeticEvaluator):
        def evaluate(self, artifact, output):
            result = super().evaluate(artifact, output)
            (artifact.root / "answer.json").write_text("99")
            return result

    with pytest.raises(ValueError, match="tampered"):
        evaluate(artifact(tmp_path), MutatingEvaluator(), tmp_path / "grade")
    assert not (tmp_path / "grade" / "evaluation.json").exists()


def test_evidence_is_verified_after_adapter_returns(tmp_path):
    """Evidence is verified after adapter returns."""
    class MutatingEvaluator(ArithmeticEvaluator):
        def evaluate(self, artifact, output):
            result = super().evaluate(artifact, output)
            (output / "grade.json").write_text("{}")
            return result

    with pytest.raises(ValueError, match="tampered"):
        evaluate(artifact(tmp_path), MutatingEvaluator(), tmp_path / "grade")


def test_invalid_adapter_result(tmp_path):
    """Invalid adapter result."""
    class InvalidEvaluator:
        def evaluate(self, artifact, output):
            return {"passed": True}

    with pytest.raises(TypeError, match="EvaluationResult"):
        evaluate(artifact(tmp_path), InvalidEvaluator(), tmp_path / "grade")


def test_exception_reserves_output_against_retries(tmp_path):
    """Exception reserves output against retries."""
    class BrokenEvaluator:
        def evaluate(self, artifact, output):
            output.mkdir()
            (output / "error.txt").write_text("evaluation interrupted")
            raise RuntimeError("failed")

    source = artifact(tmp_path)
    with pytest.raises(RuntimeError, match="failed"):
        evaluate(source, BrokenEvaluator(), tmp_path / "grade")
    assert (
        tmp_path / "grade" / "evidence" / "error.txt"
    ).read_text() == "evaluation interrupted"
    with pytest.raises(FileExistsError):
        evaluate(source, ArithmeticEvaluator(), tmp_path / "grade")


def test_verified_record_binds_evaluation_metadata(tmp_path):
    """Verified record binds evaluation metadata."""
    result = evaluate(artifact(tmp_path, 7), ArithmeticEvaluator(), tmp_path / "grade")
    record = ArtifactRef.capture(tmp_path / "grade")
    data = result.model_dump(mode="json")
    data.update(passed=True, score=1.0)
    (record.root / "evaluation.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="tampered"):
        load_evaluation(record)


def test_restored_evidence_must_still_match(tmp_path):
    """Restored evidence must still match."""
    result = evaluate(artifact(tmp_path), ArithmeticEvaluator(), tmp_path / "grade")
    data = result.model_dump(mode="json")
    (result.evidence.root / "grade.json").write_text("{}")
    with pytest.raises(ValueError, match="tampered"):
        EvaluationResult.model_validate(data)


@pytest.mark.parametrize("score", [True, float("inf"), float("nan"), "1"])
def test_invalid_scores(score):
    """Invalid scores."""
    with pytest.raises(ValueError):
        EvaluationResult(score=score)


@pytest.mark.parametrize("passed", [1, "true", [], {}])
def test_outcomes_require_booleans(passed):
    """Outcomes require booleans."""
    with pytest.raises(ValidationError):
        EvaluationResult(passed=passed)


@pytest.mark.parametrize(
    "metadata", [{"nested": (1, 2)}, {"nested": {2: "v"}}, {"value": float("nan")}]
)
def test_metadata_requires_lossless_json(metadata):
    """Metadata requires lossless json."""
    with pytest.raises(ValueError):
        EvaluationResult(metadata=metadata)


def test_metadata_snapshot_and_unrestricted_attribution():
    """Metadata snapshot and unrestricted attribution."""
    metadata = {"parameters": [1]}
    result = EvaluationResult(
        metadata=metadata, attribution="custom-domain", reason="graded"
    )
    metadata["parameters"].append(2)
    assert result.metadata == {"parameters": [1]}
    assert EvaluationResult.model_validate(result.model_dump(mode="json")) == result
