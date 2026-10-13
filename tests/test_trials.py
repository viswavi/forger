"""Runner-neutral trials keep outcomes separate from infrastructure failures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pyforger.artifacts import ArtifactRef
from pyforger.trials import (
    EvaluationResult,
    RolloutResult,
    TrialRequest,
    TrialResult,
    run_trial,
)


class ArithmeticRunner:
    """A trusted deterministic runner implementing the public adapter contract."""

    def run(
        self, request: TrialRequest, output: Path, *, environment=None
    ) -> RolloutResult:
        """Compute a real sum and retain the answer and operation trajectory."""
        output.mkdir()
        values = json.loads(request.task.path("input.json").read_text())
        answer = sum(values) + request.configuration.get("offset", 0)
        (output / "answer.json").write_text(json.dumps(answer))
        (output / "trajectory.json").write_text(json.dumps({"operation": "sum"}))
        return RolloutResult(
            status="completed",
            artifacts=ArtifactRef.capture(output),
            metadata={"answer": answer},
            usage={"cost_usd": 0.0},
        )


class ArithmeticEvaluator:
    """Grade actual runner artifacts rather than substituting function results."""

    def evaluate(self, request, rollout, output):
        """Grade the retained answer against task data and persist evidence."""
        output.mkdir()
        answer = json.loads(rollout.artifacts.path("answer.json").read_text())
        expected = sum(json.loads(request.task.path("input.json").read_text()))
        passed = answer == expected
        (output / "grade.json").write_text(json.dumps({"passed": passed}))
        return EvaluationResult(
            passed=passed,
            score=float(passed),
            metadata={"domain_flag": not passed},
            evidence=ArtifactRef.capture(output),
        )


def request(tmp_path, **configuration):
    """Create a frozen arithmetic task with public runner configuration."""
    task = tmp_path / "task"
    task.mkdir()
    (task / "input.json").write_text("[1, 2, 3]")
    return TrialRequest(
        trial_id="trial-0",
        task=ArtifactRef.capture(task),
        model="arithmetic",
        configuration=configuration,
    )


@pytest.mark.parametrize("offset,passed", [(0, True), (1, False)])
def test_real_rollout_evaluation_and_serialization(tmp_path, offset, passed):
    """Arithmetic outcomes and evidence survive trial serialization."""
    req = request(tmp_path, offset=offset)
    result = run_trial(req, ArithmeticRunner(), ArithmeticEvaluator(), tmp_path / "run")
    assert result.evaluation.passed is passed
    assert result.evaluation.score == float(passed)
    restored = TrialResult.model_validate(
        json.loads(json.dumps(result.model_dump(mode="json")))
    )
    assert restored.model_dump(mode="json") == result.model_dump(mode="json")
    assert restored.request.cache_key() == req.cache_key()
    assert json.loads(
        (tmp_path / "run" / "trial.json").read_text()
    ) == result.model_dump(mode="json")


def test_infrastructure_failure_preserves_evidence(tmp_path):
    """Infrastructure failures retain logs and skip scientific evaluation."""

    class FailedRunner:
        def run(self, request, output, *, environment=None):
            output.mkdir()
            (output / "log.txt").write_text("Transport failed")
            return RolloutResult(
                status="infrastructure_error", artifacts=ArtifactRef.capture(output)
            )

    result = run_trial(
        request(tmp_path), FailedRunner(), ArithmeticEvaluator(), tmp_path / "run"
    )
    assert result.evaluation is None
    assert not (tmp_path / "run" / "evaluation").exists()
    assert result.rollout.artifacts.path("log.txt").read_text() == "Transport failed"


def test_changed_task_fails_closed(tmp_path):
    """Runner mutations of the frozen task invalidate the trial."""

    class MutatingRunner(ArithmeticRunner):
        def run(self, request, output, *, environment=None):
            result = super().run(request, output)
            (request.task.root / "input.json").write_text("[99]")
            return result

    with pytest.raises(ValueError):
        run_trial(
            request(tmp_path), MutatingRunner(), ArithmeticEvaluator(), tmp_path / "run"
        )


def test_trial_identity_distinguishes_fresh_samples(tmp_path):
    """Different sample identifiers produce distinct trial cache keys."""
    req = request(tmp_path)
    another = TrialRequest(
        trial_id="trial-1",
        task=req.task,
        model=req.model,
        configuration=req.configuration,
    )
    assert req.cache_key() != another.cache_key()


def test_existing_output_is_not_overwritten(tmp_path):
    """Repeated execution cannot overwrite an existing trial directory."""
    req = request(tmp_path)
    output = tmp_path / "run"
    run_trial(req, ArithmeticRunner(), ArithmeticEvaluator(), output)
    with pytest.raises(FileExistsError):
        run_trial(req, ArithmeticRunner(), ArithmeticEvaluator(), output)


def test_malformed_adapter_types_fail_closed(tmp_path):
    """Runner adapters must return typed rollout results, not mappings."""

    class InvalidRunner:
        def run(self, request, output, *, environment=None):
            return {"status": "completed"}

    with pytest.raises(TypeError, match="RolloutResult"):
        run_trial(
            request(tmp_path), InvalidRunner(), ArithmeticEvaluator(), tmp_path / "run"
        )


@pytest.mark.parametrize("value", [True, -1, float("inf"), float("nan"), "1"])
def test_invalid_usage_is_rejected(value):
    """Usage records reject nonnumeric, negative, and nonfinite costs."""
    with pytest.raises(ValueError):
        RolloutResult(status="completed", usage={"cost_usd": value})


@pytest.mark.parametrize("value", [1, "true", [], {}])
def test_outcomes_require_actual_booleans(value):
    """Evaluation outcomes reject values that merely resemble booleans."""
    with pytest.raises(ValidationError):
        EvaluationResult(passed=value)


def test_restored_evidence_must_still_match(tmp_path):
    """Restoring a trial verifies that retained evidence is unchanged."""
    result = run_trial(
        request(tmp_path), ArithmeticRunner(), ArithmeticEvaluator(), tmp_path / "run"
    )
    encoded = result.model_dump(mode="json")
    (result.rollout.artifacts.root / "answer.json").write_text("99")
    with pytest.raises(ValueError):
        TrialResult.model_validate(encoded)


def test_evaluator_cannot_mutate_rollout_evidence(tmp_path):
    """Evaluator mutations of retained rollout files invalidate the trial."""

    class MutatingEvaluator(ArithmeticEvaluator):
        def evaluate(self, request, rollout, output):
            result = super().evaluate(request, rollout, output)
            (rollout.artifacts.root / "answer.json").write_text("99")
            return result

    with pytest.raises(ValueError):
        run_trial(
            request(tmp_path), ArithmeticRunner(), MutatingEvaluator(), tmp_path / "run"
        )


@pytest.mark.parametrize(
    "configuration", [{1: "v"}, {"nested": (1, 2)}, {"nested": {2: "v"}}]
)
def test_only_lossless_json_configuration_is_accepted(tmp_path, configuration):
    """Configuration rejects tuples and nonstring JSON object keys."""
    req = request(tmp_path)
    with pytest.raises(ValidationError):
        TrialRequest(
            trial_id="fresh",
            task=req.task,
            model=req.model,
            configuration=configuration,
        )


def test_configuration_is_defensively_snapshotted(tmp_path):
    """External mutations cannot change captured trial configuration."""
    configuration = {"parameters": [0.7]}
    req = request(tmp_path)
    req = TrialRequest(
        trial_id="fresh", task=req.task, model=req.model, configuration=configuration
    )
    original = req.cache_key()
    configuration["parameters"].append(1)
    req.model_dump(mode="json")["configuration"]["parameters"].append(2)
    assert req.cache_key() == original


def test_runner_may_not_change_request_identity(tmp_path):
    """Runner configuration changes invalidate sample identity."""

    class MutatingRunner(ArithmeticRunner):
        def run(self, request, output, *, environment=None):
            request.configuration["offset"] = 10
            return super().run(request, output)

    with pytest.raises(ValueError, match="identity"):
        run_trial(
            request(tmp_path), MutatingRunner(), ArithmeticEvaluator(), tmp_path / "run"
        )


def test_verified_record_binds_grading_metadata(tmp_path):
    """Verified enclosing artifacts detect edits to grading metadata."""
    from pyforger.trials import load_trial

    result = run_trial(
        request(tmp_path, offset=1),
        ArithmeticRunner(),
        ArithmeticEvaluator(),
        tmp_path / "run",
    )
    record = ArtifactRef.capture(tmp_path / "run")
    assert load_trial(record).evaluation.passed is False
    data = result.model_dump(mode="json")
    data["evaluation"].update(passed=True, score=1.0)
    (record.root / "trial.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="tampered"):
        load_trial(record)


@pytest.mark.parametrize("passed", [True, False])
def test_attributed_evaluation_roundtrip(passed):
    """Trial evaluation preserves generic outcome attribution."""
    result = EvaluationResult(passed=passed, attribution="model", reason="graded")
    assert EvaluationResult.model_validate(result.model_dump(mode="json")) == result
