"""Runner-neutral trials keep outcomes separate from infrastructure failures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pyforger.artifacts import ArtifactRef
from pyforger.trials import (
    RolloutRecord,
    RolloutResult,
    TrialRequest,
    execute_agent,
    load_rollout,
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


@pytest.mark.parametrize("offset", [0, 1])
def test_real_rollout_and_serialization(tmp_path, offset):
    """Real rollout and serialization."""
    req = request(tmp_path, offset=offset)
    rollout = execute_agent(req, ArithmeticRunner(), tmp_path / "run")
    result = RolloutRecord(request=req, rollout=rollout)
    restored = RolloutRecord.model_validate(result.model_dump(mode="json"))
    assert restored.model_dump(mode="json") == result.model_dump(mode="json")
    assert restored.request.cache_key() == req.cache_key()
    assert json.loads(
        (tmp_path / "run" / "rollout.json").read_text()
    ) == result.model_dump(mode="json")
    assert load_rollout(ArtifactRef.capture(tmp_path / "run")) == restored


def test_infrastructure_failure_preserves_evidence(tmp_path):
    """Infrastructure failures retain logs and skip scientific evaluation."""

    class FailedRunner:
        def run(self, request, output, *, environment=None):
            output.mkdir()
            (output / "log.txt").write_text("Transport failed")
            return RolloutResult(
                status="infrastructure_error", artifacts=ArtifactRef.capture(output)
            )

    result = execute_agent(request(tmp_path), FailedRunner(), tmp_path / "run")
    assert not (tmp_path / "run" / "evaluation").exists()
    assert result.artifacts.path("log.txt").read_text() == "Transport failed"


def test_changed_task_fails_closed(tmp_path):
    """Runner mutations of the frozen task invalidate the trial."""

    class MutatingRunner(ArithmeticRunner):
        def run(self, request, output, *, environment=None):
            result = super().run(request, output)
            (request.task.root / "input.json").write_text("[99]")
            return result

    with pytest.raises(ValueError):
        execute_agent(request(tmp_path), MutatingRunner(), tmp_path / "run")


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
    execute_agent(req, ArithmeticRunner(), output)
    with pytest.raises(FileExistsError):
        execute_agent(req, ArithmeticRunner(), output)


def test_malformed_adapter_types_fail_closed(tmp_path):
    """Runner adapters must return typed rollout results, not mappings."""

    class InvalidRunner:
        def run(self, request, output, *, environment=None):
            return {"status": "completed"}

    with pytest.raises(TypeError, match="RolloutResult"):
        execute_agent(request(tmp_path), InvalidRunner(), tmp_path / "run")


@pytest.mark.parametrize("value", [True, -1, float("inf"), float("nan"), "1"])
def test_invalid_usage_is_rejected(value):
    """Usage records reject nonnumeric, negative, and nonfinite costs."""
    with pytest.raises(ValueError):
        RolloutResult(status="completed", usage={"cost_usd": value})


def test_restored_evidence_must_still_match(tmp_path):
    """Restoring a trial verifies that retained evidence is unchanged."""
    request_record = request(tmp_path)
    result = execute_agent(request_record, ArithmeticRunner(), tmp_path / "run")
    encoded = RolloutRecord(request=request_record, rollout=result).model_dump(
        mode="json"
    )
    (result.artifacts.root / "answer.json").write_text("99")
    with pytest.raises(ValueError):
        RolloutRecord.model_validate(encoded)


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
        execute_agent(request(tmp_path), MutatingRunner(), tmp_path / "run")


def test_verified_record_binds_execution_metadata(tmp_path):
    """Verified record binds execution metadata."""
    req = request(tmp_path, offset=1)
    rollout = execute_agent(req, ArithmeticRunner(), tmp_path / "run")
    record = ArtifactRef.capture(tmp_path / "run")
    assert load_rollout(record).rollout.metadata["answer"] == 7
    data = RolloutRecord(request=req, rollout=rollout).model_dump(mode="json")
    data["rollout"]["metadata"]["answer"] = 6
    (record.root / "rollout.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="tampered"):
        load_rollout(record)


def test_execution_record_rejects_evaluation(tmp_path):
    """Execution record rejects evaluation."""
    req = request(tmp_path)
    rollout = execute_agent(req, ArithmeticRunner(), tmp_path / "run")
    data = RolloutRecord(request=req, rollout=rollout).model_dump(mode="json")
    data["evaluation"] = {"passed": True}
    with pytest.raises(ValidationError):
        RolloutRecord.model_validate(data)


def test_runner_exception_preserves_partial_evidence_and_reserves_output(tmp_path):
    """Runner exception preserves partial evidence and reserves output."""

    class BrokenRunner:
        def run(self, request, output, *, environment=None):
            assert not output.exists()
            output.mkdir()
            (output / "error.txt").write_text("execution interrupted")
            raise RuntimeError("failed")

    req = request(tmp_path)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="failed"):
        execute_agent(req, BrokenRunner(), output)
    assert (output / "rollout" / "error.txt").read_text() == "execution interrupted"
    assert not (output / "rollout.json").exists()
    with pytest.raises(FileExistsError):
        execute_agent(req, ArithmeticRunner(), output)


def test_runner_must_return_current_evidence(tmp_path):
    """Runner must return current evidence."""

    class StaleRunner(ArithmeticRunner):
        def run(self, request, output, *, environment=None):
            result = super().run(request, output)
            (output / "answer.json").write_text("99")
            return result

    with pytest.raises(ValueError, match="tampered"):
        execute_agent(request(tmp_path), StaleRunner(), tmp_path / "run")
    assert not (tmp_path / "run" / "rollout.json").exists()
