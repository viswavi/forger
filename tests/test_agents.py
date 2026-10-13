"""Runtime-neutral execution and replay tests using real trusted subprocesses."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from pyforger.artifacts import ArtifactRef
from pyforger.trials import (
    AgentRequest,
    EvaluationResult,
    RolloutResult,
    TrialResult,
    evaluate_agent,
    execute_agent,
    load_trial,
    replay_agent,
)


def _environment(environment):
    if environment is None:
        raise ValueError("This runner requires an explicit execution environment")
    return environment


def _limits(request):
    return request.configuration.get("command_timeout_s", 60.0)


def _collect(environment, request, output):
    outputs = tuple(request.configuration.get("outputs", []))
    if outputs:
        environment.collect(outputs, output / "files")


class CommandAgentRunner:
    """Run a public argv command in the supplied environment and retain logs."""

    def run(self, request, output, *, environment=None) -> RolloutResult:
        """Stage frozen task files, execute argv, and collect declared outputs."""
        environment = _environment(environment)
        environment.stage(request.task)
        command = request.configuration["command"]
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(arg, str) for arg in command)
        ):
            raise ValueError("command must be a nonempty JSON array of strings")
        output.mkdir()
        result = environment.run(tuple(command), timeout_s=_limits(request))
        (output / "command.json").write_text(json.dumps(command), encoding="utf-8")
        (output / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (output / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.status == "completed":
            _collect(environment, request, output)
        return RolloutResult(
            status=result.status,
            artifacts=ArtifactRef.capture(output),
            metadata={"exit_code": result.exit_code, "duration_s": result.duration_s},
        )

    def replay(self, request, rollout, output, *, environment=None) -> RolloutResult:
        """Execute the saved argv against the frozen task in a fresh environment."""
        if rollout.artifacts is None:
            raise ValueError("Replay requires retained command evidence")
        command = json.loads(rollout.artifacts.path("command.json").read_text())
        configuration = {**request.configuration, "command": command}
        replay_request = AgentRequest(
            trial_id=request.trial_id,
            task=request.task,
            model=request.model,
            configuration=configuration,
        )
        return self.run(replay_request, output, environment=environment)


class LocalEnvironment:
    """Trusted test-only subprocess environment, not a production sandbox."""

    def __init__(self, root):
        """Keep a private fixture workspace and lifecycle observations."""
        self.root = root
        self.commands = []
        self.closed = False

    def stage(self, inputs):
        """Copy verified fixture files into the new command working directory."""
        inputs.verify()
        for entry in inputs.files:
            target = self.root / entry.path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inputs.path(entry.path), target)

    def run(self, command, timeout_s, stdin=None):
        """Execute actual subprocesses against staged fixture files."""
        self.commands.append((command, timeout_s))
        started = time.monotonic()
        result = subprocess.run(
            command, cwd=self.root, input=stdin, capture_output=True, timeout=timeout_s
        )
        return SimpleNamespace(
            status="completed" if result.returncode == 0 else "failed",
            exit_code=result.returncode,
            stdout=result.stdout.decode(),
            stderr=result.stderr.decode(),
            duration_s=time.monotonic() - started,
        )

    def collect(self, outputs, output):
        """Retain declared real output files in a new artifact directory."""
        output.mkdir()
        for name in outputs:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root / name, target)
        return ArtifactRef.capture(output)


class LocalFactory:
    """Open separate working directories using the public factory contract."""

    def __init__(self, root):
        """Keep a private fixture workspace and lifecycle observations."""
        self.root = root
        self.environments = []
        self.specs = []

    @contextmanager
    def open(self, spec):
        """Retain lifecycle observations while always closing each environment."""
        root = self.root / str(len(self.environments))
        root.mkdir(parents=True)
        environment = LocalEnvironment(root)
        self.environments.append(environment)
        self.specs.append(spec)
        try:
            yield environment
        finally:
            environment.closed = True


def request(tmp_path, **configuration):
    """Create a verified tool-using arithmetic task and public limits."""
    task = tmp_path / "task"
    task.mkdir()
    (task / "input.json").write_text("[1, 2, 3]")
    (task / "instruction.md").write_text(
        "Compute the sum in input.json and write answer.txt."
    )
    return AgentRequest(
        trial_id="sample-1",
        task=ArtifactRef.capture(task),
        model="local/arithmetic",
        configuration={
            "environment": {
                "image": "test-local",
                "memory_mb": 128,
                "cpus": 0.5,
                "pids_limit": 16,
                "timeout_s": 20.0,
            },
            "command": [
                "python3",
                "-c",
                "import json; from pathlib import Path; "
                "Path('answer.txt').write_text("
                "str(sum(json.load(open('input.json')))))",
            ],
            "command_timeout_s": 7.0,
            "outputs": ["answer.txt"],
            **configuration,
        },
    )


class Evaluator:
    """Grade retained command output, not a substituted runner value."""

    def evaluate(self, request, rollout, output):
        """Compare saved arithmetic output to the frozen task data."""
        answer = int(rollout.artifacts.path("files/answer.txt").read_text())
        expected = sum(json.loads(request.task.path("input.json").read_text()))
        output.mkdir()
        (output / "grade.json").write_text(json.dumps({"answer": answer}))
        return EvaluationResult(
            score=float(answer == expected),
            passed=answer == expected,
            evidence=ArtifactRef.capture(output),
            metadata={"clean": True},
        )


def test_execute_then_grade_and_fresh_replay(tmp_path):
    """Execution is independent of grading and replay does not reuse state."""
    req = request(tmp_path)
    factory = LocalFactory(tmp_path / "envs")
    runner = CommandAgentRunner()
    rollout = execute_agent(
        req, runner, tmp_path / "execute", environment_factory=factory
    )
    record = ArtifactRef.capture(tmp_path / "execute")
    assert load_trial(record).evaluation is None
    assert rollout.artifacts.path("files/answer.txt").read_text() == "6"
    first = factory.environments[0]
    assert first.closed
    assert first.commands[0][1] == 7.0
    assert factory.specs[0]["memory_mb"] == 128
    assert factory.specs[0]["cpus"] == 0.5
    assert factory.specs[0]["pids_limit"] == 16
    assert factory.specs[0]["timeout_s"] == 20.0
    (first.root / "answer.txt").write_text("999")
    evaluation = evaluate_agent(req, rollout, Evaluator(), tmp_path / "grade")
    assert evaluation.passed
    assert evaluation.metadata == {"clean": True}
    replay = replay_agent(
        req, rollout, runner, tmp_path / "replay", environment_factory=factory
    )
    assert replay.artifacts.path("files/answer.txt").read_text() == "6"
    assert len(factory.environments) == 2
    assert all(environment.closed for environment in factory.environments)
    assert load_trial(record).model_dump(mode="json") == TrialResult(
        request=req, rollout=rollout
    ).model_dump(mode="json")


def test_no_environment_fallback(tmp_path):
    """Reusable command adapters never fall back to host execution."""
    with pytest.raises(ValueError, match="explicit execution environment"):
        execute_agent(request(tmp_path), CommandAgentRunner(), tmp_path / "run")


def test_rows_are_not_cached_and_outputs_have_safe_explicit_identity(tmp_path):
    """Repeated calls execute afresh and cannot traverse output paths."""
    from pyforger.alter.agent import execute_agent as execute_rows

    req = request(tmp_path)
    req = AgentRequest(
        trial_id="../../escape",
        task=req.task,
        model=req.model,
        configuration=req.configuration,
    )
    items = [{"request": req.model_dump(mode="json"), "upstream": 3}]
    factory = LocalFactory(tmp_path / "envs")
    runner = CommandAgentRunner()
    one = execute_rows(
        items, runner=runner, output=tmp_path / "one", environment_factory=factory
    )
    two = execute_rows(
        items, runner=runner, output=tmp_path / "two", environment_factory=factory
    )
    assert len(factory.environments) == 2
    assert one[0]["upstream"] == two[0]["upstream"] == 3
    assert "rollout" not in items[0]
    assert len(list((tmp_path / "one").iterdir())) == 1
    with pytest.raises(ValueError, match="unique"):
        execute_rows(
            items * 2,
            runner=runner,
            output=tmp_path / "bad",
            environment_factory=factory,
        )


@pytest.mark.parametrize("score", [True, float("inf"), float("nan"), "1"])
def test_invalid_scores(score):
    """Generic scores reject booleans and nonfinite or nonnumeric values."""
    with pytest.raises(ValueError):
        EvaluationResult(score=score)


def test_provider_owned_lifecycle(tmp_path):
    """An external runtime receives None and owns execution and fresh replay."""

    class ProviderRunner:
        def run(self, request, output, *, environment=None):
            assert environment is None
            output.mkdir()
            (output / "result.txt").write_text("provider execution")
            return RolloutResult(
                status="completed", artifacts=ArtifactRef.capture(output)
            )

        def replay(self, request, rollout, output, *, environment=None):
            assert environment is None
            assert (
                rollout.artifacts.path("result.txt").read_text() == "provider execution"
            )
            return self.run(request, output)

    req = request(tmp_path)
    runner = ProviderRunner()
    rollout = execute_agent(req, runner, tmp_path / "execute")
    assert replay_agent(req, rollout, runner, tmp_path / "replay").status == "completed"


def test_environment_closes_on_runner_exception(tmp_path):
    """Adapter exceptions propagate while the factory context still closes."""

    class BrokenRunner:
        def run(self, request, output, *, environment=None):
            raise RuntimeError("runner failed")

    factory = LocalFactory(tmp_path / "envs")
    with pytest.raises(RuntimeError, match="runner failed"):
        execute_agent(
            request(tmp_path),
            BrokenRunner(),
            tmp_path / "execute",
            environment_factory=factory,
        )
    assert factory.environments[0].closed


def test_grading_cannot_rewrite_rollout_metadata(tmp_path):
    """Grading preserves the snapshotted execution record as well as files."""

    class MutatingEvaluator(Evaluator):
        def evaluate(self, request, rollout, output):
            result = super().evaluate(request, rollout, output)
            rollout.metadata["exit_code"] = 99
            return result

    req = request(tmp_path)
    rollout = execute_agent(
        req,
        CommandAgentRunner(),
        tmp_path / "execute",
        environment_factory=LocalFactory(tmp_path / "envs"),
    )
    with pytest.raises(ValueError, match="metadata"):
        evaluate_agent(req, rollout, MutatingEvaluator(), tmp_path / "grade")


def test_row_evaluation_and_provider_replay(tmp_path):
    """Row operations compose without requiring a duplicate provider factory."""
    from pyforger.alter.agent import evaluate_agent as grade_rows
    from pyforger.alter.agent import replay_agent as replay_rows

    req = request(tmp_path)
    runner = CommandAgentRunner()
    rollout = execute_agent(
        req,
        runner,
        tmp_path / "execute",
        environment_factory=LocalFactory(tmp_path / "envs"),
    )
    rows = [
        {
            "request": req.model_dump(mode="json"),
            "rollout": rollout.model_dump(mode="json"),
        }
    ]
    graded = grade_rows(rows, evaluator=Evaluator(), output=tmp_path / "grade")
    assert graded[0]["evaluation"]["score"] == 1.0
    assert "evaluation" not in rows[0]

    class ProviderReplay:
        def replay(self, request, rollout, output, *, environment=None):
            assert environment is None
            output.mkdir()
            return RolloutResult(
                status="completed", artifacts=ArtifactRef.capture(output)
            )

    replayed = replay_rows(graded, runner=ProviderReplay(), output=tmp_path / "replay")
    assert replayed[0]["replay"]["status"] == "completed"
    assert replayed[0]["evaluation"] == graded[0]["evaluation"]


def test_environment_factory_requires_environment_configuration(tmp_path):
    """A factory without configuration['environment'] fails with guidance."""
    req = request(tmp_path)
    req = AgentRequest(
        trial_id=req.trial_id, task=req.task, model=req.model, configuration={}
    )
    with pytest.raises(ValueError, match="environment"):
        execute_agent(
            req,
            CommandAgentRunner(),
            tmp_path / "run",
            environment_factory=LocalFactory(tmp_path / "envs"),
        )
