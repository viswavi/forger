"""Runtime-neutral execution tests using real trusted subprocesses."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from pyforger.artifacts import ArtifactRef
from pyforger.op import clear_trace, trace
from pyforger.trials import (
    RolloutRecord,
    RolloutResult,
    TrialRequest,
    execute_agent,
    load_rollout,
)


def _environment(environment):
    """environment."""
    if environment is None:
        raise ValueError("This runner requires an explicit execution environment")
    return environment


def _limits(request):
    """limits."""
    return request.configuration.get("command_timeout_s", 60.0)


def _collect(environment, request, output):
    """collect."""
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
    return TrialRequest(
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


def test_execute_is_fresh_and_records_are_execution_only(tmp_path):
    """Execute is fresh and records are execution only."""
    req = request(tmp_path)
    factory = LocalFactory(tmp_path / "envs")
    runner = CommandAgentRunner()
    rollout = execute_agent(
        req, runner, tmp_path / "execute", environment_factory=factory
    )
    record = ArtifactRef.capture(tmp_path / "execute")
    assert rollout.artifacts.path("files/answer.txt").read_text() == "6"
    first = factory.environments[0]
    assert first.closed
    assert first.commands[0][1] == 7.0
    assert factory.specs[0] == req.configuration["environment"]
    (first.root / "answer.txt").write_text("999")
    second = execute_agent(
        req, runner, tmp_path / "second", environment_factory=factory
    )
    assert second.artifacts.path("files/answer.txt").read_text() == "6"
    assert len(factory.environments) == 2
    assert all(environment.closed for environment in factory.environments)
    assert load_rollout(record).model_dump(mode="json") == RolloutRecord(
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
    req = TrialRequest(
        trial_id="../../escape",
        task=req.task,
        model=req.model,
        configuration=req.configuration,
    )
    items = [{"request": req.model_dump(mode="json"), "upstream": 3}]
    factory = LocalFactory(tmp_path / "envs")
    runner = CommandAgentRunner()
    clear_trace()
    one = execute_rows(
        items, runner=runner, output=tmp_path / "one", environment_factory=factory
    )
    two = execute_rows(
        items, runner=runner, output=tmp_path / "two", environment_factory=factory
    )
    records = [r for r in trace() if r.op_name == "alter.execute_agent"]
    assert len(records) == 2 and not any(r.cache_hit for r in records)
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


def test_provider_owned_lifecycle(tmp_path):
    """An external runtime receives None and owns fresh execution."""

    class ProviderRunner:
        def run(self, request, output, *, environment=None):
            assert environment is None
            output.mkdir()
            (output / "result.txt").write_text("provider execution")
            return RolloutResult(
                status="completed", artifacts=ArtifactRef.capture(output)
            )

    req = request(tmp_path)
    runner = ProviderRunner()
    rollout = execute_agent(req, runner, tmp_path / "execute")
    assert rollout.status == "completed"
    assert execute_agent(req, runner, tmp_path / "second").status == "completed"


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


def test_environment_factory_requires_environment_configuration(tmp_path):
    """A factory without configuration['environment'] fails with guidance."""
    req = request(tmp_path)
    req = TrialRequest(
        trial_id=req.trial_id, task=req.task, model=req.model, configuration={}
    )
    with pytest.raises(ValueError, match="environment"):
        execute_agent(
            req,
            CommandAgentRunner(),
            tmp_path / "run",
            environment_factory=LocalFactory(tmp_path / "envs"),
        )
    assert not (tmp_path / "run").exists()
