"""Real Harbor configuration tests and an opt-in SAIL baseline pilot."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyforger.artifacts import ArtifactRef
from pyforger.harbor import (
    HarborRequest,
    HarborTaskRunner,
    _output_path,
    _redact,
    _scrub,
    _secrets,
)


@pytest.fixture
def task(tmp_path):
    """Create a minimal trusted native Harbor task package."""
    root = tmp_path / "task"
    root.mkdir()
    (root / "instruction.md").write_text("Write hello to /tmp/hello.txt.\n")
    (root / "environment").mkdir()
    (root / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (root / "task.toml").write_text(
        'version = "1.0"\n'
        "[agent]\ntimeout_sec = 60\n"
        "[verifier]\ntimeout_sec = 60\n"
        "[environment]\ncpus = 1\nmemory_mb = 1024\nstorage_mb = 10240\n"
        'network_mode = "no-network"\n'
    )
    (root / "solution").mkdir()
    (root / "solution" / "solve.sh").write_text(
        "#!/bin/bash\nprintf hello > /tmp/hello.txt\n"
    )
    (root / "tests").mkdir()
    (root / "tests" / "test.sh").write_text(
        "#!/bin/bash\nmkdir -p /logs/verifier\n"
        'if [ "$(cat /tmp/hello.txt 2>/dev/null)" = hello ]; then\n'
        "  echo 1 > /logs/verifier/reward.txt\n"
        "else\n  echo 0 > /logs/verifier/reward.txt\nfi\n"
    )
    return ArtifactRef.capture(root)


def test_import_does_not_load_harbor():
    """Keep the optional SDK out of module imports and runner construction."""
    script = (
        "import sys; from pyforger.harbor import HarborTaskRunner; "
        "HarborTaskRunner(agent={'name':'nop'}, environment={'type':'sail'}); "
        "assert 'harbor' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("backend", ["sail", "modal", "docker"])
def test_native_configuration(task, tmp_path, backend):
    """Use native Pydantic validation for each built-in backend."""
    pytest.importorskip("harbor")
    from harbor.models.trial.config import TrialConfig

    request = HarborRequest(
        trial_id="id/does-not-become-path", task=task, model="provider/user-model"
    )
    agent = {"name": "nop", "kwargs": {"api_key_env": "MY_PROVIDER_KEY"}}
    runner = HarborTaskRunner(
        agent=agent, environment={"type": backend}, max_iterations=7, timeout_s=12
    )
    config = runner._configuration(request, task.root, tmp_path / "output")
    assert isinstance(config, TrialConfig)
    assert config.agent.model_name == request.model
    assert config.agent.kwargs["max_iterations"] == 7
    assert config.agent.override_timeout_sec == 12
    assert config.environment.type.value == backend
    assert config.trial_name.startswith("trial-")
    assert (
        runner._configuration(request, task.root, tmp_path / "other").trial_name
        != config.trial_name
    )
    assert "max_iterations" not in agent["kwargs"]
    restored = TrialConfig.model_validate_json(config.model_dump_json())
    assert restored == config


def test_custom_native_configuration(task, tmp_path):
    """Support arbitrary trusted native agent and environment import paths."""
    pytest.importorskip("harbor")
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig

    agent = AgentConfig(import_path="trusted.agents:CustomAgent")
    backend = EnvironmentConfig(import_path="trusted.environments:CustomEnvironment")
    runner = HarborTaskRunner(agent=agent, environment=backend)
    agent.import_path = "changed:Agent"
    request = HarborRequest(
        trial_id="sample",
        task=task,
        model="model",
        configuration={"max_iterations": 3, "timeout_s": 4},
    )
    config = runner._configuration(request, task.root, tmp_path / "output")
    assert config.agent.import_path == "trusted.agents:CustomAgent"
    assert config.environment.import_path == backend.import_path
    assert config.agent.kwargs["max_iterations"] == 3
    assert config.agent.override_timeout_sec == 4


@pytest.mark.parametrize(
    "parameters",
    [
        {"agent": {"import_path": "evil:Agent"}},
        {"environment": {"type": "docker"}},
        {"task": "/other"},
        {"outputs": ["../../other"]},
        {"max_iterations": True},
        {"max_iterations": 0},
        {"timeout_s": -1},
    ],
)
def test_reject_request_overrides(task, tmp_path, parameters):
    """Requests cannot change trusted runtime selection or bypass limits."""
    pytest.importorskip("harbor")
    runner = HarborTaskRunner(agent={"name": "nop"}, environment={"type": "sail"})
    request = HarborRequest(
        trial_id="sample", task=task, model="model", configuration=parameters
    )
    with pytest.raises(ValueError):
        runner._configuration(request, task.root, tmp_path / "output")


def test_reject_changed_task_before_execution(task, tmp_path):
    """Fail manifest verification before an SDK or backend is created."""
    (task.root / "instruction.md").write_text("changed")
    request = HarborRequest(trial_id="sample", task=task, model="model")
    runner = HarborTaskRunner(agent={"name": "nop"}, environment={"type": "sail"})
    with pytest.raises(ValueError, match="verification failed"):
        runner.run(request, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_reject_output_aliases_and_overlap(task, tmp_path):
    """Never overwrite task packages or follow output symlinks."""
    for output in (task.root, task.root / "nested", tmp_path):
        with pytest.raises(ValueError):
            _output_path(output, task)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        _output_path(existing, task)
    alias = tmp_path / "alias"
    alias.symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        _output_path(alias / "trial", task)


def test_reject_external_environment_and_active_loop(task, tmp_path):
    """Avoid silently ignoring environments or nesting asyncio.run calls."""
    runner = HarborTaskRunner(agent={"name": "nop"}, environment={"type": "sail"})
    request = HarborRequest(trial_id="sample", task=task, model="model")
    with pytest.raises(ValueError, match="owns"):
        runner.run(request, tmp_path / "output", environment=object())

    async def check():
        """Verify check."""
        with pytest.raises(RuntimeError, match="arun"):
            runner.run(request, tmp_path / "output")

    asyncio.run(check())


def test_redact_native_files_and_metadata(tmp_path, monkeypatch):
    """Remove literal and environment-resolved secrets, including JSON escapes."""
    secret = 'secret-quote"newline\nvalue'
    monkeypatch.setenv("EXAMPLE_PROVIDER_KEY", secret)
    config = {"kwargs": {"api_key_env": "EXAMPLE_PROVIDER_KEY", "api_key": secret}}
    secrets = _secrets(config)
    output = tmp_path / "output"
    output.mkdir()
    payload = {"config": config, "message": "failure with " + secret}
    (output / "result.json").write_text(json.dumps(payload))
    (output / "trajectory.bin").write_bytes(secret.encode())
    _scrub(output, secrets)
    retained = ArtifactRef.capture(output)
    assert secret.encode() not in retained.path("trajectory.bin").read_bytes()
    serialized = json.loads(retained.path("result.json").read_text())
    assert serialized["config"]["kwargs"]["api_key"] == "[REDACTED]"
    assert serialized["config"]["kwargs"]["api_key_env"] == "EXAMPLE_PROVIDER_KEY"
    assert secret not in json.dumps(_redact(payload, secrets))
    assert secret not in serialized["message"]


def test_scrub_rejects_symlink_without_touching_target(tmp_path):
    """Validate all native files before redaction can write outside the output."""
    output = tmp_path / "output"
    output.mkdir()
    target = tmp_path / "secret.txt"
    target.write_text("credential-value")
    (output / "link").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        _scrub(output, {"credential-value"})
    assert target.read_text() == "credential-value"


@pytest.mark.skipif(
    os.environ.get("FORGER_HARBOR_SAIL_TEST") != "1",
    reason="Opt in to authorized SAIL resource usage with FORGER_HARBOR_SAIL_TEST=1",
)
@pytest.mark.parametrize("agent, reward", [("oracle", 1), ("nop", 0)])
def test_sail_native_baseline(task, tmp_path, agent, reward):
    """Run actual Harbor Oracle/Nop trials without Docker or paid LLM calls."""
    pytest.importorskip("harbor")
    request = HarborRequest(
        trial_id="sail-" + agent, task=task, model="baseline/no-llm"
    )
    runner = HarborTaskRunner(agent={"name": agent}, environment={"type": "sail"})
    result = runner.run(request, tmp_path / "output")
    assert result.status == "completed", result.metadata
    assert result.metadata["harbor_result"]["verifier_result"]["rewards"] == {
        "reward": reward
    }
    result.artifacts.verify()
    config = json.loads(
        result.artifacts.path(
            str(Path(result.metadata["harbor_result_path"]).parent / "config.json")
        ).read_text()
    )
    assert config["agent"]["model_name"] == request.model
    assert result.metadata["harbor_result_path"].endswith("/result.json")
    assert result.artifacts.path(result.metadata["harbor_result_path"]).is_file()
    assert task.cache_key() == task.digest


def test_script_execution_rejects_changed_evidence(task, tmp_path):
    """Saved program execution verifies its artifact before creating outputs."""
    from pyforger.harbor import execute_script

    script = tmp_path / "scripts"
    script.mkdir()
    (script / "analysis.py").write_text("print(1)")
    retained = ArtifactRef.capture(script)
    (script / "analysis.py").write_text("changed")
    with pytest.raises(ValueError):
        execute_script(task, retained, tmp_path / "out", environment={"type": "docker"})
    assert not (tmp_path / "out").exists()


def test_native_outcomes_and_declared_file_collection(tmp_path):
    """Verify native outcomes and declared file collection."""
    from pyforger.harbor import HarborResult, collect_trial_files, trial_outcome

    output = tmp_path / "native"
    trial = output / "trial-test"
    (trial / "artifacts/work").mkdir(parents=True)
    (trial / "agent").mkdir()
    (trial / "result.json").write_text("{}")
    (trial / "artifacts/work/answer.json").write_text("[1]")
    (trial / "artifacts/manifest.json").write_text(
        json.dumps(
            [
                {
                    "source": "/work/answer.json",
                    "destination": "artifacts/work/answer.json",
                    "status": "ok",
                }
            ]
        )
    )
    native = {"verifier_result": {"rewards": {"reward": 1}}}
    rollout = HarborResult(
        status="completed",
        artifacts=ArtifactRef.capture(output),
        metadata={
            "harbor_result": native,
            "harbor_result_path": "trial-test/result.json",
        },
    )
    assert trial_outcome(rollout)["passed"] is True
    destination = tmp_path / "collected"
    destination.mkdir()
    collect_trial_files(rollout, {"/work/answer.json": "answer.json"}, destination)
    assert (destination / "answer.json").read_text() == "[1]"
    with pytest.raises(ValueError):
        collect_trial_files(rollout, {"/work/answer.json": "../escape"}, destination)
    failed = HarborResult(
        status="failed", artifacts=rollout.artifacts, metadata=rollout.metadata
    )
    assert trial_outcome(failed)["passed"] is None


@pytest.mark.parametrize(
    "exception", ["AgentTimeoutError", "AuthenticationError", "VerifierTimeoutError"]
)
def test_native_exceptions_are_not_generic_grades(tmp_path, exception):
    """Backend diagnostics do not prescribe denominator or admission policy."""
    from pyforger.harbor import HarborResult, trial_outcome

    (tmp_path / "result.json").write_text("{}")
    result = HarborResult(
        status="failed",
        artifacts=ArtifactRef.capture(tmp_path),
        metadata={
            "harbor_result_path": "result.json",
            "harbor_result": {"exception_info": {"exception_type": exception}},
        },
    )
    outcome = trial_outcome(result)
    assert outcome["status"] == "failed" and outcome["passed"] is None
    assert outcome["exception"]["exception_type"] == exception
    assert "outcome" not in outcome


def test_native_result_lookup_verifies_retained_files(tmp_path):
    """A manifest lookup alone must not make corrupted native output trusted."""
    from pyforger.harbor import HarborResult, trial_directory

    (tmp_path / "result.json").write_text("{}")
    result = HarborResult(
        status="completed",
        artifacts=ArtifactRef.capture(tmp_path),
        metadata={"harbor_result_path": "result.json"},
    )
    (tmp_path / "result.json").write_text("tampered")
    with pytest.raises(ValueError, match="tampered"):
        trial_directory(result)


def test_scrub_preserves_unchanged_json_store_identity(tmp_path):
    """Leave clean JSON bytes and their content-addressed task digest intact."""
    from pyforger.artifacts import ArtifactStore

    source = tmp_path / "source"
    source.mkdir()
    content = b'{"unicode":"\\u00e9", "values": [1,2]}\n'
    (source / "data.json").write_bytes(content)
    output = tmp_path / "output"
    stored = ArtifactStore(output / "tasks").put(source)
    _scrub(output, set())
    assert stored.path("data.json").read_bytes() == content
    ArtifactStore(output / "tasks").get(stored.digest).verify()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_native_env_literals_redacted_before_trial_creation(
    task, tmp_path, monkeypatch, native, fail
):
    """Substitute sandbox execution only; exercise native config and results."""
    from types import SimpleNamespace

    from harbor.models.trial.config import AgentConfig, EnvironmentConfig
    from harbor.models.trial.result import AgentInfo, TrialResult
    from harbor.trial.trial import Trial

    literals = {"API_KEY": 'agent-literal-"private', "PASSWORD": "backend-private"}
    agent = {"name": "nop", "env": {"API_KEY": literals["API_KEY"]}}
    backend = {"type": "docker", "env": {"PASSWORD": literals["PASSWORD"]}}
    if native:
        agent = AgentConfig.model_validate(agent)
        backend = EnvironmentConfig.model_validate(backend)

    async def create(cls, config):
        directory = config.trials_dir / config.trial_name
        directory.mkdir()
        message = json.dumps(literals)
        (directory / "trial.log").write_text(message)
        if fail:
            raise RuntimeError("fixture provisioning failed")
        result = TrialResult(
            task_name="fixture",
            trial_name=config.trial_name,
            trial_uri=directory.as_uri(),
            task_id=config.task.get_task_id(),
            task_checksum=task.digest,
            config=config,
            agent_info=AgentInfo(name="nop", version="fixture"),
        )
        (directory / "result.json").write_text(result.model_dump_json())

        async def run():
            return result

        return SimpleNamespace(paths=SimpleNamespace(trial_dir=directory), run=run)

    monkeypatch.setattr(Trial, "create", classmethod(create))
    runner = HarborTaskRunner(agent=agent, environment=backend)
    output = tmp_path / "native"
    if fail:
        with pytest.raises(RuntimeError, match="provisioning"):
            runner.run(HarborRequest(trial_id="literal", task=task), output)
    else:
        result = runner.run(HarborRequest(trial_id="literal", task=task), output)
        assert result.status == "completed"
        assert all(
            value not in json.dumps(result.metadata) for value in literals.values()
        )
    for path in output.rglob("*"):
        if path.is_file():
            assert all(
                value.encode() not in path.read_bytes()
                and json.dumps(value)[1:-1].encode() not in path.read_bytes()
                for value in literals.values()
            )
    task.verify()


class _FakeException:
    def __init__(self, type_, message="", traceback_=""):
        self.exception_type = type_
        self.exception_message = message
        self.exception_traceback = traceback_


def _fake_result(exception=None, agent_result=None):
    from types import SimpleNamespace

    payload = SimpleNamespace(
        exception_info=exception,
        agent_result=agent_result,
        model_dump=lambda mode: {"exception_info": None},
        compute_token_cost_totals=lambda: (None, None, None, None),
    )
    return payload


def test_transient_startup_failure_classification():
    """Only pre-agent transient provider failures qualify for infra retry."""
    from pyforger.harbor import _transient_startup_failure

    assert _transient_startup_failure(
        _fake_result(_FakeException("ApiError", "upload failed: HTTP 503"))
    )
    assert _transient_startup_failure(
        _fake_result(_FakeException("ServiceUnavailableError"))
    )
    # Agent already produced a result: never retry, work was consumed.
    assert not _transient_startup_failure(
        _fake_result(_FakeException("ApiError", "HTTP 503"), agent_result=object())
    )
    # Deterministic failures are not transient.
    assert not _transient_startup_failure(
        _fake_result(_FakeException("AuthenticationError", "bad key"))
    )
    assert not _transient_startup_failure(_fake_result())


def _install_flaky_trial(monkeypatch, outcomes):
    """Substitute Trial.create/run with scripted outcomes; real runner logic stays."""
    from types import SimpleNamespace

    from harbor.trial.trial import Trial

    calls = []

    async def create(cls, config):
        index = len(calls)
        calls.append(config.trial_name)
        directory = config.trials_dir / config.trial_name
        directory.mkdir(parents=True, exist_ok=True)

        async def run():
            return outcomes[index]

        return SimpleNamespace(paths=SimpleNamespace(trial_dir=directory), run=run)

    monkeypatch.setattr(Trial, "create", classmethod(create))
    # Zero the backoff so the retry path runs at test speed.
    import pyforger.harbor as harbor_module

    monkeypatch.setattr(harbor_module, "_STARTUP_BACKOFF_S", (0.0, 0.0))
    return calls


def test_runner_retries_transient_startup_failure(task, tmp_path, monkeypatch):
    """A transient pre-agent 503 is retried and the later success returned."""
    outcomes = [
        _fake_result(_FakeException("ApiError", "upload failed: HTTP 503")),
        _fake_result(),
    ]
    calls = _install_flaky_trial(monkeypatch, outcomes)
    runner = HarborTaskRunner(agent={"name": "nop"}, environment={"type": "docker"})
    result = runner.run(HarborRequest(trial_id="retry", task=task), tmp_path / "native")
    assert result.status == "completed"
    assert len(calls) == 2


def test_runner_does_not_retry_deterministic_failure(task, tmp_path, monkeypatch):
    """A deterministic failure is returned after a single attempt."""
    outcomes = [_fake_result(_FakeException("AuthenticationError", "bad key"))]
    calls = _install_flaky_trial(monkeypatch, outcomes)
    runner = HarborTaskRunner(agent={"name": "nop"}, environment={"type": "docker"})
    result = runner.run(
        HarborRequest(trial_id="no-retry", task=task), tmp_path / "native"
    )
    assert result.status == "failed"
    assert len(calls) == 1


def test_runner_bounds_transient_retries(task, tmp_path, monkeypatch):
    """Persistent transient failures stop after the bounded attempts."""
    outcomes = [_fake_result(_FakeException("ApiError", "HTTP 503")) for _ in range(5)]
    calls = _install_flaky_trial(monkeypatch, outcomes)
    runner = HarborTaskRunner(agent={"name": "nop"}, environment={"type": "docker"})
    result = runner.run(
        HarborRequest(trial_id="bounded", task=task), tmp_path / "native"
    )
    assert result.status == "failed"
    assert len(calls) == 3


def test_real_native_missing_reward_retains_trace_and_verifier_logs(task, tmp_path):
    """Run Harbor's actual verifier failure locally, without model or provider calls."""
    from pyforger.agent_backend import AgentRunResult, AgentSpec
    from pyforger.agent_tasks import TaskSpec
    from pyforger.artifacts import FileInput
    from pyforger.evidence_journal import evidence_file_chunks
    from pyforger.harbor import trial_directory, trial_outcome
    from pyforger.task_runs import run_recorded_task

    root = task.root
    (root / "tests/test.sh").write_text(
        '#!/bin/bash\necho "native stdout before crash"\n'
        'echo "native stderr before crash" >&2\n'
        "python3 -c \"raise RuntimeError('actual grader crash')\"\n"
    )
    rollout = HarborTaskRunner(
        agent={"name": "nop"}, environment={"type": "docker"}
    ).run(
        HarborRequest(trial_id="missing-reward", task=ArtifactRef.capture(root)),
        tmp_path / "native",
    )
    outcome = trial_outcome(rollout)
    assert outcome["exception"]["exception_type"] == "RewardFileNotFoundError"
    trial = trial_directory(rollout)
    # The bridge below substitutes no execution: it feeds the real failed trial
    # through the existing host persistence API after native files are removed.
    evidence = {
        entry.path: evidence_file_chunks(
            FileInput.from_artifact(
                rollout.artifacts,
                (trial / entry.path).relative_to(rollout.artifacts.root).as_posix(),
            ).chunks()
        )[0]
        for entry in ArtifactRef.capture(trial).files
    }
    import shutil

    shutil.rmtree(tmp_path / "native")

    class CompletedNativeBackend:
        def run_agent(self, task, agent):
            return AgentRunResult(
                status="failed",
                failure_kind="infrastructure",
                diagnostics={**outcome, "native_evidence": evidence},
            )

    report = run_recorded_task(
        TaskSpec(instruction="failed native trial"),
        agent=AgentSpec(kind="nop"),
        backend=CompletedNativeBackend(),
        output=tmp_path / "retained",
    )
    retained = report["diagnostics"]["native_evidence"]

    def text(name):
        return (
            tmp_path / "retained" / retained[name]["evidence_file"]["path"]
        ).read_text()

    assert "RewardFileNotFoundError" in text("result.json")
    logs = "\n".join(text(name) for name in retained if name.startswith("verifier/"))
    assert "native stdout before crash" in logs
    assert "native stderr before crash" in logs
    assert "actual grader crash" in logs
    assert Path(report["diagnostics"]["harbor_result"]).is_file()


@pytest.mark.parametrize(
    "kind", ["file", "directory", "empty", "missing", "missing-ok"]
)
def test_declared_artifact_collection_preserves_nested_files(tmp_path, kind):
    from pyforger.harbor import HarborResult, collect_trial_files

    native = tmp_path / "native"
    trial = native / "trial"
    artifacts = trial / "artifacts/work/output"
    artifacts.mkdir(parents=True)
    source = artifacts / "submission"
    if kind == "file":
        source.write_bytes(b"\x00\xff")
    elif kind in {"directory", "empty"}:
        (source / "nested/empty").mkdir(parents=True)
        if kind == "directory":
            (source / "run.py").write_text("print('retained, not executed')")
            (source / "nested/helper.py").write_bytes(b"\x00\xff")
    (artifacts / "submission-private").write_text("not declared")
    (trial / "result.json").write_text("{}")
    (trial / "artifacts/manifest.json").write_text(
        json.dumps(
            [
                {
                    "source": "/work/output/submission",
                    "destination": "artifacts/work/output/submission",
                    "status": "failed" if kind == "missing" else "ok",
                    "type": "file" if kind == "file" else "directory",
                }
            ]
        )
    )
    rollout = HarborResult(
        status="completed",
        artifacts=ArtifactRef.capture(native),
        metadata={"harbor_result_path": "trial/result.json"},
    )
    output = tmp_path / "collected"
    output.mkdir()
    if kind == "missing-ok":
        with pytest.raises(ValueError, match="Missing declared artifact directory"):
            collect_trial_files(
                rollout, {"/work/output/submission": "submission"}, output
            )
        return
    collect_trial_files(rollout, {"/work/output/submission": "submission"}, output)
    if kind == "file":
        assert (output / "submission").read_bytes() == b"\x00\xff"
    elif kind == "directory":
        assert (
            output / "submission/run.py"
        ).read_text() == "print('retained, not executed')"
        assert (output / "submission/nested/helper.py").read_bytes() == b"\x00\xff"
        assert not (output / "submission/nested/empty").exists()
        with pytest.raises(ValueError, match="read limit"):
            collect_trial_files(
                rollout, {"/work/output/submission": "limited"}, output, max_bytes=1
            )
    else:
        assert not (output / "submission").exists()
    assert not (output / "submission-private").exists()


@pytest.mark.parametrize(
    "destination",
    [
        "../escape",
        "/tmp/escape",
        "artifacts/../result.json",
        "result.json",
        "artifacts\\escape",
    ],
)
def test_collection_rejects_manifest_destination_escape(tmp_path, destination):
    from pyforger.harbor import HarborResult, collect_trial_files

    trial = tmp_path / "native/trial"
    (trial / "artifacts").mkdir(parents=True)
    (trial / "result.json").write_text("{}")
    (trial / "artifacts/manifest.json").write_text(
        json.dumps(
            [
                {
                    "source": "/work/submission",
                    "destination": destination,
                    "status": "ok",
                    "type": "directory",
                }
            ]
        )
    )
    rollout = HarborResult(
        status="completed",
        artifacts=ArtifactRef.capture(trial.parent),
        metadata={"harbor_result_path": "trial/result.json"},
    )
    output = tmp_path / "collected"
    output.mkdir()
    with pytest.raises(ValueError):
        collect_trial_files(rollout, {"/work/submission": "submission"}, output)
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("symlink", ["source", "target", "tampered"])
def test_directory_collection_rejects_symlinks_and_tampering(tmp_path, symlink):
    from pyforger.harbor import HarborResult, collect_trial_files

    trial = tmp_path / "native/trial"
    submission = trial / "artifacts/work/submission"
    submission.mkdir(parents=True)
    (submission / "run.py").write_text("original")
    (trial / "result.json").write_text("{}")
    (trial / "artifacts/manifest.json").write_text(
        json.dumps(
            [
                {
                    "source": "/work/submission",
                    "destination": "artifacts/work/submission",
                    "status": "ok",
                    "type": "directory",
                }
            ]
        )
    )
    rollout = HarborResult(
        status="completed",
        artifacts=ArtifactRef.capture(trial.parent),
        metadata={"harbor_result_path": "trial/result.json"},
    )
    output = tmp_path / "collected"
    output.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    (private / "hidden").write_text("must stay private")
    if symlink == "source":
        (submission / "link").symlink_to(private, target_is_directory=True)
    elif symlink == "target":
        (output / "submission").symlink_to(private, target_is_directory=True)
    else:
        (submission / "run.py").write_text("changed")
    with pytest.raises(ValueError):
        collect_trial_files(rollout, {"/work/submission": "submission"}, output)
    assert (private / "hidden").read_text() == "must stay private"
    assert not (private / "run.py").exists()


@pytest.mark.parametrize("kind", ["directory", "empty", "missing"])
def test_directory_outputs_survive_cleanup_and_recording(tmp_path, kind):
    import shutil

    from pyforger.agent_backend import AgentRunResult, AgentSpec
    from pyforger.agent_tasks import TaskSpec
    from pyforger.harbor import HarborResult
    from pyforger.harbor_backend import _collect_outputs
    from pyforger.task_runs import run_recorded_task

    trial = tmp_path / "native/trial"
    submission = trial / "artifacts/work/output/submission"
    submission.mkdir(parents=True)
    if kind == "directory":
        (submission / "nested").mkdir()
        (submission / "run.py").write_text(
            "raise RuntimeError('never execute on host')"
        )
        (submission / "nested/helper.bin").write_bytes(b"\x00\xff")
    (trial / "artifacts/work/output/default.json").write_text("[1]")
    (trial / "result.json").write_text("{}")
    (trial / "artifacts/manifest.json").write_text(
        json.dumps(
            [
                {
                    "source": "/work/output/submission",
                    "destination": "artifacts/work/output/submission",
                    "status": "failed" if kind == "missing" else "ok",
                    "type": "directory",
                },
                {
                    "source": "/work/output/default.json",
                    "destination": "artifacts/work/output/default.json",
                    "status": "ok",
                    "type": "file",
                },
            ]
        )
    )
    rollout = HarborResult(
        status="completed",
        artifacts=ArtifactRef.capture(trial.parent),
        metadata={"harbor_result_path": "trial/result.json"},
    )
    outputs = ("output/submission", "output/default.json")
    collected = _collect_outputs(rollout, outputs, tmp_path / "collected")
    shutil.rmtree(trial.parent)
    shutil.rmtree(tmp_path / "collected")

    class RecordedArtifactsBackend:
        def run_agent(self, task, agent):
            return AgentRunResult(status="completed", passed=True, files=collected)

    retained = tmp_path / "retained"
    report = run_recorded_task(
        TaskSpec("retain", outputs=outputs),
        agent=AgentSpec(kind="nop"),
        backend=RecordedArtifactsBackend(),
        output=retained,
    )
    if kind == "directory":
        assert report["passed"] is True
        assert (
            retained / "output/submission/nested/helper.bin"
        ).read_bytes() == b"\x00\xff"
        assert (
            (retained / "output/submission/run.py")
            .read_text()
            .startswith("raise RuntimeError")
        )
    else:
        assert report["passed"] is None
        assert "Missing declared outputs" in report["error"]
    assert (retained / "output/default.json").read_text() == "[1]"


@pytest.mark.parametrize(
    "name",
    [
        "output/submission/../escape",
        "output/submission//run.py",
        "output/submission/./run.py",
    ],
)
def test_recorded_directory_rejects_noncanonical_descendants(tmp_path, name):
    from pyforger.agent_backend import AgentRunResult, AgentSpec
    from pyforger.agent_tasks import TaskSpec
    from pyforger.task_runs import run_recorded_task

    class InvalidArtifactsBackend:
        def run_agent(self, task, agent):
            return AgentRunResult(status="completed", passed=True, files={name: b"bad"})

    output = tmp_path / "retained"
    report = run_recorded_task(
        TaskSpec("retain", outputs=("output/submission",)),
        agent=AgentSpec(kind="nop"),
        backend=InvalidArtifactsBackend(),
        output=output,
    )
    assert report["passed"] is None
    assert report["status"] == "infrastructure_error"
    assert not (output / "output").exists()


def test_recorded_directory_scans_nested_credentials_before_writing(
    tmp_path, monkeypatch
):
    from pyforger.agent_backend import AgentRunResult, AgentSpec
    from pyforger.agent_tasks import TaskSpec
    from pyforger.task_runs import run_recorded_task

    monkeypatch.setenv("FORGER_TEST_API_KEY", "nested-sensitive-value")

    class UnsafeArtifactsBackend:
        def run_agent(self, task, agent):
            return AgentRunResult(
                status="completed",
                passed=True,
                files={
                    "output/submission/run.py": b"safe",
                    "output/submission/nested/helper.py": b"nested-sensitive-value",
                },
            )

    output = tmp_path / "retained"
    report = run_recorded_task(
        TaskSpec("retain", outputs=("output/submission",)),
        agent=AgentSpec(kind="nop"),
        backend=UnsafeArtifactsBackend(),
        output=output,
    )
    assert report["status"] == "artifact_safety_error"
    assert report["passed"] is None
    assert not (output / "output").exists()
