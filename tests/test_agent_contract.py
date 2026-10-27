"""Static and runtime contract for agent functions over any backend."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyforger.agent_backend import AgentBackend, AgentSpec
from pyforger.harbor_backend import HarborBackend
from pyforger.local_backend import LocalBackend


def test_agent_spec_validation():
    """Verify agent spec validation."""
    with pytest.raises(ValueError, match="require a Model"):
        AgentSpec()
    with pytest.raises(ValueError, match="oracle agents require"):
        AgentSpec(kind="oracle")
    with pytest.raises(ValueError, match="agent kind"):
        AgentSpec(kind="other", solution="true")
    assert AgentSpec(kind="nop").cache_key()


def test_backends_satisfy_protocol_without_models():
    """Verify backends satisfy protocol without models."""
    for backend in (LocalBackend(), HarborBackend()):
        assert isinstance(backend, AgentBackend)
        assert backend.cache_key()
        assert not hasattr(backend, "model")


def test_backend_constructors_reject_embedded_env():
    """Verify backend constructors reject embedded env."""
    with pytest.raises(ValueError, match="must not contain env"):
        HarborBackend(environment={"type": "docker", "env": {"A": "b"}})


def test_functions_accept_any_backend_statically(tmp_path):
    """Mypy proves agent functions compose with every backend type."""
    source = tmp_path / "agent_contract.py"
    source.write_text(
        "from pyforger import agents\n"
        "from pyforger.acquire.model import Model\n"
        "from pyforger.agent_backend import AgentBackend, AgentSpec\n"
        "from pyforger.agent_tasks import run_command, run_task, TaskSpec\n"
        "from pyforger.harbor_backend import HarborBackend\n"
        "from pyforger.local_backend import LocalBackend\n"
        "agent = AgentSpec(Model('openai/test'))\n"
        "backends: list[AgentBackend] = [LocalBackend(), HarborBackend()]\n"
        "for backend in backends:\n"
        "    agents.generate_source('g', {}, agent=agent, backend=backend)\n"
        "    agents.generate_files('i', {}, ('o',), agent=agent, backend=backend)\n"
        "    agents.execute_source('c', {}, backend=backend)\n"
        "    run_task(TaskSpec('i'), agent=agent, backend=backend)\n"
        "    run_command(TaskSpec('i'), 'sh solve.sh', backend=backend)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--ignore-missing-imports",
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_harbor_packaging_is_backend_internal(tmp_path):
    """Packaging produces a native task without agent-function involvement."""
    from pyforger.agent_backend import Network, WorkspaceTask
    from pyforger.harbor_backend import _package

    task = WorkspaceTask(
        instruction="do it",
        files={"data/input.txt": "x"},
        outputs=("answer.json",),
        network=Network.OFFLINE,
    )
    root = tmp_path / "task"
    artifact = _package(task, AgentSpec(kind="nop"), root, ["/work/answer.json"])
    assert (root / "environment/files/data/input.txt").read_text() == "x"
    assert 'network_mode = "no-network"' in (root / "task.toml").read_text()
    artifact.verify()


def test_harbor_backend_retains_trial_directory(tmp_path, monkeypatch):
    """Retained trials survive the backend's transient staging directory."""
    from pyforger import harbor
    from pyforger.agent_backend import WorkspaceTask
    from pyforger.artifacts import ArtifactRef
    from pyforger.harbor_backend import HarborBackend

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        async def arun(self, request, root):
            trial = Path(root) / "trial-fake"
            (trial / "agent").mkdir(parents=True)
            (trial / "result.json").write_text("{}")
            (trial / "agent" / "trajectory.json").write_text("[]")
            artifacts = ArtifactRef.capture(root)
            return SimpleNamespace(
                status="completed",
                artifacts=artifacts,
                metadata={
                    "harbor_result": {},
                    "harbor_result_path": "trial-fake/result.json",
                },
            )

    monkeypatch.setattr(harbor, "HarborTaskRunner", FakeRunner)
    retain = tmp_path / "retained"
    backend = HarborBackend(retain_trials=retain)
    result = backend.run_agent(
        WorkspaceTask(instruction="do it"),
        AgentSpec(kind="nop"),
    )
    trial_dir = Path(result.diagnostics["trial_dir"])
    assert trial_dir.parent == retain
    assert (trial_dir / "result.json").is_file()
    assert result.diagnostics["harbor_result"] == str(trial_dir / "result.json")
