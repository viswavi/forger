"""Forger owns backend policy, portable packaging, and normalized outcomes."""

import json

import pytest

from pyforger.agent_backend import AgentRunResult, AgentSpec, Network
from pyforger.agent_backends import (
    require_inner_loop_validation,
    supports_inner_loop_validation,
)
from pyforger.agent_tasks import TaskSpec, run_task
from pyforger.harbor_backend import HarborBackend, _failure_kind
from pyforger.local_backend import LocalBackend


def test_backend_capability():
    """Backend capability."""
    for backend, supported in ((LocalBackend(), True), (HarborBackend(), False)):
        assert supports_inner_loop_validation(backend) is supported
        if supported:
            require_inner_loop_validation(backend)
        else:
            with pytest.raises(NotImplementedError):
                require_inner_loop_validation(backend)


def test_neutral_task_roundtrip(tmp_path):
    """Neutral task roundtrip."""
    task = TaskSpec(
        "solve λ",
        files={"data/x": b"\x00\xff"},
        outputs=("answer",),
        packages=("numpy==2.0",),
        network=Network.PUBLIC,
        verifier="echo 1 > reward.txt",
        verifier_files={"secret": b"private"},
        solution_files={"solve.py": b"oracle"},
    )
    root = task.save(tmp_path / "task")
    assert TaskSpec.load(root) == task
    assert not (root / "solver/secret").exists()
    assert not (root / "solver/solve.py").exists()
    with pytest.raises(FileExistsError):
        task.save(root)


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/escape",
        "x/../../escape",
        "x\\escape",
        "x//a",
        "x/./a",
        "",
        "C:escape",
    ],
)
def test_reject_unsafe_paths(name):
    """Reject unsafe paths."""
    for field in ("files", "verifier_files", "solution_files"):
        with pytest.raises(ValueError):
            TaskSpec("solve", **{field: {name: b"bad"}})
    with pytest.raises(ValueError):
        TaskSpec("solve", outputs=(name,))


def test_reject_links(tmp_path):
    """Reject links."""
    root = TaskSpec("solve").save(tmp_path / "task")
    (root / "solver/leak").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="Unsafe"):
        TaskSpec.load(root)
    (tmp_path / "alias").symlink_to(root, target_is_directory=True)
    nested = TaskSpec("solve").save(tmp_path / "alias/new")
    assert nested == root / "new"
    with pytest.raises(ValueError, match="symlink"):
        TaskSpec.load(tmp_path / "alias")


def test_manifest_cannot_override_payload_roots(tmp_path):
    """Manifest cannot override payload roots."""
    root = TaskSpec("solve").save(tmp_path / "task")
    manifest = json.loads((root / "task.json").read_text())
    manifest["solver"] = "../private"
    (root / "task.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        TaskSpec.load(root)


def test_deterministic_runs_and_false_verifier():
    """Deterministic runs and false verifier."""
    task = TaskSpec("solve", outputs=("answer",), verifier="echo 0 > reward.txt")
    result = run_task(task, agent=AgentSpec(kind="oracle", solution="echo ok > answer"))
    assert result["files"] == {"answer": b"ok\n"}
    assert result["passed"] is False
    assert result["failure_kind"] is None
    assert result["trajectory"]["steps"][0]["kind"] == "oracle"
    failed = run_task(
        TaskSpec("solve"), agent=AgentSpec(kind="oracle", solution="exit 3")
    )
    assert failed["failure_kind"] == "execution"
    nop = run_task(TaskSpec("solve"), agent=AgentSpec(kind="nop"))
    assert nop["trajectory"]["steps"][0]["kind"] == "nop"


@pytest.mark.parametrize(
    "name,kind",
    [
        ("AgentTimeoutError", "budget"),
        ("AgentIterationLimitError", "budget"),
        ("AgentExecutionError", "execution"),
        ("AuthenticationError", "infrastructure"),
        ("RateLimitError", "infrastructure"),
        ("VerifierTimeoutError", "infrastructure"),
    ],
)
def test_native_failure_classification(name, kind):
    """Native failure classification."""
    assert (
        _failure_kind({"status": "failed", "exception": {"exception_type": name}})
        == kind
    )


def test_normalized_result_uses_explicit_trajectory():
    """Trajectory is an explicit backend field, never inferred from diagnostics."""
    trajectory = {"steps": [{"kind": "tool"}]}
    result = AgentRunResult("completed", passed=False, trajectory=trajectory)
    assert result.trajectory == trajectory
    assert (
        AgentRunResult("completed", diagnostics={"trajectory": trajectory}).trajectory
        is None
    )
    assert result.failure_kind is None
    assert AgentRunResult("timed_out").failure_kind == "budget"


@pytest.mark.parametrize(
    "script", ["exit 2", "echo 0 > reward.txt; exit 2", "echo 1 > reward.txt; exit 2"]
)
def test_broken_verifier_is_not_negative_evidence(script):
    """Verifier execution failures are infrastructure errors, not failed scores."""
    result = run_task(TaskSpec("solve", verifier=script), agent=AgentSpec(kind="nop"))
    assert result["status"] == "failed"
    assert result["passed"] is None
    assert result["failure_kind"] == "infrastructure"
    assert result["diagnostics"]["verifier_exit_code"] == 2
