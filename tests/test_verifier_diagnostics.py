"""Trusted packaged verifier subprocesses in private temporary root fixtures."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import RewardFileNotFoundError, Verifier

from pyforger.agent_backend import AgentSpec, WorkspaceTask
from pyforger.artifacts import ArtifactRef
from pyforger.harbor import HarborResult, _scrub, trial_outcome
from pyforger.harbor_backend import _failure_kind, _package


@pytest.mark.parametrize(
    "command,code,passed,error",
    [
        (
            "echo 0 > /work/reward.txt; echo causal >&2; exit 7",
            7,
            None,
            "VerifierExecutionError",
        ),
        ("echo causal >&2; exit 9", 9, None, "VerifierExecutionError"),
        (
            "echo 1 > /logs/verifier/reward.txt; exit 4",
            4,
            None,
            "VerifierExecutionError",
        ),
        ("echo 0 > /work/reward.txt", 0, False, None),
        ("echo 1 > /work/reward.txt", 0, True, None),
        (
            "echo bad-json > /logs/verifier/reward.json",
            0,
            None,
            "VerifierRewardProtocolError",
        ),
        ("echo NaN > /work/reward.txt", 0, None, "VerifierRewardProtocolError"),
        ("true", 0, None, "VerifierRewardProtocolError"),
        ("echo nonsense > /work/reward.txt", 0, None, "VerifierRewardProtocolError"),
    ],
)
def test_packaged_verifier_execution(tmp_path, command, code, passed, error):
    """Never interpret a provisional grade after a nonzero verifier exit."""
    artifact = _package(
        WorkspaceTask(
            instruction="fixture",
            verifier="echo fixture-secret; echo fixture-secret >&2; " + command,
        ),
        AgentSpec(kind="nop"),
        tmp_path / "task",
        [],
    )
    work = tmp_path / "work"
    logs = tmp_path / "verifier"
    work.mkdir()
    logs.mkdir()
    tests = artifact.root / "tests"
    # Relocate only fixed container roots; execute no generated learner code.
    for name in ("launch.py", "verifier-command.sh"):
        path = tests / name
        source = path.read_text()
        for old, new in (
            ("/logs/verifier", str(logs)),
            ("/tests", str(tests)),
            ("/work", str(work)),
        ):
            source = source.replace(old, new)
        path.write_text(source)
    process = subprocess.run(
        [sys.executable, "-I", str(tests / "launch.py")],
        capture_output=True,
        check=False,
    )
    assert process.returncode == code
    record = json.loads((logs / "execution.json").read_text())
    assert record["exit_code"] == code
    if code:
        assert not (logs / "reward.txt").exists()
        assert not (logs / "reward.json").exists()
    _scrub(logs, {"fixture-secret"})
    assert "fixture-secret" not in (logs / "stdout.txt").read_text()
    assert "fixture-secret" not in (logs / "stderr.txt").read_text()
    if code:
        assert "causal" in (logs / "stderr.txt").read_text() or code == 4
    (tmp_path / "result.json").write_text("{}")
    verifier = Verifier(task=None, trial_paths=TrialPaths(tmp_path), environment=None)
    try:
        if not (logs / "reward.txt").exists() and not (logs / "reward.json").exists():
            raise RewardFileNotFoundError("No reward after verifier execution")
        rewards = (
            verifier._parse_reward_json()
            if (logs / "reward.json").exists()
            else verifier._parse_reward_text()
        )
        native = {"verifier_result": {"rewards": rewards}}
    except Exception as exc:
        native = {
            "exception_info": {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        }
    rollout = HarborResult(
        status="completed" if passed is not None else "failed",
        artifacts=ArtifactRef.capture(tmp_path),
        metadata={"harbor_result_path": "result.json", "harbor_result": native},
    )
    outcome = trial_outcome(rollout)
    assert outcome["passed"] is passed
    assert outcome["verifier_execution"]["exit_code"] == code
    if error:
        assert outcome["status"] == "failed"
        assert outcome["exception"]["exception_type"] == error
        assert _failure_kind(outcome) == "infrastructure"
        assert outcome["native_exception"] == native["exception_info"]
    else:
        assert outcome["status"] == "completed"
        assert _failure_kind(outcome) is None
    assert os.stat(logs / "execution.json").st_mode & 0o077 == 0


@pytest.mark.parametrize(
    "record", [{"exit_code": None}, {"exit_code": "0"}, [], {"exit_code": 5}]
)
def test_unfinished_or_invalid_execution_never_uses_native_reward(tmp_path, record):
    """Fail closed even if Harbor already parsed a provisional native reward."""
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier/execution.json").write_text(json.dumps(record))
    (tmp_path / "result.json").write_text("{}")
    rollout = HarborResult(
        status="completed",
        artifacts=ArtifactRef.capture(tmp_path),
        metadata={
            "harbor_result_path": "result.json",
            "harbor_result": {"verifier_result": {"rewards": {"reward": 0}}},
        },
    )
    outcome = trial_outcome(rollout)
    assert outcome["status"] == "failed"
    assert outcome["passed"] is None
    assert outcome["exception"]["exception_type"] == "VerifierExecutionError"
