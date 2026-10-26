"""Recorded execution uses real local task and verifier paths."""

import json
from types import SimpleNamespace

import pytest

from pyforger.acquire.model import Model
from pyforger.agent_backend import AgentSpec
from pyforger.agent_tasks import TaskSpec
from pyforger.local_backend import LocalBackend
from pyforger.task_runs import run_controls, run_recorded_task


def test_recorded_nested_artifacts_and_fresh_paths(tmp_path):
    """Retain only declared nested files and refuse reused attempt roots."""
    task = TaskSpec(
        "write answer",
        outputs=("nested/answer.txt", "binary"),
        solution_files={"binary": b"\x00\xff"},
        verifier="test -f nested/answer.txt && echo 1 > reward.txt",
    )
    root = tmp_path / "new/attempt"
    report = run_recorded_task(
        task,
        agent=AgentSpec(
            kind="oracle", solution="mkdir nested; echo yes > nested/answer.txt"
        ),
        backend=LocalBackend(),
        output=root,
    )
    assert report["passed"] is True
    assert report["status"] == "completed"
    assert report["failure_kind"] is None
    assert (root / "nested/answer.txt").read_text() == "yes\n"
    assert (root / "binary").read_bytes() == b"\x00\xff"
    assert not (root / "reward.txt").exists()
    assert json.loads((root / "run-report.json").read_text()) == report
    assert (
        json.loads((root / "agent/trajectory.json").read_text())["steps"][0]["kind"]
        == "oracle"
    )
    with pytest.raises(FileExistsError):
        run_recorded_task(task, agent=AgentSpec(kind="nop"), output=root)


def test_controls_and_negative_verification(tmp_path):
    """Nop verification failure is scored false, never an infrastructure pass."""
    task = TaskSpec(
        "create answer",
        verifier="if test -f answer; then echo 1; else echo 0; fi > reward.txt",
    )
    reports = run_controls(
        task,
        {
            "oracle": AgentSpec(kind="oracle", solution="touch answer"),
            "nop": AgentSpec(kind="nop"),
        },
        backend=LocalBackend(),
        output=tmp_path / "controls",
        concurrency=2,
    )
    assert reports["oracle"]["passed"] is True
    assert reports["nop"]["passed"] is False
    assert reports["nop"]["failure_kind"] is None
    assert reports["oracle"]["trial_dir"] != reports["nop"]["trial_dir"]


def test_missing_outputs_are_unscored(tmp_path):
    """A verifier pass cannot mask a missing declared artifact."""
    result = run_recorded_task(
        TaskSpec("no answer", outputs=("missing",), verifier="echo 1 > reward.txt"),
        agent=AgentSpec(kind="nop"),
        backend=LocalBackend(),
        output=tmp_path / "missing",
    )
    assert result["status"] == "infrastructure_error"
    assert result["passed"] is None
    assert result["failure_kind"] == "infrastructure"
    assert "Missing declared outputs" in result["error"]
    assert result["diagnostics"]["verifier_exit_code"] == 0


@pytest.mark.parametrize(
    "solution,timeout,kind", [("exit 7", 10, "execution"), ("sleep 2", 0.05, "budget")]
)
def test_deterministic_failures_keep_attribution(tmp_path, solution, timeout, kind):
    """Execution and budget failures remain distinct from infrastructure errors."""
    result = run_recorded_task(
        TaskSpec("fail", outputs=("missing",)),
        agent=AgentSpec(kind="oracle", solution=solution),
        backend=LocalBackend(),
        output=tmp_path / kind,
        timeout_s=timeout,
    )
    assert result["failure_kind"] == kind
    assert result["passed"] is None


@pytest.mark.parametrize("malformed", [False, True])
def test_sdk_boundary_errors_recorded_and_redacted(tmp_path, monkeypatch, malformed):
    """Only the external SDK session is replaced for provider/artifact failures."""
    secret = "example-private-credential-123"
    monkeypatch.setenv("TEST_PROVIDER_TOKEN", secret)

    class Session:
        state = SimpleNamespace(execution_status="finished", events=[object()])

        def send_message(self, message):
            pass

        async def arun(self):
            if not malformed:
                raise RuntimeError(secret)

        def close(self):
            pass

    monkeypatch.setattr(LocalBackend, "_sdk_session", lambda *a, **kw: Session())
    root = tmp_path / "provider"
    report = run_recorded_task(
        TaskSpec("run provider"),
        agent=AgentSpec(Model("openai/test")),
        backend=LocalBackend(),
        output=root,
    )
    assert report["failure_kind"] == "infrastructure"
    assert report["status"] == "infrastructure_error"
    assert report["passed"] is None
    assert ("AttributeError" if malformed else "[REDACTED]") in report["error"]
    assert secret not in (root / "run-report.json").read_text()


def test_text_evidence_redacted(tmp_path, monkeypatch):
    """Declared text and diagnostics use generic environment credential redaction."""
    secret = "example-token-123456"
    monkeypatch.setenv("TEST_SECRET", secret)
    report = run_recorded_task(
        TaskSpec("retain", files={"answer": secret}, outputs=("answer",)),
        agent=AgentSpec(kind="oracle", solution="cat answer"),
        backend=LocalBackend(),
        output=tmp_path / "redacted",
    )
    assert report["diagnostics"]["stdout"] == "[REDACTED]"
    assert report["passed"] is None
    assert report["status"] == "artifact_safety_error"
    assert report["failure_kind"] == "artifact_safety"
    assert "Credential detected" in report["error"]
    assert not (tmp_path / "redacted/answer").exists()


@pytest.mark.parametrize(
    "name",
    [
        "run-report.json",
        "run-report.json/child",
        "agent",
        "agent/trajectory.json",
        "agent/trajectory.json/child",
    ],
)
def test_metadata_collisions_rejected(tmp_path, name):
    """Output metadata collisions fail before an attempt directory is made."""
    with pytest.raises(ValueError, match="collide"):
        run_recorded_task(
            TaskSpec("collision", outputs=(name,)),
            agent=AgentSpec(kind="nop"),
            output=tmp_path / "attempt",
        )
    assert not (tmp_path / "attempt").exists()


def test_duplicate_and_prefix_outputs_rejected(tmp_path):
    """Declared files cannot overwrite each other or become parent directories."""
    for outputs in [("x", "x"), ("x", "x/y")]:
        with pytest.raises(ValueError, match="collide"):
            run_recorded_task(
                TaskSpec("collision", outputs=outputs),
                agent=AgentSpec(kind="nop"),
                output=tmp_path / "attempt",
            )


def test_unsafe_paths_and_controls_rejected(tmp_path):
    """Reject traversal and symlinked evidence roots without touching targets."""
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_recorded_task(
            TaskSpec("safe"), agent=AgentSpec(kind="nop"), output=tmp_path / "link"
        )
    for name in ["../escape", "a/b", ".", "", "agent.json"]:
        with pytest.raises(ValueError, match="safe IDs"):
            run_controls(
                TaskSpec("safe"),
                {name: AgentSpec(kind="nop")},
                backend=LocalBackend(),
                output=tmp_path / "controls",
            )
    for name in ["../escape", "/absolute", "a/../b"]:
        with pytest.raises(ValueError):
            TaskSpec("unsafe", outputs=(name,))
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_invalid_control_limits(tmp_path, concurrency):
    """Control executor limits are validated before creating any output."""
    with pytest.raises(ValueError):
        run_controls(
            TaskSpec("safe"),
            {},
            backend=LocalBackend(),
            output=tmp_path / "controls",
            concurrency=concurrency,
        )
    assert not (tmp_path / "controls").exists()


def test_nop_missing_submissions_stay_scored_negative(tmp_path):
    """Missing Nop answer and analysis remain verifier failures through controls."""
    reports = run_controls(
        TaskSpec(
            "submit answer and analysis",
            outputs=("answer.txt", "analysis.txt"),
            verifier=(
                "if test -f answer.txt && test -f analysis.txt; "
                "then echo 1; else echo 0; fi > reward.txt"
            ),
        ),
        {
            "oracle": AgentSpec(
                kind="oracle", solution="touch answer.txt analysis.txt"
            ),
            "nop": AgentSpec(kind="nop"),
        },
        backend=LocalBackend(),
        output=tmp_path / "missing-submissions",
    )
    assert reports["oracle"]["passed"] is True
    assert reports["nop"]["status"] == "completed"
    assert reports["nop"]["passed"] is False
    assert reports["nop"]["failure_kind"] is None
    assert not (tmp_path / "missing-submissions/nop/answer.txt").exists()


def test_ancestor_alias_and_exact_source_bytes(tmp_path):
    """Normalize trusted parent aliases while preserving retained source bytes."""
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    source = b"# source with unicode: \xce\xbb\r\nprint('answer')\r\n"
    report = run_recorded_task(
        TaskSpec(
            "retain source",
            files={"nested/program.py": source},
            outputs=("nested/program.py",),
        ),
        agent=AgentSpec(kind="nop"),
        backend=LocalBackend(),
        output=alias / "attempt",
    )
    assert report["status"] == "completed"
    assert report["trial_dir"] == str(real / "attempt")
    assert (real / "attempt/nested/program.py").read_bytes() == source


def test_native_sdk_evidence_structure(tmp_path, monkeypatch):
    """External SDK event transport retains nested native fields without flattening."""
    event = {"kind": "MessageEvent", "llm_message": {"content": [{"text": "native"}]}}

    class Session:
        state = SimpleNamespace(execution_status="finished", events=[event])

        def send_message(self, message):
            pass

        async def arun(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(LocalBackend, "_sdk_session", lambda *a, **kw: Session())
    report = run_recorded_task(
        TaskSpec("retain native evidence"),
        agent=AgentSpec(Model("openai/test")),
        backend=LocalBackend(),
        output=tmp_path / "native",
    )
    trajectory = json.loads((tmp_path / "native/agent/trajectory.json").read_text())
    from pyforger.evidence_journal import iter_trajectory_events

    assert list(iter_trajectory_events(trajectory, root=tmp_path / "native")) == [event]
    assert report["diagnostics"]["trajectory"] == trajectory


@pytest.mark.parametrize(
    "content", [b"token_count = 42\nsecret = 'ordinary science'\n", b"\xff\x00binary"]
)
def test_benign_artifact_assignments_are_exact(tmp_path, content):
    report = run_recorded_task(
        TaskSpec("retain", files={"answer": content}, outputs=("answer",)),
        agent=AgentSpec(kind="nop"),
        backend=LocalBackend(),
        output=tmp_path / "run",
    )
    assert report["status"] == "completed"
    assert (tmp_path / "run/answer").read_bytes() == content


@pytest.mark.parametrize("binary", [False, True])
def test_all_outputs_checked_before_any_publication(tmp_path, monkeypatch, binary):
    secret = "synthetic-known-credential-838383"
    monkeypatch.setenv("ARTIFACT_TEST_SECRET", secret)
    content = secret.encode()
    if binary:
        content = b"\xff\x00" + content + b"\x00\xfe"
    report = run_recorded_task(
        TaskSpec(
            "retain",
            files={"first": b"safe", "last": content},
            outputs=("first", "last"),
        ),
        agent=AgentSpec(kind="oracle", solution="echo oracle-once"),
        backend=LocalBackend(),
        output=tmp_path / "run",
    )
    assert report["failure_kind"] == "artifact_safety"
    assert report["passed"] is None
    assert not (tmp_path / "run/first").exists()
    assert not (tmp_path / "run/last").exists()
    assert report["diagnostics"]["stdout"] == "oracle-once\n"
    assert secret not in (tmp_path / "run/run-report.json").read_text()
    assert secret not in (tmp_path / "run/agent/trajectory.json").read_text()


@pytest.mark.parametrize("repair", [True, False])
def test_sdk_artifact_repair_in_existing_loop(tmp_path, monkeypatch, repair):
    """Replace only provider transport; use real host loop, collection and verifier."""
    secret = "synthetic-sdk-credential-939393"
    monkeypatch.setenv("ARTIFACT_TEST_SECRET", secret)
    messages = []

    class Session:
        state = SimpleNamespace(execution_status="finished", events=[])

        def __init__(self, root):
            self.root = root
            self.turn = 0

        def send_message(self, message):
            messages.append(message)
            assert secret not in message

        async def arun(self):
            self.turn += 1
            (self.root / "answer").write_bytes(
                b"safe" if repair and self.turn > 1 else b"\xff" + secret.encode()
            )

        def close(self):
            pass

    monkeypatch.setattr(
        LocalBackend, "_sdk_session", lambda self, root, *a, **kw: Session(root)
    )
    report = run_recorded_task(
        TaskSpec("produce", outputs=("answer",), verifier="echo 1 > reward.txt"),
        agent=AgentSpec(Model("openai/test")),
        backend=LocalBackend(),
        output=tmp_path / "run",
    )
    assert len(messages) == (2 if repair else 3)
    assert secret not in json.dumps(report)
    if repair:
        assert report["status"] == "completed"
        assert report["passed"] is True
        assert (tmp_path / "run/answer").read_bytes() == b"safe"
    else:
        assert report["failure_kind"] == "artifact_safety"
        assert "exhausted" in report["error"]
        assert report["passed"] is None
        assert not (tmp_path / "run/answer").exists()


def test_configured_non_environment_credential_is_blocked(tmp_path):
    secret = "synthetic-configured-credential-737373"
    report = run_recorded_task(
        TaskSpec(
            "retain", files={"answer": b"\xff" + secret.encode()}, outputs=("answer",)
        ),
        agent=AgentSpec(Model("openai/test", api_key=secret), kind="nop"),
        backend=LocalBackend(),
        output=tmp_path / "run",
    )
    assert report["failure_kind"] == "artifact_safety"
    assert secret not in json.dumps(report)
    assert not (tmp_path / "run/answer").exists()


@pytest.mark.parametrize("leak", [False, True])
def test_recorded_validated_task_collects_post_verifier_bytes(
    tmp_path, monkeypatch, leak
):
    secret = "synthetic-verifier-credential-838383"
    monkeypatch.setenv("VERIFIER_TEST_SECRET", secret)

    class Session:
        state = SimpleNamespace(execution_status="finished", events=[])

        def __init__(self, root):
            self.root = root

        def send_message(self, message):
            pass

        async def arun(self):
            (self.root / "answer").write_text("before")

        def close(self):
            pass

    monkeypatch.setattr(
        LocalBackend, "_sdk_session", lambda self, root, *a, **kw: Session(root)
    )
    final = secret if leak else "after"
    report = run_recorded_task(
        TaskSpec(
            "produce",
            outputs=("answer",),
            verifier=f"printf '%s' '{final}' > answer; echo 1 > reward.txt",
        ),
        agent=AgentSpec(Model("openai/test")),
        backend=LocalBackend(),
        output=tmp_path / "run",
    )
    if leak:
        assert report["failure_kind"] == "artifact_safety"
        assert report["passed"] is None
        assert not (tmp_path / "run/answer").exists()
    else:
        assert report["passed"] is True
        assert (tmp_path / "run/answer").read_text() == "after"
    assert secret not in json.dumps(report)


@pytest.mark.parametrize("created", [False, True])
def test_sdk_missing_submission_reaches_verifier_without_repair(
    tmp_path, monkeypatch, created
):
    turns = []

    class Session:
        state = SimpleNamespace(execution_status="finished", events=[])

        def send_message(self, message):
            turns.append(message)

        async def arun(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(LocalBackend, "_sdk_session", lambda *a, **kw: Session())
    report = run_recorded_task(
        TaskSpec(
            "produce",
            outputs=("answer",),
            verifier="echo verified > answer; echo 1 > reward.txt"
            if created
            else "echo 0 > reward.txt",
        ),
        agent=AgentSpec(Model("openai/test")),
        backend=LocalBackend(),
        output=tmp_path / "run",
    )
    assert len(turns) == 1
    assert report["status"] == "completed"
    assert report["passed"] is created
    if created:
        assert (tmp_path / "run/answer").read_text() == "verified\n"
    else:
        assert not (tmp_path / "run/answer").exists()
