"""Review regressions run real shells/workers behind offline service boundaries."""

import ast
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_sail_backend import LocalSailbox

import pyforger.harbor as harbor
import pyforger.sail_backend as sail_backend
from pyforger.agent_backend import AgentSpec, ProgramValidationError, WorkspaceTask
from pyforger.agent_diagnostics import trajectory
from pyforger.agents import generate_source
from pyforger.harbor_backend import HarborBackend
from pyforger.local_backend import LocalBackend
from pyforger.measurements import Measurement
from pyforger.sail_backend import SailBackend


@pytest.mark.parametrize("scheme", ["Basic", "basic", "BASIC", "Bearer"])
def test_authorization_redacts_entire_credential(scheme):
    """Authorization credentials are fully redacted for each scheme."""
    result = trajectory(
        [{"text": f'curl -H "Authorization: {scheme} dXNlcjpwYXNz"'}],
        AgentSpec(kind="nop"),
    )
    assert "dXNlcjpwYXNz" not in json.dumps(result)


@pytest.mark.parametrize("first", ["true", "printf '\\377' > program.py"])
def test_source_artifact_rejection_retries(tmp_path, first):
    """Rejected source artifacts are retried with a fresh generation."""
    marker = tmp_path / "calls"
    source = "def entry(item):\n    return ['ok']\n"
    solution = (
        f"echo call >> {marker}; "
        f"if test $(wc -l < {marker}) = 1; then {first}; "
        f"else printf %s {shlex.quote(source)} > program.py; fi"
    )
    assert (
        generate_source("test", {}, agent=AgentSpec(kind="oracle", solution=solution))
        == source
    )
    assert marker.read_text() == "call\ncall\n"


def test_source_backend_failure_not_retried(tmp_path):
    """Backend infrastructure failures are not retried as validation."""
    marker = tmp_path / "calls"
    with pytest.raises(RuntimeError):
        generate_source(
            "test",
            {},
            agent=AgentSpec(kind="oracle", solution=f"echo call >> {marker}; exit 9"),
        )
    assert marker.read_text() == "call\n"


def test_exited_shell_leader_children_are_killed(tmp_path):
    """Children of an exited shell leader are killed on timeout."""
    marker = tmp_path / "late"
    with pytest.raises(TimeoutError):
        LocalBackend().run_agent_validated(
            WorkspaceTask("write", timeout_s=0.2),
            AgentSpec(
                kind="oracle", solution=f"(sleep 0.7; echo orphan > {marker}) & exit 0"
            ),
            validate=lambda _: Measurement(passed=True),
        )
    time.sleep(0.8)
    assert not marker.exists()


def test_hostile_proc_name_cannot_evade_verifier_cleanup(tmp_path):
    """A hostile process name cannot evade verifier cleanup."""
    tree = ast.parse(Path(sail_backend.__file__).read_text())
    script = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "deadline=time.monotonic()+5" in node.value
    )
    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import ctypes,time; from pathlib import Path; "
            "ctypes.CDLL(None).prctl(15,b'worker) Z',0,0,0); "
            f"Path({str(ready)!r}).touch(); time.sleep(30)",
        ]
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        script = script.replace(
            "p.stat().st_uid != 65534", f"int(p.name) != {child.pid}"
        )
        subprocess.run([sys.executable, "-I", "-c", script], check=True, timeout=8)
        assert child.wait(timeout=0.5) == -9
    finally:
        child.kill()
        child.wait()


@pytest.fixture
def offline_sail(tmp_path, monkeypatch):
    """Offline Sail fixture using a local Sailbox substitute."""
    # Substitute only provisioning/UIDs and absolute VM paths. Commands, file
    # transport, validation, and verifier lifecycle are production code.
    class Box(LocalSailbox):
        sailbox_id = "offline-regression"

        def run(self, argv, **kwargs):
            if argv[0] == "bash":
                return SimpleNamespace(exit_code=0, timed_out=False)
            if argv[0] == "/usr/bin/env":
                argv = [part.replace("/work", str(self.root)) for part in argv]
                result = subprocess.run(argv, capture_output=True, text=True)
                return SimpleNamespace(
                    exit_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    timed_out=False,
                )
            if "-c" in argv and "deadline=time.monotonic()+5" in argv[-1]:
                # Never terminate unrelated host processes owned by nobody.
                argv = [
                    *argv[:-1],
                    argv[-1].replace("p.stat().st_uid != 65534", "True"),
                ]
            elif "-c" in argv and "/work/reward.txt" in argv[-1]:
                argv = [*argv[:-1], argv[-1].replace("/work", str(self.root))]
            return super().run(argv, **kwargs)

        def terminate(self):
            pass

    box = Box(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "sail",
        SimpleNamespace(
            Image=SimpleNamespace(from_registry=lambda _: object()),
            App=SimpleNamespace(find=lambda *a, **k: object()),
            Sailbox=SimpleNamespace(create=lambda **k: box),
        ),
    )
    return SailBackend()


def test_sail_oracle_feedback_is_literal_and_same_workspace(offline_sail, tmp_path):
    """Sail repair feedback is literal and stays in the same workspace."""
    reason = 'expected-content "quotes"\n$(touch injected); `touch injected2`'
    result = offline_sail.run_agent_validated(
        WorkspaceTask("write output", outputs=("x",)),
        AgentSpec(
            kind="oracle",
            solution=(
                'if test -f marker; then printf "%s" "$FORGER_FEEDBACK" > x; '
                "else touch marker; echo bad > x; fi"
            ),
        ),
        validate=lambda files: Measurement(
            passed=reason.encode() in files["x"], reason=reason
        ),
    )
    assert len(result.validation_attempts) == 2
    assert result.validation_attempts[0].output == {"x": b"bad\n"}
    assert not (tmp_path / "injected").exists()
    assert not (tmp_path / "injected2").exists()


@pytest.mark.parametrize("validated", [False, True])
def test_sail_verifier_outputs_and_accepted_snapshot(offline_sail, validated):
    """Sail collects verifier outputs and the accepted snapshot."""
    task = WorkspaceTask(
        "write", outputs=("x",), verifier="echo verified > x; echo 1 > reward.txt"
    )
    spec = AgentSpec(kind="oracle", solution="echo generated > x")
    if validated:
        result = offline_sail.run_agent_validated(
            task, spec, validate=lambda _: Measurement(passed=True)
        )
    else:
        result = offline_sail.run_agent(task, spec)
    assert result.passed is True
    assert result.files == {"x": b"generated\n" if validated else b"verified\n"}


@pytest.fixture
def offline_harbor(tmp_path, monkeypatch):
    """Offline Harbor fixture executing the replay worker locally."""
    # No Harbor service is required: execute the packaged replay worker in a
    # real subprocess, retaining the real request and collection protocol.
    def execute(task, scripts, output, **kwargs):
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        shutil.copytree(task.root / "environment/files", work, dirs_exist_ok=True)
        script = scripts.path("worker.py").read_text().replace("/work", str(work))
        run = subprocess.run(
            [sys.executable, "-I", "-c", script], capture_output=True, text=True
        )
        return SimpleNamespace(work=work, returncode=run.returncode)

    def collect(rollout, files, destination):
        for remote, local in files.items():
            (destination / local).write_bytes(
                (rollout.work / remote.removeprefix("/work/")).read_bytes()
            )

    monkeypatch.setattr(harbor, "execute_script", execute)
    monkeypatch.setattr(
        harbor,
        "trial_outcome",
        lambda run: {
            "status": "completed" if run.returncode == 0 else "failed",
            "passed": run.returncode == 0,
        },
    )
    monkeypatch.setattr(harbor, "collect_trial_files", collect)
    return HarborBackend()


@pytest.mark.parametrize(
    "code",
    [
        "def broken(:",
        "def entry(item): return 4",
        'def entry(item): raise RuntimeError("bad code")',
    ],
)
def test_harbor_program_failure_is_validation(offline_harbor, code):
    """Harbor program failures are classified as validation errors."""
    with pytest.raises(ProgramValidationError):
        offline_harbor.execute_program(code, {}, 2)


def test_harbor_valid_program_compatibility(offline_harbor):
    """Valid Harbor programs keep their compatibility behavior."""
    assert offline_harbor.execute_program("def entry(item): return ['ok']", {}, 2) == [
        "ok"
    ]


def test_harbor_infrastructure_failure_not_validation(offline_harbor, monkeypatch):
    """Harbor infrastructure failures are not validation errors."""
    monkeypatch.setattr(
        harbor, "trial_outcome", lambda _: {"status": "failed", "passed": False}
    )
    with pytest.raises(RuntimeError, match="replay failed"):
        offline_harbor.execute_program("def entry(item): return ['ok']", {}, 2)


@pytest.mark.parametrize("backend_name", ["local", "sail", "harbor"])
def test_validation_exception_contract(backend_name, offline_sail, offline_harbor):
    """Validation errors keep the ProgramValidationError contract."""
    backend = {"local": LocalBackend(), "sail": offline_sail, "harbor": offline_harbor}[
        backend_name
    ]
    with pytest.raises(ProgramValidationError) as caught:
        backend.execute_program("def broken(:", {}, 2)
    assert isinstance(caught.value, ValueError)


def test_harbor_source_retries_program_validation(offline_harbor, tmp_path):
    """Harbor source generation retries program validation failures."""

    class Backend(HarborBackend):
        def run_agent(self, task, agent):
            return LocalBackend().run_agent(task, agent)

    marker = tmp_path / "calls"
    source = "def entry(item): return ['ok']"
    spec = AgentSpec(
        kind="oracle",
        solution=(
            f"echo call >> {marker}; if test $(wc -l < {marker}) = 1; "
            'then echo "def broken(:" > program.py; '
            f"else printf %s {shlex.quote(source)} > program.py; fi"
        ),
    )
    assert generate_source("test", {}, agent=spec, backend=Backend()) == source
    assert marker.read_text() == "call\ncall\n"


@pytest.mark.parametrize("first", ["true", "printf '\\377' > program.py"])
def test_invalid_source_exhaustion_is_bounded(tmp_path, first):
    """Invalid source exhaustion is bounded by the attempt limit."""
    marker = tmp_path / "calls"
    with pytest.raises(RuntimeError, match="did not produce a working program"):
        generate_source(
            "test",
            {},
            attempts=2,
            agent=AgentSpec(kind="oracle", solution=f"echo call >> {marker}; {first}"),
        )
    assert marker.read_text() == "call\ncall\n"


def test_sail_collects_new_verifier_report_and_reward(offline_sail):
    """Sail collects new verifier report and reward files."""
    result = offline_sail.run_agent(
        WorkspaceTask(
            "write",
            outputs=("report", "reward.txt"),
            verifier="echo report > report; echo 1 > reward.txt",
        ),
        AgentSpec(kind="nop"),
    )
    assert result.files == {"report": b"report\n", "reward.txt": b"1\n"}


def test_harbor_replay_infrastructure_failure_aborts_generation(
    offline_harbor, monkeypatch, tmp_path
):
    """Harbor replay infrastructure failure aborts generation."""

    class Backend(HarborBackend):
        def run_agent(self, task, agent):
            return LocalBackend().run_agent(task, agent)

    monkeypatch.setattr(
        harbor, "trial_outcome", lambda _: {"status": "failed", "passed": False}
    )
    marker = tmp_path / "calls"
    with pytest.raises(RuntimeError, match="replay failed"):
        generate_source(
            "test",
            {},
            backend=Backend(),
            agent=AgentSpec(
                kind="oracle",
                solution=(
                    f"echo call >> {marker}; "
                    'echo "def entry(item): return []" > program.py'
                ),
            ),
        )
    assert marker.read_text() == "call\n"
