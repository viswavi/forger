"""Real transport file code and opt-in direct Sail lifecycle tests."""

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import pyforger.sail_transport as sail_transport
from pyforger.agent_backend import AgentBackend, AgentSpec, Network, WorkspaceTask
from pyforger.agent_session import SessionBudget
from pyforger.sail_backend import SailBackend
from pyforger.sail_transport import (
    _FILE_IO,
    SailTransport,
    egress_policy,
    workspace_path,
)


def test_backend_protocol():
    """Backend protocol."""
    assert isinstance(SailBackend(), AgentBackend)
    assert SailBackend().cache_key() != SailBackend(image="other").cache_key()


@pytest.mark.parametrize(
    "name", ["", "/tmp/x", "../x", "a/../x", "a//x", "a/./x", "x/", "a\x00b"]
)
def test_unsafe_paths(name):
    """Unsafe paths."""
    with pytest.raises(ValueError):
        workspace_path(name)


def test_egress():
    """Egress."""
    assert egress_policy(Network.MODEL) == {"no_network": True}
    assert egress_policy(Network.OFFLINE) == {"no_network": True}
    assert egress_policy(Network.PUBLIC) == {}
    with pytest.raises(ValueError):
        egress_policy("public")


def transfer(tmp_path, files, write=False, limit=100, request_file=False):
    """Transfer."""
    # Run the actual remote script locally, only substituting the disposable
    # root and ownership operation (tests need no root privileges).
    script = _FILE_IO.replace("'/work'", repr(str(tmp_path)))
    script = script.replace("os.fchown(stream.fileno(), 65534, 65534)", "pass")
    script = script.replace("os.fchown(child, 65534, 65534)", "pass")
    request = json.dumps({"files": files, "write": write, "limit": limit})
    if request_file:
        spool = tmp_path.parent / (tmp_path.name + "-request.json")
        spool.write_text(request)
        arguments = ["--request-file", str(spool)]
    else:
        arguments = [request]
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            *arguments,
        ],
        capture_output=True,
        text=True,
    )


def test_file_roundtrip(tmp_path):
    """File roundtrip."""
    value = base64.b64encode(b"\x00hello\xff").decode()
    assert transfer(tmp_path, {"nested/a": value}, True).returncode == 0
    result = transfer(tmp_path, {"nested/a": "", "missing": ""})
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"nested/a": value}


@pytest.mark.parametrize("kind", ["leaf", "directory", "hardlink", "fifo"])
def test_collection_rejects_nonregular_or_escaping_files(tmp_path, kind):
    """Collection rejects nonregular or escaping files."""
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "secret").write_text("hidden")
    name = "output"
    if kind == "leaf":
        (tmp_path / name).symlink_to(outside / "secret")
    elif kind == "directory":
        (tmp_path / name).symlink_to(outside, target_is_directory=True)
        name += "/secret"
    elif kind == "hardlink":
        os.link(outside / "secret", tmp_path / name)
    else:
        os.mkfifo(tmp_path / name)
    result = transfer(tmp_path, {name: ""})
    assert result.returncode != 0
    assert "hidden" not in result.stdout


def test_staging_refuses_symlink_and_overwrite(tmp_path):
    """Staging refuses symlink and overwrite."""
    (tmp_path / "verifier").symlink_to(tmp_path.parent, target_is_directory=True)
    assert transfer(tmp_path, {"verifier/secret": "YQ=="}, True).returncode != 0
    (tmp_path / "existing").write_text("preserve")
    assert transfer(tmp_path, {"existing": "YQ=="}, True).returncode != 0
    assert (tmp_path / "existing").read_text() == "preserve"


def test_oversize_output(tmp_path):
    """Oversize output."""
    (tmp_path / "large").write_bytes(b"x" * 101)
    assert transfer(tmp_path, {"large": ""}).returncode != 0


live = pytest.mark.skipif(
    os.environ.get("FORGER_SAIL_LIVE") != "1",
    reason="explicit FORGER_SAIL_LIVE=1 required",
)


@live
def test_live_oracle_verifier_and_offline_replay():
    """Live oracle verifier and offline replay."""
    backend = SailBackend()
    result = backend.run_agent(
        WorkspaceTask(
            instruction="copy",
            files={"input": "hello"},
            outputs=("output",),
            network=Network.MODEL,
            timeout_s=180,
            verifier_files={"secret": "hidden"},
            verifier="test $(cat verifier/secret) = hidden && echo 1 > reward.txt",
        ),
        AgentSpec(
            kind="oracle", solution="test ! -e verifier/secret && cp input output"
        ),
    )
    assert result.files == {"output": b"hello"}
    assert result.passed is True
    assert backend.execute_program(
        "def run(item): return [item['text'].upper()]", {"text": "abc"}, 180
    ) == ["ABC"]


@live
def test_live_nop():
    """Live nop."""
    result = SailBackend().run_agent(
        WorkspaceTask(
            instruction="nothing", files={"x": "y"}, outputs=("x",), timeout_s=180
        ),
        AgentSpec(kind="nop"),
    )
    assert result.files == {"x": b"y"}


@live
def test_live_packages_built_before_offline_run():
    """Live packages built before offline run."""
    result = SailBackend().run_agent(
        WorkspaceTask(
            instruction="Use prebuilt package offline",
            packages=("packaging==25.0",),
            outputs=("version",),
            network=Network.OFFLINE,
            timeout_s=240,
        ),
        AgentSpec(
            kind="oracle",
            solution=(
                "python3 -c 'import packaging; "
                'open("version", "w").write(packaging.__version__)'
                "'"
            ),
            kwargs={"load_skills": False},
        ),
    )
    assert result.files["version"] == b"25.0"
    assert result.diagnostics["cleanup"] == "terminated"


def test_command_observation_roundtrips_sdk_schema():
    """Command observation roundtrips sdk schema."""
    from openhands.sdk.tool import Observation

    from pyforger.sail_transport import SailCommandObservation

    value = SailCommandObservation.from_text("exit_code=0")
    assert isinstance(
        Observation.model_validate(value.model_dump()), SailCommandObservation
    )


def test_large_file_request_roundtrip(tmp_path):
    """Transfer binary payloads beyond Linux's per-argument size limit."""
    data = bytes(range(256)) * 2048
    value = base64.b64encode(data).decode()
    result = transfer(
        tmp_path, {"large.bin": value}, True, limit=len(data), request_file=True
    )
    assert result.returncode == 0, result.stderr
    result = transfer(tmp_path, {"large.bin": ""}, limit=len(data), request_file=True)
    assert result.returncode == 0, result.stderr
    assert base64.b64decode(json.loads(result.stdout)["large.bin"]) == data


@live
def test_live_large_file_roundtrip():
    """Stage and collect multiple binary files larger than 128 KiB remotely."""
    data = bytes(range(256)) * 2048
    result = SailBackend().run_agent(
        WorkspaceTask(
            instruction="Copy large binary input",
            timeout_s=180,
            files={"input.bin": data, "second.bin": data},
            outputs=("output.bin", "second.bin"),
            network=Network.OFFLINE,
        ),
        AgentSpec(kind="oracle", solution="cp input.bin output.bin"),
    )
    assert result.files == {"output.bin": data, "second.bin": data}
    assert result.diagnostics["cleanup"] == "terminated"


@live
@pytest.mark.parametrize("exit_code,passed", [(0, False), (2, None)])
def test_live_verifier_execution_failure_is_not_a_score(exit_code, passed):
    """Sail distinguishes a successful negative grade from broken verification."""
    result = SailBackend().run_agent(
        WorkspaceTask(
            instruction="nothing",
            timeout_s=180,
            verifier=f"echo 0 > reward.txt; exit {exit_code}",
        ),
        AgentSpec(kind="nop"),
    )
    assert result.passed is passed
    assert result.status == ("completed" if exit_code == 0 else "failed")
    assert result.failure_kind == (None if exit_code == 0 else "infrastructure")
    assert result.diagnostics["verifier_exit_code"] == exit_code


class LocalSailbox:
    """Execute real transport scripts with only the remote boundary substituted."""

    def __init__(self, root):
        """Use a disposable local workspace instead of a remote microVM."""
        self.root = root
        self.fs = self
        self.requests = 0

    def write(self, path, source, **kwargs):
        """Consume the streaming request as Sail's filesystem API does."""
        Path(path).write_bytes(source.read())

    def run(self, argv, **kwargs):
        """Run the actual helper with local ownership and workspace substitution."""
        self.requests += 1
        argv = list(argv)
        if _FILE_IO in argv:
            script = _FILE_IO.replace("'/work'", repr(str(self.root)))
            script = script.replace("os.fchown(stream.fileno(), 65534, 65534)", "pass")
            script = script.replace("os.fchown(child, 65534, 65534)", "pass")
            argv[argv.index(_FILE_IO)] = script
        argv[0] = sys.executable
        result = subprocess.run(argv, capture_output=True, text=True)
        return SimpleNamespace(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            stdout_truncated=False,
            timed_out=False,
        )


def local_transport(tmp_path):
    """Bind the production transport to real local subprocesses for offline tests."""
    return SailTransport(LocalSailbox(tmp_path), SessionBudget(10, 120))


def test_transport_stages_original_document_size(tmp_path):
    """Stage the failed run's payload size without changing output limits."""
    data = b"x" * 10_412_539
    transport = local_transport(tmp_path)
    assert transport.files({"raw-data/s1.docx": data}, write=True) == {}
    assert hashlib.sha256((tmp_path / "raw-data/s1.docx").read_bytes()).digest() == (
        hashlib.sha256(data).digest()
    )


def test_transport_input_bounds_preflight_and_across_calls(tmp_path, monkeypatch):
    """Reject oversized batches before remote work and bound cumulative staging."""
    monkeypatch.setattr(sail_transport, "MAX_INPUT_FILE_BYTES", 10)
    monkeypatch.setattr(sail_transport, "MAX_INPUT_BYTES", 16)
    transport = local_transport(tmp_path)
    with pytest.raises(ValueError, match="Input exceeds"):
        transport.files({"large": b"x" * 11}, write=True)
    with pytest.raises(ValueError, match="Input.*total"):
        transport.files({"a": b"x" * 9, "b": b"x" * 8}, write=True)
    assert transport.box.requests == 0
    transport.files({"a": b"x" * 10}, write=True)
    transport.files({"b": "é" * 3}, write=True)
    requests = transport.box.requests
    with pytest.raises(ValueError, match="Input.*total"):
        transport.files({"c": b"x"}, write=True)
    assert transport.box.requests == requests


def test_transport_output_bounds_and_repeated_collection(tmp_path, monkeypatch):
    """Bound each collection without charging unchanged snapshots repeatedly."""
    monkeypatch.setattr(sail_transport, "MAX_OUTPUT_FILE_BYTES", 10)
    monkeypatch.setattr(sail_transport, "MAX_OUTPUT_BYTES", 16)
    transport = local_transport(tmp_path)
    for name, size in (("a", 10), ("b", 6), ("c", 1), ("large", 11)):
        (tmp_path / name).write_bytes(b"x" * size)
    for _ in range(2):
        assert transport.files({"a": b"", "b": b""}, write=False) == {
            "a": b"x" * 10,
            "b": b"x" * 6,
        }
    with pytest.raises(RuntimeError, match="Output exceeds size limit"):
        transport.files({"large": b""}, write=False)
    with pytest.raises(RuntimeError, match="Output exceeds size limit"):
        transport.files({"a": b"", "b": b"", "c": b""}, write=False)
    assert transport.files({"missing": b"", "a": b""}, write=False) == {"a": b"x" * 10}


@pytest.mark.parametrize("size", [32 * 1024 * 1024, 32 * 1024 * 1024 + 1])
def test_transport_production_upload_boundary(tmp_path, size):
    """Accept the public-download ceiling exactly and reject its first extra byte."""
    transport = local_transport(tmp_path)
    data = b"x" * size
    if size > sail_transport.MAX_INPUT_FILE_BYTES:
        with pytest.raises(ValueError, match="Input exceeds size limit"):
            transport.files({"input": data}, write=True)
        assert transport.box.requests == 0
    else:
        transport.files({"input": data}, write=True)
        assert (tmp_path / "input").read_bytes() == data


def test_transport_production_output_limit_unchanged(tmp_path):
    """Do not allow the larger staging budget to expand collected artifacts."""
    (tmp_path / "output").write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    with pytest.raises(RuntimeError, match="Output exceeds size limit"):
        local_transport(tmp_path).files({"output": b""}, write=False)


def test_transport_failed_batch_keeps_staging_reservation(tmp_path, monkeypatch):
    """A partially staged failed batch must not reset the session byte budget."""
    monkeypatch.setattr(sail_transport, "MAX_INPUT_BYTES", 4)
    (tmp_path / "existing").write_bytes(b"old")
    transport = local_transport(tmp_path)
    with pytest.raises(RuntimeError, match="failed Sail file transfer"):
        transport.files({"new": b"ab", "existing": b"cd"}, write=True)
    assert (tmp_path / "new").read_bytes() == b"ab"
    assert (tmp_path / "existing").read_bytes() == b"old"
    requests = transport.box.requests
    with pytest.raises(ValueError, match="Input exceeds total size limit"):
        transport.files({"another": b"x"}, write=True)
    assert transport.box.requests == requests


@pytest.mark.parametrize("validated", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize(
    "outcome", ["limit", "provider", "raised", "timeout", "finished", "rejected"]
)
def test_sdk_failure_evidence(tmp_path, monkeypatch, validated, outcome, cleanup_fails):
    """Exercise backend orchestration, faking only external Sail/SDK boundaries."""
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation
    from openhands.sdk.event.conversation_error import ConversationErrorEvent

    from pyforger.acquire.model import Model
    from pyforger.measurements import Measurement

    class Box(LocalSailbox):
        sailbox_id = "offline-box"
        terminated = False

        def run(self, argv, **kwargs):
            if argv[0] == "bash":
                return SimpleNamespace(exit_code=0, timed_out=False)
            return super().run(argv, **kwargs)

        def terminate(self):
            self.terminated = True
            if cleanup_fails and outcome not in {"finished", "rejected"}:
                raise RuntimeError("termination failed")

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
    sessions = []

    def initialize(self, **kwargs):
        self.agent = SimpleNamespace(close=lambda: None)
        self.id = "offline-session"
        self.state = SimpleNamespace(events=[], execution_status="running")
        self.closed = False
        assert kwargs["max_iteration_per_run"] == 30
        sessions.append(self)

    async def arun(self):
        if outcome == "timeout":
            raise TimeoutError("deadline exhausted")
        if outcome == "raised":
            raise ConnectionError("provider disconnected")
        if outcome in {"finished", "rejected"}:
            self.state.execution_status = "finished"
            return
        self.state.events.append(
            ConversationErrorEvent(
                source="agent",
                code=(
                    "MaxIterationsReached" if outcome == "limit" else "ProviderError"
                ),
                detail="stopped api_key=test-secret",
            )
        )
        self.state.execution_status = "error"

    monkeypatch.setattr(LocalConversation, "id", None)
    monkeypatch.setattr(LocalConversation, "state", None)
    monkeypatch.setattr(LocalConversation, "__del__", lambda self: None)
    monkeypatch.setattr(LocalConversation, "__init__", initialize)
    monkeypatch.setattr(LocalConversation, "send_message", lambda *a: None)
    monkeypatch.setattr(LocalConversation, "arun", arun)

    def close(self):
        self.closed = True
        if cleanup_fails and outcome not in {"finished", "rejected"}:
            raise RuntimeError("close failed")

    monkeypatch.setattr(LocalConversation, "close", close)
    checks = []

    def validate(files):
        checks.append(files)
        return Measurement(passed=outcome != "rejected")

    task = WorkspaceTask(
        instruction="science",
        outputs=("answer",),
        verifier=(
            "must not execute" if outcome not in {"finished", "rejected"} else None
        ),
        verifier_files={"hidden": b"secret"},
    )
    (tmp_path / "answer").write_bytes(b"unfinished artifact")
    spec = AgentSpec(Model("test-model", api_key="test-secret"), max_iterations=30)
    backend = SailBackend()
    if outcome == "rejected" and validated:
        from pyforger.structured import ValidationExhausted

        with pytest.raises(ValidationExhausted):
            backend.run_agent_validated(task, spec, validate=validate, max_repairs=0)
        assert sessions[0].closed and box.terminated
        return
    result = (
        backend.run_agent_validated(task, spec, validate=validate)
        if validated
        else backend.run_agent(task, spec)
    )
    assert sessions[0].closed and box.terminated
    assert result.diagnostics["cleanup"] == (
        "failed"
        if cleanup_fails and outcome not in {"finished", "rejected"}
        else "terminated"
    )
    assert result.diagnostics["session_cleanup"] == (
        "failed"
        if cleanup_fails and outcome not in {"finished", "rejected"}
        else "closed"
    )
    assert result.trajectory is result.diagnostics["trajectory"]
    assert "test-secret" not in json.dumps(result.diagnostics)
    if outcome in {"finished", "rejected"}:
        assert result.status == "completed"
        assert result.files == {"answer": b"unfinished artifact"}
        assert len(checks) == int(validated)
    else:
        assert result.status == ("timed_out" if outcome == "timeout" else "failed")
        assert result.failure_kind == (
            "budget" if outcome in {"limit", "timeout"} else "infrastructure"
        )
        assert result.passed is None and result.files == {}
        assert not checks and not result.validation_attempts
        assert "snapshots" not in result.diagnostics
        assert result.diagnostics["execution_status"] in {"error", "running"}
        assert not (tmp_path / "verifier").exists()
        if outcome not in {"raised", "timeout"}:
            assert result.trajectory["steps"][0]["code"] == (
                "MaxIterationsReached" if outcome == "limit" else "ProviderError"
            )
