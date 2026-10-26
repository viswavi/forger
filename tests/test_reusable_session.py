"""Real persistent workspaces and SDK conversations, substituting external I/O only."""

import json
import time
from dataclasses import replace

import pytest
from completion import completion_server

from pyforger.acquire.model import Model
from pyforger.agent_backend import AgentSpec, WorkspaceTask
from pyforger.evidence_journal import iter_trajectory_events
from pyforger.local_backend import LocalBackend
from pyforger.measurements import Measurement
from pyforger.structured import ValidationExhausted


def test_reusable_oracle_outer_feedback_and_inputs():
    agent = AgentSpec(
        kind="oracle",
        solution='echo "$FORGER_FEEDBACK" > feedback; cat input >> output',
    )
    with LocalBackend().open_session() as session:
        first = session.run_agent(
            WorkspaceTask("first", files={"input": "a"}, outputs=("output",)), agent
        )
        second = session.run_agent(
            WorkspaceTask(
                "repair", files={"input": "b"}, outputs=("output", "feedback")
            ),
            replace(agent),
        )
        assert first.files == {"output": b"a"}
        assert second.files == {"output": b"ab", "feedback": b"repair\n"}
        assert second.diagnostics["iterations"] == 2
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.run_agent(WorkspaceTask("again"), agent)


def test_reusable_validation_exhaustion_is_repairable():
    agent = AgentSpec(kind="oracle", solution="cat input > output")
    with LocalBackend().open_session() as session:
        with pytest.raises(ValidationExhausted):
            session.run_agent_validated(
                WorkspaceTask("write", files={"input": "bad"}, outputs=("output",)),
                agent,
                validate=lambda _: Measurement(passed=False),
                max_repairs=0,
            )
        result = session.run_agent(
            WorkspaceTask("repair", files={"input": "good"}, outputs=("output",)),
            replace(agent),
        )
        assert result.files == {"output": b"good"}
        assert result.diagnostics["iterations"] == 2


def test_reusable_iteration_cap_cannot_be_reset():
    agent = AgentSpec(kind="nop", max_iterations=1)
    with LocalBackend().open_session() as session:
        session.run_agent(WorkspaceTask("one"), agent)
        with pytest.raises(RuntimeError, match="iteration budget"):
            session.run_agent(WorkspaceTask("two"), replace(agent, max_iterations=None))
        assert session._closed


def test_absolute_deadline_allows_idle_beyond_first_task_timeout():
    agent = AgentSpec(kind="nop")
    with LocalBackend().open_session(deadline=time.monotonic() + 5) as session:
        session.run_agent(WorkspaceTask("first", timeout_s=0.05), agent)
        time.sleep(0.1)
        assert (
            session.run_agent(
                WorkspaceTask("second", timeout_s=0.5), replace(agent)
            ).status
            == "completed"
        )


def test_reusable_expired_deadline_closes():
    with LocalBackend().open_session(deadline=time.monotonic() - 1) as session:
        with pytest.raises(TimeoutError) as caught:
            session.run_agent(WorkspaceTask("first"), AgentSpec(kind="nop"))
        assert session._closed
        current = caught.value.diagnostics["current_trajectory"]
        assert current["steps"] == []
        assert current["conversation_id"] is None
        assert current["no_current_history_reason"] == "turn_not_started"
        assert current["incomplete"] is True


@pytest.mark.parametrize(
    "task",
    [
        WorkspaceTask("no", verifier="true"),
        WorkspaceTask("no", verifier_files={"hidden": "x"}),
        WorkspaceTask("no", solution_files={"secret": "x"}),
    ],
)
def test_reusable_sealed_material_rejected(task):
    with LocalBackend().open_session() as session:
        with pytest.raises(ValueError, match="hidden"):
            session.run_agent(task, AgentSpec(kind="nop"))
        assert session._agent is None


def tool(name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-" + name,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def test_real_sdk_model_switch_keeps_conversation():
    responses = [
        tool("finish", {"message": "first-done"}),
        tool("finish", {"message": "second-done"}),
    ]
    with completion_server(responses) as (url, requests):

        def spec(model, key):
            return AgentSpec(
                Model(
                    model,
                    api_key=key,
                    api_base=url,
                    api_mode="chat",
                    capability_overrides={"vision": False},
                ),
                kwargs={"agent_context": None},
            )

        first, second = (
            spec("openai/gpt-4o", "test-first-credential"),
            spec("openai/gpt-4o-mini", "test-second-credential"),
        )
        with LocalBackend().open_session() as session:
            a = session.run_agent(WorkspaceTask("remember first-task-marker"), first)
            b = session.run_agent(WorkspaceTask("second-task-marker"), second)
        assert a.diagnostics["session_id"] == b.diagnostics["session_id"]
        assert b.diagnostics["iterations"] == 2
        assert requests[0]["model"] == "gpt-4o"
        assert requests[1]["model"] == "gpt-4o-mini"
        assert "first-task-marker" in json.dumps(requests[1]["messages"])
        assert "test-first-credential" not in json.dumps(b.trajectory)
        assert "test-second-credential" not in json.dumps(b.trajectory)


@pytest.mark.parametrize(
    "fail_worker", [False, True, "timeout", "later_timeout", "nested_timeout"]
)
def test_sail_real_worker_persists_and_switches_model(
    tmp_path, monkeypatch, fail_worker
):
    """Run the actual sealed worker and transport, replacing only Sail and HTTP."""
    import os
    import shutil
    import socket
    import struct
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from test_sail_backend import LocalSailbox

    import pyforger.sail_sdk as remote
    from pyforger.sail_backend import SailBackend

    if (
        not shutil.which("docker")
        or subprocess.run(
            ["docker", "image", "inspect", "python:3.13.17-slim-bookworm"],
            capture_output=True,
        ).returncode
    ):
        pytest.skip(
            "Native isolated terminal integration requires local Docker image python:3.13.17-slim-bookworm"
        )
    boxes = []
    workers = []

    class Box(LocalSailbox):
        sailbox_id = "workspace-box"
        terminated = False

        def __init__(self, root):
            super().__init__(root)
            root.chmod(0o777)
            deps = str(
                Path(sys.executable).parent.parent / "lib/python3.13/site-packages"
            )
            self.container = subprocess.check_output(
                [
                    "docker",
                    "run",
                    "-d",
                    "--network",
                    "none",
                    "-v",
                    f"{root}:/work",
                    "-v",
                    f"{deps}:/usr/local/lib/python3.13/site-packages:ro",
                    "python:3.13.17-slim-bookworm",
                    "sleep",
                    "300",
                ],
                text=True,
            ).strip()

        def write(self, path, source, **kwargs):
            if path in ("/tmp/_sail_terminal.py", "/tmp/_sail_sdk_transport.py"):
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        "-i",
                        self.container,
                        "python",
                        "-c",
                        "import sys;open(sys.argv[1],'wb').write(sys.stdin.buffer.read())",
                        path,
                    ],
                    input=source.read(),
                    check=True,
                )
            else:
                super().write(path, source, **kwargs)

        def run(self, argv, **kwargs):
            if argv[0] == "bash":
                return SimpleNamespace(exit_code=0, timed_out=False)
            if argv[0] == "setpriv":
                result = subprocess.run(
                    ["docker", "exec", self.container, *argv],
                    capture_output=True,
                    text=True,
                    timeout=25,
                )
                return SimpleNamespace(
                    exit_code=result.returncode, stderr=result.stderr
                )
            # The local transfer adapter is not container root; make task-created
            # files readable/writable by the host solely at this test boundary.
            subprocess.run(
                ["docker", "exec", self.container, "chmod", "-R", "a+rwX", "/work"],
                check=True,
            )
            return super().run(argv, **kwargs)

        def exec(self, argv, **kwargs):
            process = subprocess.Popen(
                [
                    "docker",
                    "exec",
                    "-i",
                    "--user",
                    "65534:65534",
                    self.container,
                    *argv,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            class Input:
                def write(self, value):
                    if "current-action" in value:
                        target = later if fail_worker == "nested_timeout" else session
                        target._budget.deadline = time.monotonic() + 1
                    process.stdin.write(value)
                    process.stdin.flush()

                def close(self):
                    process.stdin.close()

            def output():
                yield from process.stdout

            return SimpleNamespace(
                stdin=Input(),
                stdout=output(),
                wait=lambda: SimpleNamespace(exit_code=process.wait(timeout=35)),
            )

        def terminate(self):
            subprocess.run(
                ["docker", "rm", "-f", self.container], capture_output=True, check=True
            )
            self.terminated = True

    class Worker:
        sailbox_id = "worker-box"
        terminated = False

        def __init__(self):
            self.directory = tempfile.TemporaryDirectory()
            self.fs = self
            self.process = None

        def write(self, path, source, **kwargs):
            content = source.read()
            if fail_worker is True and path.endswith("_sail_sdk_worker.py"):
                content = content.replace(
                    b"                session.run()",
                    b"                print('early-worker-evidence\\n' + 'x' * 20000, flush=True)\n"
                    b"                class ConversationRunError(RuntimeError): pass\n"
                    b"                try:\n"
                    b"                    raise ValueError('provider causal detail first-key')\n"
                    b"                except ValueError as cause:\n"
                    b"                    raise ConversationRunError('conversation failed with detail') from cause",
                )
            Path(self.directory.name, Path(path).name).write_bytes(content)

        def read(self, path):
            return Path(self.directory.name, Path(path).name).read_bytes()

        def local_argv(self, argv):
            index = argv.index("python3")
            return [sys.executable] + [
                (
                    self.directory.name + "/private/socket"
                    if arg.endswith("/socket")
                    else str(Path(self.directory.name, Path(arg).name))
                )
                if arg.startswith("/tmp/")
                else arg
                for arg in argv[index + 1 :]
            ]

        def environment(self):
            return {
                "PATH": os.environ["PATH"],
                "HOME": self.directory.name,
                "FORGER_WORKER_LOG": self.directory.name + "/forger-worker.log",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            }

        def run(self, argv, **kwargs):
            result = subprocess.run(
                self.local_argv(argv),
                env=self.environment(),
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            assert result.returncode == 0, result.stderr
            with socket.socket(socket.AF_UNIX) as connection:
                connection.connect(self.directory.name + "/private/socket")
                self.daemon_pid = struct.unpack(
                    "3i",
                    connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12),
                )[0]
            return SimpleNamespace(exit_code=0)

        def exec(self, argv, **kwargs):
            process = subprocess.Popen(
                self.local_argv(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.environment(),
            )

            class Input:
                def write(self, value):
                    process.stdin.write(value)
                    process.stdin.flush()

                def close(self):
                    process.stdin.close()

            def stdout():
                yield from process.stdout
                process.wait(timeout=5)
                process.stdout.close()
                process.stderr.close()

            return SimpleNamespace(
                stdin=Input(),
                stdout=stdout(),
                wait=lambda: SimpleNamespace(exit_code=process.wait(timeout=5)),
            )

        def terminate(self):
            if hasattr(self, "daemon_pid"):
                try:
                    os.kill(self.daemon_pid, 15)
                except ProcessLookupError:
                    pass
            journal = Path(self.directory.name, "forger-worker.log.events")
            if journal.exists():
                assert journal.stat().st_mode & 0o777 == 0o600
                assert b"first-key" not in journal.read_bytes()
                assert b"second-key" not in journal.read_bytes()
            self.directory.cleanup()
            self.terminated = True
            if fail_worker is True:
                monkeypatch.setenv("REMOTE_SERVICE_TOKEN", "rotated-service-credential")
                raise RuntimeError("worker cleanup failed old-service-credential")

    class Image:
        def pip_install(self, *args):
            return self

    def create(**kwargs):
        if kwargs["name"].startswith("forger-sdk-"):
            worker = Worker()
            workers.append(worker)
            return worker
        box = Box(tmp_path)
        boxes.append(box)
        return box

    monkeypatch.setitem(
        sys.modules,
        "sail",
        SimpleNamespace(
            Image=SimpleNamespace(from_registry=lambda _: Image()),
            App=SimpleNamespace(find=lambda *a, **k: None),
            Sailbox=SimpleNamespace(create=create),
        ),
    )
    monkeypatch.setenv("REMOTE_SERVICE_TOKEN", "old-service-credential")
    split = remote.urlsplit
    monkeypatch.setattr(
        remote, "urlsplit", lambda url: split(url)._replace(scheme="https")
    )
    responses = [
        tool("terminal", {"command": "cat input > output"}),
        tool("finish", {"message": "initial"}),
        tool("terminal", {"command": "cat input >> output"}),
        tool("finish", {"message": "repaired"}),
    ]
    timeout_case = fail_worker in {"timeout", "later_timeout", "nested_timeout"}
    if timeout_case:
        timeout_messages = [
            tool(
                "terminal",
                {"command": "printf current-action; sleep 30", "timeout": 30},
            )
        ]
        responses = (
            responses[:2] if fail_worker != "timeout" else []
        ) + timeout_messages
    with completion_server(responses) as (url, requests):

        def spec(model, key):
            return AgentSpec(
                Model(
                    model,
                    api_key=key,
                    api_base=url,
                    api_mode="chat",
                    capability_overrides={"vision": False},
                ),
                max_iterations=8,
            )

        if fail_worker is True:
            session = SailBackend().open_session(deadline=time.monotonic() + 120)
            with pytest.raises(
                RuntimeError, match="conversation failed with detail"
            ) as caught:
                with session:
                    session.run_agent(
                        WorkspaceTask("first-marker", outputs=("output",)),
                        spec("openai/gpt-4o", "first-key"),
                    )
            error = caught.value
            assert "provider causal detail" in str(error)
            assert "direct cause" in str(error)
            assert "first-key" not in str(error)
            diagnostics = error.diagnostics
            assert diagnostics["cleanup"] == "terminated"
            assert diagnostics["sdk_cleanup"] == "failed"
            assert (
                diagnostics["sdk_cleanup_error"] == "worker cleanup failed [REDACTED]"
            )
            assert "old-service-credential" not in json.dumps(diagnostics)
            events = list(iter_trajectory_events(diagnostics["trajectory"]))
            assert events
            assert "first-marker" in json.dumps(events)
            log_ref = diagnostics["sdk_worker_log"]["evidence_file"]
            log = Path(log_ref["path"]).read_text()
            assert "early-worker-evidence" in log
            assert len(log) > 20000
            assert "ConversationRunError" in diagnostics["sdk_error"]["traceback"]
            assert "provider causal detail" in diagnostics["sdk_error"]["traceback"]
            assert "first-key" not in json.dumps(diagnostics)
            assert workers[0].terminated and boxes[0].terminated
            return

        if timeout_case:
            first = None
            session = SailBackend().open_session(deadline=time.monotonic() + 120)
            with pytest.raises(TimeoutError) as caught:
                with session:
                    if fail_worker != "timeout":
                        first = session.run_agent(
                            WorkspaceTask(
                                "earlier-finished",
                                files={"input": "a"},
                                outputs=("output",),
                            ),
                            spec("openai/gpt-4o", "first-key"),
                        )
                    if fail_worker == "nested_timeout":
                        with SailBackend().open_session(
                            deadline=time.monotonic() + 120
                        ) as later:
                            later.run_agent(
                                WorkspaceTask("current-timeout"),
                                spec("openai/gpt-4o", "second-key"),
                            )
                    else:
                        session.run_agent(
                            WorkspaceTask("current-timeout"),
                            spec("openai/gpt-4o", "second-key"),
                        )
            retained = caught.value.diagnostics
            path = tmp_path / "failure-diagnostics.json"
            path.write_text(json.dumps(retained))
            saved = json.loads(path.read_text())
            current = saved["current_trajectory"]
            assert current["incomplete"] is True
            assert current["conversation_id"]
            assert current["call_id"] and current["turn_id"]
            assert current["last_event_at"] and current["captured_at"]
            events = list(iter_trajectory_events(retained["current_trajectory"]))
            assert "current-timeout" in json.dumps(events)
            assert "current-action" in json.dumps(events)
            assert "earlier-finished" not in json.dumps(events)
            assert saved["trajectory"] == current
            if first is not None:
                if fail_worker == "nested_timeout":
                    assert (
                        current["conversation_id"]
                        != first.diagnostics["conversation_id"]
                    )
                    earlier = retained["enclosing_sessions"][0]["trajectory"]
                    assert "earlier-finished" in json.dumps(
                        list(iter_trajectory_events(earlier))
                    )
                else:
                    assert (
                        current["conversation_id"]
                        == first.diagnostics["conversation_id"]
                    )
                    earlier = retained["sdk_attempts"][0]["turns"][0]
                    assert "earlier-finished" in json.dumps(
                        list(iter_trajectory_events(earlier))
                    )
            assert "first-key" not in path.read_text()
            assert "second-key" not in path.read_text()
            assert all(worker.terminated for worker in workers)
            return

        with SailBackend().open_session(deadline=time.monotonic() + 120) as session:
            a = session.run_agent(
                WorkspaceTask(
                    "first-marker", files={"input": "a"}, outputs=("output",)
                ),
                spec("openai/gpt-4o", "first-key"),
            )
            b = session.run_agent(
                WorkspaceTask(
                    "outer feedback", files={"input": "b"}, outputs=("output",)
                ),
                spec("openai/gpt-4o-mini", "second-key"),
            )
            assert a.files == {"output": b"a"}
            assert b.files == {"output": b"ab"}
            assert a.diagnostics["conversation_id"] == b.diagnostics["conversation_id"]
            assert b.diagnostics["iterations"] == 4
            assert len(boxes) == len(workers) == 1
            assert "first-marker" in json.dumps(requests[2]["messages"])
            assert requests[2]["model"] == "gpt-4o-mini"
            assert not workers[0].terminated
        assert workers[0].terminated and boxes[0].terminated


def test_policy_change_is_rejected_without_losing_original_session():
    from pyforger.agent_backend import Network

    with LocalBackend().open_session() as session:
        agent = AgentSpec(kind="nop")
        session.run_agent(WorkspaceTask("first"), agent)
        with pytest.raises(ValueError, match="network"):
            session.run_agent(WorkspaceTask("unsafe", network=Network.PUBLIC), agent)
        assert (
            session.run_agent(WorkspaceTask("original"), replace(agent)).status
            == "completed"
        )


def test_concurrent_use_and_close_rejected():
    import threading

    entered, release = threading.Event(), threading.Event()
    failures = []
    session = LocalBackend().open_session()
    agent = AgentSpec(kind="nop")

    def validate(_):
        entered.set()
        assert release.wait(5)
        return Measurement(passed=True)

    def run():
        try:
            session.run_agent_validated(
                WorkspaceTask("first"), agent, validate=validate
            )
        except BaseException as exc:  # noqa: BLE001 - forward thread failures to test
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="already in use"):
            session.run_agent(WorkspaceTask("overlap"), agent)
        with pytest.raises(RuntimeError, match="already in use"):
            session.close()
    finally:
        release.set()
        thread.join(5)
        session.close()
    assert not failures


def test_sail_update_replaces_leaf_without_following_link(tmp_path):
    from test_sail_backend import local_transport

    outside = tmp_path.parent / "protected-target"
    outside.write_bytes(b"protected")
    (tmp_path / "input").symlink_to(outside)
    transport = local_transport(tmp_path)
    transport.files({"input": b"updated"}, write=True, replace=True)
    assert outside.read_bytes() == b"protected"
    assert (tmp_path / "input").read_bytes() == b"updated"
    (tmp_path / "directory").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="failed Sail file transfer"):
        transport.files(
            {"directory/protected-target": b"bad"}, write=True, replace=True
        )
    assert outside.read_bytes() == b"protected"


def test_task_timeout_caps_active_call_even_with_long_candidate_deadline():
    with LocalBackend().open_session(deadline=time.monotonic() + 30) as session:
        with pytest.raises(TimeoutError):
            session.run_agent(
                WorkspaceTask("wait", timeout_s=0.05),
                AgentSpec(kind="oracle", solution="sleep 1"),
            )
        assert session._closed


def test_default_backend_calls_still_have_fresh_workspaces():
    backend = LocalBackend()
    task = WorkspaceTask("counter", outputs=("count",))
    agent = AgentSpec(
        kind="oracle",
        solution="if test -f count; then echo reused > count; else echo fresh > count; fi",
    )
    assert backend.run_agent(task, agent).files == {"count": b"fresh\n"}
    assert backend.run_agent(task, agent).files == {"count": b"fresh\n"}


def test_cleanup_error_does_not_replace_original_session_exception():
    from types import SimpleNamespace

    from pyforger.reusable_session import ReusableSession

    original = RuntimeError("original causal failure")

    class Session(ReusableSession):
        def _initialize(self, task, agent):
            def cleanup():
                self._diagnostics["cleanup"] = "failed"
                raise OSError("cleanup causal detail")

            self._stack.callback(cleanup)

            def run(*args):
                raise original

            return run

    session = Session(SimpleNamespace(name="test"))
    with pytest.raises(RuntimeError) as caught, session:
        session.run_agent(WorkspaceTask("fail"), AgentSpec(kind="nop"))
    assert caught.value is original
    assert original.diagnostics["cleanup"] == "failed"
    assert "cleanup causal detail" in original.diagnostics["cleanup_traceback"]


def test_real_local_journal_snapshots_rotated_credentials_and_lazy_replacements(
    tmp_path, monkeypatch
):
    from pyforger.artifacts import FileInput

    first_key, second_key = "old-short-key", "new-short-key"
    responses = [
        tool("finish", {"message": first_key}),
        tool("finish", {"message": first_key + " " + second_key}),
    ]
    source = tmp_path / "input"
    source.write_text("first")
    with completion_server(responses) as (url, requests):
        spec = AgentSpec(
            Model(
                "openai/gpt-4o",
                api_base=url,
                api_mode="chat",
                capability_overrides={"vision": False},
            ),
            api_key_env="INNOCUOUS_NAME",
            kwargs={"agent_context": None},
        )
        monkeypatch.setenv("INNOCUOUS_NAME", first_key)
        with LocalBackend().open_session() as session:
            a = session.run_agent(
                WorkspaceTask("first", files={"input": FileInput.capture(source)}), spec
            )
            prefix = list(iter_trajectory_events(a.trajectory))
            monkeypatch.setenv("INNOCUOUS_NAME", second_key)
            source.write_text("second")
            b = session.run_agent(
                WorkspaceTask("later", files={"input": FileInput.capture(source)}), spec
            )
        events = list(iter_trajectory_events(b.trajectory))
        assert len(events) > len(prefix)
        assert list(iter_trajectory_events(a.trajectory)) == prefix
        assert first_key not in json.dumps(events)
        assert second_key not in json.dumps(events)
        assert "[REDACTED]" in json.dumps(events)
