"""Concurrent native SDK sessions with only HTTP and microVM boundaries local."""

import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_sail_backend import LocalSailbox

from pyforger.acquire.model import Model
from pyforger.agent_backend import AgentSpec, WorkspaceTask
from pyforger.agent_session import SessionBudget
from pyforger.artifacts import ArtifactStore
from pyforger.measurements import Measurement
from pyforger.sail_backend import SailBackend
from pyforger.sail_transport import SailTransport
from pyforger.structured import GenerationInconclusive, ValidationExhausted


@pytest.mark.parametrize(
    "mode",
    [
        "plain",
        "switch",
        "repair",
        "iterations",
        "deadline",
        "http_timeout",
        "rejected",
        "inconclusive",
        "validator_error",
        "missing",
        "stdin_stall",
        "stdin_ack_after_output",
    ],
)
def test_concurrent_native_sessions(tmp_path, caplog, capsys, monkeypatch, mode):
    """Keep concurrent native sessions, repair and failure budgets isolated."""
    barrier = threading.Barrier(4)
    http_counts = {}
    count_lock = threading.Lock()
    worker_state = threading.local()

    class Box(LocalSailbox):
        def __init__(self, root):
            super().__init__(root)
            self.sailbox_id = root.name

        def run(self, argv, **kwargs):
            if argv[0] != "/usr/bin/env":
                return super().run(argv, **kwargs)
            assert "offline-test" not in str(argv) + str(kwargs)
            barrier.wait(timeout=30)
            result = subprocess.run(
                ["bash", "-c", argv[-1].removeprefix("cd /work && ")],
                cwd=self.root,
                env={"PATH": "/usr/bin:/bin"},
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
            return SimpleNamespace(
                exit_code=result.returncode,
                stdout=(
                    ("data line\n" * 43932)[:395385]
                    if mode == "stdin_ack_after_output"
                    else result.stdout
                ),
                stderr=result.stderr,
                timed_out=False,
            )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            user = next(m for m in body["messages"] if m["role"] == "user")
            content = user["content"]
            text = content if isinstance(content, str) else content[0]["text"]
            marker = next(
                word for word in text.split() if word.startswith("candidate-")
            )
            with count_lock:
                http_counts[marker] = http_counts.get(marker, 0) + 1
            if mode == "http_timeout":
                time.sleep(2)
            last_user = max(
                i for i, m in enumerate(body["messages"]) if m["role"] == "user"
            )
            repair = "Validation feedback" in str(
                body["messages"][last_user]["content"]
            )
            if repair and mode != "missing":
                assert "repair-needed" in str(body["messages"][last_user]["content"])
            done = any(m["role"] == "tool" for m in body["messages"][last_user + 1 :])
            name = (
                "finish"
                if done
                else next(
                    t["function"]["name"]
                    for t in body["tools"]
                    if t["function"]["name"] != "finish"
                )
            )
            arguments = (
                {"message": marker}
                if done
                else {
                    "command": (
                        f"test -f answer && printf %s {marker}-repaired > answer"
                        if repair
                        else f"printf %s {marker} > answer"
                    )
                }
            )
            response = {
                "id": "chatcmpl-" + marker,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-" + marker + str(done),
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    serving.start()

    stall_first = mode == "stdin_stall"
    stall_state = threading.local()

    class Worker:
        sailbox_id = "local-sdk-worker"

        def __init__(self):
            self.directory = tempfile.TemporaryDirectory()
            self.fs = self
            self.process = None
            self.stall = False
            self.clients = 0

        def write(self, path, source, **kwargs):
            Path(self.directory.name, Path(path).name).write_bytes(source.read())

        def read(self, path):
            return Path(self.directory.name, Path(path).name).read_bytes()

        def local_argv(self, argv):
            index = argv.index("python3")
            args = argv[index + 1 :]
            return [sys.executable] + [
                (
                    self.directory.name + "/private/socket"
                    if arg.endswith("/socket")
                    else str(Path(self.directory.name, Path(arg).name))
                )
                if arg.startswith("/tmp/")
                else arg
                for arg in args
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
            self.clients += 1
            process = subprocess.Popen(
                self.local_argv(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.environment(),
            )
            worker = self

            class Input:
                def close(self):
                    process.stdin.close()

                def write(self, value):
                    assert "offline-test" not in str(argv) + str(kwargs)
                    process.stdin.write(value)
                    process.stdin.flush()
                    # Lose an accepted turn's response; the reconnect must use
                    # the same ID, not create another SDK/model turn.
                    if worker.stall and '"kind": "turn"' in value:
                        worker.stall = False
                        time.sleep(0.3)
                        process.kill()
                        process.communicate(timeout=5)
                        raise TimeoutError("lost short-lived client")

            def stdout():
                yield from process.stdout
                process.wait(timeout=5)
                process.stdin.close()
                process.stdout.close()
                process.stderr.close()

            return SimpleNamespace(
                stdin=Input(),
                stdout=stdout(),
                wait=lambda: SimpleNamespace(exit_code=process.wait(timeout=5)),
            )

        def terminate(self):
            if hasattr(self, "daemon_pid"):
                os.kill(self.daemon_pid, 15)
            for path in Path(self.directory.name).rglob("*"):
                if path.is_file() and not path.is_socket():
                    assert b"offline-test" not in path.read_bytes(), str(path)
            self.directory.cleanup()

    def make_worker(**kwargs):
        worker = Worker()
        worker_state.current = worker
        if stall_first:
            # Each concurrent run stalls only its own first worker, tracked per
            # thread so parallel runs cannot race on shared state.
            first = not getattr(stall_state, "created", False)
            stall_state.created = True
            worker.stall = first
        return worker

    class Image:
        def pip_install(self, *args):
            return self

    fake_sail = SimpleNamespace(
        Image=SimpleNamespace(from_registry=lambda _: Image()),
        App=SimpleNamespace(find=lambda *a, **k: None),
        Sailbox=SimpleNamespace(create=make_worker),
    )
    monkeypatch.setitem(sys.modules, "sail", fake_sail)
    # Only the provider boundary is local; production requires HTTPS.
    import pyforger.sail_sdk as remote

    original_split = remote.urlsplit
    monkeypatch.setattr(
        remote, "urlsplit", lambda url: original_split(url)._replace(scheme="https")
    )
    cwd, env = os.getcwd(), dict(os.environ)

    def run(index):
        marker = f"candidate-{index}"
        root = tmp_path / marker
        root.mkdir()
        model = Model(
            "openai/gpt-4o",
            api_key="offline-test",
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            api_mode="chat",
            num_retries=0,
            timeout=1 if mode == "http_timeout" else 15,
            capability_overrides={"vision": False},
        )
        diagnostics = {}
        budget = SessionBudget(
            2 if mode == "iterations" else 6, 20 if mode == "http_timeout" else 120
        )
        checks = []

        def validate(files):
            checks.append(files)
            if mode == "deadline":
                # Expire the real shared budget during host validation.
                budget.timeout_s = 0.001
            if mode == "validator_error":
                raise OSError("validator unavailable")
            passed = files["answer"].endswith(b"-repaired")
            if mode == "rejected":
                passed = False
            elif mode == "inconclusive":
                passed = None
            return Measurement(passed=passed, reason="repair-needed")

        transport_box = Box(root)

        def execute():
            if stall_first:
                stall_state.created = False
            if mode == "switch":
                from pyforger.sail_sdk import remote_sdk_session

                task = WorkspaceTask(instruction=marker, outputs=("answer",))
                spec = AgentSpec(model=model, max_iterations=budget.max_iterations)
                transport = SailTransport(transport_box, budget)
                with remote_sdk_session(
                    SailBackend(), transport, task, spec, None, 0, diagnostics
                ) as run:
                    first, _ = run(task, None, 0)
                    time.sleep(0.1)
                    clients = worker_state.current.clients
                    time.sleep(0.3)
                    assert worker_state.current.clients == clients
                    second, attempts = run(task, None, 0, spec)
                    assert first == second
                    assert (
                        len({s["conversation_id"] for s in diagnostics["snapshots"]})
                        == 1
                    )
                return second, attempts
            return SailBackend()._sdk(
                SailTransport(transport_box, budget),
                WorkspaceTask(
                    instruction=marker,
                    outputs=("missing",) if mode == "missing" else ("answer",),
                ),
                AgentSpec(model=model, max_iterations=budget.max_iterations),
                None if mode in {"plain", "http_timeout"} else validate,
                1,
                diagnostics,
            )

        if mode in {"iterations", "deadline", "http_timeout"}:
            started = time.monotonic()
            with pytest.raises((RuntimeError, TimeoutError)):
                execute()
            if mode == "http_timeout":
                assert time.monotonic() - started < 30
                assert not checks
            else:
                assert len(checks) == 1
                assert budget.iterations == 2
        elif mode in {"rejected", "inconclusive", "validator_error", "missing"}:
            error = {
                "rejected": ValidationExhausted,
                "missing": ValidationExhausted,
                "inconclusive": GenerationInconclusive,
                "validator_error": OSError,
            }[mode]
            with pytest.raises(error) as caught:
                execute()
            count = 2 if mode in {"rejected", "missing"} else 1
            assert len(checks) == (0 if mode == "missing" else count)
            assert budget.iterations == count * 2
            if mode != "validator_error":
                assert len(caught.value.attempts) == count
                assert caught.value.attempts[0].output == (
                    {} if mode == "missing" else {"answer": marker.encode()}
                )
        else:
            result, attempts = execute()
            suffix = (
                "-repaired"
                if mode in {"repair", "stdin_stall", "stdin_ack_after_output"}
                else ""
            )
            assert result == {"answer": (marker + suffix).encode()}
            assert len(attempts) == (
                2 if mode in {"repair", "stdin_stall", "stdin_ack_after_output"} else 0
            )
            assert budget.iterations == (
                4
                if mode in {"repair", "stdin_stall", "stdin_ack_after_output", "switch"}
                else 2
            )
            if mode == "stdin_ack_after_output":
                assert diagnostics["sdk_worker_attempts"] == 1
                assert "sdk_stdin_retries" not in diagnostics
            if mode == "stdin_stall":
                assert "sdk_stdin_retries" not in diagnostics
                assert diagnostics["sdk_worker_attempts"] == 1
                assert diagnostics["sdk_transport"]["reconnects"] == 1
            if attempts:
                assert attempts[0].output == {"answer": marker.encode()}
                assert (
                    len({v["conversation_id"] for v in diagnostics["snapshots"]}) == 1
                )
                assert (
                    diagnostics["snapshots"][1]["events"]
                    > diagnostics["snapshots"][0]["events"]
                )
            assert diagnostics["execution_status"] == "finished"
        assert diagnostics["sdk_cleanup"] == "terminated"
        if mode == "stdin_stall":
            assert http_counts[marker] == 4
        return diagnostics.get("conversation_id", str(root))

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(run, range(8)))
        assert len(set(ids)) == 8
        assert os.getcwd() == cwd
        assert dict(os.environ) == env
        time.sleep(0.2)
        captured = capsys.readouterr()
        assert not any(
            error in captured.err + captured.out
            for error in (
                "cannot reuse already awaited coroutine",
                "task_done() called too many times",
                "was never awaited",
            )
        )
        assert not any(
            any(
                error in r.getMessage()
                for error in (
                    "Duplicate tool name registered",
                    "cannot reuse already awaited coroutine",
                    "task_done() called too many times",
                    "was never awaited",
                )
            )
            for r in caplog.records
        )
    finally:
        server.shutdown()
        server.server_close()
        serving.join()


def test_concurrent_artifact_publication(tmp_path):
    """Publish the same digest concurrently without partial artifacts."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer").write_bytes(b"immutable artifact" * 1000)
    store = ArtifactStore(tmp_path / "store")
    barrier = threading.Barrier(8)

    def publish(_):
        barrier.wait(timeout=10)
        ref = store.put(source)
        ref.verify()
        return ref

    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(publish, range(8)))
    assert len({ref.digest for ref in refs}) == 1
    assert list(store.root.iterdir()) == [refs[0].root]
