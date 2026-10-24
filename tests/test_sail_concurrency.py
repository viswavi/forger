"""Concurrent native SDK sessions with only HTTP and microVM boundaries local."""

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from test_sail_backend import LocalSailbox

from pyforger.acquire.model import Model
from pyforger.agent_backend import AgentSpec, WorkspaceTask
from pyforger.agent_session import SessionBudget
from pyforger.artifacts import ArtifactStore
from pyforger.measurements import Measurement
from pyforger.sail_backend import SailBackend
from pyforger.sail_transport import TRANSPORTS, SailCommandTool, SailTransport
from pyforger.structured import GenerationInconclusive, ValidationExhausted


@pytest.mark.parametrize(
    "mode",
    [
        "plain",
        "repair",
        "iterations",
        "deadline",
        "http_timeout",
        "rejected",
        "inconclusive",
        "validator_error",
    ],
)
def test_concurrent_native_sessions(tmp_path, caplog, capsys, mode):
    """Keep concurrent native sessions, repair and failure budgets isolated."""
    barrier = threading.Barrier(4)

    class Box(LocalSailbox):
        def __init__(self, root):
            super().__init__(root)
            self.sailbox_id = root.name

        def run(self, argv, **kwargs):
            if argv[0] != "/usr/bin/env":
                return super().run(argv, **kwargs)
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
                stdout=result.stdout,
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
            if mode == "http_timeout":
                time.sleep(2)
            last_user = max(
                i for i, m in enumerate(body["messages"]) if m["role"] == "user"
            )
            repair = "Validation feedback" in str(
                body["messages"][last_user]["content"]
            )
            if repair:
                assert "repair-needed" in str(body["messages"][last_user]["content"])
            done = any(m["role"] == "tool" for m in body["messages"][last_user + 1 :])
            name = "finish" if done else SailCommandTool.name
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
    before = dict(TRANSPORTS)
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
            timeout=15,
            capability_overrides={"vision": False},
        )
        diagnostics = {}
        budget = SessionBudget(
            2 if mode == "iterations" else 6, 1 if mode == "http_timeout" else 60
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

        def execute():
            return SailBackend()._sdk(
                SailTransport(Box(root), budget),
                WorkspaceTask(instruction=marker, outputs=("answer",)),
                AgentSpec(model=model, max_iterations=6),
                None if mode in {"plain", "http_timeout"} else validate,
                1,
                diagnostics,
            )

        if mode in {"iterations", "deadline", "http_timeout"}:
            started = time.monotonic()
            with pytest.raises((RuntimeError, TimeoutError)):
                execute()
            if mode == "http_timeout":
                assert time.monotonic() - started < 2
                assert not checks
            else:
                assert len(checks) == 1
                assert budget.iterations == 2
        elif mode in {"rejected", "inconclusive", "validator_error"}:
            error = {
                "rejected": ValidationExhausted,
                "inconclusive": GenerationInconclusive,
                "validator_error": OSError,
            }[mode]
            with pytest.raises(error) as caught:
                execute()
            count = 2 if mode == "rejected" else 1
            assert len(checks) == count
            assert budget.iterations == count * 2
            if mode != "validator_error":
                assert len(caught.value.attempts) == count
                assert caught.value.attempts[0].output == {"answer": marker.encode()}
        else:
            result, attempts = execute()
            suffix = "-repaired" if mode == "repair" else ""
            assert result == {"answer": (marker + suffix).encode()}
            assert len(attempts) == (2 if mode == "repair" else 0)
            assert budget.iterations == (4 if mode == "repair" else 2)
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
        assert diagnostics["session_cleanup"] == "closed"
        return diagnostics["conversation_id"]

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(run, range(8)))
        assert len(set(ids)) == 8
        assert TRANSPORTS == before
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
