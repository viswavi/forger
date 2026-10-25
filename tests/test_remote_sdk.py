"""Exercise the real remote worker against a local OpenAI HTTP server."""

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pyforger.acquire.model import Model
from pyforger.agent_backend import AgentSpec
from pyforger.sail_sdk import frames, worker_request


@pytest.mark.parametrize("limit,turns", [(4, 2), (None, 12)])
def test_real_worker_persists_repair_and_relays_commands(tmp_path, limit, turns):
    """A real worker persists repair state and relays commands."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            last_user = max(
                i for i, m in enumerate(body["messages"]) if m["role"] == "user"
            )
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
            args = {"message": "done"} if done else {"command": "printf result"}
            data = json.dumps(
                {
                    "id": "test",
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
                                        "id": str(len(requests)),
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(args),
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
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    worker = Path(__file__).parents[1] / "src/pyforger/_sail_sdk_worker.py"
    for name in (
        "_sail_sdk_worker.py",
        "_sail_sdk_transport.py",
        "_sdk_runtime.py",
        "agent_diagnostics.py",
    ):
        (tmp_path / name).write_bytes(worker.with_name(name).read_bytes())
    transport = tmp_path / "_sail_sdk_transport.py"
    path = str(tmp_path / "private/socket")
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "FORGER_WORKER_LOG": str(tmp_path / "worker.log"),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    argv = [sys.executable, "-u", str(transport)]
    subprocess.run(argv + ["start", path, "120"], env=env, check=True, timeout=15)
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(path)
        pid = struct.unpack(
            "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )[0]
    number = ack = 0
    pending = deque()

    def exchange(value=None):
        nonlocal number, ack
        request = {"ack": ack}
        if value is not None:
            number += 1
            request.update(id=number, frame=value)
        response = subprocess.run(
            argv + ["client", path],
            env=env,
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        reply = json.loads(response.stdout)
        assert reply["accepted"] == number
        for sequence, encoded in reply["outputs"]:
            assert sequence == ack + 1
            ack = sequence
            pending.append(json.loads(encoded))
        return reply

    def send(value):
        exchange(value)

    def receive():
        deadline = time.monotonic() + 30
        while not pending:
            assert time.monotonic() < deadline
            reply = exchange()
            assert pending or not reply["done"], reply
        return pending.popleft()

    try:
        spec = AgentSpec(
            model=Model(
                "openai/gpt-4o",
                api_key="test-only",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
                api_mode="chat",
                capability_overrides={"vision": False},
            ),
            max_iterations=limit,
        )
        send(worker_request(spec, 30))
        ready = receive()
        assert ready["kind"] == "ready"
        assert ready["conversation_id"]
        ids = []
        for turn in ["create output"] + ["repair output"] * (turns - 1):
            send({"kind": "turn", "instruction": turn})
            command = receive()
            while command["kind"] in {"event_chunk", "event_end"}:
                assert "test-only" not in json.dumps(command)
                command = receive()
            assert command["kind"] == "command", command
            assert command["command"] == "printf result"
            assert command["timeout"] == 300
            send({"kind": "observation", "text": "exit_code=0\nresult"})
            result = receive()
            while result["kind"] in {"event_chunk", "event_end"}:
                assert "test-only" not in json.dumps(result)
                result = receive()
            assert result["kind"] == "result", result
            assert result["status"] == "finished"
            ids.append(result["conversation_id"])
        assert ids[0] == ids[1]
        assert result["iterations"] == 2 * turns
        assert len(requests) == 2 * turns
        for request in requests:
            users = [message for message in request["messages"] if message["role"] == "user"]
            assert "run time consuming commands in the background and monitor them" in json.dumps(users[-1])
            assert "job.pid" in json.dumps(users[-1])
            assert "never broad pkill" in json.dumps(users[-1])
        send({"kind": "close"})
        deadline = time.monotonic() + 20
        while not exchange()["done"]:
            assert time.monotonic() < deadline
        assert "test-only" not in (tmp_path / "worker.log").read_text()
    finally:
        os.kill(pid, 15)
        server.shutdown()
        thread.join()
        server.server_close()


def test_protocol_chunk_boundaries():
    """Protocol frames split on chunk boundaries."""
    assert list(frames(['{"kind":', '"result"}\n{"a":1}\n'])) == [
        {"kind": "result"},
        {"a": 1},
    ]


def test_serialization_preserves_explicit_reasoning_and_routing():
    """Serialization preserves explicit reasoning and routing."""
    spec = AgentSpec(
        model=Model(
            "openai/gpt-4o",
            reasoning_effort=None,
            extra_body={"provider": {"only": ["together"], "allow_fallbacks": False}},
            custom_tokenizer="meta-models/Muse-Glimmer-30B",
        )
    )
    options = worker_request(spec, 20)["llm"]
    assert options["reasoning_effort"] is None
    assert options["custom_tokenizer"] == "meta-models/Muse-Glimmer-30B"
    assert options["litellm_extra_body"]["provider"] == {
        "only": ["together"],
        "allow_fallbacks": False,
    }


def test_controller_import_and_serialization_do_not_import_sdk():
    """Controller import and serialization do not import the SDK."""
    code = """
import sys
from pyforger.acquire.model import Model
from pyforger.agent_backend import AgentSpec
from pyforger.sail_backend import SailBackend
from pyforger.sail_sdk import worker_request
from pyforger.sail_transport import SailTransport
spec = AgentSpec(
    model=Model(
        'openai/gpt-4o',
        custom_tokenizer='meta-models/Muse-Glimmer-30B',
    )
)
worker_request(spec, 30)
assert 'openhands.sdk' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_worker_timeout_is_integer_when_episode_limits_provider_timeout():
    """Worker timeout is an integer when episode limits provider timeout."""
    from pyforger.acquire.model import Model
    from pyforger.agent_backend import AgentSpec
    from pyforger.sail_sdk import worker_request

    spec = AgentSpec(model=Model("openai/test", timeout=3600, api_key="test"))
    request = worker_request(spec, 599.75)
    assert type(request["llm"]["timeout"]) is int
    assert request["llm"]["timeout"] <= 600


def test_worker_allows_five_total_llm_attempts():
    from pyforger.acquire.model import Model
    from pyforger.agent_backend import AgentSpec

    spec = AgentSpec(model=Model("openai/test", api_key="test"))
    assert worker_request(spec, 600)["llm"]["num_retries"] == 5


def test_current_event_recovery_is_bounded_and_rejects_stale_turn(tmp_path):
    from types import SimpleNamespace

    from pyforger.sail_sdk import _recover_current_events

    prior = {
        "kind": "event",
        "call_id": "old",
        "turn_id": "old",
        "event": {"text": "stale"},
    }
    path = tmp_path / "events"
    path.write_text(json.dumps(prior) + "\n")
    current = {"call_id": "new", "turn_id": "new", "steps": []}
    worker = SimpleNamespace(fs=SimpleNamespace(read=lambda _: path.read_bytes()))
    _recover_current_events(worker, current)
    assert current["steps"] == []
    assert current["recovery"] == "no_matching_current_turn_journal"

    released = threading.Event()

    def blocked_read(_):
        released.wait(10)
        return b""

    worker.fs.read = blocked_read
    started = time.monotonic()
    try:
        _recover_current_events(worker, current)
        assert time.monotonic() - started < 4
        assert current["recovery"] == "read_deadline_exceeded"
    finally:
        released.set()


def test_expired_remote_setup_records_no_current_history():
    from types import SimpleNamespace

    from pyforger.sail_sdk import remote_sdk_session
    from pyforger.session_budget import SessionBudget

    budget = SessionBudget(None, 0.001)
    time.sleep(0.01)
    diagnostics = {
        "conversation_id": "stale",
        "trajectory": {"steps": [{"text": "old"}]},
    }
    with pytest.raises(TimeoutError) as caught:
        with remote_sdk_session(
            None, SimpleNamespace(budget=budget), None, None, None, 0, diagnostics
        ):
            pytest.fail("Expired setup must not start a worker")
    current = caught.value.diagnostics["current_trajectory"]
    assert current["steps"] == []
    assert current["conversation_id"] is None
    assert current["no_current_history_reason"] == "session_not_started"
    assert current["captured_at"]


def test_worker_stack_snapshot_survives_active_python_callbacks(tmp_path):
    import shutil

    interpreter = shutil.which("python3.12") or sys.executable
    worker = Path(__file__).parents[1] / "src/pyforger/_sail_sdk_worker.py"
    code = r"""
import importlib.util, json, resource, sys, threading, time
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
spec = importlib.util.spec_from_file_location("worker", sys.argv[1])
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
stop = threading.Event()
def callback(x):
    private_value = "credential-sentinel-not-for-logs"
    return x % 17
def busy():
    while not stop.is_set():
        sorted(range(1000), key=callback)
thread = threading.Thread(target=busy)
thread.start()
try:
    with open(sys.argv[2], "w") as sink:
        for _ in range(100):
            worker.thread_snapshot(sink)
finally:
    stop.set()
    thread.join()
"""
    path = tmp_path / "stacks.jsonl"
    result = subprocess.run(
        [interpreter, "-c", code, str(worker), str(path)],
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr.decode()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 100
    assert all(record["event"] == "worker_stack_snapshot" for record in records)
    assert any(
        frame["function"] == "callback"
        for record in records
        for stack in record["threads"]
        for frame in stack["frames"]
    )
    assert all(len(record["threads"]) <= 32 for record in records)
    assert all(
        len(stack["frames"]) <= 32 for record in records for stack in record["threads"]
    )
    assert "credential-sentinel-not-for-logs" not in path.read_text()
