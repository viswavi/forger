"""Real socket clients and detached process lifecycle, without a Sail account."""

import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from pyforger._sail_sdk_transport import MAX_OUTPUT_BYTES, Mailbox


def test_mailbox_dedup_replay_ack_and_order():
    mailbox = Mailbox()
    mailbox.send({"kind": "ready"})
    request = {"ack": 0, "id": 1, "frame": {"kind": "turn"}}
    first = mailbox.exchange(request)
    assert mailbox.exchange(request) == first
    assert mailbox.inputs.qsize() == 1
    assert mailbox.receive() == {"kind": "turn"}
    with pytest.raises(ValueError, match="Out of order"):
        mailbox.exchange({"ack": 0, "id": 3, "frame": {}})
    mailbox.done = True
    assert mailbox.exchange({"ack": 1})["outputs"] == []
    assert mailbox.output_bytes == 0
    with pytest.raises(ValueError, match="acknowledgment"):
        mailbox.exchange({"ack": 2})


def test_mailbox_backpressure_is_bounded():
    mailbox = Mailbox()
    payload = {"kind": "event_chunk", "text": "x" * 32000}
    for _ in range(130):
        mailbox.send(payload)
    finished = threading.Event()

    def produce():
        mailbox.send({"kind": "event_chunk", "text": "y" * 200000})
        finished.set()

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    assert not finished.wait(0.05)
    assert mailbox.output_bytes <= MAX_OUTPUT_BYTES
    mailbox.exchange({"ack": 130})
    assert finished.wait(1)
    thread.join()


def test_detached_subprocess_disconnect_and_secret_disk(tmp_path):
    source = Path(__file__).parents[1] / "src/pyforger"
    for filename in (
        "_sail_sdk_transport.py",
        "_sail_sdk_worker.py",
        "_sdk_runtime.py",
        "agent_diagnostics.py",
    ):
        shutil.copyfile(source / filename, tmp_path / filename)
    path = str(tmp_path / "private/socket")
    argv = [sys.executable, str(tmp_path / "_sail_sdk_transport.py")]
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "FORGER_WORKER_LOG": str(tmp_path / "worker.log"),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    subprocess.run(argv + ["start", path, "30"], env=env, check=True, timeout=15)
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(path)
        pid = struct.unpack(
            "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )[0]
    try:
        assert (tmp_path / "private").stat().st_mode & 0o777 == 0o700
        assert Path(path).stat().st_mode & 0o777 == 0o600
        # An accepted initialization with invalid options deliberately fails before
        # model HTTP. Its credential must never reach logs, argv, or files.
        secret = "detached-secret-" + os.urandom(16).hex()
        request = {
            "ack": 0,
            "id": 1,
            "frame": {"timeout_s": 10, "llm": {"api_key": secret}},
        }
        with socket.socket(socket.AF_UNIX) as connection:
            connection.connect(path)
            connection.sendall(json.dumps(request).encode() + b"\n")
            # Disconnect without consuming the response, after acceptance.
        time.sleep(0.3)
        response = subprocess.run(
            argv + ["client", path],
            env=env,
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        reply = json.loads(response.stdout)
        assert reply["accepted"] == 1
        deadline = time.monotonic() + 15
        while not reply["done"]:
            assert time.monotonic() < deadline
            response = subprocess.run(
                argv + ["client", path],
                env=env,
                input=json.dumps(request) + "\n",
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            reply = json.loads(response.stdout)
        assert len(reply["outputs"]) == 1
        assert json.loads(reply["outputs"][0][1])["kind"] == "error"
        assert secret not in response.stdout + response.stderr
        assert secret.encode() not in Path(f"/proc/{pid}/cmdline").read_bytes()
        for file in tmp_path.rglob("*"):
            if file.is_file():
                assert secret.encode() not in file.read_bytes(), str(file)
    finally:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("field", ["id", "ack"])
def test_mailbox_rejects_boolean_sequence(field):
    mailbox = Mailbox()
    request = {"ack": 0, "id": 1, "frame": {}}
    request[field] = True
    with pytest.raises(ValueError):
        mailbox.exchange(request)


def test_client_length_framing_unicode_controls(tmp_path):
    from pyforger._sail_sdk_transport import BATCH_BYTES

    mailbox = Mailbox()
    text = '\x00\n\\"🦀' * 1000
    for _ in range(10):
        mailbox.send({"kind": "event_chunk", "text": text})
    mailbox.done = True
    path = str(tmp_path / "socket")
    expected = mailbox.exchange({"ack": 0})
    assert len(json.dumps(expected).encode()) > BATCH_BYTES
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(path)
        listener.listen(1)

        def serve():
            with listener.accept()[0] as connection:
                with connection.makefile("rwb") as stream:
                    request = json.loads(stream.readline())
                    response = json.dumps(mailbox.exchange(request)).encode()
                    # Deliberately fragment the length header and body.
                    framed = struct.pack("!I", len(response)) + response
                    for offset in range(0, len(framed), 333):
                        stream.write(framed[offset : offset + 333])
                        stream.flush()

        server = threading.Thread(target=serve)
        server.start()
        source = Path(__file__).parents[1] / "src/pyforger/_sail_sdk_transport.py"
        result = subprocess.run(
            [sys.executable, str(source), "client", path],
            input='{"ack": 0}\n',
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        server.join(5)
    assert json.loads(result.stdout) == expected
    for _, encoded in expected["outputs"]:
        assert json.loads(encoded)["text"] == text


def test_detached_daemon_lifetime_cleanup(tmp_path):
    source = Path(__file__).parents[1] / "src/pyforger/_sail_sdk_transport.py"
    path = str(tmp_path / "private/socket")
    result = subprocess.run(
        [sys.executable, str(source), "start", path, "1"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout == result.stderr == ""
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(path)
    time.sleep(1.2)
    with socket.socket(socket.AF_UNIX) as connection:
        with pytest.raises(ConnectionRefusedError):
            connection.connect(path)
