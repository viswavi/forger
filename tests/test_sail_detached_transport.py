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
        "FORGER_TRANSPORT_LOG": str(tmp_path / "transport.log"),
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
        env=dict(os.environ, FORGER_TRANSPORT_LOG=str(tmp_path / "transport.log")),
    )
    assert result.stdout == result.stderr == ""
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(path)
    time.sleep(1.2)
    with socket.socket(socket.AF_UNIX) as connection:
        with pytest.raises(ConnectionRefusedError):
            connection.connect(path)


@pytest.mark.parametrize("failure", ["socket", "stderr"])
def test_controller_retains_each_nonzero_client_attempt(tmp_path, failure):
    from types import SimpleNamespace

    from pyforger.sail_sdk import _DetachedConnection
    from pyforger.session_budget import SessionBudget

    secret = "client-credential-" + os.urandom(16).hex()
    source = Path(__file__).parents[1] / "src/pyforger/_sail_sdk_transport.py"

    # Only the Sail exec boundary is adapted: the client and missing socket are real.
    class LocalWorker:
        def exec(self, argv, **kwargs):
            argv = [sys.executable, str(source), "client", str(Path("/tmp") / secret)]
            if failure == "stderr":
                argv = [
                    sys.executable,
                    "-c",
                    (
                        "import json,sys; frame=json.load(sys.stdin)['frame']; "
                        "sys.stderr.write('x'*5000 + frame['llm']['api_key'] + '\\n'); "
                        "sys.exit(7)"
                    ),
                ]
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            def wait():
                code = process.wait(timeout=5)
                stderr = process.stderr.read()
                process.stdout.close()
                process.stderr.close()
                return SimpleNamespace(exit_code=code, stderr=stderr)

            return SimpleNamespace(
                stdin=process.stdin, stdout=process.stdout, wait=wait
            )

    connection = _DetachedConnection(LocalWorker(), SessionBudget(None, 10))
    connection.write(json.dumps({"llm": {"api_key": secret}}))
    exit_code = 1 if failure == "socket" else 7
    with pytest.raises(ConnectionError, match=f"exit_code={exit_code}") as error:
        next(connection.stdout)
    assert len(connection.failures) == 3
    assert connection.reconnects == 2
    assert [failure["attempt"] for failure in connection.failures] == [1, 2, 3]
    assert all(item["exit_code"] == exit_code for item in connection.failures)
    assert all(len(item["stderr"]) <= 4000 for item in connection.failures)
    if failure == "socket":
        assert all(
            "FileNotFoundError" in item["stderr"] for item in connection.failures
        )
    else:
        assert all("[REDACTED]" in item["stderr"] for item in connection.failures)
    assert secret not in json.dumps(list(connection.failures)) + str(error.value)


def test_daemon_diagnostics(tmp_path):
    source = Path(__file__).parents[1] / "src/pyforger/_sail_sdk_transport.py"
    shutil.copyfile(source, tmp_path / source.name)
    # A minimal worker keeps this real daemon/socket test independent of SDK HTTP.
    (tmp_path / "_sail_sdk_worker.py").write_text(
        'def main(wire):\n    wire.receive()\n    raise RuntimeError("worker-secret")\n'
    )
    path = str(tmp_path / "socket")
    log = tmp_path / "transport.log"
    env = dict(os.environ, FORGER_TRANSPORT_LOG=str(log))
    process = subprocess.Popen(
        [sys.executable, str(tmp_path / source.name), "daemon", path, "2"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    secret = "protocol-secret-" + os.urandom(16).hex()
    try:
        deadline = time.monotonic() + 5
        while not Path(path).exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        for _ in range(150):
            with socket.socket(socket.AF_UNIX) as connection:
                connection.connect(path)
                connection.sendall(json.dumps({"ack": secret}).encode() + b"\n")
                assert connection.recv(4) == b""
        with socket.socket(socket.AF_UNIX) as connection:
            connection.connect(path)
            connection.sendall(
                json.dumps(
                    {"ack": 0, "id": 1, "frame": {"llm": {"api_key": secret}}}
                ).encode()
                + b"\n"
            )
            connection.recv(4096)
        process.communicate(timeout=5)
        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(records) <= 128
        assert any(
            r["event"] == "request_failed" and r["error_type"] == "ValueError"
            for r in records
        )
        assert any(r["event"] == "worker_failed" for r in records)
        assert any(r["event"] == "worker_done" for r in records)
        assert records[-1]["event"] == "lifetime_expired"
        assert log.stat().st_mode & 0o777 == 0o600
        assert secret not in log.read_text()
        assert "worker-secret" not in log.read_text()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_diagnostic_redaction_precedes_tail_truncation():
    from pyforger._sail_sdk_transport import DIAGNOSTIC_CHARS, diagnostic_text

    secret = "inline-key-" + os.urandom(16).hex()
    raw = secret + "x" * (DIAGNOSTIC_CHARS - 2)
    safe = diagnostic_text(raw, [secret])
    assert len(safe) <= DIAGNOSTIC_CHARS
    assert secret[-2:] not in safe
    safe = diagnostic_text('Bearer unknown-value api_key="another-value"', [])
    assert "unknown-value" not in safe
    assert "another-value" not in safe


def test_daemon_bind_failure_retains_exception_metadata(tmp_path):
    source = Path(__file__).parents[1] / "src/pyforger/_sail_sdk_transport.py"
    log = tmp_path / "transport.log"
    path = str(tmp_path / "socket")
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(path)
        result = subprocess.run(
            [sys.executable, str(source), "daemon", path, "10"],
            env=dict(os.environ, FORGER_TRANSPORT_LOG=str(log)),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    assert result.returncode != 0
    records = [json.loads(line) for line in log.read_text().splitlines()]
    failure = next(record for record in records if record["event"] == "daemon_failed")
    assert failure["error_type"] == "OSError"
    assert failure["errno"] is not None
    assert failure["stack"][-1]["function"] == "daemon"
    assert records[-1]["event"] == "daemon_stopped"


@pytest.mark.parametrize("termination", ["SIGKILL", "SIGTERM", "SIGSEGV", "lifetime"])
def test_supervisor_retains_daemon_exit_without_payloads(tmp_path, termination):
    source = Path(__file__).parents[1] / "src/pyforger/_sail_sdk_transport.py"
    shutil.copyfile(source, tmp_path / source.name)
    # Only the SDK/provider boundary is replaced; subprocess supervision and socket
    # framing exercise production code, with a worker that never calls the network.
    (tmp_path / "_sail_sdk_worker.py").write_text(
        "def main(wire):\n    wire.receive()\n    wire.send({'kind': 'ready'})\n"
        "    wire.receive()\n"
    )
    path = str(tmp_path / "socket")
    log = tmp_path / "transport.log.process"
    process = subprocess.Popen(
        [
            sys.executable,
            str(tmp_path / source.name),
            "supervise",
            path,
            "1" if termination == "lifetime" else "15",
        ],
        env=dict(os.environ, FORGER_TRANSPORT_LOG=str(tmp_path / "transport.log")),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid = None
    secret = "memory-only-" + os.urandom(16).hex()
    try:
        deadline = time.monotonic() + 5
        while not Path(path).exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        pid = json.loads(log.read_text().splitlines()[0])["daemon_pid"]
        if termination == "SIGKILL":
            deadline = time.monotonic() + 7
            while "daemon_alive" not in log.read_text():
                assert time.monotonic() < deadline
                time.sleep(0.02)
        result = subprocess.run(
            [sys.executable, str(tmp_path / source.name), "client", path],
            input=json.dumps({"id": 1, "ack": 0, "frame": {"llm": {"api_key": secret}}})
            + "\n",
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        assert json.loads(result.stdout)["accepted"] == 1
        if termination != "lifetime":
            os.kill(pid, getattr(signal, termination))
        assert process.wait(timeout=5) == 0
        records = [json.loads(line) for line in log.read_text().splitlines()]
        final = records[-1]
        assert final["event"] == "daemon_exit"
        assert final["signal_name"] == (
            None if termination == "lifetime" else termination
        )
        assert final["exit_code"] == (
            0 if termination == "lifetime" else -getattr(signal, termination)
        )
        assert "VmRSS" in final["resources"]
        assert log.stat().st_mode & 0o777 == 0o600
        assert secret not in log.read_text()
        assert "api_key" not in log.read_text()
    finally:
        if process.poll() is None:
            if pid is not None:
                os.kill(pid, signal.SIGKILL)
            process.kill()
        process.wait(timeout=5)


def test_supervisor_records_uncaught_daemon_failure(tmp_path):
    source = Path(__file__).parents[1] / "src/pyforger/_sail_sdk_transport.py"
    log = tmp_path / "transport.log.process"
    path = str(tmp_path / "occupied")
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(path)
        subprocess.run(
            [sys.executable, str(source), "supervise", path, "10"],
            env=dict(os.environ, FORGER_TRANSPORT_LOG=str(tmp_path / "transport.log")),
            capture_output=True,
            timeout=5,
            check=True,
        )
    final = json.loads(log.read_text().splitlines()[-1])
    assert final["event"] == "daemon_exit"
    assert final["exit_code"] == 1
    assert final["signal_number"] is None


def test_resource_probe_tolerates_missing_process():
    from pyforger._sail_sdk_transport import process_resources

    assert process_resources(2**30) == {}
