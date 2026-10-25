"""Private, memory-only mailbox for the detached SDK process."""

from __future__ import annotations

import collections
import json
import os
import queue
import re
import resource
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

SOCKET_PATH = "/tmp/forger-sdk/socket"
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
BATCH_BYTES = 256 * 1024

DIAGNOSTIC_CHARS = 4000
TRANSPORT_LOG_PATH = "/tmp/forger-transport.log"


def diagnostic_text(text: str, secrets: list[str]) -> str:
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)(Bearer|Basic)\s+[^\s\"']+", r"\1 [REDACTED]", text)
    text = re.sub(
        r"(?i)((?:api[_-]?key|token|secret|password|authorization|credential)"
        r"[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)",
        r"\1[REDACTED]",
        text,
    )
    # Redact before truncation, so a boundary cannot expose part of a key.
    return text[-DIAGNOSTIC_CHARS:]


class TransportLog:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.records: collections.deque[dict[str, Any]] = collections.deque(maxlen=128)
        try:
            self.sink = os.fdopen(
                os.open(
                    os.environ.get("FORGER_TRANSPORT_LOG", TRANSPORT_LOG_PATH),
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                    0o600,
                ),
                "w",
            )
        except OSError:
            self.sink = open(os.devnull, "w")  # noqa: SIM115 — daemon-lifetime sink

    def record(self, event: str, error: BaseException | None = None) -> None:
        record: dict[str, Any] = {
            "event": event,
            "at": time.monotonic(),
            "pid": os.getpid(),
        }
        if error is not None:
            # Exception messages and traceback source lines can contain payloads.
            record.update(
                error_type=type(error).__name__, errno=getattr(error, "errno", None)
            )
            record["stack"] = [
                {"function": frame.name, "line": frame.lineno}
                for frame in traceback.extract_tb(error.__traceback__)[-8:]
            ]
        with self.lock:
            self.records.append(record)
            try:
                self.sink.seek(0)
                self.sink.write(
                    "".join(json.dumps(item) + "\n" for item in self.records)
                )
                self.sink.truncate()
                self.sink.flush()
            except OSError:
                pass


class Mailbox:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.inputs: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=64)
        self.accepted = 0
        self.sequence = 0
        self.acked = 0
        self.outputs: collections.deque[tuple[int, str]] = collections.deque()
        self.output_bytes = 0
        self.done = False
        self.secrets: list[str] = []

    def receive(self) -> dict[str, Any]:
        return self.inputs.get()

    def send(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value) + "\n"
        size = len(encoded.encode())
        if size > BATCH_BYTES:
            raise ValueError("Oversize SDK output frame")
        with self.condition:
            while self.output_bytes + size > MAX_OUTPUT_BYTES:
                self.condition.wait()
            self.sequence += 1
            self.outputs.append((self.sequence, encoded))
            self.output_bytes += size
            self.condition.notify_all()

    def exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.condition:
            ack = request["ack"]
            if type(ack) is not int or not self.acked <= ack <= self.sequence:
                raise ValueError("Invalid SDK output acknowledgment")
            self.acked = ack
            while self.outputs and self.outputs[0][0] <= ack:
                _, encoded = self.outputs.popleft()
                self.output_bytes -= len(encoded.encode())
            self.condition.notify_all()
            number = request.get("id")
            if number is not None:
                if type(number) is not int or number < 1:
                    raise ValueError("Invalid SDK request ID")
                if number == self.accepted + 1:
                    frame = request["frame"]
                    key = frame.get("llm", {}).get("api_key")
                    if isinstance(key, str) and key:
                        self.secrets.append(key)
                    self.inputs.put_nowait(frame)
                    self.accepted = number
                elif number > self.accepted:
                    raise ValueError("Out of order SDK request")
                # IDs at or below the watermark were already accepted, even if
                # the client lost the response. Never enqueue them a second time.
            if not self.outputs and not self.done:
                self.condition.wait(5 if number is None else 0.05)
            batch = []
            size = 0
            for sequence, encoded in self.outputs:
                length = len(encoded.encode())
                if size + length > BATCH_BYTES:
                    break
                batch.append([sequence, encoded])
                size += length
            return {"accepted": self.accepted, "outputs": batch, "done": self.done}


def daemon(path: str, lifetime: float) -> None:
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    log = TransportLog()
    log.record("daemon_started")

    def expire() -> None:
        log.record("lifetime_expired")
        os._exit(0)

    # The box lifetime is the final safety net if the controller disappears.
    timer = threading.Timer(lifetime, expire)
    timer.daemon = True
    timer.start()
    mailbox = Mailbox()

    def run() -> None:
        try:
            from _sail_sdk_worker import main

            log.record("worker_started")
            main(mailbox)
        except BaseException as exc:  # noqa: BLE001 — report worker termination
            log.record("worker_failed", exc)
            error = traceback.format_exc()
            for secret in mailbox.secrets:
                error = error.replace(secret, "[REDACTED]")
            mailbox.send(
                {
                    "kind": "error",
                    "error_type": "WorkerCrashed",
                    "iterations": 0,
                    "traceback": error,
                }
            )
        finally:
            log.record("worker_done")
            with mailbox.condition:
                mailbox.done = True
                mailbox.condition.notify_all()

    threading.Thread(target=run, name="sdk-conversation", daemon=True).start()
    try:
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(path)
            os.chmod(path, 0o600)
            listener.listen(4)
            log.record("listening")
            while True:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(30)
                    try:
                        stream = connection.makefile("rwb")
                        with stream:
                            line = stream.readline(MAX_REQUEST_BYTES + 1)
                            if len(line) > MAX_REQUEST_BYTES or not line.endswith(
                                b"\n"
                            ):
                                raise ValueError("Oversize or truncated request")
                            response = mailbox.exchange(json.loads(line))
                            encoded = json.dumps(response).encode()
                            stream.write(struct.pack("!I", len(encoded)) + encoded)
                            stream.flush()
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        TypeError,
                        queue.Full,
                    ) as exc:
                        log.record("request_failed", exc)
                        # A lost connection must not end the conversation or replay work.
                        continue
    except BaseException as exc:
        log.record("daemon_failed", exc)
        raise
    finally:
        log.record("daemon_stopped")


def start(path: str, lifetime: float) -> None:
    os.mkdir(os.path.dirname(path), mode=0o700)
    child = subprocess.Popen(
        [sys.executable, "-u", __file__, "daemon", path, str(lifetime)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    deadline = time.monotonic() + 10
    while True:
        if child.poll() is not None or time.monotonic() >= deadline:
            raise RuntimeError(f"SDK daemon failed to start: exit_code={child.poll()}")
        try:
            with socket.socket(socket.AF_UNIX) as connection:
                connection.connect(path)
            return
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.01)


def client(path: str) -> None:
    line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
    if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
        raise ValueError("Oversize or truncated request")
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(30)
        connection.connect(path)
        connection.sendall(line)
        with connection.makefile("rb") as stream:
            header = stream.read(4)
            if len(header) != 4:
                raise ValueError("Truncated SDK response header")
            length = struct.unpack("!I", header)[0]
            if length > 8 * BATCH_BYTES:
                raise ValueError("Oversize SDK response")
            response = stream.read(length)
        if len(response) != length:
            raise ValueError("Truncated SDK response")
        sys.stdout.buffer.write(response + b"\n")
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    mode, path = sys.argv[1:3]
    if mode == "start":
        start(path, float(sys.argv[3]))
    elif mode == "daemon":
        daemon(path, float(sys.argv[3]))
    elif mode == "client":
        client(path)
    else:
        raise ValueError("Invalid SDK transport mode")
