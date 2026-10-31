"""Lightweight controller for a persistent SDK worker in a separate Sail VM."""

from __future__ import annotations

import io
import json
import math
import os
import queue
import re
import shlex
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pyforger._sail_sdk_transport import (
    PROCESS_LOG_PATH,
    TRANSPORT_LOG_PATH,
    diagnostic_text,
)
from pyforger._sdk_install import (
    SDK_COMMIT,
    SDK_REQUIREMENT,
    SDK_VERSION,
    TOOLS_REQUIREMENT,
    sdk_provenance_command,
)
from pyforger.agent_diagnostics import credential_values, sanitizer
from pyforger.evidence_journal import (
    EvidenceJournal,
    evidence_file_chunks,
    iter_trajectory_events,
)
from pyforger.measurements import Measurement
from pyforger.sail_provision import create_sailbox
from pyforger.sdk_config import llm_options
from pyforger.structured import generate_validated
from pyforger.usage import combine_usage, usage_record
from pyforger.utils import redact

MAX_FRAME_BYTES = 1024 * 1024
# Keep write health independent of model latency and the overall session budget.
# A timeout is only meaningful while stdout is also being consumed.
STDIN_WRITE_TIMEOUT_S = 60.0
# Bound on frames queued to a guest that has stopped reading. Reaching it means
# the worker is not draining, so the controller must stop accepting output.
OUTBOUND_QUEUE_FRAMES = 64
PROTOCOL_TIMELINE_FRAMES = 128
# The worker redirects SDK stdout/stderr here, away from protocol frames.
WORKER_LOG_PATH = "/tmp/forger-worker.log"
WORKER_LOG_TAIL_CHARS = 8000


def worker_request(spec: Any, timeout: float) -> dict[str, Any]:
    """Resolve only the intended model credential, without SDK construction."""
    options = llm_options(spec.model)
    if spec.api_key_env:
        options["api_key"] = os.environ[spec.api_key_env]
    options["timeout"] = max(1, math.ceil(min(options["timeout"], timeout)))
    options["num_retries"] = 5
    return {
        "timeout_s": timeout,
        "redaction_secrets": [
            value
            for key, value in os.environ.items()
            if value
            and re.search(
                r"api.?key|token|secret|password|authorization|credential", key, re.I
            )
        ],
        "llm": options,
        "max_iterations": spec.max_iterations,
        "disable_prompt_cache_key": bool(
            spec.model._kwargs.get("disable_prompt_cache_key")
        ),
        "runtime": asdict(spec.runtime) if spec.runtime else None,
    }


def _frame_kind(value: dict[str, Any]) -> str:
    kind = value.get("kind", "request")
    return (
        kind
        if isinstance(kind, str)
        and kind
        in {
            "request",
            "turn",
            "configure",
            "configured",
            "ready",
            "close",
            "terminal_observation",
            "terminal",
            "event_chunk",
            "event_end",
            "result",
            "error",
        }
        else "unknown"
    )


class _FrameWriter:
    """Serialize frames to the guest's stdin off the reader thread.

    Waiting for write completion on the reader thread recreates the duplex
    deadlock: the guest can emit its next command or trajectory while the Sail
    write is still in flight. Queue sends without waiting, then poll their
    completion while consuming stdout so real stalls still fail promptly.
    """

    def __init__(self, stdin: Any, timeout_s: float = STDIN_WRITE_TIMEOUT_S) -> None:
        self._stdin = stdin
        self._timeout_s = timeout_s
        self._queue: queue.Queue[
            tuple[str, threading.Event, dict[str, Any]] | None
        ] = queue.Queue(maxsize=OUTBOUND_QUEUE_FRAMES)
        self._error: BaseException | None = None
        self._pending: deque[tuple[threading.Event, float, dict[str, Any]]] = deque()
        self._lock = threading.Lock()
        self._timeline: deque[dict[str, Any]] = deque(maxlen=PROTOCOL_TIMELINE_FRAMES)
        self._sequence = 0
        self._reader: dict[str, Any] = {
            "last_stdout_at": None,
            "last_stdout_bytes": None,
            "last_frame_kind": None,
            "last_frame_at": None,
            "reader_error_type": None,
        }
        self._reader_thread: threading.Thread | None = None
        self._output: queue.Queue[Any] | None = None
        self._thread = threading.Thread(
            target=self._drain, name="sail-stdin-writer", daemon=True
        )
        self._thread.start()

    def _drain(self) -> None:
        while True:
            value = self._queue.get()
            if value is None:
                return
            encoded, completed, metadata = value
            with self._lock:
                metadata["write_started_at"] = time.monotonic()
                metadata["queue_wait_s"] = (
                    metadata["write_started_at"] - metadata["enqueued_at"]
                )
            try:
                self._stdin.write(encoded)
            except BaseException as exc:  # recorded, re-raised on the caller
                with self._lock:
                    self._error = exc
                    metadata["error_type"] = type(exc).__name__
                return
            finally:
                with self._lock:
                    metadata["write_finished_at"] = time.monotonic()
                    metadata["write_duration_s"] = (
                        metadata["write_finished_at"] - metadata["write_started_at"]
                    )
                    metadata["completed"] = "error_type" not in metadata
                completed.set()

    def send(self, value: dict[str, Any]) -> None:
        """Queue without waiting: the calling thread must keep reading stdout."""
        self.check()
        completed = threading.Event()
        encoded = json.dumps(value) + "\n"
        enqueued = time.monotonic()
        with self._lock:
            self._sequence += 1
            metadata = {
                "direction": "stdin",
                "sequence": self._sequence,
                "kind": _frame_kind(value),
                "encoded_bytes": len(encoded.encode()),
                "enqueued_at": enqueued,
            }
            self._timeline.append(metadata)
        try:
            self._queue.put_nowait((encoded, completed, metadata))
        except queue.Full:
            with self._lock:
                metadata["queue_rejected"] = True
            self._raise_with_snapshot(
                TimeoutError(
                    "stdin write timed out: guest stopped draining the protocol pipe"
                )
            )
        self._pending.append((completed, enqueued, metadata))

    def check(self) -> None:
        """Surface failed or stalled writes without stopping stdout consumption."""
        if self._error is not None:
            self._raise_with_snapshot(self._error)
        while self._pending and self._pending[0][0].is_set():
            self._pending.popleft()
        if self._pending and time.monotonic() - self._pending[0][1] >= self._timeout_s:
            self._raise_with_snapshot(
                TimeoutError("stdin write timed out: protocol write did not complete")
            )

    def _raise_with_snapshot(self, error: BaseException) -> None:
        error.protocol_snapshot = self.snapshot()  # type: ignore[attr-defined]
        raise error

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        pending = [
            (started, metadata)
            for done, started, metadata in self._pending
            if not done.is_set()
        ]
        with self._lock:
            return {
                "snapshot_at": now,
                "timeline": [dict(item) for item in self._timeline],
                "pending_count": len(pending),
                "queue_count": self._queue.qsize(),
                "oldest_pending_age_s": now - pending[0][0] if pending else None,
                "pending_frames": [dict(metadata) for _, metadata in pending],
                "writer_alive": self._thread.is_alive(),
                "writer_error_type": type(self._error).__name__
                if self._error
                else None,
                "reader_alive": self._reader_thread.is_alive()
                if self._reader_thread
                else None,
                "reader_backlog": self._output.qsize()
                if self._output is not None
                else 0,
                **self._reader,
            }

    def close(self) -> None:
        """Stop the writer without waiting on a wedged guest."""
        if self._thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass


def frames(chunks: Any, *, on_frame: Any = None) -> Any:
    """Decode bounded JSON lines from arbitrary Sail stream boundaries."""
    pending = ""
    for chunk in chunks:
        pending += chunk
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            if len(line.encode()) > MAX_FRAME_BYTES:
                raise ValueError("Oversize SDK protocol frame")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Invalid SDK protocol frame")
            if on_frame is not None:
                on_frame(value, len(line.encode()) + 1)
            yield value
        if len(pending.encode()) > MAX_FRAME_BYTES:
            raise ValueError("Oversize SDK protocol frame")
    if pending:
        raise ValueError("Truncated SDK protocol frame")


def _incoming_frames(chunks: Any, writer: _FrameWriter, budget: Any) -> Any:
    """Poll write health even when the guest emits nothing after a failed send."""
    output: queue.Queue[Any] = queue.Queue(maxsize=64)
    stopped = threading.Event()

    def put(value: Any) -> None:
        while not stopped.is_set():
            try:
                output.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def read() -> None:
        try:
            for chunk in chunks:
                if stopped.is_set():
                    return
                with writer._lock:
                    writer._reader.update(
                        last_stdout_at=time.monotonic(),
                        last_stdout_bytes=len(chunk.encode()),
                    )
                put(chunk)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the controller
            with writer._lock:
                writer._reader["reader_error_type"] = type(exc).__name__
            put(exc)
        finally:
            put(None)

    thread = threading.Thread(target=read, name="sail-stdout-reader", daemon=True)
    writer._reader_thread = thread
    writer._output = output
    thread.start()

    def checked_chunks() -> Any:
        while True:
            writer.check()
            budget.remaining_s()
            try:
                value = output.get(timeout=0.1)
            except queue.Empty:
                continue
            if value is None:
                return
            if isinstance(value, BaseException):
                raise value
            yield value

    sequence = 0

    def record_frame(frame: dict[str, Any], encoded_bytes: int) -> None:
        nonlocal sequence
        sequence += 1
        now = time.monotonic()
        with writer._lock:
            writer._reader.update(last_frame_kind=_frame_kind(frame), last_frame_at=now)
            writer._timeline.append(
                {
                    "direction": "stdout",
                    "sequence": sequence,
                    "kind": _frame_kind(frame),
                    "received_at": now,
                    "encoded_bytes": encoded_bytes,
                }
            )

    try:
        yield from frames(checked_chunks(), on_frame=record_frame)
    finally:
        stopped.set()


class _DetachedConnection:
    """Adapt short-lived Sail socket clients to the existing duplex frame reader."""

    def __init__(self, worker: Any, budget: Any) -> None:
        self.worker = worker
        self.budget = budget
        self.pending: queue.Queue[str] = queue.Queue(maxsize=OUTBOUND_QUEUE_FRAMES)
        self.stopped = threading.Event()
        self.stdin = self
        self.stdout = self.read()
        self.requests = 0
        self.reconnects = 0
        self.failures: deque[dict[str, Any]] = deque(maxlen=PROTOCOL_TIMELINE_FRAMES)
        self.secrets: list[str] = []

    def write(self, encoded: str) -> None:
        self.pending.put_nowait(encoded)

    @staticmethod
    def argv(mode: str, *args: str) -> list[str]:
        return [
            "/usr/bin/env",
            "-i",
            "PATH=/usr/local/bin:/usr/bin:/bin",
            "HOME=/tmp",
            "HF_HUB_OFFLINE=1",
            "python3",
            "-u",
            "/tmp/_sail_sdk_transport.py",
            mode,
            "/tmp/forger-sdk/socket",
            *args,
        ]

    def read(self) -> Any:
        number = 0
        ack = 0
        awaiting = False
        while not self.stopped.is_set():
            envelope: dict[str, Any] = {"ack": ack}
            try:
                encoded = self.pending.get(timeout=0.1 if not awaiting else 0.05)
            except queue.Empty:
                if not awaiting:
                    continue
            else:
                awaiting = True
                number += 1
                frame = json.loads(encoded)
                key = frame.get("llm", {}).get("api_key")
                if isinstance(key, str) and key:
                    self.secrets.append(key)
                self.secrets.extend(
                    value
                    for value in frame.get("redaction_secrets", ())
                    if isinstance(value, str) and value
                )
                envelope.update(id=number, frame=frame)
            payload = json.dumps(envelope) + "\n"
            # Retry the SAME ID/cursor against the same daemon, never the SDK turn.
            for attempt in range(3):
                self.budget.remaining_s()
                exit_code = None
                stderr = ""
                try:
                    process = self.worker.exec(
                        self.argv("client"),
                        user="65534",
                        open_stdin=True,
                        timeout=max(1, min(40, math.ceil(self.budget.remaining_s()))),
                        output_buffer_bytes=MAX_FRAME_BYTES,
                    )
                    process.stdin.write(payload)
                    process.stdin.close()
                    raw = "".join(process.stdout)
                    result = process.wait()
                    exit_code = result.exit_code
                    if exit_code != 0:
                        stderr = diagnostic_text(
                            redact(getattr(result, "stderr", "")), self.secrets
                        )
                        raise ConnectionError(
                            f"SDK socket client failed: exit_code={exit_code}; stderr={stderr}"
                        )
                    response = json.loads(raw)
                    if envelope.get("id", 0) > response["accepted"]:
                        raise ValueError("SDK request not acknowledged")
                    self.requests += 1
                    break
                except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
                    self.failures.append(
                        {
                            "request_id": number,
                            "ack": ack,
                            "attempt": attempt + 1,
                            "at": time.monotonic(),
                            "exit_code": exit_code,
                            "stderr": stderr,
                            "error_type": type(exc).__name__,
                            "error_message": diagnostic_text(
                                redact(str(exc))
                                if isinstance(exc, (OSError, EOFError))
                                else type(exc).__name__,
                                self.secrets,
                            ),
                        }
                    )
                    if (
                        not isinstance(exc, (OSError, EOFError))
                        or attempt == 2
                        or self.stopped.is_set()
                    ):
                        raise
                    self.reconnects += 1
                    time.sleep(0.1)
            for sequence, encoded in response["outputs"]:
                if sequence != ack + 1:
                    raise ValueError("Invalid SDK output sequence")
                ack = sequence
                kind = json.loads(encoded).get("kind")
                if kind in {
                    "ready",
                    "configured",
                    "result",
                    "error",
                    "terminal",
                    "terminal_observation",
                    "closed",
                }:
                    awaiting = False
                yield encoded
            if response["done"] and not response["outputs"]:
                return

    def close(self) -> None:
        self.stopped.set()


class _TerminalConnection(_DetachedConnection):
    @staticmethod
    def argv(mode: str, *args: str) -> list[str]:
        argv = _DetachedConnection.argv(mode, *args)
        argv.insert(3, "FORGER_TASK_TERMINAL=1")
        return argv


def _retain_worker_log(
    worker: Any, diagnostics: dict[str, Any], attempt_record: dict[str, Any]
) -> None:
    """Retain the full worker log before terminating its box, plus a UI tail."""
    try:

        def chunks():
            # Worker RedactedLog sanitizes before persistence, including forwarded
            # service/environment credentials registered by the private mailbox.
            if hasattr(worker.fs, "read_stream"):
                yield from worker.fs.read_stream(WORKER_LOG_PATH)
            else:
                yield worker.fs.read(WORKER_LOG_PATH)
            if "protocol" in attempt_record:
                yield (
                    "\nHost protocol metadata: "
                    + json.dumps(attempt_record["protocol"])
                    + "\n"
                ).encode()

        log_ref, tail = evidence_file_chunks(chunks())
        attempt_record["worker_log"] = log_ref
        diagnostics["sdk_worker_log"] = log_ref
        attempt_record["worker_log_tail"] = tail
        diagnostics["sdk_worker_log_tail"] = tail
    except Exception as exc:
        diagnostics.setdefault("sdk_worker_log_errors", []).append(
            {"error_type": type(exc).__name__, "error_message": str(exc)}
        )


def _recover_current_events(
    worker: Any, current: dict[str, Any], journal: EvidenceJournal | None = None
) -> None:
    """Recover a bounded private worker suffix using sequence identity, not content."""
    result: queue.Queue[Any] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            result.put(worker.fs.read(WORKER_LOG_PATH + ".events"))
        except Exception as exc:
            result.put(type(exc).__name__)

    threading.Thread(target=read, name="sdk-event-recovery", daemon=True).start()
    try:
        raw = result.get(timeout=3)
    except queue.Empty:
        current["recovery"] = "read_deadline_exceeded"
        return
    if not isinstance(raw, bytes):
        current["recovery"] = "unavailable:" + str(raw)
        return
    matched = False
    existing = (
        iter(iter_trajectory_events(current)) if journal is not None else iter(())
    )
    existing_sequence = -1
    existing_event = None
    for line in raw[: 16 * 1024 * 1024 + 256].splitlines():
        try:
            record = json.loads(line)
        except (ValueError, UnicodeError):
            current["recovery_truncated"] = True
            continue
        if record.get("kind") == "omitted":
            current["recovery_truncated"] = True
        if (
            not current.get("turn_id")
            or record.get("turn_id") != current["turn_id"]
            or record.get("call_id") != current.get("call_id")
        ):
            continue
        matched = True
        current["conversation_id"] = record.get("conversation_id")
        current["last_event_at"] = record.get("at")
        if record.get("kind") == "event":
            sequence = record.get("sequence")
            if (
                journal is not None
                and type(sequence) is int
                and sequence <= current.get("last_sequence", -1)
            ):
                while existing_sequence < sequence:
                    existing_event = next(existing, None)
                    existing_sequence += 1
                if existing_event != record["event"]:
                    current["recovery"] = "conflicting_sequence"
                    current["sequence_error"] = {"conflict": sequence}
                    return
            if (
                journal is not None
                and isinstance(sequence, int)
                and sequence > current.get("last_sequence", -1)
            ):
                if (
                    "last_sequence" in current
                    and sequence != current["last_sequence"] + 1
                ):
                    current["sequence_error"] = {
                        "expected": current["last_sequence"] + 1,
                        "received": sequence,
                    }
                    current["recovery"] = "sequence_gap"
                    return
                if record["event"].get("kind") == "ConversationErrorEvent":
                    current["last_error_code"] = record["event"].get("code")
                journal.append_sanitized(record["event"])
                current.update(journal.trajectory())
                current["last_sequence"] = sequence
            elif journal is None:
                # Legacy workers cannot prove duplicate identity; preserve every record.
                current.setdefault("steps", []).append(record["event"])
    current["recovery"] = (
        "current_turn_journal" if matched else "no_matching_current_turn_journal"
    )


@contextmanager
def remote_sdk_session(
    backend: Any,
    transport: Any,
    task: Any,
    spec: Any,
    validate: Any,
    max_repairs: int,
    diagnostics: dict[str, Any],
    *,
    require_outputs: bool = True,
) -> Any:
    """Own one real remote conversation across multiple host generation calls."""
    import sail

    budget = transport.budget
    redaction_specs = [spec]
    credentials = credential_values([spec])
    active_journal: EvidenceJournal | None = None
    current: dict[str, Any] = {
        "call_id": uuid.uuid4().hex,
        "turn_id": None,
        "conversation_id": None,
        "steps": [],
        "incomplete": True,
        "phase": "worker_setup",
        "started_at": time.time(),
        "no_current_history_reason": "session_not_started",
    }
    diagnostics["current_trajectory"] = current
    diagnostics["trajectory"] = current
    diagnostics.pop("conversation_id", None)
    try:
        request = worker_request(spec, budget.remaining_s())
    except BaseException as exc:
        current["captured_at"] = time.time()
        exc.diagnostics = dict(diagnostics)
        raise
    endpoint = request["llm"]["base_url"]
    if (
        not endpoint
        or urlsplit(endpoint).scheme != "https"
        or not urlsplit(endpoint).hostname
    ):
        raise ValueError("Remote Sail SDK requires an explicit HTTPS model endpoint")
    image = (
        sail.Image.from_registry(backend.image)
        .apt_install("git")
        .pip_install(
            SDK_REQUIREMENT,
            TOOLS_REQUIREMENT,
            "transformers==5.20.2",
            "mcp==1.30.0",
            "fastmcp==3.4.8",
            "litellm==1.103.4",
        )
        .run_commands(sdk_provenance_command("python3"))
    )
    tokenizer = request["llm"]["custom_tokenizer"]
    if tokenizer:
        # Download only public tokenizer assets at build time, without credentials.
        patterns = (
            "['*token*','*vocab*','merges.txt','config.json',"
            "'*.model','*.tiktoken','*.jinja']"
        )
        script = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download({tokenizer!r}, local_dir='/opt/tokenizer', "
            f"allow_patterns={patterns}, token=False)"
        )
        image = image.run_commands("python3 -c " + shlex.quote(script))
        request["llm"]["custom_tokenizer"] = "/opt/tokenizer"

    def create_worker() -> Any:
        attempt = diagnostics.get("sdk_worker_attempts", 0) + 1
        diagnostics["sdk_worker_attempts"] = attempt
        instance = create_sailbox(
            sail.Sailbox.create,
            budget=budget,
            setup_times=diagnostics.setdefault("sdk_worker_setup_s", []),
            app=sail.App.find(backend.app, mint_if_missing=True),
            image=image,
            name="forger-sdk-" + uuid.uuid4().hex,
            memory_limit_gib=8,
            egress_policy={"allowlist": [urlsplit(endpoint).hostname]},
            max_lifetime_seconds=max(60, math.ceil(budget.remaining_s()) + 60),
        )
        diagnostics.update(
            sdk_sailbox_id=instance.sailbox_id,
            execution_mode="remote-sdk",
            sdk_version=SDK_VERSION,
            sdk_commit=SDK_COMMIT,
            sdk_cleanup="pending",
        )
        diagnostics["sdk_worker_sailbox_ids"] = diagnostics.get(
            "sdk_worker_sailbox_ids", []
        ) + [instance.sailbox_id]
        return instance

    try:
        worker = create_worker()
    except BaseException as exc:
        current["captured_at"] = time.time()
        exc.diagnostics = dict(diagnostics)
        raise
    writer: _FrameWriter | None = None
    attempt_record: dict[str, Any] = {}
    incoming: Any = None
    connection: _DetachedConnection | None = None
    terminal_connection: _TerminalConnection | None = None
    try:
        for filename in ("_sail_sdk_transport.py", "_sail_terminal.py"):
            transport.box.fs.write(
                "/tmp/" + filename,
                io.BytesIO(Path(__file__).with_name(filename).read_bytes()),
                mode=0o644,
                user="root",
            )
        started_terminal = transport.box.run(
            [
                "setpriv",
                "--reuid=65534",
                "--regid=65534",
                "--clear-groups",
                "--no-new-privs",
                "--bounding-set=-all",
                "--inh-caps=-all",
                "--ambient-caps=-all",
                *_TerminalConnection.argv(
                    "start", str(max(1, math.ceil(budget.remaining_s())))
                ),
            ],
            user="root",
            timeout=20,
        )
        if started_terminal.exit_code:
            raise RuntimeError("Cannot start isolated native task terminal")
        terminal_connection = _TerminalConnection(transport.box, budget)
        terminal_incoming = iter(terminal_connection.stdout)
        terminal_connection.stdin.write(json.dumps({"kind": "initialize"}) + "\n")
        if json.loads(next(terminal_incoming)).get("kind") != "ready":
            raise RuntimeError("Native task terminal failed isolation/setup checks")
        for filename in (
            "_sail_sdk_worker.py",
            "_sail_sdk_transport.py",
            "_sdk_runtime.py",
            "usage.py",
            "agent_diagnostics.py",
        ):
            worker.fs.write(
                "/tmp/" + filename,
                io.BytesIO(Path(__file__).with_name(filename).read_bytes()),
                mode=0o644,
                user="root",
            )
        # Attempt-scoped trajectory: the retry path must not erase the evidence
        # of the attempt that stalled, so each attempt keeps its own record.
        attempt_index = diagnostics.get("sdk_worker_attempts", 1) - 1
        attempts = diagnostics.setdefault("sdk_attempts", [])
        while len(attempts) < attempt_index:
            attempts.append({})
        attempt_record = {
            "attempt": attempt_index + 1,
            "sailbox_id": worker.sailbox_id,
            "trajectory": {"turns": []},
            "stdin_write_stalls": 0,
        }
        attempts.append(attempt_record)
        attempt_record["turns"] = attempt_record["trajectory"]["turns"]
        lifetime = max(1, math.ceil(budget.remaining_s()))
        started = worker.run(
            _DetachedConnection.argv("start", str(lifetime)),
            user="65534",
            timeout=min(20, lifetime),
        )
        if started.exit_code != 0:
            stderr = diagnostic_text(
                redact(started.stderr), [request["llm"].get("api_key", "")]
            )
            attempt_record["transport_start_failure"] = {
                "exit_code": started.exit_code,
                "stderr": stderr,
            }
            raise RuntimeError(
                f"Remote SDK daemon failed to start: exit_code={started.exit_code}; "
                f"stderr={stderr}"
            )
        connection = _DetachedConnection(worker, budget)
        writer = _FrameWriter(connection.stdin)
        incoming = _incoming_frames(connection.stdout, writer, budget)
        event_parts: list[str] = []
        usage_parts: list[dict[str, Any]] = []

        def send(value: dict[str, Any]) -> None:
            budget.remaining_s()
            writer.send(value)

        def handle_error(frame: dict[str, Any]) -> None:
            if "usage" in frame:
                usage_parts.append(frame["usage"])
                diagnostics["usage"] = combine_usage(
                    usage_parts, call_id=current["call_id"]
                )
            budget.iterations = frame["iterations"]
            diagnostics["budget_exhausted"] = frame.get("budget_exhausted", False)
            diagnostics["execution_status"] = "error"
            frame = sanitizer([], secrets=credentials)(frame)
            attempt_record["error"] = frame
            diagnostics["sdk_error"] = frame
            message = "Remote SDK failed: " + frame["error_type"]
            if frame.get("error_message"):
                message += ": " + frame["error_message"]
            if frame.get("traceback"):
                message += "\n" + frame["traceback"]
            raise RuntimeError(message)

        request["timeout_s"] = budget.remaining_s()
        send(request)
        del request

        budget.remaining_s()
        ready_frame = next(incoming, None)
        if ready_frame is None:
            raise RuntimeError("Remote SDK disconnected")
        if ready_frame.get("kind") == "error":
            handle_error(ready_frame)
        if ready_frame.get("kind") != "ready":
            raise ValueError("Unknown remote SDK protocol frame")

        diagnostics["conversation_id"] = ready_frame.get("conversation_id")
        current["conversation_id"] = ready_frame.get("conversation_id")

        def generate(instruction: str) -> dict[str, bytes]:
            nonlocal current, active_journal
            current = {
                "call_id": call_id,
                "turn_id": uuid.uuid4().hex,
                "conversation_id": diagnostics.get("conversation_id"),
                "steps": [],
                "incomplete": True,
                "phase": "agent_turn",
                "started_at": time.time(),
            }
            journal = EvidenceJournal(redaction_specs)
            active_journal = journal
            current.pop("steps")
            current["last_sequence"] = -1
            current.update(journal.trajectory())
            diagnostics["current_trajectory"] = current
            diagnostics["trajectory"] = current
            attempt_record["turns"].append(current)
            diagnostics["usage"] = combine_usage(
                [*usage_parts, usage_record()], call_id=call_id
            )
            send(
                {
                    "kind": "turn",
                    "call_id": call_id,
                    "turn_id": current["turn_id"],
                    "instruction": instruction,
                    "timeout_s": budget.remaining_s(),
                }
            )
            diagnostics["execution_status"] = "running"
            for frame in incoming:
                budget.remaining_s()
                kind = frame.get("kind")
                if kind == "event_chunk":
                    event_parts.append(frame["text"])
                elif kind == "event_end":
                    sequence = frame.get("sequence")
                    expected = current.get("last_sequence", -1) + 1
                    if type(sequence) is not int or sequence != expected:
                        current["sequence_error"] = {
                            "expected": expected,
                            "received": sequence,
                        }
                        raise RuntimeError("Out-of-order SDK event sequence")
                    event = json.loads("".join(event_parts))
                    journal.append_sanitized(event)
                    if event.get("kind") == "ConversationErrorEvent":
                        current["last_error_code"] = event.get("code")
                    current.update(journal.trajectory())
                    current["last_sequence"] = frame.get(
                        "sequence", current.get("last_sequence", -1) + 1
                    )
                    current["last_event_at"] = time.time()
                    event_parts.clear()
                elif kind == "terminal":
                    current["phase"] = "terminal"
                    current["last_activity_at"] = time.time()
                    terminal_connection.stdin.write(json.dumps(frame) + "\n")
                    reply = json.loads(next(terminal_incoming))
                    if reply.get("kind") != "terminal_observation":
                        raise RuntimeError("Task terminal failed")
                    current["phase"] = "agent_turn"
                    send(reply)
                elif kind == "error":
                    handle_error(frame)
                elif kind == "result":
                    if "usage" in frame:
                        usage_parts.append(frame["usage"])
                        diagnostics["usage"] = combine_usage(
                            usage_parts, call_id=current["call_id"]
                        )
                    budget.iterations = frame["iterations"]
                    diagnostics["budget_exhausted"] = (
                        frame["status"] != "finished"
                        and budget.max_iterations is not None
                        and budget.iterations >= budget.max_iterations
                    )
                    diagnostics["execution_status"] = frame["status"]
                    diagnostics["conversation_id"] = frame["conversation_id"]
                    current.update(
                        conversation_id=frame["conversation_id"],
                        incomplete=frame["status"] != "finished",
                        phase="result",
                        captured_at=time.time(),
                    )
                    diagnostics.setdefault("snapshots", []).append(
                        {
                            "conversation_id": frame["conversation_id"],
                            # The worker ran the session, so record its box, not
                            # the transport box that only holds the workspace.
                            "sailbox_id": worker.sailbox_id,
                            "events": frame["events"],
                        }
                    )
                    if frame["status"] != "finished":
                        raise RuntimeError("Remote SDK did not finish")
                    return dict(
                        transport.files(
                            {name: b"" for name in task.outputs}, write=False
                        )
                    )
                else:
                    raise ValueError("Unknown remote SDK protocol frame")
            raise RuntimeError("Remote SDK disconnected")

        def check(files: dict[str, bytes]) -> Measurement:
            current["phase"] = "host_validation"
            current["last_activity_at"] = time.time()
            missing = [name for name in task.outputs if name not in files]
            value = (
                Measurement(passed=False, reason=f"Missing declared outputs: {missing}")
                if missing and require_outputs
                else validate(files)
            )
            budget.remaining_s()
            return value

        def run_session() -> tuple[Any, Any]:
            if validate is None:
                return generate(task.instruction), ()
            result = generate_validated(
                task.instruction,
                generate=generate,
                validate=check,
                revise=lambda request, attempt: (
                    "Repair the outputs in the same workspace. Validation feedback:\n"
                    + attempt.measurement.model_dump_json()
                ),
                max_attempts=max_repairs + 1,
            )
            return result.output, result.attempts

        def run(
            next_task: Any,
            next_validate: Any,
            next_max_repairs: int,
            next_spec: Any = None,
        ) -> Any:
            nonlocal task, validate, max_repairs, spec, call_id, current
            usage_parts.clear()
            call_id = uuid.uuid4().hex
            diagnostics["usage"] = usage_record({"call_id": call_id})
            current = {
                "call_id": call_id,
                "turn_id": None,
                "conversation_id": diagnostics.get("conversation_id"),
                "steps": [],
                "incomplete": True,
                "phase": "configure",
                "started_at": time.time(),
                "no_current_history_reason": "turn_not_started",
            }
            diagnostics["current_trajectory"] = current
            diagnostics["trajectory"] = current
            task, validate, max_repairs = next_task, next_validate, next_max_repairs
            if next_spec is not None:
                credentials.update(credential_values([next_spec]))
                redaction_specs.append(next_spec)
                updated = worker_request(next_spec, budget.remaining_s())
                updated_endpoint = urlsplit(updated["llm"]["base_url"] or "")
                if (
                    updated_endpoint.scheme != "https"
                    or updated_endpoint.hostname != urlsplit(endpoint).hostname
                ):
                    raise ValueError(
                        "Reusable Sail session cannot change model endpoint host"
                    )
                if updated["llm"]["custom_tokenizer"] != tokenizer:
                    raise ValueError(
                        "Reusable Sail session cannot change custom tokenizer"
                    )
                if tokenizer:
                    updated["llm"]["custom_tokenizer"] = "/opt/tokenizer"
                send(
                    {
                        "kind": "configure",
                        "llm": updated["llm"],
                        "redaction_secrets": updated["redaction_secrets"],
                        "disable_prompt_cache_key": updated["disable_prompt_cache_key"],
                    }
                )
                configured = next(incoming, None)
                if configured is None or configured.get("kind") != "configured":
                    if configured is not None and configured.get("kind") == "error":
                        handle_error(configured)
                    raise RuntimeError("Remote SDK model reconfiguration failed")
                spec = next_spec
            return run_session()

        call_id = current["call_id"]
        yield run
    finally:
        active_error = sys.exc_info()[1]
        if locals().get("event_parts"):
            current["partial_event_frame"] = True
        cleanup_errors: list[BaseException] = []

        def cleanup(label: str, action: Any) -> None:
            try:
                action()
            except BaseException as exc:
                cleanup_errors.append(exc)
                diagnostics.setdefault("sdk_cleanup_errors", []).append(
                    {
                        "operation": label,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        active_current = diagnostics.get("current_trajectory", current)
        if active_current.get("phase") == "call_setup":
            current = active_current
        cleanup(
            "current_events",
            lambda: _recover_current_events(worker, current, active_journal),
        )
        if active_error is not None:
            current["incomplete"] = True
        current["captured_at"] = time.time()
        if not current.get("journal", {}).get("count", len(current.get("steps", []))):
            current.setdefault(
                "no_current_history_reason", "no_current_events_recoverable"
            )
        else:
            current.pop("no_current_history_reason", None)
        diagnostics["current_trajectory"] = current
        diagnostics["trajectory"] = current
        if terminal_connection is not None:

            def close_terminal() -> None:
                terminal_connection.stdin.write(json.dumps({"kind": "close"}) + "\n")
                result: queue.Queue[Any] = queue.Queue(maxsize=1)

                def drain_terminal() -> None:
                    try:
                        for encoded in terminal_incoming:
                            if json.loads(encoded).get("kind") == "closed":
                                result.put(True)
                                return
                    except (OSError, ValueError, StopIteration):
                        pass
                    result.put(False)

                threading.Thread(target=drain_terminal, daemon=True).start()
                try:
                    diagnostics["terminal_cleanup_acknowledged"] = result.get(timeout=3)
                except queue.Empty:
                    diagnostics["terminal_cleanup_acknowledged"] = False
                finally:
                    terminal_connection.close()

            cleanup("terminal.close", close_terminal)
        if connection is not None:
            diagnostics["sdk_transport"] = {
                "mode": "detached-socket",
                "requests": connection.requests,
                "reconnects": connection.reconnects,
                "failures": list(connection.failures),
            }
            attempt_record["transport"] = diagnostics["sdk_transport"]
            cleanup("connection.close", connection.close)
        if writer is not None:
            snapshot = (
                getattr(active_error, "protocol_snapshot", None) or writer.snapshot()
            )
            attempt_record["protocol"] = snapshot
            diagnostics["sdk_protocol"] = snapshot
            cleanup("writer.close", writer.close)
        if incoming is not None:
            cleanup("incoming.close", incoming.close)
        cleanup(
            "worker_log",
            lambda: _retain_worker_log(worker, diagnostics, attempt_record),
        )

        def retain_transport_log(path: str, key: str) -> None:
            try:
                raw = worker.fs.read(path)
            except Exception as exc:  # noqa: BLE001 — diagnostics must not fail cleanup
                attempt_record[key.removesuffix("_tail") + "_error_type"] = type(
                    exc
                ).__name__
                return
            text = diagnostic_text(
                redact(raw.decode("utf-8", errors="replace")),
                connection.secrets if connection is not None else [],
            )
            attempt_record[key] = text
            diagnostics["sdk_" + key] = text

        for path, key in (
            (TRANSPORT_LOG_PATH, "transport_log_tail"),
            (PROCESS_LOG_PATH, "process_log_tail"),
        ):
            cleanup(key, lambda path=path, key=key: retain_transport_log(path, key))

        def terminate() -> None:
            try:
                worker.terminate()
                diagnostics["sdk_cleanup"] = "terminated"
            except BaseException as exc:
                diagnostics["sdk_cleanup"] = "failed"
                diagnostics["sdk_cleanup_error_type"] = type(exc).__name__
                diagnostics["sdk_cleanup_error"] = str(exc)
                raise

        cleanup("worker.terminate", terminate)
        if connection is not None:
            credentials.update(connection.secrets)
        safe = sanitizer([], secrets=credentials)(diagnostics)
        diagnostics.clear()
        diagnostics.update(safe)
        if active_error is not None:
            existing = getattr(active_error, "diagnostics", {})
            if existing.get("current_trajectory", {}).get("call_id") not in (
                None,
                current["call_id"],
            ):
                existing.setdefault("enclosing_sessions", []).append(dict(diagnostics))
            else:
                active_error.diagnostics = {**existing, **diagnostics}
        elif cleanup_errors:
            cleanup_errors[0].diagnostics = dict(diagnostics)
            raise cleanup_errors[0]


def _run_remote_sdk_once(
    backend: Any,
    transport: Any,
    task: Any,
    spec: Any,
    validate: Any,
    max_repairs: int,
    diagnostics: dict[str, Any],
    *,
    require_outputs: bool = True,
) -> tuple[Any, Any]:
    with remote_sdk_session(
        backend,
        transport,
        task,
        spec,
        validate,
        max_repairs,
        diagnostics,
        require_outputs=require_outputs,
    ) as run:
        return run(task, validate, max_repairs)


def run_remote_sdk(
    backend: Any,
    transport: Any,
    task: Any,
    spec: Any,
    validate: Any,
    max_repairs: int,
    diagnostics: dict[str, Any],
    *,
    require_outputs: bool = True,
) -> tuple[Any, Any]:
    """Run one conversation; transport retries reconnect only to its daemon."""
    return _run_remote_sdk_once(
        backend,
        transport,
        task,
        spec,
        validate,
        max_repairs,
        diagnostics,
        require_outputs=require_outputs,
    )
