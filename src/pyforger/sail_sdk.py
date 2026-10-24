"""Lightweight controller for a persistent SDK worker in a separate Sail VM."""

from __future__ import annotations

import io
import json
import math
import os
import queue
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

from pyforger.measurements import Measurement
from pyforger.sdk_config import llm_options
from pyforger.structured import generate_validated
from pyforger.utils import redact

SDK_VERSION = "1.58.0"
# Pin the OpenRouter affinity fix and safe retry diagnostics while awaiting
# released SDK versions containing both changes.
SDK_REQUIREMENT = (
    "openhands-sdk @ git+https://github.com/neubig/software-agent-sdk.git"
    "@3e83b0734add43f40b45f89f16cdfb75356ff01e#subdirectory=openhands-sdk"
)
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
            "observation",
            "command",
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
        self._queue: queue.Queue[tuple[str, threading.Event, dict[str, Any]] | None] = (
            queue.Queue(maxsize=OUTBOUND_QUEUE_FRAMES)
        )
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
                envelope.update(id=number, frame=json.loads(encoded))
            payload = json.dumps(envelope) + "\n"
            # Retry the SAME ID/cursor against the same daemon, never the SDK turn.
            for attempt in range(3):
                self.budget.remaining_s()
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
                    if result.exit_code != 0:
                        raise ConnectionError("SDK socket client failed")
                    response = json.loads(raw)
                    if envelope.get("id", 0) > response["accepted"]:
                        raise ValueError("SDK request not acknowledged")
                    self.requests += 1
                    break
                except (OSError, EOFError):
                    if attempt == 2 or self.stopped.is_set():
                        raise
                    self.reconnects += 1
                    time.sleep(0.1)
            for sequence, encoded in response["outputs"]:
                if sequence != ack + 1:
                    raise ValueError("Invalid SDK output sequence")
                ack = sequence
                kind = json.loads(encoded).get("kind")
                if kind in {"ready", "configured", "result", "error", "command"}:
                    awaiting = False
                yield encoded
            if response["done"] and not response["outputs"]:
                return

    def close(self) -> None:
        self.stopped.set()


def _retain_worker_log(
    worker: Any, diagnostics: dict[str, Any], attempt_record: dict[str, Any]
) -> None:
    """Retain the full worker log before terminating its box, plus a UI tail."""
    try:
        raw = worker.fs.read(WORKER_LOG_PATH)
    except Exception as exc:
        diagnostics.setdefault("sdk_worker_log_errors", []).append(
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return
    text = redact(raw.decode("utf-8", errors="replace"))
    if "protocol" in attempt_record:
        text += (
            "\nHost protocol metadata: " + json.dumps(attempt_record["protocol"]) + "\n"
        )
    attempt_record["worker_log"] = text
    diagnostics["sdk_worker_log"] = text
    attempt_record["worker_log_tail"] = text[-WORKER_LOG_TAIL_CHARS:]
    diagnostics["sdk_worker_log_tail"] = text[-WORKER_LOG_TAIL_CHARS:]


@contextmanager
def remote_sdk_session(
    backend: Any,
    transport: Any,
    task: Any,
    spec: Any,
    validate: Any,
    max_repairs: int,
    diagnostics: dict[str, Any],
) -> Any:
    """Own one real remote conversation across multiple host generation calls."""
    import sail

    budget = transport.budget
    redaction_specs = [spec]
    request = worker_request(spec, budget.remaining_s())
    endpoint = request["llm"]["base_url"]
    if (
        not endpoint
        or urlsplit(endpoint).scheme != "https"
        or not urlsplit(endpoint).hostname
    ):
        raise ValueError("Remote Sail SDK requires an explicit HTTPS model endpoint")
    image = sail.Image.from_registry(backend.image).pip_install(
        SDK_REQUIREMENT,
        "transformers==5.20.2",
        "mcp==1.30.0",
        "fastmcp==3.4.8",
        "litellm==1.103.4",
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
        started = time.monotonic()
        instance = sail.Sailbox.create(
            app=sail.App.find(backend.app, mint_if_missing=True),
            image=image,
            image_build_timeout=max(1, math.ceil(budget.remaining_s())),
            name="forger-sdk-" + uuid.uuid4().hex,
            memory_limit_gib=8,
            egress_policy={"allowlist": [urlsplit(endpoint).hostname]},
            timeout=budget.remaining_s(),
            max_lifetime_seconds=max(60, math.ceil(budget.remaining_s()) + 60),
        )
        diagnostics.update(
            sdk_sailbox_id=instance.sailbox_id,
            execution_mode="remote-sdk",
            sdk_version=SDK_VERSION,
            sdk_cleanup="pending",
        )
        # Provisioning is infrastructure, not agent work: credit it back so a
        # slow image build cannot eat the time the agent needs to do the task.
        diagnostics.setdefault("sdk_worker_setup_s", []).append(
            round(budget.exempt_elapsed_s(started), 3)
        )
        diagnostics["sdk_worker_sailbox_ids"] = diagnostics.get(
            "sdk_worker_sailbox_ids", []
        ) + [instance.sailbox_id]
        return instance

    worker = create_worker()
    writer: _FrameWriter | None = None
    attempt_record: dict[str, Any] = {}
    incoming: Any = None
    connection: _DetachedConnection | None = None
    try:
        for filename in (
            "_sail_sdk_worker.py",
            "_sail_sdk_transport.py",
            "_sdk_runtime.py",
            "agent_diagnostics.py",
        ):
            worker.fs.write(
                "/tmp/" + filename,
                io.BytesIO(Path(__file__).with_name(filename).read_bytes()),
                mode=0o644,
                user="root",
            )
        lifetime = max(1, math.ceil(budget.remaining_s()))
        started = worker.run(
            _DetachedConnection.argv("start", str(lifetime)),
            user="65534",
            timeout=min(20, lifetime),
        )
        if started.exit_code != 0:
            raise RuntimeError("Remote SDK daemon failed to start")
        connection = _DetachedConnection(worker, budget)
        # Attempt-scoped trajectory: the retry path must not erase the evidence
        # of the attempt that stalled, so each attempt keeps its own record.
        attempt_index = diagnostics.get("sdk_worker_attempts", 1) - 1
        attempts = diagnostics.setdefault("sdk_attempts", [])
        while len(attempts) < attempt_index:
            attempts.append({})
        attempt_record = {
            "attempt": attempt_index + 1,
            "sailbox_id": worker.sailbox_id,
            "trajectory": {"steps": []},
            "stdin_write_stalls": 0,
        }
        attempts.append(attempt_record)
        # Downstream readers (sail_backend) read diagnostics["trajectory"], so
        # keep it pointing at the live attempt while the per-attempt record
        # preserves what a later retry would otherwise erase.
        diagnostics["trajectory"] = attempt_record["trajectory"]
        writer = _FrameWriter(connection.stdin)
        incoming = _incoming_frames(connection.stdout, writer, budget)
        event_parts: list[str] = []

        def send(value: dict[str, Any]) -> None:
            budget.remaining_s()
            writer.send(value)

        def handle_error(frame: dict[str, Any]) -> None:
            budget.iterations = frame["iterations"]
            diagnostics["budget_exhausted"] = frame.get("budget_exhausted", False)
            diagnostics["execution_status"] = "error"
            from pyforger.agent_diagnostics import trajectory

            for redaction_spec in redaction_specs:
                frame = trajectory([frame], redaction_spec)["steps"][0]
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

        def generate(instruction: str) -> dict[str, bytes]:
            send(
                {
                    "kind": "turn",
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
                    attempt_record["trajectory"]["steps"].append(
                        json.loads("".join(event_parts))
                    )
                    event_parts.clear()
                elif kind == "command":
                    result = transport.command(frame["command"])
                    send(
                        {
                            "kind": "observation",
                            "text": (
                                f"exit_code={result.exit_code}\nstdout:\n"
                                f"{result.stdout}\nstderr:\n{result.stderr}"
                            ),
                        }
                    )
                elif kind == "error":
                    handle_error(frame)
                elif kind == "result":
                    budget.iterations = frame["iterations"]
                    diagnostics["budget_exhausted"] = (
                        frame["status"] != "finished"
                        and budget.max_iterations is not None
                        and budget.iterations >= budget.max_iterations
                    )
                    diagnostics["execution_status"] = frame["status"]
                    diagnostics["conversation_id"] = frame["conversation_id"]
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
            missing = [name for name in task.outputs if name not in files]
            value = (
                Measurement(passed=False, reason=f"Missing declared outputs: {missing}")
                if missing
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
            nonlocal task, validate, max_repairs, spec
            task, validate, max_repairs = next_task, next_validate, next_max_repairs
            if next_spec is not None:
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

        yield run
    finally:
        from pyforger.agent_diagnostics import trajectory

        active_error = sys.exc_info()[1]
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

        if connection is not None:
            diagnostics["sdk_transport"] = {
                "mode": "detached-socket",
                "requests": connection.requests,
                "reconnects": connection.reconnects,
            }
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
        for redaction_spec in redaction_specs:
            safe = trajectory([diagnostics], redaction_spec)["steps"][0]
            diagnostics.clear()
            diagnostics.update(safe)
        if active_error is not None:
            active_error.diagnostics = dict(diagnostics)
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
) -> tuple[Any, Any]:
    with remote_sdk_session(
        backend, transport, task, spec, validate, max_repairs, diagnostics
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
) -> tuple[Any, Any]:
    """Run one conversation; transport retries reconnect only to its daemon."""
    return _run_remote_sdk_once(
        backend, transport, task, spec, validate, max_repairs, diagnostics
    )
