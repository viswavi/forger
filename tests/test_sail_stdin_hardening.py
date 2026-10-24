"""Bound real OS pipe writes without requiring a Sail VM."""

import io
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
from pyforger.sail_sdk import (
    PROTOCOL_TIMELINE_FRAMES,
    _FrameWriter,
    _incoming_frames,
    _retain_worker_log,
)
from pyforger.session_budget import SessionBudget


def test_writer_sends_complete_frame():
    sink = io.StringIO()
    writer = _FrameWriter(sink, timeout_s=0.2)
    try:
        writer.send({"kind": "turn", "instruction": "hello"})
        writer.close()
        writer._thread.join(timeout=1)
        writer.check()
        assert json.loads(sink.getvalue()) == {"kind": "turn", "instruction": "hello"}
        snapshot = writer.snapshot()
        frame = snapshot["timeline"][0]
        assert frame["kind"] == "turn"
        assert frame["sequence"] == 1
        assert frame["encoded_bytes"] == len(sink.getvalue().encode())
        assert (
            frame["enqueued_at"]
            <= frame["write_started_at"]
            <= frame["write_finished_at"]
        )
        assert frame["queue_wait_s"] >= 0
        assert frame["write_duration_s"] >= 0
        assert frame["completed"] is True
        assert snapshot["pending_count"] == 0
        assert "hello" not in json.dumps(snapshot)
    finally:
        writer.close()


def test_blocked_real_pipe_write_times_out_and_close_does_not_wait():
    reader, output = os.pipe()
    sink = os.fdopen(output, "w", buffering=1)
    writer = _FrameWriter(sink, timeout_s=0.05)
    try:
        started = time.monotonic()
        writer.send({"payload": "x" * (2 * 1024 * 1024)})
        time.sleep(0.06)
        with pytest.raises(TimeoutError, match="stdin write timed out") as error:
            writer.check()
        snapshot = error.value.protocol_snapshot
        assert snapshot["pending_count"] == 1
        assert snapshot["queue_count"] == 0
        assert snapshot["oldest_pending_age_s"] >= 0.05
        assert snapshot["writer_alive"] is True
        assert snapshot["writer_error_type"] is None
        frame = snapshot["timeline"][0]
        assert frame["encoded_bytes"] > 2 * 1024 * 1024
        assert frame["write_started_at"] >= frame["enqueued_at"]
        assert "write_finished_at" not in frame
        assert "payload" not in json.dumps(snapshot)
        writer.close()
        assert time.monotonic() - started < 1
    finally:

        def drain():
            while os.read(reader, 65536):
                pass

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        writer._thread.join(timeout=2)
        assert not writer._thread.is_alive()
        sink.close()
        thread.join(timeout=2)
        os.close(reader)


def test_writer_propagates_closed_pipe():
    sink = io.StringIO()
    sink.close()
    writer = _FrameWriter(sink, timeout_s=0.2)
    try:
        writer.send({"kind": "turn"})
        writer._thread.join(timeout=1)
        with pytest.raises(ValueError, match="closed") as error:
            writer.check()
        assert error.value.protocol_snapshot["writer_error_type"] == "ValueError"
        assert error.value.protocol_snapshot["timeline"][0]["completed"] is False
    finally:
        writer.close()


def test_budget_exemption_credits_only_requested_interval():
    budget = SessionBudget(None, 10)
    before = budget.remaining_s()
    credit = budget.exempt_elapsed_s(time.monotonic() - 2)
    assert 1.9 < credit < 2.1
    assert budget.remaining_s() > before + 1.9


def test_failed_write_wakes_silent_stdout_reader():
    reader, output = os.pipe()
    chunks = os.fdopen(reader)
    sink = io.StringIO()
    sink.close()
    writer = _FrameWriter(sink, timeout_s=0.1)
    incoming = _incoming_frames(chunks, writer, SessionBudget(None, 5))
    try:
        writer.send({"kind": "turn"})
        with pytest.raises(ValueError, match="closed"):
            next(incoming)
    finally:
        incoming.close()
        writer.close()
        os.close(output)
        chunks.close()


def test_stalled_write_wakes_silent_stdout_reader():
    reader, output = os.pipe()
    input_reader, input_output = os.pipe()
    chunks = os.fdopen(reader)
    sink = os.fdopen(input_output, "w", buffering=1)
    writer = _FrameWriter(sink, timeout_s=0.05)
    incoming = _incoming_frames(chunks, writer, SessionBudget(None, 5))
    try:
        writer.send({"payload": "x" * (2 * 1024 * 1024)})
        with pytest.raises(TimeoutError, match="stdin write timed out") as error:
            next(incoming)
        assert error.value.protocol_snapshot["reader_alive"] is True
        assert error.value.protocol_snapshot["reader_backlog"] == 0
    finally:
        incoming.close()
        writer.close()
        os.close(output)
        chunks.close()
        os.close(input_reader)
        writer._thread.join(timeout=2)
        sink.close()


@pytest.mark.skipif(
    os.environ.get("FORGER_SAIL_LIVE") != "1",
    reason="explicit FORGER_SAIL_LIVE=1 required",
)
def test_live_sail_large_duplex_protocol():
    import sail

    box = sail.Sailbox.create(
        app=sail.App.find("forger", mint_if_missing=True),
        image=sail.Image.from_registry("python:3.13-slim"),
        name="forger-stdin-test-" + uuid.uuid4().hex,
        timeout=180,
        max_lifetime_seconds=240,
    )
    writer = None
    incoming = None
    try:
        process = box.exec(
            [
                "python3",
                "-u",
                "-c",
                (
                    "import sys,json; print(json.dumps({'kind':'ready'}),flush=True)\n"
                    "for line in sys.stdin:\n"
                    " value=json.loads(line)\n"
                    " for i in range(100):\n"
                    "  print(json.dumps({'kind':'event_chunk','text':'x'*32000}),flush=True)\n"
                    " print(json.dumps({'kind':'result','size':len(value['text'])}),flush=True)\n"
                ),
            ],
            open_stdin=True,
            timeout=120,
            output_buffer_bytes=65536,
        )
        writer = _FrameWriter(process.stdin)
        incoming = _incoming_frames(process.stdout, writer, SessionBudget(None, 120))
        assert next(incoming) == {"kind": "ready"}
        for size in (395385, 421298, 1100000):
            writer.send({"kind": "observation", "text": "x" * size})
            for _ in range(100):
                assert next(incoming) == {"kind": "event_chunk", "text": "x" * 32000}
            assert next(incoming) == {"kind": "result", "size": size}
        assert not writer._pending or writer._pending[0][0].wait(5)
        writer.check()
    finally:
        if incoming is not None:
            incoming.close()
        if writer is not None:
            writer.close()
        box.terminate()


def test_real_duplex_protocol_retains_bounded_metadata():
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import sys,json\n"
                "for line in sys.stdin:\n"
                " for i in range(140):\n"
                "  print(json.dumps({'kind':'event_chunk','text':'private'*6000}),flush=True)\n"
                " print(json.dumps({'kind':'result'}),flush=True)\n"
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    writer = _FrameWriter(process.stdin, timeout_s=3)
    incoming = _incoming_frames(process.stdout, writer, SessionBudget(None, 10))
    try:
        for _ in range(2):
            writer.send({"kind": "observation", "text": "secret" * 70000})
            for _ in range(140):
                assert next(incoming)["kind"] == "event_chunk"
            assert next(incoming) == {"kind": "result"}
        writer.check()
        snapshot = writer.snapshot()
        assert len(snapshot["timeline"]) == PROTOCOL_TIMELINE_FRAMES
        assert snapshot["last_stdout_bytes"] > 0
        assert snapshot["last_frame_kind"] == "result"
        assert snapshot["last_frame_at"] <= snapshot["snapshot_at"]
        assert snapshot["timeline"][-1]["sequence"] == 282
        assert snapshot["timeline"][-1]["encoded_bytes"] == len('{"kind": "result"}\n')
        assert "secret" not in json.dumps(snapshot)
        assert "private" not in json.dumps(snapshot)
    finally:
        incoming.close()
        writer.close()
        writer._thread.join(timeout=2)
        process.stdin.close()
        process.wait(timeout=3)
        process.stdout.close()


def test_protocol_metadata_is_appended_to_full_worker_log(tmp_path):
    log = tmp_path / "worker.log"
    log.write_text("phase log\n")
    worker = SimpleNamespace(fs=SimpleNamespace(read=lambda path: log.read_bytes()))
    diagnostics = {}
    attempt = {"protocol": {"pending_count": 1, "timeline": []}}
    _retain_worker_log(worker, diagnostics, attempt)
    assert 'Host protocol metadata: {"pending_count": 1' in attempt["worker_log"]
    assert diagnostics["sdk_worker_log"] == attempt["worker_log"]
