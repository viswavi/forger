#!/usr/bin/env python3
"""Native Sail stdin diagnostic; NOT a confirmed infrastructure reproducer.

Requires only sail and the standard library; SAIL_API_KEY stays in the host env.
Default: continuously drain stdout/stderr. --consumer pause deliberately stops
stdout consumption after READY, exercising documented pipe backpressure; a delay
in that mode is not infrastructure evidence. JSONL logs contain synthetic-data
sizes/timing only. No Forger or OpenHands imports, fixtures, or acknowledgment mocks.
"""

# Exceptions at remote/thread boundaries are sanitized to protect credentials.
# ruff: noqa: BLE001

import argparse
import importlib.metadata
import json
import os
import queue
import sys
import threading
import time
import uuid

import sail

GUEST = r"""
import sys
sizes = [int(x) for x in sys.argv[1].split(',')]
reply = int(sys.argv[2])
for turn, size in enumerate(sizes):
    print('READY %d' % turn, flush=True)
    first = sys.stdin.buffer.read(1)
    if not first:
        raise SystemExit(2)
    left = reply
    while left:
        n = min(left, 32000)
        sys.stdout.write('x' * n + '\n')
        sys.stdout.flush()
        left -= n
    rest = sys.stdin.buffer.readline()
    if first + rest != b'x' * size + b'\n':
        raise SystemExit(3)
    print('RECEIVED %d %d' % (turn, size + 1), flush=True)
"""
START = time.monotonic()
LOG_LOCK = threading.Lock()


def log(event, **fields):
    with LOG_LOCK:
        print(
            json.dumps(
                dict(t=round(time.monotonic() - START, 6), event=event, **fields)
            ),
            flush=True,
        )


def launch(target):
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def bounded_call(call, timeout):
    result = queue.Queue()

    def run():
        try:
            result.put((True, call()))
        except Exception as exc:
            result.put((False, type(exc).__name__))

    launch(run)
    try:
        ok, value = result.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError("watchdog") from None
    if not ok:
        raise RuntimeError(value)
    return value


def probe(args, index, results):
    box = None
    status = "error"
    try:
        # Server lifetime bounds remote resources even if provisioning/cleanup
        # fails to return; the outer watchdog also bounds the local process.
        box = sail.Sailbox.create(
            app=sail.App.find(args.app, mint_if_missing=True),
            image=sail.Image.from_registry("python:3.13-slim"),
            name="stdin-diagnostic-" + uuid.uuid4().hex,
            timeout=args.watchdog,
            max_lifetime_seconds=args.watchdog + 60,
        )
        log("created", probe=index, box_id=box.sailbox_id)
        proc = box.exec(
            [
                "python3",
                "-u",
                "-c",
                GUEST,
                ",".join(map(str, args.sizes)),
                str(args.reply_bytes),
            ],
            open_stdin=True,
            timeout=args.watchdog,
            output_mode="pipe",
            output_buffer_bytes=args.output_buffer_bytes,
        )
        events = queue.Queue()
        counts = {"stdout": 0, "stderr": 0}

        def drain_stdout():
            pending = ""
            try:
                for chunk in proc.stdout:
                    counts["stdout"] += len(chunk.encode())
                    pending += chunk
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        if line.startswith(("READY ", "RECEIVED ")):
                            log(
                                "guest_marker",
                                probe=index,
                                marker=line,
                                stdout_bytes=counts["stdout"],
                            )
                            events.put(line)
                            if line.startswith("READY ") and args.consumer == "pause":
                                log(
                                    "consumer_pause",
                                    probe=index,
                                    seconds=args.pause_seconds,
                                )
                                time.sleep(args.pause_seconds)
                                log("consumer_resumed", probe=index)
                events.put("EOF")
            except Exception as exc:
                events.put("reader_error:" + type(exc).__name__)

        def drain_stderr():
            try:
                for chunk in proc.stderr:
                    counts["stderr"] += len(chunk.encode())
            except Exception as exc:
                log("stderr_error", probe=index, error=type(exc).__name__)

        launch(drain_stdout)
        launch(drain_stderr)

        def expect(expected):
            try:
                actual = events.get(timeout=args.turn_timeout)
            except queue.Empty:
                raise TimeoutError("guest_marker") from None
            if actual != expected:
                raise RuntimeError("unexpected_marker")

        for turn, size in enumerate(args.sizes):
            expect(f"READY {turn}")
            if turn and args.idle_seconds:
                log("idle", probe=index, turn=turn, seconds=args.idle_seconds)
                time.sleep(args.idle_seconds)
            payload = b"x" * size + b"\n"
            started = time.monotonic()
            log(
                "native_write_start",
                probe=index,
                turn=turn,
                bytes=len(payload),
                consumer=args.consumer,
                stdout_bytes=counts["stdout"],
            )
            bounded_call(
                lambda data=payload: proc.stdin.write(data), args.write_timeout
            )
            log(
                "native_write_done",
                probe=index,
                turn=turn,
                elapsed=round(time.monotonic() - started, 6),
                stdout_bytes=counts["stdout"],
            )
            expect(f"RECEIVED {turn} {size + 1}")
        result = bounded_call(proc.wait, args.turn_timeout)
        if result.exit_code != 0 or result.timed_out:
            raise RuntimeError("guest_exit")
        expected = sum(
            len(f"READY {t}\nRECEIVED {t} {s + 1}\n")
            + args.reply_bytes
            + (args.reply_bytes + 31999) // 32000
            for t, s in enumerate(args.sizes)
        )
        if counts["stdout"] != expected or counts["stderr"]:
            raise RuntimeError("output_byte_count")
        status = "passed"
        log("verified", probe=index, **counts)
    except Exception as exc:
        # Do not emit exception messages: provider errors may contain credentials.
        log(
            "probe_error", probe=index, error=type(exc).__name__, consumer=args.consumer
        )
    finally:
        if box is not None:
            try:
                bounded_call(box.terminate, args.cleanup_timeout)
                log("terminated", probe=index)
            except Exception as exc:
                status = "cleanup_error"
                log(
                    "cleanup_error",
                    probe=index,
                    error=type(exc).__name__,
                    box_id=box.sailbox_id,
                )
        results.put(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", default="forger")
    parser.add_argument("--sizes", default="395385,421298,1100000")
    parser.add_argument("--reply-bytes", type=int, default=3200000)
    parser.add_argument("--output-buffer-bytes", type=int, default=65536)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--consumer", choices=["drain", "pause"], default="drain")
    parser.add_argument("--pause-seconds", type=float, default=3)
    parser.add_argument("--idle-seconds", type=float, default=0)
    parser.add_argument("--write-timeout", type=float, default=15)
    parser.add_argument("--turn-timeout", type=float, default=20)
    parser.add_argument("--cleanup-timeout", type=float, default=15)
    parser.add_argument("--watchdog", type=int, default=150)
    args = parser.parse_args()
    args.sizes = [int(s) for s in args.sizes.split(",")]
    if (
        not args.sizes
        or min(args.sizes) < 1
        or args.concurrency < 1
        or args.reply_bytes < 0
        or args.idle_seconds < 0
        or args.pause_seconds < 0
        or min(
            args.write_timeout, args.turn_timeout, args.cleanup_timeout, args.watchdog
        )
        <= 0
    ):
        parser.error("invalid sizes, concurrency, byte count, or timing")
    if not os.environ.get("SAIL_API_KEY"):
        parser.error("SAIL_API_KEY must be set in the environment")
    log(
        "diagnostic_not_confirmed_reproducer",
        sail=importlib.metadata.version("sail"),
        python=sys.version.split()[0],
        config=vars(args),
    )
    results = queue.Queue()
    for index in range(args.concurrency):
        launch(lambda i=index: probe(args, i, results))
    deadline = time.monotonic() + args.watchdog + args.cleanup_timeout
    statuses = []
    try:
        for _ in range(args.concurrency):
            statuses.append(results.get(timeout=max(0.01, deadline - time.monotonic())))
    except queue.Empty:
        log("global_watchdog", remote_lifetime_limit=args.watchdog + 60)
        return 2
    log(
        "summary",
        statuses=statuses,
        native_stall_confirmed=False,
        note="Failures require investigation; intentional pauses are not infrastructure evidence.",
    )
    return 0 if all(s == "passed" for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
