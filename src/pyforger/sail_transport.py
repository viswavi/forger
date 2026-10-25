"""Direct Sail microVM transport and the sole tool exposed to SDK agents."""

from __future__ import annotations

import io
import json
import math
import shlex
import tempfile
from collections.abc import Iterator
from typing import Any

from pyforger.agent_backend import Network

WORKSPACE = "/work"
# Optional compatibility policies, disabled on the default execution path.
MAX_INPUT_FILE_BYTES: int | None = None
MAX_INPUT_BYTES: int | None = None
MAX_OUTPUT_FILE_BYTES: int | None = None
MAX_OUTPUT_BYTES: int | None = None


def workspace_path(name: str) -> str:
    """Reject ambiguous and escaping workspace paths before any remote request."""
    if not name or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"Invalid workspace path: {name!r}")
    if "\x00" in name or name.startswith("/"):
        raise ValueError(f"Invalid workspace path: {name!r}")
    return name


def egress_policy(network: Network) -> dict[str, Any]:
    """Sail blocks private destinations and IPv6 internet access platform-wide."""
    if network in (Network.OFFLINE, Network.MODEL):
        return {"no_network": True}
    if network is Network.PUBLIC:
        return {}
    raise ValueError(f"Unsupported network policy: {network!r}")


# Walk using directory descriptors: an agent can rename paths while collection
# runs, so resolve()/lstat() followed by fs.read() would be a TOCTOU escape.
_FILE_IO = r"""
import os, stat, json, sys
if sys.argv[1] == '--request-file':
    with open(sys.argv[2]) as source: request = json.load(source)
else:
    request = json.loads(sys.argv[1])
root = os.open('/work', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
result = {}
for name, value in request['files'].items():
    fd = os.dup(root)
    try:
        parts = name.split('/')
        for part in parts[:-1]:
            if request['write']:
                try: os.mkdir(part, 0o777, dir_fd=fd)
                except FileExistsError: pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=fd)
            if request['write']:
                os.fchown(child, 65534, 65534)
            os.close(fd)
            fd = child
        if request.get('replace') and request['write']:
            try: os.unlink(parts[-1], dir_fd=fd)
            except FileNotFoundError: pass
        flags = os.O_NOFOLLOW | os.O_NONBLOCK
        flags |= ((os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                  if request['write'] else os.O_RDONLY)
        leaf = os.open(parts[-1], flags, 0o644, dir_fd=fd)
        with os.fdopen(leaf, 'wb' if request['write'] else 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('Not a single-link regular workspace file: ' + name)
            if request['write']:
                with open(request['source'], 'rb') as source:
                    info = os.fstat(source.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise ValueError('Unsafe transfer source')
                    if info.st_size != request['size']:
                        raise ValueError('Transfer source size mismatch')
                    remaining = request['size']
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError('Incomplete transfer source')
                        stream.write(chunk)
                        remaining -= len(chunk)
                    if source.read(1):
                        raise ValueError('Transfer source grew')
                os.fchown(stream.fileno(), 65534, 65534)
            else:
                size = 0
                with open(request['source'], 'xb') as target:
                    while True:
                        chunk = stream.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if request.get('limit') is not None and size > request['limit']:
                            raise ValueError('Output exceeds size limit: ' + name)
                        target.write(chunk)
                after = os.fstat(stream.fileno())
                if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns
                ) or size != after.st_size or after.st_nlink != 1:
                    raise ValueError('Output changed during transfer: ' + name)
                result[name] = size
    except FileNotFoundError:
        if request['write']: raise
    finally:
        os.close(fd)
os.close(root)
print(json.dumps(result))
"""


def _input_chunks(value: str | bytes) -> Iterator[bytes | memoryview]:
    # Bound UTF-8 expansion as well as binary writes without copying the payload.
    if isinstance(value, str):
        for offset in range(0, len(value), 256 * 1024):
            yield value[offset : offset + 256 * 1024].encode()
    else:
        view = memoryview(value)
        for offset in range(0, len(view), 1024 * 1024):
            yield view[offset : offset + 1024 * 1024]


class SailTransport:
    """A single microVM handle with private host upload spooling."""

    def __init__(self, box: Any, budget: Any):
        """Bind a remote handle to the shared session budget."""
        self.box = box
        self.budget = budget
        self.input_bytes = 0

    def run(self, argv: list[str], *, user: str = "root") -> Any:
        """Bound each remote command by the shared remaining wall budget."""
        result = self.box.run(
            argv,
            user=user,
            timeout=max(1, math.ceil(self.budget.remaining_s())),
            output_buffer_bytes=24 * 1024 * 1024,
        )
        self.budget.remaining_s()
        if result.timed_out:
            raise TimeoutError("Sail command timed out")
        return result

    def command(
        self, command: str, *, feedback: str = "", timeout_s: float | None = None
    ) -> Any:
        """Run generated commands unprivileged, with no host credentials."""
        if timeout_s is not None:
            if not math.isfinite(timeout_s) or timeout_s <= 0:
                raise ValueError("Command timeout must be finite and positive")
            timeout_s = min(timeout_s, self.budget.remaining_s())
            # GNU timeout owns the command process group; escalate even when a
            # descendant ignores TERM. EOF prevents unattended stdin waits.
            command = (
                "exec timeout --verbose --kill-after=1s "
                + shlex.quote(str(timeout_s) + "s")
                + " bash -c "
                + shlex.quote(command)
                + " < /dev/null"
            )
        return self.run(
            [
                "/usr/bin/env",
                "-i",
                "PATH=/usr/local/bin:/usr/bin:/bin",
                "HOME=/work",
                "FORGER_FEEDBACK=" + feedback,
                "bash",
                "-c",
                "cd /work && " + command,
            ],
            user="65534",
        )

    def files(
        self, files: dict[str, str | bytes], *, write: bool, replace: bool = False
    ) -> dict[str, bytes]:
        """Stage or snapshot files without following symlinks, using bounded chunks."""
        total = 0
        for name, value in files.items():
            workspace_path(name)
            size = 0
            if write:
                size = (
                    sum(len(chunk) for chunk in _input_chunks(value))
                    if isinstance(value, str)
                    else len(value)
                )
            if MAX_INPUT_FILE_BYTES is not None and size > MAX_INPUT_FILE_BYTES:
                raise ValueError(
                    f"Input exceeds size limit: {name}: {size} bytes; "
                    f"limit {MAX_INPUT_FILE_BYTES} bytes"
                )
            total += size
        if (
            write
            and MAX_INPUT_BYTES is not None
            and self.input_bytes + total > MAX_INPUT_BYTES
        ):
            raise ValueError(
                f"Input exceeds total size limit: {self.input_bytes + total} bytes; "
                f"limit {MAX_INPUT_BYTES} bytes"
            )
        # Reserve before remote writes: a failed batch may already have staged bytes.
        self.input_bytes += total
        # The root-only spool is outside the agent-writable workspace. Payloads
        # travel through Sail's streaming file API, never the exec argv limit.
        created = self.run(
            [
                "python3",
                "-I",
                "-c",
                "import tempfile; print(tempfile.mkdtemp(prefix='forger-transfer-'))",
            ]
        )
        if created.exit_code:
            raise RuntimeError("Cannot create protected Sail transfer directory")
        directory = created.stdout.strip()
        collected: dict[str, bytes] = {}
        try:
            output_bytes = 0
            for name, value in files.items():
                size = 0
                if write:
                    # TemporaryFile is private and unlinked, even if upload fails.
                    with tempfile.TemporaryFile(mode="w+b") as source:
                        for chunk in _input_chunks(value):
                            self.budget.remaining_s()
                            source.write(chunk)
                            size += len(chunk)
                        source.seek(0)
                        self.budget.remaining_s()
                        self.box.fs.write(
                            directory + "/payload", source, mode=0o600, user="root"
                        )
                payload = json.dumps(
                    {
                        "files": {name: ""},
                        "source": directory + "/payload",
                        "size": size,
                        "write": write,
                        "replace": replace,
                        "limit": min(
                            (
                                limit
                                for limit in (
                                    MAX_OUTPUT_FILE_BYTES,
                                    None
                                    if MAX_OUTPUT_BYTES is None
                                    else MAX_OUTPUT_BYTES - output_bytes,
                                )
                                if limit is not None
                            ),
                            default=None,
                        ),
                    }
                ).encode()
                self.budget.remaining_s()
                self.box.fs.write(
                    directory + "/request.json",
                    io.BytesIO(payload),
                    mode=0o600,
                    user="root",
                )
                result = self.run(
                    [
                        "python3",
                        "-I",
                        "-c",
                        _FILE_IO,
                        "--request-file",
                        directory + "/request.json",
                    ]
                )
                if result.exit_code or result.stdout_truncated:
                    raise RuntimeError(
                        "Unsafe or failed Sail file transfer: " + result.stderr[-2000:]
                    )
                received = {}
                for key, expected_size in json.loads(result.stdout).items():
                    with tempfile.TemporaryFile(mode="w+b") as target:
                        for chunk in self.box.fs.read_stream(directory + "/payload"):
                            self.budget.remaining_s()
                            target.write(chunk)
                        if target.tell() != expected_size:
                            raise RuntimeError("Incomplete Sail output transfer")
                        target.seek(0)
                        received[key] = target.read()
                    self.budget.remaining_s()
                # Each request has one file; remove the snapshot before reuse.
                if received:
                    self.run(
                        [
                            "python3",
                            "-I",
                            "-c",
                            "import os,sys; os.unlink(sys.argv[1])",
                            directory + "/payload",
                        ]
                    )
                output_bytes += sum(len(data) for data in received.values())
                collected.update(received)
            return collected
        finally:
            # Cleanup must still run when the shared execution budget expires.
            self.box.run(
                [
                    "python3",
                    "-I",
                    "-c",
                    "import shutil,sys; shutil.rmtree(sys.argv[1])",
                    directory,
                ],
                user="root",
                timeout=10,
                check=True,
            )
