"""Direct Sail microVM transport and the sole tool exposed to SDK agents."""

from __future__ import annotations

import base64
import io
import json
import math
from typing import Any

from openhands.sdk.tool import Action, Observation, ToolDefinition, ToolExecutor
from pydantic import Field

from pyforger.agent_backend import Network

WORKSPACE = "/work"
MAX_INPUT_FILE_BYTES = 32 * 1024 * 1024
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_FILE_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
TRANSPORTS: dict[str, Any] = {}


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
import os, stat, json, base64, sys
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
        flags = os.O_NOFOLLOW | os.O_NONBLOCK
        flags |= ((os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                  if request['write'] else os.O_RDONLY)
        leaf = os.open(parts[-1], flags, 0o644, dir_fd=fd)
        with os.fdopen(leaf, 'wb' if request['write'] else 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('Not a single-link regular workspace file: ' + name)
            if request['write']:
                stream.write(base64.b64decode(value))
                os.fchown(stream.fileno(), 65534, 65534)
            else:
                data = stream.read(request['limit'] + 1)
                if len(data) > request['limit']:
                    raise ValueError('Output exceeds size limit: ' + name)
                result[name] = base64.b64encode(data).decode()
    except FileNotFoundError:
        if request['write']: raise
    finally:
        os.close(fd)
os.close(root)
print(json.dumps(result))
"""


class SailTransport:
    """A single microVM handle; no host filesystem or shell operations."""

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

    def command(self, command: str, *, feedback: str = "") -> Any:
        """Run generated commands unprivileged, with no host credentials."""
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

    def files(self, files: dict[str, str | bytes], *, write: bool) -> dict[str, bytes]:
        """Stage new files or collect bounded bytes without following symlinks."""
        total = 0
        for name, value in files.items():
            workspace_path(name)
            size = (
                len(value.encode() if isinstance(value, str) else value) if write else 0
            )
            if size > MAX_INPUT_FILE_BYTES:
                raise ValueError("Input exceeds size limit: " + name)
            total += size
        if write and self.input_bytes + total > MAX_INPUT_BYTES:
            raise ValueError("Input exceeds total size limit")
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
                data = value.encode() if isinstance(value, str) else value
                payload = json.dumps(
                    {
                        "files": {
                            name: base64.b64encode(data).decode() if write else ""
                        },
                        "write": write,
                        "limit": min(
                            MAX_OUTPUT_FILE_BYTES, MAX_OUTPUT_BYTES - output_bytes
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
                received = {
                    key: base64.b64decode(data)
                    for key, data in json.loads(result.stdout).items()
                }
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


class SailCommandAction(Action):
    """Execute a shell command inside the remote workspace."""

    command: str = Field(description="Bash command in /work of the Sail microVM")


class SailCommandObservation(Observation):
    """Output from a remote Sail command."""


class SailCommandExecutor(ToolExecutor[SailCommandAction, SailCommandObservation]):
    """Dispatch SDK actions exclusively through the remote transport."""

    def __init__(self, transport: SailTransport):
        """Bind the command transport."""
        self.transport = transport

    def __call__(
        self, action: SailCommandAction, conversation: Any = None
    ) -> SailCommandObservation:
        """Execute remotely and render a bounded observation."""
        result = self.transport.command(action.command)
        return SailCommandObservation.from_text(
            f"exit_code={result.exit_code}\nstdout:\n{result.stdout[-16000:]}"
            f"\nstderr:\n{result.stderr[-16000:]}"
        )


class SailCommandTool(ToolDefinition[SailCommandAction, SailCommandObservation]):
    """Remote command tool; never resolves a local terminal or file editor."""

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> list[Any]:
        """Resolve the per-conversation transport without serializing credentials."""
        return [
            cls(
                description=(
                    "Execute bash in the isolated Sail microVM at /work. "
                    "Use shell commands to read and edit files. "
                    "No commands execute on the host."
                ),
                action_type=SailCommandAction,
                observation_type=SailCommandObservation,
                executor=SailCommandExecutor(TRANSPORTS[params["token"]]),
            )
        ]
