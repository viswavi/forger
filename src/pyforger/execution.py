"""Pluggable execution of commands in fresh, resource-limited containers.

Images are trusted, locally installed images with /bin/sh, sleep and tar. Docker is
an isolation boundary, not a substitute for a hardened daemon or VM sandbox.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, Optional, Protocol

if TYPE_CHECKING:
    from pyforger.artifacts import ArtifactRef


ExecutionStatus = Literal["completed", "failed", "timed_out", "infrastructure_error"]


def _relative_file(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
    ):
        raise ValueError("outputs must be relative POSIX file paths")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("outputs must not contain traversal or empty components")
    return value


@dataclass(frozen=True)
class ExecutionSpec:
    """Immutable command, artifact paths, and limits for isolated execution."""

    image: str
    command: tuple[str, ...]
    inputs: Optional[ArtifactRef] = None
    outputs: tuple[str, ...] = ()
    timeout_s: float = 60.0
    memory_mb: int = 256
    cpus: float = 1.0
    pids_limit: int = 64
    workdir: str = "/work"

    def __post_init__(self) -> None:
        """Validate command arguments, output paths, and resource limits."""
        if (
            not isinstance(self.image, str)
            or not self.image
            or self.image.startswith("-")
            or any(c.isspace() for c in self.image)
            or "\x00" in self.image
        ):
            raise ValueError("image must be a trusted local image reference")
        if (
            not isinstance(self.command, tuple)
            or not self.command
            or any(not isinstance(arg, str) or "\x00" in arg for arg in self.command)
            or not self.command[0]
        ):
            raise ValueError("command must be a nonempty tuple of arguments")
        if not isinstance(self.outputs, tuple):
            raise ValueError("outputs must be a tuple")
        for name in self.outputs:
            _relative_file(name)
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("duplicate outputs")
        if any(
            a.startswith(b + "/") for a in self.outputs for b in self.outputs if a != b
        ):
            raise ValueError("output files cannot be ancestors of other outputs")
        for name in ("timeout_s", "cpus", "memory_mb", "pids_limit"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(name + " must be finite and positive")
        for name in ("memory_mb", "pids_limit"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(name + " must be an integer")
        if self.workdir != "/work":
            raise ValueError("workdir is fixed at /work")


@dataclass(frozen=True)
class ExecutionResult:
    """Command status, bounded logs, retained outputs, and elapsed time."""

    status: ExecutionStatus
    exit_code: Optional[int]
    stdout: str
    stderr: str
    artifacts: Optional[ArtifactRef]
    duration_s: float


class ExecutionBackend(Protocol):
    """Contract for executing specifications and publishing output evidence."""

    def execute(self, spec: ExecutionSpec, output: Path) -> ExecutionResult:
        """Execute the specification and return status and retained outputs."""
        ...


class _Deadline(Exception):
    pass


class _Infrastructure(Exception):
    pass


class DockerBackend:
    """Run commands in fresh containers using trusted local Docker images."""

    def __init__(
        self,
        docker: str = "docker",
        *,
        max_log_bytes: int = 65536,
        cleanup_timeout_s: float = 10.0,
    ) -> None:
        """Configure the Docker executable, log limit, and cleanup deadline."""
        if (
            isinstance(max_log_bytes, bool)
            or not isinstance(max_log_bytes, int)
            or max_log_bytes <= 0
        ):
            raise ValueError("max_log_bytes must be a positive integer")
        if not math.isfinite(cleanup_timeout_s) or cleanup_timeout_s <= 0:
            raise ValueError("cleanup_timeout_s must be finite and positive")
        self.docker = docker
        self.max_log_bytes = max_log_bytes
        self.cleanup_timeout_s = cleanup_timeout_s

    def _run(
        self,
        args: list[str],
        deadline: float,
        *,
        archive: Optional[Path] = None,
        archive_limit: int = 0,
        stdin_file: Optional[Path] = None,
    ) -> tuple[int, str, str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _Deadline()
        source = stdin_file.open("rb") if stdin_file is not None else None
        try:
            proc = subprocess.Popen(
                [self.docker, *args],
                stdin=source or subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
        finally:
            if source is not None:
                source.close()
        buffers = [bytearray(), bytearray()]
        errors = []

        def drain(stream, index):
            try:
                with (
                    archive.open("wb")
                    if archive is not None and index == 0
                    else tempfile.TemporaryFile()
                ) as sink:
                    total = 0
                    while True:
                        chunk = stream.read(65536)
                        if not chunk:
                            break
                        total += len(chunk)
                        if archive is not None and index == 0:
                            if total > archive_limit:
                                errors.append("output archive exceeds size limit")
                                proc.kill()
                                break
                            sink.write(chunk)
                        else:
                            buffers[index].extend(
                                chunk[
                                    : max(0, self.max_log_bytes - len(buffers[index]))
                                ]
                            )
            except Exception as exc:
                errors.append(str(exc))
                proc.kill()
            finally:
                stream.close()

        threads = [
            threading.Thread(target=drain, args=(stream, i), daemon=True)
            for i, stream in enumerate((proc.stdout, proc.stderr))
        ]
        for thread in threads:
            thread.start()
        expired = False
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            expired = True
            proc.kill()
            proc.wait()
        finally:
            for thread in threads:
                thread.join()
        texts = tuple(bytes(buf).decode("utf-8", errors="replace") for buf in buffers)
        if expired:
            raise _Deadline(*texts)
        if errors:
            raise _Infrastructure("; ".join(errors))
        return proc.returncode, texts[0], texts[1]

    def _check(self, args: list[str], deadline: float, **kwargs) -> str:
        code, stdout, stderr = self._run(args, deadline, **kwargs)
        if code:
            raise _Infrastructure(stderr or stdout or "Docker CLI failed")
        return stdout

    @staticmethod
    def _extract(
        archive: Path, destination: Path, outputs: tuple[str, ...], limit: int
    ) -> None:
        # Never call extractall: Docker's tar stream is untrusted, including its
        # ancestor directories, links, sparse metadata and duplicate members.
        with tarfile.open(archive, mode="r:") as tar:
            files = {}
            seen = set()
            total = 0
            for member in tar:
                name = member.name
                if name in (".", "./") and member.isdir():
                    continue
                if name.startswith("./"):
                    name = name[2:]
                if member.isdir():
                    name = name.rstrip("/")
                _relative_file(name)
                if (
                    name in seen
                    or not (member.isfile() or member.isdir())
                    or member.issparse()
                ):
                    raise ValueError("unsafe output archive member: " + name)
                seen.add(name)
                total += member.size
                if total > limit or len(seen) > 100000:
                    raise ValueError("output archive exceeds limits")
                if member.isfile():
                    files[name] = member
            for name in seen:
                if any(
                    str(parent) in files
                    for parent in PurePosixPath(name).parents
                    if str(parent) != "."
                ):
                    raise ValueError("non-directory output ancestor")
            if any(name not in files for name in outputs):
                raise ValueError("declared output is missing or is not a regular file")
            for name in outputs:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(files[name])
                if source is None:
                    raise ValueError("output has no file data")
                with source, target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)

    @staticmethod
    def _stage_inputs(inputs: ArtifactRef, stage: Path) -> None:
        from pyforger.artifacts import ArtifactRef

        inputs.verify()
        for entry in inputs.files:
            relative = _relative_file(entry.path)
            source = inputs.root / relative
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            flags = (
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            with os.fdopen(os.open(source, flags), "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("input is not a regular file")
                with target.open("xb") as sink:
                    shutil.copyfileobj(stream, sink)
            target.chmod(0o666)
        inputs.verify()
        staged = ArtifactRef.capture(stage)
        if staged.digest != inputs.digest:
            raise ValueError("staged inputs differ from the input artifact")
        staged.verify()

    @staticmethod
    def _input_archive(stage: Path, archive: Path) -> None:
        from pyforger.artifacts import ArtifactRef

        ref = ArtifactRef.capture(stage)
        directories: set[str] = set()
        for entry in ref.files:
            directories.update(
                str(parent)
                for parent in PurePosixPath(entry.path).parents
                if str(parent) != "."
            )
        with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as tar:
            for name in sorted(directories):
                member = tarfile.TarInfo(name)
                member.type = tarfile.DIRTYPE
                member.mode = 0o777
                tar.addfile(member)
            for entry in ref.files:
                member = tarfile.TarInfo(entry.path)
                member.size = entry.size
                member.mode = 0o666
                with (stage / entry.path).open("rb") as stream:
                    tar.addfile(member, stream)

    @staticmethod
    def _output_path(output: Path) -> Path:
        output = Path(os.path.abspath(output))
        if output.is_symlink():
            raise ValueError("output directory must not be a symlink")
        # Resolve trusted caller-owned ancestor aliases, but never follow a
        # symlink at the publication leaf.
        return output.parent.resolve() / output.name

    def execute(self, spec: ExecutionSpec, output: Path) -> ExecutionResult:
        """Execute once; nonzero exits skip collection and remain failed.

        Successful commands with missing/unsafe outputs are infrastructure
        errors. Timeout and cleanup errors take precedence over command status.
        The caller must prevent concurrent mutation of input/output directories.
        Collection is not a point-in-time snapshot of background container writers;
        tar errors fail closed and every received member is validated on host.
        """
        from pyforger.artifacts import ArtifactRef

        started = time.monotonic()
        deadline = started + spec.timeout_s
        container = "forger-" + uuid.uuid4().hex
        status: ExecutionStatus = "infrastructure_error"
        code = None
        stdout = stderr = ""
        artifacts = None
        create_attempted = False
        try:
            output = self._output_path(output)
            if output.exists() or output.is_symlink():
                if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
                    raise ValueError("output must be absent or an empty directory")
            with tempfile.TemporaryDirectory(prefix="forger-execution-") as temporary:
                temp = Path(temporary)
                stage = temp / "inputs"
                stage.mkdir()
                if spec.inputs is not None:
                    self._stage_inputs(spec.inputs, stage)
                details = json.loads(
                    self._check(
                        [
                            "image",
                            "inspect",
                            "--format",
                            '{"Id":{{json .Id}},"Config":{'
                            '"Volumes":{{json (index .Config "Volumes")}}}}',
                            spec.image,
                        ],
                        deadline,
                    )
                )
                if details.get("Config", {}).get("Volumes"):
                    raise ValueError("images declaring volumes are not supported")
                image = details["Id"]
                create_attempted = True
                self._check(
                    [
                        "create",
                        "--pull=never",
                        "--name",
                        container,
                        "--network=none",
                        "--read-only",
                        "--cap-drop=ALL",
                        "--security-opt=no-new-privileges",
                        "--user=65534:65534",
                        "--memory",
                        str(spec.memory_mb) + "m",
                        "--memory-swap",
                        str(spec.memory_mb) + "m",
                        "--cpus",
                        str(spec.cpus),
                        "--pids-limit",
                        str(spec.pids_limit),
                        "--tmpfs",
                        "/work:rw,nosuid,nodev,mode=1777,size="
                        + str(spec.memory_mb)
                        + "m",
                        "--workdir=/work",
                        "--log-driver=none",
                        "--entrypoint=/bin/sh",
                        image,
                        "-c",
                        "while :; do sleep 3600; done",
                    ],
                    deadline,
                )
                self._check(["start", container], deadline)
                if spec.inputs is not None:
                    input_archive = temp / "inputs.tar"
                    self._input_archive(stage, input_archive)
                    self._check(
                        [
                            "exec",
                            "-i",
                            "--workdir=/work",
                            container,
                            "tar",
                            "--no-same-owner",
                            "-xf",
                            "-",
                        ],
                        deadline,
                        stdin_file=input_archive,
                    )
                code, stdout, stderr = self._run(
                    ["exec", "--workdir=/work", container, *spec.command], deadline
                )
                status = "completed" if code == 0 else "failed"
                if spec.outputs and code == 0:
                    archive = temp / "outputs.tar"
                    # Docker cp reads the underlying rootfs, not the live tmpfs.
                    # Tar does not dereference links; its entire untrusted stream
                    # is validated before any declared file is written on host.
                    self._check(
                        [
                            "exec",
                            "--workdir=/work",
                            container,
                            "tar",
                            "-cf",
                            "-",
                            "--",
                            ".",
                        ],
                        deadline,
                        archive=archive,
                        archive_limit=spec.memory_mb * 1024 * 1024 * 2 + 1048576,
                    )
                    collected = temp / "collected"
                    collected.mkdir()
                    self._extract(
                        archive, collected, spec.outputs, spec.memory_mb * 1024 * 1024
                    )
                    ArtifactRef.capture(collected).verify()
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if output.exists():
                        output.rmdir()
                    shutil.move(str(collected), str(output))
                    artifacts = ArtifactRef.capture(output)
                    artifacts.verify()
        except _Deadline as exc:
            status = "timed_out"
            if exc.args:
                stdout, stderr = exc.args
        except Exception as exc:
            status = "infrastructure_error"
            stderr = (stderr + "\n" + str(exc))[-self.max_log_bytes :]
        finally:
            if create_attempted:
                try:
                    self._check(
                        ["rm", "--force", container],
                        time.monotonic() + self.cleanup_timeout_s,
                    )
                except (OSError, _Deadline, _Infrastructure) as exc:
                    status = "infrastructure_error"
                    stderr = (stderr + "\nContainer cleanup failed: " + str(exc))[
                        -self.max_log_bytes :
                    ]
        return ExecutionResult(
            status, code, stdout, stderr, artifacts, time.monotonic() - started
        )
