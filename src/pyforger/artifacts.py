"""Verified, content-addressed directory artifacts for pipeline inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict, StrictInt, model_validator

_CHUNK_SIZE = 1024 * 1024


def _relative(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
        or value.startswith("/")
        or PureWindowsPath(value).drive
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise ValueError(f"Invalid artifact relative path: {value!r}")
    return value


def _sha256(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid artifact SHA-256 digest: {value!r}")


def _normalize_root(root: Path) -> Path:
    root = Path(os.path.abspath(root))
    # Ancestor aliases are trusted only at construction; never resolve the root
    # itself, so root symlinks remain visible to strict directory validation.
    try:
        return root.parent.resolve() / root.name
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot normalize artifact root {root}: {exc}") from exc


def _directory(root: Path) -> Path:
    root = Path(os.path.abspath(root))
    for part in reversed((root, *root.parents)):
        mode = part.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Artifact symlink is not allowed: {part}")
        if not stat.S_ISDIR(mode):
            raise ValueError(f"Artifact root must be a directory: {part}")
    return root


class ArtifactFile(BaseModel):
    """One immutable manifest entry; paths use forward slashes."""

    model_config = ConfigDict(frozen=True)

    path: str
    size: StrictInt
    sha256: str

    @model_validator(mode="after")
    def validate_manifest(self) -> ArtifactFile:
        """Validate the relative path, byte count, and content digest."""
        _relative(self.path)
        if type(self.size) is not int or self.size < 0:
            raise ValueError("Artifact file size must be a nonnegative integer")
        _sha256(self.sha256)
        return self


def _digest(files: tuple[ArtifactFile, ...]) -> str:
    payload = [[f.path, f.size, f.sha256] for f in files]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _open_file(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    stream = os.fdopen(os.open(path, flags), "rb")
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        stream.close()
        raise ValueError(f"Artifact must be a single-link regular file: {path}")
    return stream


def _manifest(root: Path) -> tuple[ArtifactFile, ...]:
    try:
        root = _directory(root)
        files = []
        pending = [root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    mode = entry.stat(follow_symlinks=False).st_mode
                    path = Path(entry.path)
                    if stat.S_ISLNK(mode):
                        raise ValueError(f"Artifact symlink is not allowed: {path}")
                    if stat.S_ISDIR(mode):
                        pending.append(path)
                    elif stat.S_ISREG(mode):
                        with _open_file(path) as stream:
                            before = os.fstat(stream.fileno())
                            digest = hashlib.sha256()
                            size = 0
                            while True:
                                chunk = stream.read(_CHUNK_SIZE)
                                if not chunk:
                                    break
                                size += len(chunk)
                                digest.update(chunk)
                            after = os.fstat(stream.fileno())
                        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                            after.st_size,
                            after.st_mtime_ns,
                            after.st_ctime_ns,
                        ) or size != after.st_size:
                            raise ValueError(
                                f"Artifact file changed while hashing: {path}"
                            )
                        files.append(
                            ArtifactFile(
                                path=path.relative_to(root).as_posix(),
                                size=size,
                                sha256=digest.hexdigest(),
                            )
                        )
                    else:
                        raise ValueError(
                            f"Artifact special file is not allowed: {path}"
                        )
        return tuple(sorted(files, key=lambda f: f.path))
    except OSError as exc:
        raise ValueError(f"Cannot read artifact {root}: {exc}") from exc


class ArtifactRef(BaseModel):
    """A snapshot of regular files, independent of root and timestamps.

    Empty directories are excluded. Verification is not a filesystem lock:
    callers must prevent concurrent mutation while capturing or consuming files.
    Trusted ancestor aliases are normalized at construction; root and descendant
    symlinks are rejected during verification.
    """

    model_config = ConfigDict(frozen=True)

    root: Path
    digest: str
    files: tuple[ArtifactFile, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> ArtifactRef:
        """Normalize the root and validate the immutable manifest and digest."""
        object.__setattr__(self, "root", _normalize_root(self.root))
        object.__setattr__(self, "files", tuple(self.files))
        _sha256(self.digest)
        if any(not isinstance(f, ArtifactFile) for f in self.files):
            raise ValueError("Invalid artifact manifest entries")
        paths = [f.path for f in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("Artifact manifest paths must be unique and sorted")
        if _digest(self.files) != self.digest:
            raise ValueError("Artifact manifest digest mismatch")
        return self

    @classmethod
    def capture(cls, root: Path) -> ArtifactRef:
        """Hash regular files under a validated root into an immutable reference."""
        root = _normalize_root(root)
        files = _manifest(root)
        return cls(root=root, digest=_digest(files), files=files)

    def verify(self) -> None:
        """Reject missing, additional, or changed files against the manifest."""
        current = {f.path: f for f in _manifest(self.root)}
        expected = {f.path: f for f in self.files}
        missing = sorted(expected.keys() - current.keys())
        additional = sorted(current.keys() - expected.keys())
        tampered = sorted(
            p for p in expected.keys() & current.keys() if expected[p] != current[p]
        )
        if missing or additional or tampered:
            raise ValueError(
                f"Artifact verification failed: missing={missing[:10]}, "
                f"additional={additional[:10]}, tampered={tampered[:10]}"
            )

    def cache_key(self) -> str:
        """Verify current contents before returning the location-independent digest."""
        self.verify()
        return self.digest

    def path(self, relative: str) -> Path:
        """Return the path of a manifested file without rehashing the tree.

        This is a lookup, not an integrity check: call ``verify()`` explicitly
        at trust boundaries (before reuse, after untrusted code runs).
        """
        relative = _relative(relative)
        if relative not in {f.path for f in self.files}:
            raise ValueError(f"File is not in artifact manifest: {relative!r}")
        return self.root / relative


class ArtifactStore:
    """Atomic directory publication in a trusted, content-addressed local store."""

    def __init__(self, root: Path):
        """Create or validate the trusted artifact publication directory."""
        self.root = _normalize_root(root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            _directory(self.root)
        except OSError as exc:
            raise ValueError(f"Cannot create artifact store: {exc}") from exc

    def get(self, digest: str) -> ArtifactRef:
        """Verify stored contents against the requested digest."""
        _sha256(digest)
        _directory(self.root)
        destination = self.root / digest
        try:
            _directory(destination)
        except OSError as exc:
            raise ValueError(f"Cannot read stored artifact {digest}: {exc}") from exc
        ref = ArtifactRef.capture(destination)
        if ref.digest != digest:
            raise ValueError(f"Artifact store digest mismatch: {digest}")
        return ref

    def put(self, source: Path) -> ArtifactRef:
        """Publish a verified copy atomically, reusing matching stored contents.

        The source is hashed once by ``capture``; the staged copy is verified
        once before the atomic rename. The returned reference is constructed
        for the destination directly instead of recapturing it.
        """
        ref = ArtifactRef.capture(source)
        _directory(self.root)
        destination = self.root / ref.digest
        if os.path.lexists(destination):
            return self.get(ref.digest)
        with tempfile.TemporaryDirectory(
            prefix=".artifact-", dir=self.root
        ) as temporary:
            staged = Path(temporary) / "contents"
            staged.mkdir()
            for entry in ref.files:
                target = staged / entry.path
                target.parent.mkdir(parents=True, exist_ok=True)
                with (
                    _open_file(ref.root / entry.path) as stream,
                    target.open("wb") as output,
                ):
                    shutil.copyfileobj(stream, output, _CHUNK_SIZE)
            ArtifactRef(root=staged, digest=ref.digest, files=ref.files).verify()
            try:
                staged.rename(destination)
            except OSError:
                if not os.path.lexists(destination):
                    raise
                return self.get(ref.digest)
            return ArtifactRef(root=destination, digest=ref.digest, files=ref.files)


def read_text(path: Path, max_bytes: int | None = None) -> str:
    """Read single-link regular UTF-8 content, with an optional byte limit.

    Ancestor aliases are trusted and normalized at entry, as for ArtifactRef.

    Callers must prevent concurrent path or content mutation during the read.
    """
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
        raise ValueError("max_bytes must be positive")
    path = Path(path)
    try:
        path = _normalize_root(path)
        path = _directory(path.parent) / path.name
        with _open_file(path) as stream:
            data = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
    except OSError as exc:
        raise ValueError(f"Expected readable regular artifact: {path}") from exc
    if max_bytes is not None and len(data) > max_bytes:
        raise ValueError("Artifact exceeds read limit")
    return data.decode("utf-8")


class FileInput(BaseModel):
    """Integrity-checked file-backed workspace input; does not sanitize content."""

    model_config = ConfigDict(frozen=True)
    source: Path
    size: StrictInt
    sha256: str

    @model_validator(mode="after")
    def validate_input(self) -> FileInput:
        object.__setattr__(self, "source", _normalize_root(self.source))
        if self.size < 0:
            raise ValueError("File size must be nonnegative")
        _sha256(self.sha256)
        return self

    @classmethod
    def capture(cls, path: Path) -> FileInput:
        """Capture a regular file without loading its contents into memory."""
        path = _normalize_root(Path(path))
        _directory(path.parent)
        with _open_file(path) as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            size = os.fstat(stream.fileno()).st_size
        return cls(source=path, size=size, sha256=digest)

    @classmethod
    def from_artifact(cls, ref: ArtifactRef, relative: str) -> FileInput:
        """Select a manifested file; integrity is checked again during staging."""
        path = ref.path(relative)
        entry = next(f for f in ref.files if f.path == relative)
        return cls(source=path, size=entry.size, sha256=entry.sha256)

    def chunks(self):
        """Verify before yielding, then detect concurrent source mutation."""
        _directory(self.source.parent)
        with _open_file(self.source) as stream:
            before = os.fstat(stream.fileno())
            if (
                before.st_size != self.size
                or hashlib.file_digest(stream, "sha256").hexdigest() != self.sha256
            ):
                raise ValueError("File input integrity mismatch")
            stream.seek(0)
            digest = hashlib.sha256()
            size = 0
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
                size += len(chunk)
                yield chunk
            after = os.fstat(stream.fileno())
            if (
                size != self.size
                or digest.hexdigest() != self.sha256
                or (before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_mtime_ns, after.st_ctime_ns)
            ):
                raise ValueError("File input changed during staging")

    def copy_to(self, path: Path) -> None:
        """Stream into a newly created destination."""
        with Path(path).open("xb") as target:
            for chunk in self.chunks():
                target.write(chunk)

    def cache_key(self) -> object:
        for _ in self.chunks():
            pass
        return (self.size, self.sha256)


def write_input(path: Path, content: str | bytes | FileInput) -> None:
    """Materialize a workspace value without buffering file-backed inputs."""
    if isinstance(content, FileInput):
        content.copy_to(path)
    else:
        path.write_bytes(content.encode() if isinstance(content, str) else content)
