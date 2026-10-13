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

from pydantic import BaseModel, ConfigDict, StrictInt, model_validator

_MAX_ENTRIES = 100_000
_MAX_DEPTH = 100
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


def _open_file(path: Path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    stream = os.fdopen(os.open(path, flags), "rb")
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        stream.close()
        raise ValueError(f"Artifact special file is not allowed: {path}")
    return stream


def _manifest(root: Path) -> tuple[ArtifactFile, ...]:
    try:
        root = _directory(root)
        files = []
        pending = [(root, 0)]
        count = 0
        while pending:
            directory, depth = pending.pop()
            if depth > _MAX_DEPTH:
                raise ValueError("Artifact exceeds maximum directory depth")
            with os.scandir(directory) as entries:
                for entry in entries:
                    count += 1
                    if count > _MAX_ENTRIES:
                        raise ValueError("Artifact exceeds maximum entry count")
                    mode = entry.stat(follow_symlinks=False).st_mode
                    path = Path(entry.path)
                    if stat.S_ISLNK(mode):
                        raise ValueError(f"Artifact symlink is not allowed: {path}")
                    if stat.S_ISDIR(mode):
                        pending.append((path, depth + 1))
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
        if len(self.files) > _MAX_ENTRIES or any(
            not isinstance(f, ArtifactFile) for f in self.files
        ):
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
        """Return a manifested file path after validating the entire artifact."""
        relative = _relative(relative)
        self.verify()
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
        """Publish a verified copy atomically, reusing matching stored contents."""
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
            ref.verify()
            ArtifactRef(root=staged, digest=ref.digest, files=ref.files).verify()
            try:
                staged.rename(destination)
            except OSError:
                if not os.path.lexists(destination):
                    raise
            return self.get(ref.digest)


def read_text(path: Path, max_bytes: int = 2 * 1024 * 1024) -> str:
    """Read bounded regular UTF-8 content, rejecting symlinked path components.

    Callers must prevent concurrent path or content mutation during the read.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    path = Path(path)
    try:
        path = _directory(path.parent) / path.name
        with _open_file(path) as stream:
            data = stream.read(max_bytes + 1)
    except OSError as exc:
        raise ValueError(f"Expected readable regular artifact: {path}") from exc
    if len(data) > max_bytes:
        raise ValueError("Artifact exceeds read limit")
    return data.decode("utf-8")
