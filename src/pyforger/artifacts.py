"""Verified, content-addressed directory artifacts for pipeline inputs."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Tuple

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


@dataclass(frozen=True)
class ArtifactFile:
    """One immutable manifest entry; paths use forward slashes."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _relative(self.path)
        if type(self.size) is not int or self.size < 0:
            raise ValueError("Artifact file size must be a nonnegative integer")
        _sha256(self.sha256)


def _digest(files: Tuple[ArtifactFile, ...]) -> str:
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


def _manifest(root: Path) -> Tuple[ArtifactFile, ...]:
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
                                path.relative_to(root).as_posix(),
                                size,
                                digest.hexdigest(),
                            )
                        )
                    else:
                        raise ValueError(
                            f"Artifact special file is not allowed: {path}"
                        )
        return tuple(sorted(files, key=lambda f: f.path))
    except OSError as exc:
        raise ValueError(f"Cannot read artifact {root}: {exc}") from exc


@dataclass(frozen=True)
class ArtifactRef:
    """A snapshot of regular files, independent of root and timestamps.

    Empty directories are excluded. Verification is not a filesystem lock:
    callers must prevent concurrent mutation while capturing or consuming files.
    Trusted ancestor aliases are normalized at construction; root and descendant
    symlinks are rejected during verification.
    """

    root: Path
    digest: str
    files: Tuple[ArtifactFile, ...]

    def __post_init__(self) -> None:
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

    @classmethod
    def capture(cls, root: Path) -> ArtifactRef:
        root = _normalize_root(root)
        files = _manifest(root)
        return cls(root, _digest(files), files)

    def verify(self) -> None:
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
        self.verify()
        return self.digest

    def path(self, relative: str) -> Path:
        relative = _relative(relative)
        self.verify()
        if relative not in {f.path for f in self.files}:
            raise ValueError(f"File is not in artifact manifest: {relative!r}")
        return self.root / relative

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "digest": self.digest,
            "files": [
                {"path": f.path, "size": f.size, "sha256": f.sha256} for f in self.files
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactRef:
        try:
            if not isinstance(data["root"], str) or not isinstance(data["files"], list):
                raise ValueError("Invalid artifact ledger root or files")
            return cls(
                Path(data["root"]),
                data["digest"],
                tuple(ArtifactFile(**f) for f in data["files"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid artifact ledger: {exc}") from exc


class ArtifactStore:
    """Atomic directory publication in a trusted, content-addressed local store."""

    def __init__(self, root: Path):
        self.root = _normalize_root(root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            _directory(self.root)
        except OSError as exc:
            raise ValueError(f"Cannot create artifact store: {exc}") from exc

    def get(self, digest: str) -> ArtifactRef:
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
                with _open_file(ref.root / entry.path) as stream, target.open(
                    "wb"
                ) as output:
                    shutil.copyfileobj(stream, output, _CHUNK_SIZE)
            ref.verify()
            ArtifactRef(staged, ref.digest, ref.files).verify()
            try:
                staged.rename(destination)
            except OSError:
                if not os.path.lexists(destination):
                    raise
            return self.get(ref.digest)
