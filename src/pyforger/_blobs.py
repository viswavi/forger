"""Store large values in ``@op`` cache entries once, as content-addressed blobs.

Rows often carry the same large payloads through every step, e.g. a 47K-char
thinking trace or a list of 50 tests. Pickled as-is, each step's cache entry
holds its own copy. Here, when an entry is pickled, each large ``str`` or
``bytes`` and each large ``list`` or ``tuple`` is written once to
``<cache root>/blobs/<sha256>`` and the entry keeps only a reference, so the same
trace carried through six steps is stored once.

Entries pickled without blobs (by older Forger versions) load unchanged. A
reference whose blob file is gone raises ``MissingBlobError`` on load, which
``@op`` treats as a cache miss.

.. warning::
    Blobs of lists and tuples are pickles. Like the cache entries, they are only
    safe to load from a trusted cache directory.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import threading
from pathlib import Path
from typing import Any

# A str or bytes value at least this long (in characters or bytes) is hashed
# into cache keys by its digest, and the digest is memoized.
DIGEST_MIN_SIZE = 4096
# A str or bytes at least this long, or a list or tuple whose contents add up to
# it, is stored as a blob. Smaller str / bytes stay inline: a file per 5K-char
# test (13,000 for 260 rows) took 10s to write on macOS; a list of 50 such tests
# is one blob instead.
BLOB_MIN_SIZE = 16 * 1024

_PROTOCOL = 4

# sha256 digests of large str / bytes objects, by id. Each entry holds the object
# itself, so its id cannot be reused by another object while it is here. The same
# trace is hashed for a step's key, its ops' keys, the trace, and its blob name;
# this makes that one pass. Bounded by the total length of the objects held,
# which can keep up to that much of otherwise freed strings alive.
_DIGEST_MEMO_MAX = 256 * 1024 * 1024
_DIGESTS: dict[int, tuple[str | bytes, bytes]] = {}
_DIGESTS_SIZE = 0
_DIGESTS_LOCK = threading.Lock()


class MissingBlobError(FileNotFoundError):
    """A cache entry refers to a blob that is no longer on disk."""


def encoded(value: str | bytes) -> bytes:
    """The bytes a str or bytes value is hashed and stored as."""
    if isinstance(value, str):
        # surrogatepass: a lone surrogate (e.g. from a bad decode) still round-trips.
        return value.encode("utf-8", "surrogatepass")
    return bytes(value)


def digest(value: str | bytes) -> bytes:
    """sha256 of ``encoded(value)``; memoized for large values of exact type."""
    if len(value) < DIGEST_MIN_SIZE or type(value) not in (str, bytes):
        return hashlib.sha256(encoded(value)).digest()
    global _DIGESTS_SIZE
    hit = _DIGESTS.get(id(value))
    if hit is not None and hit[0] is value:
        return hit[1]
    result = hashlib.sha256(encoded(value)).digest()
    with _DIGESTS_LOCK:
        if _DIGESTS_SIZE + len(value) > _DIGEST_MEMO_MAX:
            # Drop the oldest half, so a run over more than the bound still hits.
            for key in list(_DIGESTS)[: len(_DIGESTS) // 2 + 1]:
                _DIGESTS_SIZE -= len(_DIGESTS.pop(key)[0])
        if id(value) not in _DIGESTS:
            _DIGESTS[id(value)] = (value, result)
            _DIGESTS_SIZE += len(value)
    return result


def _at_least(obj: Any, limit: int) -> bool:
    """True when obj's contents add up to ``limit`` bytes or more, roughly.

    Each node counts 8 plus its length for str / bytes, so the walk stops after
    at most ``limit / 8`` nodes. A str or bytes that is itself a blob counts as
    its reference, so a list of two thinking traces stays inline.
    """
    total = 0
    stack = [obj]
    while stack:
        x = stack.pop()
        total += 8
        if isinstance(x, (str, bytes)):
            total += 64 if len(x) >= BLOB_MIN_SIZE else len(x)
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        elif isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())
        if total >= limit:
            return True
    return False


def _write_blob(blob_dir: Path, data: bytes, hex_digest: str) -> None:
    path = blob_dir / hex_digest
    if path.exists():
        return
    tmp = blob_dir / f"{hex_digest}.tmp.{os.getpid()}.{threading.get_ident()}"
    tmp.write_bytes(data)
    os.replace(tmp, path)


class _BlobPickler(pickle.Pickler):
    """Pickle obj, writing its large values to ``blob_dir`` as references."""

    def __init__(
        self, file: io.BytesIO, blob_dir: Path, root: Any, active: set[int]
    ) -> None:
        super().__init__(file, protocol=_PROTOCOL)
        self._blob_dir = blob_dir
        self._root = root
        # Lists being pickled into blobs right now: a list that contains
        # itself is pickled inline (with pickle's own memo) inside its blob.
        self._active = active

    def persistent_id(self, obj: Any) -> tuple[str, str] | None:
        tp = type(obj)
        if tp is str or tp is bytes:
            if len(obj) < BLOB_MIN_SIZE:
                return None
            hex_digest = digest(obj).hex()
            _write_blob(self._blob_dir, encoded(obj), hex_digest)
            return ("s" if tp is str else "b", hex_digest)
        if (tp is list or tp is tuple) and obj is not self._root:
            if id(obj) in self._active or not _at_least(obj, BLOB_MIN_SIZE):
                return None
            data = _dumps(obj, self._blob_dir, self._active)
            hex_digest = hashlib.sha256(data).hexdigest()
            _write_blob(self._blob_dir, data, hex_digest)
            return ("p", hex_digest)
        return None


def dumps(obj: Any, blob_dir: Path) -> bytes:
    """Pickle obj like ``pickle.dumps``, storing its large values in ``blob_dir``."""
    blob_dir.mkdir(parents=True, exist_ok=True)
    return _dumps(obj, blob_dir, set())


def _dumps(obj: Any, blob_dir: Path, active: set[int]) -> bytes:
    active.add(id(obj))
    try:
        buf = io.BytesIO()
        _BlobPickler(buf, blob_dir, obj, active).dump(obj)
        return buf.getvalue()
    finally:
        active.discard(id(obj))


class _BlobUnpickler(pickle.Unpickler):
    def __init__(self, file: io.BytesIO, blob_dir: Path) -> None:
        super().__init__(file)
        self._blob_dir = blob_dir

    def persistent_load(self, pid: Any) -> Any:
        kind, hex_digest = pid
        if kind not in ("s", "b", "p") or not _is_hex_digest(hex_digest):
            raise pickle.UnpicklingError(f"unknown blob reference {pid!r}")
        path = self._blob_dir / hex_digest
        try:
            with path.open("rb") as f:
                # Like cache entries: only load blobs the current user wrote.
                if hasattr(os, "getuid") and os.fstat(f.fileno()).st_uid != os.getuid():
                    raise PermissionError(f"Blob {path} is owned by another user.")
                data = f.read()
        except FileNotFoundError as exc:
            raise MissingBlobError(f"cached value {path} is missing") from exc
        if kind == "s":
            return data.decode("utf-8", "surrogatepass")
        if kind == "b":
            return data
        return loads(data, self._blob_dir)


def _is_hex_digest(text: Any) -> bool:
    return (
        isinstance(text, str)
        and len(text) == 64
        and all(c in "0123456789abcdef" for c in text)
    )


def loads(data: bytes, blob_dir: Path) -> Any:
    """Unpickle what ``dumps`` wrote, reading its blobs from ``blob_dir``."""
    return _BlobUnpickler(io.BytesIO(data), blob_dir).load()
