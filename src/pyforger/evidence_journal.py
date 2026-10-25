"""Append-only private event storage and integrity-checked portable readers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import weakref
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pyforger.agent_diagnostics import sanitizer
from pyforger.artifacts import FileInput, _directory, _open_file


class _PrivateStorage:
    def __init__(self, path: Path):
        self.path = path
        self.cleanup = weakref.finalize(self, path.unlink, missing_ok=True)


class _JournalReference(dict):
    __slots__ = ("__weakref__", "_sealed", "_storage")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sealed = tuple(sorted(self.items()))


_PRIVATE_JOURNALS: weakref.WeakValueDictionary[int, _JournalReference] = (
    weakref.WeakValueDictionary()
)


def is_private_reference(value: Any) -> bool:
    return (
        isinstance(value, _JournalReference)
        and _PRIVATE_JOURNALS.get(id(value)) is value
        and value._sealed == tuple(sorted(value.items()))
    )


class EvidenceJournal:
    """Constant-size controller state, with one JSON event per durable line."""

    def __init__(self, specs: Iterable[Any] = ()):
        fd, name = tempfile.mkstemp(prefix="forger-evidence-", suffix=".jsonl")
        os.close(fd)
        self.path = Path(name)
        self._storage = _PrivateStorage(self.path)
        self.clean = sanitizer(specs)
        self._digest = hashlib.sha256()
        self.ref = {
            "path": name,
            "size": 0,
            "sha256": self._digest.hexdigest(),
            "count": 0,
        }

    def append(self, event: Any) -> None:
        self.append_sanitized(
            self.clean(
                event if isinstance(event, dict) else event.model_dump(mode="json")
            )
        )

    def append_sanitized(self, event: Any) -> None:
        """Accept only events from the credential-sanitized private worker channel."""
        encoded = (
            json.dumps(event, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode()
        with self.path.open("ab", buffering=0) as target:
            target.write(encoded)
            os.fsync(target.fileno())
        self._digest.update(encoded)
        self.ref.update(
            size=self.ref["size"] + len(encoded),
            sha256=self._digest.hexdigest(),
            count=self.ref["count"] + 1,
        )

    def trajectory(self) -> dict[str, Any]:
        snapshot = _JournalReference(self.ref)
        snapshot._storage = self._storage
        _PRIVATE_JOURNALS[id(snapshot)] = snapshot
        return {"journal": snapshot}


def iter_trajectory_events(
    value: dict[str, Any], *, root: Path | None = None
) -> Iterator[Any]:
    """Read legacy steps or verify a journal before yielding ordered events."""
    if "journal" not in value:
        yield from value.get("steps", ())
        return
    ref = value["journal"]
    if (
        not isinstance(ref, dict)
        or set(ref) != {"path", "size", "sha256", "count"}
        or not isinstance(ref["path"], str)
        or type(ref["count"]) is not int
        or ref["count"] < 0
    ):
        raise ValueError("Invalid journal reference")
    if root is None and not is_private_reference(ref):
        raise ValueError(
            "Unknown private journal; an explicit contained root is required"
        )
    path = Path(ref["path"])
    if root is not None:
        from pyforger.agent_backend import validate_workspace_path

        validate_workspace_path(ref["path"])
        path = Path(root).absolute() / path
        _directory(path.parent)
    source = FileInput(source=path, size=ref["size"], sha256=ref["sha256"])
    _directory(source.source.parent)
    with _open_file(source.source) as stream:
        remaining = source.size
        digest = hashlib.sha256()
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Incomplete journal")
            digest.update(chunk)
            remaining -= len(chunk)
        if digest.hexdigest() != source.sha256:
            raise ValueError("Journal digest mismatch")
        stream.seek(0)
        remaining = source.size
        count = 0
        digest = hashlib.sha256()
        while remaining:
            line = stream.readline(remaining)
            remaining -= len(line)
            if not line.endswith(b"\n"):
                raise ValueError("Incomplete journal record")
            digest.update(line)
            count += 1
            yield json.loads(line)
        if count != ref["count"] or digest.hexdigest() != source.sha256:
            raise ValueError("Journal changed or record count mismatch")


def journal_events(events: Iterable[Any], specs: Iterable[Any]) -> dict[str, Any]:
    journal = EvidenceJournal(specs)
    for event in events:
        journal.append(event)
    return journal.trajectory()


def evidence_file_chunks(chunks: Iterable[bytes]) -> tuple[dict[str, Any], str]:
    """Persist already sanitized worker bytes with a bounded display tail."""
    fd, name = tempfile.mkstemp(prefix="forger-log-", suffix=".txt")
    digest = hashlib.sha256()
    size = 0
    tail = b""
    with os.fdopen(fd, "wb") as stream:
        for chunk in chunks:
            stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            tail = (tail + chunk)[-32000:]
        stream.flush()
        os.fsync(stream.fileno())
    ref = _JournalReference(path=name, size=size, sha256=digest.hexdigest())
    ref._storage = _PrivateStorage(Path(name))
    _PRIVATE_JOURNALS[id(ref)] = ref
    return {"evidence_file": ref}, tail.decode("utf-8", errors="replace")[-8000:]


def evidence_file(text: str) -> dict[str, Any]:
    return evidence_file_chunks([text.encode()])[0]


def materialize_journals(
    value: Any, output: Path, specs: Iterable[Any], _memo: dict | None = None
) -> Any:
    """Relocate only provenance-verified private journals into a trial directory."""
    if _memo is None:
        _memo = {}
    if isinstance(value, list):
        return [materialize_journals(v, output, specs, _memo) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "evidence_file":
            if not is_private_reference(item):
                raise ValueError("Unknown private evidence file")
            identity = (item["path"], item["size"], item["sha256"])
            if identity not in _memo:
                destination = output / "agent/logs" / (item["sha256"] + ".txt")
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                _directory(destination.parent)
                if not destination.exists():
                    fd, temporary = tempfile.mkstemp(
                        dir=destination.parent, prefix=".log-"
                    )
                    try:
                        with os.fdopen(fd, "wb") as stream:
                            source = FileInput(
                                source=Path(item["path"]),
                                size=item["size"],
                                sha256=item["sha256"],
                            )
                            for chunk in source.chunks():
                                stream.write(chunk)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, destination)
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
                else:
                    captured = FileInput.capture(destination)
                    if (
                        captured.size != item["size"]
                        or captured.sha256 != item["sha256"]
                    ):
                        raise ValueError("Published log integrity mismatch")
                _memo[identity] = {
                    **item,
                    "path": destination.relative_to(output).as_posix(),
                }
            result[key] = _memo[identity]
            continue
        if key != "journal":
            result[key] = materialize_journals(item, output, specs, _memo)
            continue
        # A schema-shaped object never grants permission to copy raw content.
        trusted = is_private_reference(item)
        if not trusted:
            raise ValueError("Unknown private journal reference")
        identity = (item["path"], item["size"], item["sha256"], item["count"])
        if identity in _memo:
            result[key] = _memo[identity]
            continue
        destination = output / "agent/journals" / (item["sha256"] + ".jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _directory(destination.parent)
        if not destination.exists():
            fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".journal-")
            try:
                with os.fdopen(fd, "wb") as stream:
                    for event in iter_trajectory_events({"journal": item}):
                        stream.write(
                            (
                                json.dumps(event, ensure_ascii=True, allow_nan=False)
                                + "\n"
                            ).encode()
                        )
                    stream.flush()
                    os.fsync(stream.fileno())
                # The private canonical JSON encoding is preserved exactly.
                captured = FileInput.capture(Path(temporary))
                if captured.size != item["size"] or captured.sha256 != item["sha256"]:
                    raise ValueError("Journal changed during publication")
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        else:
            captured = FileInput.capture(destination)
            if captured.size != item["size"] or captured.sha256 != item["sha256"]:
                raise ValueError("Published journal integrity mismatch")
        result[key] = {**item, "path": destination.relative_to(output).as_posix()}
        _memo[identity] = result[key]
    return result


def has_private_evidence(value: Any) -> bool:
    """Detect capabilities that must be published before JSON normalization."""
    if is_private_reference(value):
        return True
    if isinstance(value, dict):
        return any(has_private_evidence(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_private_evidence(item) for item in value)
    return False


def publish_private_evidence(value: Any, output: Path) -> Any:
    """Publish live private capabilities before report serialization.

    Unrelated objects and portable/external schema-shaped references are preserved;
    only exact in-process identity capabilities authorize filesystem reads.
    """
    memo: dict = {}

    def publish(item: Any) -> Any:
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                if key in {"journal", "evidence_file"} and is_private_reference(child):
                    result[key] = materialize_journals({key: child}, output, [], memo)[
                        key
                    ]
                else:
                    result[key] = publish(child)
            return result
        if isinstance(item, (list, tuple)):
            return [publish(child) for child in item]
        return item

    return publish(value)
