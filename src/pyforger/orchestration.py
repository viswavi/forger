"""SQLite-backed verified stage attempts with explicit retry and recovery."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from pyforger.artifacts import ArtifactRef


class StageConflict(RuntimeError):
    """An attempt requires explicit retry/recovery, or has lost ownership."""


def _json(value: Any) -> str:
    def validate(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, list):
            for child in item:
                validate(child)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise ValueError("Value must be JSON-compatible")

    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class _Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            schema = """
                CREATE TABLE IF NOT EXISTS stage_attempts (
                    identity TEXT NOT NULL, attempt INTEGER NOT NULL,
                    stage_id TEXT NOT NULL, inputs TEXT NOT NULL, version TEXT NOT NULL,
                    output TEXT NOT NULL UNIQUE, token TEXT NOT NULL,
                    state TEXT NOT NULL,
                    result TEXT, artifact TEXT, error TEXT,
                    PRIMARY KEY(identity, attempt));
            """
            for statement in schema.split(";"):
                if statement.strip():
                    db.execute(statement)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()


class RunLedger(_Store):
    """Run actions in fresh output directories; never implicitly retry an attempt.

    ``retry=True`` permits a new failed-stage attempt. ``recover=True`` fences an
    interrupted running attempt; the caller must first stop that action. Both
    require a new output path, preserving earlier partial outputs. Inspection
    records contain results and serialized ArtifactRefs, without altering results.
    For an initialized stage directory, pass ``action_subdir="attempt-1"``;
    the action receives that freshly created child root, not the stage directory.
    """

    def __init__(self, path: Path):
        """Initialize the ledger and migrate optional result-artifact metadata."""
        super().__init__(path)
        with self._transaction() as db:
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(stage_attempts)")
            }
            if "result_artifacts" not in columns:
                db.execute(
                    "ALTER TABLE stage_attempts ADD COLUMN result_artifacts TEXT"
                )
            db.execute(
                "CREATE TABLE IF NOT EXISTS run_binding ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                "configuration TEXT NOT NULL, version TEXT NOT NULL)"
            )

    def bind_run(
        self, configuration: dict[str, Any], implementation_version: str
    ) -> None:
        """Atomically bind this ledger to exact JSON configuration and version.

        Binding is optional and applies to the entire ledger, not one stage.
        Rebinding identical values is idempotent; changes raise StageConflict.
        """
        if not isinstance(configuration, dict):
            raise ValueError("configuration must be a JSON object")
        self._validate_version(implementation_version)
        encoded = _json(configuration)
        with self._transaction() as db:
            previous = db.execute(
                "SELECT * FROM run_binding WHERE singleton=1"
            ).fetchone()
            if previous is None:
                db.execute(
                    "INSERT INTO run_binding VALUES (1,?,?)",
                    (encoded, implementation_version),
                )
            elif (previous["configuration"], previous["version"]) != (
                encoded,
                implementation_version,
            ):
                raise StageConflict(
                    "Ledger is bound to a different run configuration or version"
                )

    @staticmethod
    def _validate_version(version: str) -> None:
        if not isinstance(version, str) or not version:
            raise ValueError("implementation_version must be a nonempty string")

    @classmethod
    def _identity(cls, stage_id: str, inputs: dict[str, Any], version: str) -> str:
        if not isinstance(stage_id, str) or not stage_id:
            raise ValueError("stage_id must be a nonempty string")
        cls._validate_version(version)
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be a JSON object")
        return hashlib.sha256(_json([stage_id, inputs, version]).encode()).hexdigest()

    @staticmethod
    def _decode_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        for field in ("inputs", "result", "artifact", "result_artifacts"):
            if record[field] is not None:
                record[field] = json.loads(record[field])
        return record

    def get_record(
        self, stage_id: str, inputs: dict[str, Any], implementation_version: str
    ) -> Optional[dict[str, Any]]:
        """Return the latest attempt for an exact identity without verifying files."""
        identity = self._identity(stage_id, inputs, implementation_version)
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM stage_attempts WHERE identity=? "
                "ORDER BY attempt DESC LIMIT 1",
                (identity,),
            ).fetchone()
        return self._decode_record(row) if row is not None else None

    def stage(
        self,
        stage_id: str,
        inputs: dict[str, Any],
        action: Callable[[Path], dict[str, Any]],
        *,
        output: Path,
        implementation_version: str,
        retry: bool = False,
        recover: bool = False,
        artifact_refs: Optional[
            Callable[[dict[str, Any]], Iterable[ArtifactRef]]
        ] = None,
    ) -> dict[str, Any]:
        """Run an artifact-producing function in an absent attempt payload.

        The ledger owns the attempt directory; the action creates its ``payload``
        child. Failed payloads are preserved and explicit retries receive a fresh
        directory. This is a convenience over run_stage, not a pipeline engine.
        """
        record = self.get_record(stage_id, inputs, implementation_version)
        attempt = record["attempt"] if record else 1
        if record and record["state"] != "completed" and (retry or recover):
            attempt += 1
        directory = Path(output)
        if attempt > 1:
            directory = directory.with_name(directory.name + f"-attempt-{attempt}")
        return self.run_stage(
            stage_id,
            inputs,
            lambda root: action(root / "payload"),
            output=directory,
            implementation_version=implementation_version,
            retry=retry,
            recover=recover,
            artifact_refs=artifact_refs,
        )

    def run_stage(
        self,
        stage_id: str,
        inputs: dict[str, Any],
        action: Callable[[Path], dict[str, Any]],
        *,
        output: Path,
        implementation_version: str,
        retry: bool = False,
        recover: bool = False,
        action_subdir: Optional[str] = None,
        artifact_refs: Optional[
            Callable[[dict[str, Any]], Iterable[ArtifactRef]]
        ] = None,
    ) -> dict[str, Any]:
        """Run a fresh stage attempt or reuse its verified completed result.

        ``artifact_refs`` explicitly selects ArtifactRef objects from a JSON result.
        It runs once after a successful action, on a decoded copy of the result;
        returned refs are verified and persisted independently of user JSON.
        Resume verifies stored refs even if the callback is omitted, and never
        invokes either callback. Adding a contract to an untracked completed
        attempt fails closed; use a new implementation version instead.
        """
        identity = self._identity(stage_id, inputs, implementation_version)
        encoded = _json(inputs)
        if artifact_refs is not None and not callable(artifact_refs):
            raise ValueError("artifact_refs must be a callable")
        output = Path(os.path.abspath(output))
        if output.is_symlink():
            raise ValueError(f"Output symlink is not allowed: {output}")
        if action_subdir is not None:
            if (
                not isinstance(action_subdir, str)
                or not action_subdir
                or action_subdir in (".", "..")
                or "/" in action_subdir
                or "\\" in action_subdir
            ):
                raise ValueError("action_subdir must be a single directory name")
            output = output / action_subdir
        if output.is_symlink():
            raise ValueError(f"Output symlink is not allowed: {output}")
        output = output.parent.resolve() / output.name
        token = uuid.uuid4().hex
        with self._transaction() as db:
            previous = db.execute(
                "SELECT * FROM stage_attempts WHERE identity=? "
                "ORDER BY attempt DESC LIMIT 1",
                (identity,),
            ).fetchone()
            if previous and previous["state"] == "completed":
                if previous["output"] != str(output):
                    raise StageConflict(
                        "Completed stage is bound to a different output path"
                    )
                ref = ArtifactRef.model_validate(json.loads(previous["artifact"]))
                ref.verify()
                if previous["result_artifacts"] is None:
                    if artifact_refs is not None:
                        raise StageConflict(
                            "Completed stage has no result artifact contract"
                        )
                else:
                    for data in json.loads(previous["result_artifacts"]):
                        ArtifactRef.model_validate(data).verify()
                cached_result: dict[str, Any] = json.loads(previous["result"])
                return cached_result
            if previous:
                if previous["state"] == "running" and not recover:
                    raise StageConflict("Running stage requires explicit recovery")
                if previous["state"] != "running" and not retry:
                    raise StageConflict("Failed stage requires explicit retry")
                db.execute(
                    "UPDATE stage_attempts SET state='superseded' "
                    "WHERE identity=? AND attempt=? AND state='running'",
                    (identity, previous["attempt"]),
                )
            attempt = previous["attempt"] + 1 if previous else 1
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)
            try:
                db.execute(
                    "INSERT INTO stage_attempts "
                    "(identity,attempt,stage_id,inputs,version,output,token,state) "
                    "VALUES (?,?,?,?,?,?,?,'running')",
                    (
                        identity,
                        attempt,
                        stage_id,
                        encoded,
                        implementation_version,
                        str(output),
                        token,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StageConflict("Output path is already reserved") from exc
        try:
            output.mkdir(parents=True, exist_ok=False)
            result = action(output)
            if not isinstance(result, dict):
                raise ValueError("Stage result must be a JSON object")
            serialized = _json(result)
            result_artifacts = None
            if artifact_refs is not None:
                selected = artifact_refs(json.loads(serialized))
                if isinstance(selected, (dict, str, bytes)):
                    raise ValueError(
                        "artifact_refs must return an iterable of ArtifactRef objects"
                    )
                try:
                    refs = list(selected)
                except TypeError as exc:
                    raise ValueError(
                        "artifact_refs must return an iterable of ArtifactRef objects"
                    ) from exc
                for selected_ref in refs:
                    if not isinstance(selected_ref, ArtifactRef):
                        raise ValueError(
                            "artifact_refs must return ArtifactRef objects"
                        )
                    selected_ref.verify()
                result_artifacts = _json(
                    [selected_ref.model_dump(mode="json") for selected_ref in refs]
                )
            ref = ArtifactRef.capture(output)
            artifact = _json(ref.model_dump(mode="json"))
            with self._transaction() as db:
                updated = db.execute(
                    "UPDATE stage_attempts SET state='completed',result=?,artifact=?,"
                    "result_artifacts=? "
                    "WHERE identity=? AND attempt=? AND token=? AND state='running'",
                    (serialized, artifact, result_artifacts, identity, attempt, token),
                ).rowcount
                if not updated:
                    raise StageConflict("Stage attempt lost ownership")
            decoded_result: dict[str, Any] = json.loads(serialized)
            return decoded_result
        except BaseException as exc:
            with self._transaction() as db:
                db.execute(
                    "UPDATE stage_attempts SET state='failed',error=? "
                    "WHERE identity=? AND attempt=? AND token=? AND state='running'",
                    (f"{type(exc).__name__}: {exc}", identity, attempt, token),
                )
            raise

    def inspect(self) -> list[dict[str, Any]]:
        """Return decoded attempt records without executing or altering stages."""
        with self._transaction() as db:
            return [
                self._decode_record(row)
                for row in db.execute(
                    "SELECT * FROM stage_attempts ORDER BY stage_id,identity,attempt"
                )
            ]
