"""Durability and contention tests using actual SQLite files and artifacts."""
import hashlib
import json
import multiprocessing
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pyforger.artifacts import ArtifactRef
from pyforger.orchestration import RunLedger, StageConflict


def write_output(path):
    """Write a deterministic result into a fresh stage output directory."""
    assert path.is_dir()
    assert not list(path.iterdir())
    (path / "data.json").write_text('{"value": 1}')
    return {"answer": [1, True, None], "resumed": "user supplied"}


def interrupt_stage(database, output):
    """Leave a partial stage artifact by terminating the worker process."""

    def die(path):
        (path / "partial").write_text("retained")
        os._exit(7)

    RunLedger(Path(database)).run_stage(
        "interrupted", {}, die, output=Path(output), implementation_version="1"
    )


def test_stage_restart_identity_and_artifact(tmp_path):
    """Restarted ledgers reuse verified results only for matching identities."""
    database = tmp_path / "shared.db"
    output = tmp_path / "output"
    ledger = RunLedger(database)
    expected = ledger.run_stage(
        "build",
        {"b": 2, "a": 1},
        write_output,
        output=output,
        implementation_version="1",
    )

    def forbidden(path):
        pytest.fail("Resumed stage must not execute")

    restarted = RunLedger(database)
    assert (
        restarted.run_stage(
            "build",
            {"a": 1, "b": 2},
            forbidden,
            output=output,
            implementation_version="1",
        )
        == expected
    )
    record = restarted.inspect()[0]
    assert record["state"] == "completed"
    assert ArtifactRef.model_validate(record["artifact"]).verify() is not False
    for stage, inputs, version in [
        ("other", {"a": 1, "b": 2}, "1"),
        ("build", {"a": 2, "b": 2}, "1"),
        ("build", {"a": 1, "b": 2}, "2"),
    ]:
        restarted.run_stage(
            stage,
            inputs,
            write_output,
            output=tmp_path / f"new-{len(restarted.inspect())}",
            implementation_version=version,
        )
    assert len(restarted.inspect()) == 4
    with pytest.raises(StageConflict):
        restarted.run_stage(
            "build",
            {"a": 1, "b": 2},
            forbidden,
            output=tmp_path / "different",
            implementation_version="1",
        )


@pytest.mark.parametrize("damage", ["tamper", "missing"])
def test_stage_integrity_fails_closed(tmp_path, damage):
    """Missing or tampered stage artifacts prevent completed-result reuse."""
    ledger = RunLedger(tmp_path / "db")
    output = tmp_path / "output"
    ledger.run_stage(
        "stage", {}, write_output, output=output, implementation_version="v"
    )
    data = output / "data.json"
    if damage == "tamper":
        data.write_text("modified")
    else:
        data.unlink()
    with pytest.raises(ValueError, match="Artifact verification failed"):
        RunLedger(tmp_path / "db").run_stage(
            "stage",
            {},
            write_output,
            output=output,
            implementation_version="v",
            retry=True,
            recover=True,
        )
    assert not data.exists() if damage == "missing" else data.read_text() == "modified"
    assert ledger.inspect()[0]["state"] == "completed"


def test_failed_attempt_preserved_and_explicit_retry(tmp_path):
    """Failed attempts retain partial outputs and require explicit fresh retries."""
    ledger = RunLedger(tmp_path / "db")
    output = tmp_path / "partial"

    def fail(path):
        (path / "partial").write_text("keep")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        ledger.run_stage("stage", {}, fail, output=output, implementation_version="v")
    assert ledger.inspect()[0]["error"] == "RuntimeError: boom"
    assert (output / "partial").read_text() == "keep"
    with pytest.raises(StageConflict):
        RunLedger(tmp_path / "db").run_stage(
            "stage",
            {},
            write_output,
            output=tmp_path / "next",
            implementation_version="v",
        )
    with pytest.raises(FileExistsError):
        ledger.run_stage(
            "stage",
            {},
            write_output,
            output=output,
            implementation_version="v",
            retry=True,
        )
    ledger.run_stage(
        "stage",
        {},
        write_output,
        output=tmp_path / "next",
        implementation_version="v",
        retry=True,
    )
    assert [r["state"] for r in ledger.inspect()] == ["failed", "completed"]


def test_interrupted_process_requires_recovery(tmp_path):
    """Interrupted stages require explicit recovery and preserve partial outputs."""
    database = tmp_path / "db"
    output = tmp_path / "partial"
    process = multiprocessing.get_context("spawn").Process(
        target=interrupt_stage, args=(str(database), str(output))
    )
    process.start()
    process.join(20)
    assert not process.is_alive()
    assert process.exitcode == 7
    ledger = RunLedger(database)
    with pytest.raises(StageConflict):
        ledger.run_stage(
            "interrupted",
            {},
            write_output,
            output=tmp_path / "next",
            implementation_version="1",
            retry=True,
        )
    ledger.run_stage(
        "interrupted",
        {},
        write_output,
        output=tmp_path / "next",
        implementation_version="1",
        recover=True,
    )
    assert [r["state"] for r in ledger.inspect()] == ["superseded", "completed"]
    assert (output / "partial").read_text() == "retained"


def test_stage_claim_and_action_do_not_hold_database_lock(tmp_path):
    """Running actions fence duplicate claims without locking unrelated work."""
    database = tmp_path / "db"
    started, release = threading.Event(), threading.Event()
    calls = []

    def action(path):
        calls.append(path)
        started.set()
        assert release.wait(10)
        return write_output(path)

    first, second = RunLedger(database), RunLedger(database)
    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(
            first.run_stage,
            "stage",
            {},
            action,
            output=tmp_path / "first",
            implementation_version="v",
        )
        try:
            assert started.wait(10)
            with pytest.raises(StageConflict):
                second.run_stage(
                    "stage",
                    {},
                    action,
                    output=tmp_path / "second",
                    implementation_version="v",
                )
            second.run_stage(
                "unrelated",
                {},
                write_output,
                output=tmp_path / "unrelated",
                implementation_version="v",
            )
        finally:
            release.set()
        future.result(10)
    assert len(calls) == 1


def test_recovery_fences_old_stage_completion(tmp_path):
    """Recovered attempts prevent old workers from publishing completion."""
    database = tmp_path / "db"
    started, release = threading.Event(), threading.Event()

    def slow(path):
        started.set()
        assert release.wait(10)
        return write_output(path)

    with ThreadPoolExecutor(1) as pool:
        old = pool.submit(
            RunLedger(database).run_stage,
            "stage",
            {},
            slow,
            output=tmp_path / "old",
            implementation_version="v",
        )
        try:
            assert started.wait(10)
            RunLedger(database).run_stage(
                "stage",
                {},
                write_output,
                output=tmp_path / "new",
                implementation_version="v",
                recover=True,
            )
        finally:
            release.set()
        with pytest.raises(StageConflict):
            old.result(10)
    assert [r["state"] for r in RunLedger(database).inspect()] == [
        "superseded",
        "completed",
    ]


def expire(database):
    # Expire persisted leases directly rather than mocking clocks or sleeping.
    """Expire persisted running leases directly in the queue database."""
    with sqlite3.connect(database) as db:
        db.execute(
            "UPDATE work_jobs SET expires=? WHERE state='running'", (time.time() - 1,)
        )


@pytest.mark.parametrize(
    "invalid", [{1: "key"}, {"nan": float("nan")}, {"tuple": (1,)}]
)
def test_stage_invalid_json_prevents_execution(tmp_path, invalid):
    """Invalid JSON prevents stage execution or fails result publication."""
    ledger = RunLedger(tmp_path / "db")
    with pytest.raises(ValueError):
        ledger.run_stage(
            "stage",
            invalid,
            write_output,
            output=tmp_path / "output",
            implementation_version="v",
        )
    assert ledger.inspect() == []
    with pytest.raises(ValueError):
        ledger.run_stage(
            "result",
            {},
            lambda path: invalid,
            output=tmp_path / "result",
            implementation_version="v",
        )
    assert ledger.inspect()[0]["state"] == "failed"


def test_missing_output_and_preexisting_directory_fail_closed(tmp_path):
    """Existing output paths and missing completed artifacts fail closed."""
    ledger = RunLedger(tmp_path / "db")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("untouched")
    with pytest.raises(FileExistsError):
        ledger.run_stage(
            "stage", {}, write_output, output=output, implementation_version="v"
        )
    assert ledger.inspect() == []
    output = tmp_path / "output"
    ledger.run_stage(
        "stage", {}, write_output, output=output, implementation_version="v"
    )
    (output / "data.json").unlink()
    output.rmdir()
    with pytest.raises(ValueError, match="Cannot read artifact"):
        ledger.run_stage(
            "stage", {}, write_output, output=output, implementation_version="v"
        )
    assert not output.exists()


def test_initialized_stage_directory_uses_reserved_action_subdir(tmp_path):
    """Reserved child directories preserve initialized stage metadata."""
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    (stage_dir / "spec.json").write_text('{"initialized": true}')
    ledger = RunLedger(tmp_path / "db")
    result = ledger.run_stage(
        "stage",
        {},
        write_output,
        output=stage_dir,
        implementation_version="v",
        action_subdir="attempt-1",
    )
    assert (stage_dir / "attempt-1" / "data.json").is_file()
    assert (stage_dir / "spec.json").read_text() == '{"initialized": true}'
    assert Path(ledger.inspect()[0]["artifact"]["root"]) == stage_dir / "attempt-1"
    assert (
        RunLedger(tmp_path / "db").run_stage(
            "stage",
            {},
            write_output,
            output=stage_dir,
            implementation_version="v",
            action_subdir="attempt-1",
        )
        == result
    )
    for invalid in [
        "",
        ".",
        "..",
        "../escape",
        "/absolute",
        "nested/child",
        "nested\\child",
    ]:
        with pytest.raises(ValueError, match="action_subdir"):
            ledger.run_stage(
                "stage",
                {},
                write_output,
                output=stage_dir,
                implementation_version="v",
                action_subdir=invalid,
            )


@pytest.mark.parametrize("damage", ["tamper", "additional", "missing"])
def test_result_artifacts_verified_on_resume(tmp_path, damage):
    """Explicit external references are retained and verified without callbacks."""
    external = tmp_path / "external"
    external.mkdir()
    (external / "data").write_text("original")
    ref = ArtifactRef.capture(external)
    calls = []

    def action(path):
        calls.append("action")
        return {"external": ref.model_dump(mode="json")}

    def references(result):
        calls.append("references")
        return [ArtifactRef.model_validate(result["external"])]

    options = dict(
        output=tmp_path / "output", implementation_version="v", artifact_refs=references
    )
    ledger = RunLedger(tmp_path / "db")
    expected = ledger.run_stage("stage", {}, action, **options)
    assert (
        RunLedger(tmp_path / "db").run_stage("stage", {}, action, **options) == expected
    )
    assert calls == ["action", "references"]
    assert ledger.get_record("stage", {}, "v")["result_artifacts"] == [
        ref.model_dump(mode="json")
    ]
    if damage == "tamper":
        (external / "data").write_text("modified")
    elif damage == "additional":
        (external / "extra").write_text("new")
    else:
        (external / "data").unlink()
    with pytest.raises(ValueError, match="Artifact verification failed"):
        RunLedger(tmp_path / "db").run_stage("stage", {}, action, **options)
    assert calls == ["action", "references"]


def test_run_binding_exact_and_concurrent(tmp_path):
    """Canonical configuration binding is atomic and rejects changed runs."""
    database = tmp_path / "db"
    ledgers = [RunLedger(database) for _ in range(8)]
    barrier = threading.Barrier(8)

    def bind(index):
        barrier.wait()
        try:
            ledgers[index].bind_run({"choice": index % 2}, "v")
            return index % 2
        except StageConflict:
            return None

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(bind, range(8)))
    winners = [value for value in results if value is not None]
    assert len(winners) == 4
    assert len(set(winners)) == 1
    restarted = RunLedger(database)
    restarted.bind_run({"choice": winners[0]}, "v")
    with pytest.raises(StageConflict):
        restarted.bind_run({"choice": winners[0]}, "changed")
    with pytest.raises(StageConflict):
        restarted.bind_run({"choice": 1 - winners[0]}, "v")


@pytest.mark.parametrize("invalid", [None, {}, "reference", ["reference"]])
def test_artifact_callback_requires_explicit_refs(tmp_path, invalid):
    """Arbitrary JSON is never interpreted as an artifact reference."""
    ledger = RunLedger(tmp_path / "db")
    with pytest.raises(ValueError, match="ArtifactRef"):
        ledger.run_stage(
            "stage",
            {},
            write_output,
            output=tmp_path / "output",
            implementation_version="v",
            artifact_refs=lambda result: invalid,
        )
    assert ledger.inspect()[0]["state"] == "failed"


def test_external_artifact_verified_before_completion(tmp_path):
    """Stale references cannot be published as a completed result."""
    external = tmp_path / "external"
    external.mkdir()
    (external / "data").write_text("original")
    ref = ArtifactRef.capture(external)
    (external / "data").write_text("modified")
    ledger = RunLedger(tmp_path / "db")
    with pytest.raises(ValueError, match="Artifact verification failed"):
        ledger.run_stage(
            "stage",
            {},
            write_output,
            output=tmp_path / "output",
            implementation_version="v",
            artifact_refs=lambda result: [ref],
        )
    assert ledger.inspect()[0]["state"] == "failed"


def test_run_binding_and_record_identity(tmp_path):
    """Lookups distinguish input/version identities and bindings preserve types."""
    ledger = RunLedger(tmp_path / "db")
    ledger.bind_run({"b": [1, True], "a": None}, "v")
    ledger.bind_run({"a": None, "b": [1, True]}, "v")
    with pytest.raises(StageConflict):
        ledger.bind_run({"b": [True, 1], "a": None}, "v")
    assert ledger.get_record("stage", {}, "v") is None
    for index, (inputs, version) in enumerate([({}, "v"), ({"x": 1}, "v"), ({}, "v2")]):
        ledger.run_stage(
            "stage",
            inputs,
            write_output,
            output=tmp_path / str(index),
            implementation_version=version,
        )
        record = ledger.get_record("stage", inputs, version)
        assert record["inputs"] == inputs
        assert record["version"] == version
        assert record["attempt"] == 1


def test_output_ancestor_alias_normalized_and_leaf_symlink_rejected(tmp_path):
    """Trusted ancestor aliases share reservations; leaf aliases are forbidden."""
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    ledger = RunLedger(tmp_path / "db")
    expected = ledger.run_stage(
        "stage", {}, write_output, output=alias / "output", implementation_version="v"
    )
    assert (
        ledger.run_stage(
            "stage",
            {},
            write_output,
            output=real / "output",
            implementation_version="v",
        )
        == expected
    )
    leaf = tmp_path / "leaf"
    leaf.symlink_to(real / "output", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        ledger.run_stage(
            "leaf", {}, write_output, output=leaf, implementation_version="v"
        )


def test_result_json_not_implicitly_parsed_or_mutated(tmp_path):
    """Only explicit refs are tracked, and selectors cannot mutate results."""
    ledger = RunLedger(tmp_path / "db")
    result = {"root": "not an artifact", "digest": "invalid", "files": []}

    def select(value):
        value.clear()
        return iter([])

    options = dict(output=tmp_path / "output", implementation_version="v")
    assert (
        ledger.run_stage(
            "stage", {}, lambda path: result, artifact_refs=select, **options
        )
        == result
    )
    assert ledger.run_stage("stage", {}, write_output, **options) == result
    assert ledger.get_record("stage", {}, "v")["result_artifacts"] == []


def test_stored_refs_verified_when_callback_omitted(tmp_path):
    """Omitting a selector on resume cannot bypass a persisted contract."""
    external = tmp_path / "external"
    external.mkdir()
    ref = ArtifactRef.capture(external)
    ledger = RunLedger(tmp_path / "db")
    options = dict(output=tmp_path / "output", implementation_version="v")
    ledger.run_stage(
        "stage", {}, write_output, artifact_refs=lambda result: [ref], **options
    )
    external.rmdir()
    with pytest.raises(ValueError, match="Cannot read artifact"):
        ledger.run_stage("stage", {}, write_output, **options)


def test_legacy_ledger_migration_and_untracked_contract(tmp_path):
    """Legacy records migrate without inventing contracts or running callbacks."""
    database = tmp_path / "db"
    output = tmp_path / "output"
    output.mkdir()
    expected = write_output(output)
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE TABLE stage_attempts (identity TEXT NOT NULL, "
            "attempt INTEGER NOT NULL, "
            "stage_id TEXT NOT NULL, inputs TEXT NOT NULL, version TEXT NOT NULL, "
            "output TEXT NOT NULL UNIQUE, token TEXT NOT NULL, state TEXT NOT NULL, "
            "result TEXT, artifact TEXT, error TEXT, "
            "PRIMARY KEY(identity, attempt))"
        )
        identity = hashlib.sha256(b'["stage",{},"v"]').hexdigest()
        db.execute(
            "INSERT INTO stage_attempts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                identity,
                1,
                "stage",
                "{}",
                "v",
                str(output),
                "token",
                "completed",
                json.dumps(expected),
                json.dumps(ArtifactRef.capture(output).model_dump(mode="json")),
                None,
            ),
        )
    ledger = RunLedger(database)
    assert ledger.get_record("stage", {}, "v")["result_artifacts"] is None
    assert (
        RunLedger(database).run_stage(
            "stage", {}, write_output, output=output, implementation_version="v"
        )
        == expected
    )

    def forbidden(result):
        pytest.fail("Selectors must not run on completed attempts")

    with pytest.raises(StageConflict, match="no result artifact contract"):
        ledger.run_stage(
            "stage",
            {},
            write_output,
            output=output,
            implementation_version="v",
            artifact_refs=forbidden,
        )


@pytest.mark.parametrize(
    "configuration,version", [([], "v"), ({"nan": float("nan")}, "v"), ({}, "")]
)
def test_invalid_run_binding_does_not_bind(tmp_path, configuration, version):
    """Invalid binding data does not reserve the ledger's run identity."""
    ledger = RunLedger(tmp_path / "db")
    with pytest.raises(ValueError):
        ledger.bind_run(configuration, version)
    ledger.bind_run({"valid": True}, "v")


def test_stage_payload_retry_and_resume(tmp_path):
    """Artifact-producing callers get absent payloads and preserved failed attempts."""
    ledger = RunLedger(tmp_path / "ledger.sqlite")
    calls = []

    def action(path):
        assert not path.exists()
        path.mkdir()
        (path / "result.txt").write_text("evidence")
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("failure")
        return {"path": str(path)}

    options = dict(
        stage_id="example",
        inputs={},
        action=action,
        output=tmp_path / "stage",
        implementation_version="1",
    )
    with pytest.raises(RuntimeError):
        ledger.stage(**options)
    result = ledger.stage(**options, retry=True)
    assert calls == [tmp_path / "stage/payload", tmp_path / "stage-attempt-2/payload"]
    assert ledger.stage(**options) == result
    assert len(calls) == 2
    assert (calls[0] / "result.txt").read_text() == "evidence"
    (calls[1] / "result.txt").write_text("changed")
    with pytest.raises(ValueError):
        ledger.stage(**options)
