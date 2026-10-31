"""Exercise durable journals and lazy inputs without provider or backend mocks."""

import gc
import json
import os
from types import SimpleNamespace

import pytest

from pyforger.agent_backend import AgentSpec, WorkspaceTask
from pyforger.agent_tasks import TaskSpec
from pyforger.artifacts import ArtifactRef, FileInput
from pyforger.evidence_journal import (
    _PRIVATE_JOURNALS,
    EvidenceJournal,
    iter_trajectory_events,
    materialize_journals,
)
from pyforger.local_backend import LocalBackend, _materialize
from pyforger.sail_sdk import _recover_current_events
from pyforger.sail_transport import _input_chunks


def test_many_events_once_sanitization_order_and_frozen_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_SERVICE_TOKEN", "short")
    journal = EvidenceJournal()
    clean = journal.clean
    calls = []

    def counted(event):
        calls.append(1)
        return clean(event)

    journal.clean = counted
    for index in range(2000):
        journal.append({"text": "short", "index": index % 3, "token": "anything"})
        if index == 99:
            prefix = journal.trajectory()
    retained = journal.trajectory()
    assert len(json.dumps(retained)) < 300
    assert len(calls) == 2000
    assert len(list(iter_trajectory_events(prefix))) == 100
    for index, event in enumerate(iter_trajectory_events(retained)):
        assert event == {
            "text": "[REDACTED]",
            "index": index % 3,
            "token": "[REDACTED]",
        }
    journal.clean = lambda _: pytest.fail("trusted journal sanitized twice")
    memo = {}
    portable = materialize_journals({"a": retained, "b": retained}, tmp_path, [], memo)
    assert portable["a"] == portable["b"]
    assert len(list((tmp_path / "agent/journals").iterdir())) == 1
    assert len(list(iter_trajectory_events(portable["a"], root=tmp_path))) == 2000
    assert os.stat(tmp_path / portable["a"]["journal"]["path"]).st_mode & 0o777 == 0o600


def test_provenance_containment_integrity_and_lifetime(tmp_path):
    journal = EvidenceJournal()
    journal.append({"value": "safe"})
    ref = journal.trajectory()
    assert id(ref["journal"]) in _PRIVATE_JOURNALS
    forged = json.loads(json.dumps(ref))
    with pytest.raises(ValueError, match="Unknown"):
        materialize_journals(forged, tmp_path, [])
    with pytest.raises(ValueError, match="Unknown"):
        list(iter_trajectory_events(forged))
    portable = materialize_journals(ref, tmp_path, [])
    broken = {"journal": {**portable["journal"], "path": "../escape"}}
    with pytest.raises(ValueError):
        list(iter_trajectory_events(broken, root=tmp_path))
    path = tmp_path / portable["journal"]["path"]
    path.write_text('{"value":"evil"}\n')
    with pytest.raises(ValueError, match="digest|Incomplete"):
        list(iter_trajectory_events(portable, root=tmp_path))
    identity = id(ref["journal"])
    del ref
    gc.collect()
    assert identity not in _PRIVATE_JOURNALS


def test_recovery_equal_payloads_sequence_and_stale_turn(tmp_path):
    journal = EvidenceJournal()
    journal.append_sanitized({"text": "same"})
    current = {
        "turn_id": "turn",
        "call_id": "call",
        "last_sequence": 0,
        **journal.trajectory(),
    }
    rows = [
        {
            "kind": "event",
            "turn_id": turn,
            "call_id": "call",
            "sequence": seq,
            "event": {"text": "same"},
        }
        for turn, seq in [("old", 99), ("turn", 0), ("turn", 1), ("turn", 2)]
    ]
    path = tmp_path / "worker.events"
    path.write_bytes(
        b"\n".join(json.dumps(r).encode() for r in rows) + b'\n{"partial":'
    )

    class Files:
        def read(self, name):
            return path.read_bytes()

    _recover_current_events(SimpleNamespace(fs=Files()), current, journal)
    assert list(iter_trajectory_events(current)) == [{"text": "same"}] * 3
    assert current["recovery_truncated"] is True


def test_file_input_real_local_staging_and_task_roundtrip(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    payload = bytes(range(256)) * 10000
    (source / "data").write_bytes(payload)
    value = FileInput.from_artifact(ArtifactRef.capture(source), "data")
    assert b"".join(_input_chunks(value)) == payload
    local = tmp_path / "local"
    local.mkdir()
    _materialize(local, {"nested/data": value})
    assert (local / "nested/data").read_bytes() == payload
    task = TaskSpec("copy", files={"data": value}, outputs=("out",))
    saved = task.save(tmp_path / "task")
    assert TaskSpec.load(saved).files["data"] == payload
    result = LocalBackend().run_agent(
        WorkspaceTask("copy", files={"data": value}, outputs=("out",)),
        AgentSpec(kind="oracle", solution="cp data out"),
    )
    assert result.files["out"] == payload
    (source / "data").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        b"".join(_input_chunks(value))


def test_script_streaming_matches_inline_for_every_chunk_boundary(tmp_path):
    from pyforger.local_backend import _script_chunks, _workspace_text

    text = " /work /work/abc x/work /workx '/work' /work\nhttps://host/work "
    path = tmp_path / "script"
    path.write_text(text)
    value = FileInput.capture(path)
    for size in range(1, 15):

        class Chunks:
            def chunks(self):
                data = text.encode()
                for offset in range(0, len(data), size):
                    yield data[offset : offset + size]

        assert b"".join(_script_chunks(Chunks(), tmp_path)).decode() == _workspace_text(
            tmp_path, text
        )
    root = tmp_path / "workspace"
    root.mkdir()
    _materialize(root, {"run.sh": value})
    _materialize(root, {"run.sh": value})
    assert (root / "run.sh").read_text() == _workspace_text(root, text)
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        _materialize(root, {"escape/overwrite": value})
    assert not (tmp_path / "overwrite").exists()


def test_snapshot_capability_cannot_be_modified(tmp_path):
    journal = EvidenceJournal()
    journal.append({"text": "safe"})
    reference = journal.trajectory()
    reference["journal"]["path"] = "/etc/passwd"
    with pytest.raises(ValueError, match="Unknown"):
        materialize_journals(reference, tmp_path, [])


def test_recovery_missing_first_or_middle_sequence_is_explicit(tmp_path):
    journal = EvidenceJournal()
    current = {
        "turn_id": "turn",
        "call_id": "call",
        "last_sequence": -1,
        **journal.trajectory(),
    }

    class Files:
        def read(self, name):
            return json.dumps(
                {
                    "kind": "event",
                    "turn_id": "turn",
                    "call_id": "call",
                    "sequence": 1,
                    "event": {"text": "later"},
                }
            ).encode()

    _recover_current_events(SimpleNamespace(fs=Files()), current, journal)
    assert current["recovery"] == "sequence_gap"
    assert current["sequence_error"] == {"expected": 0, "received": 1}
    assert list(iter_trajectory_events(current)) == []


def test_file_log_portability(tmp_path):
    from pyforger.evidence_journal import evidence_file_chunks

    ref, tail = evidence_file_chunks(b"abcd" * 5000 for _ in range(20))
    portable = materialize_journals({"log": ref, "other": ref}, tmp_path, [])
    assert portable["log"] == portable["other"]
    entry = portable["log"]["evidence_file"]
    lazy = FileInput(
        source=tmp_path / entry["path"], size=entry["size"], sha256=entry["sha256"]
    )
    assert sum(len(chunk) for chunk in lazy.chunks()) == 400000
    assert len(tail) == 8000


def test_published_evidence_survives_owner_cleanup_and_portable_move(tmp_path):
    import shutil

    journal = EvidenceJournal()
    journal.append({"value": "retained"})
    reference = journal.trajectory()
    private = journal.path
    root = tmp_path / "trial"
    root.mkdir()
    portable = materialize_journals(reference, root, [])
    manifest = root / "trajectory.json"
    manifest.write_text(json.dumps(portable))
    del reference, journal
    gc.collect()
    assert not private.exists()
    moved = tmp_path / "moved"
    shutil.move(root, moved)
    loaded = json.loads((moved / "trajectory.json").read_text())
    assert list(iter_trajectory_events(loaded, root=moved)) == [{"value": "retained"}]


def test_metadata_redaction_preserves_private_identity(monkeypatch, tmp_path):
    from pyforger.agent_diagnostics import sanitizer

    journal = EvidenceJournal()
    journal.append({"value": "safe"})
    value = journal.trajectory()
    safe = sanitizer([])({"nested": value, "error": "Bearer hidden"})
    assert safe["nested"]["journal"] is value["journal"]
    assert safe["error"] == "Bearer [REDACTED]"
    assert materialize_journals(safe, tmp_path, [])["nested"]["journal"]["count"] == 1


def test_publish_private_only_leaves_unrelated_and_portable_refs(tmp_path):
    from pyforger.evidence_journal import publish_private_evidence

    journal = EvidenceJournal()
    journal.append({"text": "safe"})
    value = {
        "live": journal.trajectory(),
        "ordinary": {"journal": {"anything": [1]}},
        "external": {
            "journal": {
                "path": "/etc/passwd",
                "count": 1,
                "size": 1,
                "sha256": "0" * 64,
            }
        },
    }
    safe = publish_private_evidence(value, tmp_path)
    assert safe["ordinary"] == value["ordinary"]
    assert safe["external"] == value["external"]
    assert list(iter_trajectory_events(safe["live"], root=tmp_path)) == [
        {"text": "safe"}
    ]
    assert publish_private_evidence(safe, tmp_path) == safe


def test_model_service_and_prior_snapshot_credential_redaction(monkeypatch):
    from pyforger.acquire.model import Model
    from pyforger.agent_diagnostics import credential_values, sanitizer

    monkeypatch.setenv("INNOCENT", "old-key")
    monkeypatch.setenv("SERVICE_TOKEN", "service")
    first = AgentSpec(Model("openai/test", api_key="model-key"), api_key_env="INNOCENT")
    registry = credential_values([first])
    monkeypatch.setenv("INNOCENT", "new-key")
    registry.update(credential_values([first]))
    journal = EvidenceJournal()
    journal.clean = sanitizer([], secrets=registry)
    journal.append(
        {
            "text": "old-key new-key model-key service Bearer arbitrary",
            "password": "hidden",
        }
    )
    assert list(iter_trajectory_events(journal.trajectory())) == [
        {
            "text": "[REDACTED] [REDACTED] [REDACTED] [REDACTED] Bearer [REDACTED]",
            "password": "[REDACTED]",
        }
    ]


def test_recovery_conflicting_sequence_fails_explicitly():
    journal = EvidenceJournal()
    journal.append_sanitized({"text": "original"})
    current = {
        "turn_id": "turn",
        "call_id": "call",
        "last_sequence": 0,
        **journal.trajectory(),
    }

    class Files:
        def read(self, name):
            return json.dumps(
                {
                    "kind": "event",
                    "turn_id": "turn",
                    "call_id": "call",
                    "sequence": 0,
                    "event": {"text": "different"},
                }
            ).encode()

    _recover_current_events(SimpleNamespace(fs=Files()), current, journal)
    assert current["recovery"] == "conflicting_sequence"
    assert list(iter_trajectory_events(current)) == [{"text": "original"}]


def test_portable_journal_rejects_symlink_ancestor_before_normalization(tmp_path):
    journal = EvidenceJournal()
    journal.append({"text": "safe"})
    source = tmp_path / "source"
    source.mkdir()
    portable = materialize_journals(journal.trajectory(), source, [])
    root = tmp_path / "reader"
    root.mkdir()
    (root / "agent").symlink_to(source / "agent", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        list(iter_trajectory_events(portable, root=root))
