"""Real retained-file paths: integrity, secret boundaries, ownership and peak RSS."""

import gc
import io
import json
import subprocess
import sys

import pytest

from pyforger._streaming_json import JSONReader, trajectory_events
from pyforger.agent_backend import AgentSpec
from pyforger.agent_diagnostics import ArtifactSafetyError, validate_artifact_bytes
from pyforger.artifacts import (
    ArtifactRef,
    FileInput,
    OwnedFileInput,
    input_bytes,
    replace_file_secrets,
)
from pyforger.evidence_journal import (
    iter_trajectory_events,
    journal_events,
    materialize_journals,
)
from pyforger.harbor import HarborResult, _redact, _scrub, collect_trial_files


@pytest.mark.parametrize("offset", [-20, -1, 0, 1, 20])
def test_secret_chunk_boundaries(tmp_path, offset, monkeypatch):
    path = tmp_path / "binary"
    secret = b'private-key-with-quote"and-newline\n'
    body = b"x" * (1024 * 1024 + offset) + secret + b"\x00\xffsuffix"
    path.write_bytes(body)
    replace_file_secrets(path, {secret, secret[:9]})
    assert path.read_bytes() == body.replace(secret, b"[REDACTED]")
    path.write_bytes(body)
    source = FileInput.capture(path)
    monkeypatch.setenv("FORGER_TEST_SECRET", secret.decode())
    with pytest.raises(ArtifactSafetyError):
        validate_artifact_bytes({"data": source}, [AgentSpec(kind="nop")])


@pytest.mark.parametrize(
    "value",
    [
        {
            "nested": [{"api_key": {"value": "hidden"}, "api_key_env": "NAME"}],
            "text": 'é\\"\nkey-secret',
            "escape": "😀",
        },
        [1, -1.2e10, True, None, {}, [], "", "\\", '"'],
    ],
)
def test_incremental_json_matches_native_redaction(value):
    text = json.dumps(value, ensure_ascii=True, indent=2)

    class SmallReads(io.StringIO):
        def read(self, size=-1):
            return super().read(min(size, 3))

    target = io.StringIO()
    reader = JSONReader(SmallReads(text))
    reader.rewrite(
        target, lambda v: _redact(v, {"key-secret"}), lambda k: k == "api_key"
    )
    assert json.loads(target.getvalue()) == _redact(value, {"key-secret"})
    assert JSONReader(SmallReads(text)).value() == value


def test_scrub_escaped_secrets_and_unchanged_identity(tmp_path):
    secret = 'private-"\n😀credential'
    path = tmp_path / "trajectory.json"
    value = {
        "steps": [
            {"api_\u006bey": "hidden", "message": secret, "api_key_env": "MY_ENV"}
        ],
        "meta": {"token": [1, 2]},
    }
    path.write_text(json.dumps(value, ensure_ascii=True))
    _scrub(tmp_path, {secret})
    assert json.loads(path.read_text()) == _redact(value, {secret})
    before = path.read_bytes()
    _scrub(tmp_path, set())
    assert path.read_bytes() == before


def test_owned_outputs_survive_native_cleanup_and_are_integrity_checked(tmp_path):
    path = tmp_path / "native"
    path.write_bytes(b"\x00\xffscientific evidence")
    owned = OwnedFileInput.snapshot(path)
    path.unlink()
    assert input_bytes(owned) == b"\x00\xffscientific evidence"
    owned.source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        input_bytes(owned)
    private = owned.source
    del owned
    gc.collect()
    assert not private.exists()


def test_trajectory_journal_full_portable_snapshot(tmp_path):
    path = tmp_path / "native.json"
    with path.open("w") as stream:
        stream.write('{"metadata":{"note":"not an event"},"steps":[')
        for index in range(128):
            if index:
                stream.write(",")
            json.dump(
                {"index": index, "message": "z" * 65536, "api_key": "hidden"}, stream
            )
        stream.write("]}")
    retained = journal_events(trajectory_events(path), [AgentSpec(kind="nop")])
    path.unlink()
    output = tmp_path / "portable"
    output.mkdir()
    portable = materialize_journals(retained, output, [])
    for index, event in enumerate(iter_trajectory_events(portable, root=output)):
        assert event == {
            "index": index,
            "message": "z" * 65536,
            "api_key": "[REDACTED]",
        }
    assert index == 127
    assert retained["journal"]["count"] == 128


def test_binary_collect_preserves_all_bytes(tmp_path):
    native = tmp_path / "native"
    trial = native / "trial"
    artifacts = trial / "artifacts"
    artifacts.mkdir(parents=True)
    body = bytes(range(256)) * 8192
    (artifacts / "data").write_bytes(body)
    (artifacts / "manifest.json").write_text(
        json.dumps(
            [{"source": "/work/data", "status": "ok", "destination": "artifacts/data"}]
        )
    )
    (trial / "result.json").write_text("{}")
    rollout = HarborResult(
        status="completed",
        artifacts=ArtifactRef.capture(native),
        metadata={"harbor_result_path": "trial/result.json"},
    )
    output = tmp_path / "collected"
    output.mkdir()
    collect_trial_files(rollout, {"/work/data": "data"}, output)
    assert (output / "data").read_bytes() == body
    with pytest.raises(ValueError, match="read limit"):
        collect_trial_files(rollout, {"/work/data": "data"}, output, max_bytes=10)


# Each process generates its own evidence incrementally, then exercises real code.
_MEMORY_SCRIPT = r"""
import gc, hashlib, json, resource, sys, tempfile, tracemalloc
from pathlib import Path
from pyforger.harbor import _scrub
from pyforger.artifacts import OwnedFileInput, FileInput
from pyforger._streaming_json import trajectory_events
from pyforger.evidence_journal import journal_events, iter_trajectory_events
from pyforger.agent_backend import AgentSpec
from pyforger.agent_diagnostics import validate_artifact_bytes
count = int(sys.argv[1])
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    row = json.dumps({'message': 'x' * 65536, 'api_key': 'hidden', 'index': 1})
    with (root / 'trajectory.json').open('w') as stream:
        stream.write('{"steps":[')
        for index in range(count):
            if index: stream.write(',')
            stream.write(row)
        stream.write(']}')
    with (root / 'large.bin').open('wb') as stream:
        for index in range(count): stream.write(b'z' * 65536)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    _scrub(root, {'concrete-credential-not-in-evidence'})
    retained = journal_events(trajectory_events(root / 'trajectory.json'), [AgentSpec(kind='nop')])
    assert retained['journal']['count'] == count
    assert sum(1 for _ in iter_trajectory_events(retained)) == count
    output = OwnedFileInput.snapshot(root / 'large.bin')
    validate_artifact_bytes({'data': output}, [AgentSpec(kind='nop')])
    assert sum(map(len, output.chunks())) == count * 65536
    print(json.dumps({'peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'baseline_kib': baseline, 'bytes_each': count * 65536,
                      'traced_peak': tracemalloc.get_traced_memory()[1]}))
"""


@pytest.mark.skipif(sys.platform != "linux", reason="ru_maxrss is KiB on Linux")
def test_large_retained_evidence_memory_is_not_proportional_to_total_size():
    measurements = []
    for count in (128, 1536):
        completed = subprocess.run(
            [sys.executable, "-c", _MEMORY_SCRIPT, str(count)],
            check=True,
            capture_output=True,
            text=True,
        )
        measurements.append(json.loads(completed.stdout.splitlines()[-1]))
    small, large = measurements
    print("Harbor retained evidence RSS:", measurements)
    assert large["traced_peak"] < 16 * 1024 * 1024
    assert large["traced_peak"] - small["traced_peak"] < 4 * 1024 * 1024
    assert large["peak_kib"] - large["baseline_kib"] < 32 * 1024
    assert large["peak_kib"] - small["peak_kib"] < 24 * 1024


def test_native_context_usage_matches_upstream_without_retaining_steps(tmp_path):
    pytest.importorskip("harbor")
    from harbor.agents.factory import AgentFactory
    from harbor.models.agent.context import AgentContext
    from harbor.models.trial.config import AgentConfig
    from harbor.utils.trajectory_utils import load_model_usage_from_trajectory

    config = AgentConfig(
        import_path="pyforger._harbor_model:ForgerOpenHandsSDK",
        model_name="openai/test",
    )
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
    payload = {
        "agent": {"name": "test", "version": "1", "model_name": "openai/test"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "instruction"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "large" * 200000,
                "metrics": {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "cost_usd": 0.1,
                },
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "unpriced",
                "model_name": "other",
                "metrics": {"prompt_tokens": 7},
            },
        ],
        "final_metrics": {
            "total_prompt_tokens": 19,
            "total_completion_tokens": 5,
            "total_cost_usd": 0.1,
        },
    }
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(payload))
    expected = load_model_usage_from_trajectory(path)
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert context.model_usage == expected
    assert context.n_input_tokens == 19
    assert context.n_output_tokens == 5
    assert context.cost_usd == 0.1
    # TrialResult embeds only compact context, not messages or trajectory objects.
    assert len(context.model_dump_json()) < 2000


def test_recorded_task_streams_owned_outputs_and_portable_journal(tmp_path):
    from pyforger.agent_backend import AgentRunResult
    from pyforger.agent_tasks import TaskSpec
    from pyforger.task_runs import run_recorded_task

    source = tmp_path / "native-data"
    with source.open("wb") as stream:
        for _ in range(32):
            stream.write(b"scientific evidence\x00\xff" * 65536)
    digest = FileInput.capture(source).sha256
    owned = OwnedFileInput.snapshot(source)
    source.unlink()
    trajectory = journal_events(
        ({"index": index, "message": "x" * 65536} for index in range(16)),
        [AgentSpec(kind="nop")],
    )

    class CompletedEvidence:
        def run_agent(self, task, agent):
            return AgentRunResult(
                status="completed",
                files={"data": owned},
                passed=True,
                trajectory=trajectory,
            )

    output = tmp_path / "recorded"
    report = run_recorded_task(
        TaskSpec(instruction="retain", outputs=("data",)),
        agent=AgentSpec(kind="nop"),
        backend=CompletedEvidence(),
        output=output,
    )
    assert report["passed"] is True
    assert FileInput.capture(output / "data").sha256 == digest
    retained = json.loads((output / "agent/trajectory.json").read_text())
    assert sum(1 for _ in iter_trajectory_events(retained, root=output)) == 16


@pytest.mark.parametrize(
    "content",
    ['{"steps":null}', '{"steps":[1,]}', "[1] trailing", '{"steps":[],"events":[]}'],
)
def test_malformed_native_trajectory_rejected(tmp_path, content):
    path = tmp_path / "trajectory.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        list(trajectory_events(path))


def test_scrub_long_whitespace_preserves_exact_bytes(tmp_path):
    path = tmp_path / "spaced.json"
    content = b" " * (2 * 1024 * 1024) + b'{ "steps" : [] }\n'
    path.write_bytes(content)
    _scrub(tmp_path, set())
    assert path.read_bytes() == content
