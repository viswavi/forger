"""Tests for multi-GPU embedding (no GPU: fake encoders and fake workers)."""

from __future__ import annotations

import multiprocessing
import os
import sys
import textwrap
import time
import types

import numpy as np
import pytest

from pyforger.inspect import _embed_dp, cluster, embedding_dispersion

TEXTS = [f"text {i} " + "word " * (i % 7) for i in range(23)]
ENCODE_KWARGS = dict(batch_size=64, normalize_embeddings=True, convert_to_numpy=True)


def _vectors(texts: list[str]) -> np.ndarray:
    """A deterministic, unit-norm vector per text."""
    raw = np.array([[len(t), sum(map(ord, t)) % 97, 1.0] for t in texts], np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


class _FakeEncoder:
    """Stands in for SentenceTransformer; counts constructions."""

    device = types.SimpleNamespace(type="cuda")
    loads = 0

    def __init__(self, model_id):
        type(self).loads += 1

    def encode(self, docs, **kwargs):
        assert kwargs == ENCODE_KWARGS
        return _vectors(docs)


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """A `sentence_transformers` module whose encoder is `_FakeEncoder`."""
    _FakeEncoder.loads = 0
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


class _FakeProcess:
    """In-process stand-in for `EmbedProcess`; records every instance."""

    started: list[_FakeProcess] = []
    fail_on_gpu: str | None = None

    def __init__(self, gpu):
        self.gpu = gpu
        self.texts = None
        self.closed = False
        self.connection, self._worker_end = multiprocessing.Pipe()
        type(self).started.append(self)

    def send(self, model_id, texts, encode_kwargs):
        assert encode_kwargs == ENCODE_KWARGS
        self.texts = texts
        self._worker_end.send(None)  # makes `connection` ready

    def result(self):
        if self.gpu == type(self).fail_on_gpu:
            raise RuntimeError(f"embedding worker on GPU {self.gpu} failed")
        return _vectors(self.texts)

    def close(self):
        self.closed = True
        self.connection.close()
        self._worker_end.close()


@pytest.fixture
def fake_processes(monkeypatch):
    """Replace worker processes with `_FakeProcess`."""
    _FakeProcess.started = []
    _FakeProcess.fail_on_gpu = None
    monkeypatch.setattr(_embed_dp, "EmbedProcess", _FakeProcess)
    return _FakeProcess


def test_embedding_gpus_needs_two_gpus_and_enough_docs(monkeypatch):
    """Multi-GPU only with 2+ visible GPUs, CUDA, and at least the minimum docs."""
    fake_torch = types.ModuleType("torch")
    emptied = []
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        empty_cache=lambda: emptied.append(True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    many = cluster._MULTI_GPU_MIN_DOCS

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    assert cluster._embedding_gpus(many) == ["2", "5"]
    assert emptied  # cached GPU memory is released for the workers
    assert cluster._embedding_gpus(many - 1) == []

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert cluster._embedding_gpus(many) == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert cluster._embedding_gpus(many) == []

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    assert cluster._embedding_gpus(many) == []


def test_default_embed_shards_across_gpus_in_input_order(
    monkeypatch, fake_sentence_transformers, fake_processes
):
    """Each GPU embeds a share; the result matches one encoder, row for row."""
    monkeypatch.setattr(cluster, "_embedding_gpus", lambda n: ["0", "1", "2"])

    embeddings = cluster._default_embed(TEXTS, "some/model")

    np.testing.assert_array_equal(embeddings, _vectors(TEXTS))
    assert embeddings.dtype == np.float32
    assert [p.gpu for p in fake_processes.started] == ["0", "1", "2"]
    shares = [len(p.texts) for p in fake_processes.started]
    assert sum(shares) == len(TEXTS) and max(shares) - min(shares) <= 1
    assert all(p.closed for p in fake_processes.started)
    assert _FakeEncoder.loads == 0  # the parent loads no model of its own


def test_default_embed_single_gpu_runs_in_process(
    monkeypatch, fake_sentence_transformers, fake_processes
):
    """With one GPU (or too few docs) nothing changes: one encoder, no workers."""
    monkeypatch.setattr(cluster, "_embedding_gpus", lambda n: [])

    embeddings = cluster._default_embed(TEXTS, "some/model")

    np.testing.assert_array_equal(embeddings, _vectors(TEXTS))
    assert _FakeEncoder.loads == 1
    assert fake_processes.started == []


def test_encode_with_more_gpus_than_texts(fake_processes):
    """A GPU without a share of the texts starts no worker."""
    embeddings = _embed_dp.encode(TEXTS[:2], "m", ["0", "1", "2", "3"], ENCODE_KWARGS)

    np.testing.assert_array_equal(embeddings, _vectors(TEXTS[:2]))
    assert len(fake_processes.started) == 2


def test_encode_closes_every_worker_when_one_fails(fake_processes):
    """A failing worker raises, and the other workers are stopped too."""
    fake_processes.fail_on_gpu = "0"

    with pytest.raises(RuntimeError, match="GPU 0 failed"):
        _embed_dp.encode(TEXTS, "m", ["0", "1"], ENCODE_KWARGS)

    assert len(fake_processes.started) == 2
    assert all(p.closed for p in fake_processes.started)


def test_cache_key_does_not_depend_on_gpu_count(
    monkeypatch, fake_sentence_transformers, fake_processes
):
    """A result embedded on one GPU is reused on two, without embedding again."""
    rows = [{"text": t} for t in TEXTS]
    monkeypatch.setattr(cluster, "_embedding_gpus", lambda n: [])
    first = embedding_dispersion(rows, "text", embedding_model="some/model")
    assert _FakeEncoder.loads == 1

    monkeypatch.setattr(cluster, "_embedding_gpus", lambda n: ["0", "1"])
    second = embedding_dispersion(rows, "text", embedding_model="some/model")

    assert fake_processes.started == []  # a cache hit: no workers started
    assert second.vendi == first.vendi


# A `sentence_transformers` for real worker processes: each vector holds the
# text's length and the worker's CUDA_VISIBLE_DEVICES.
_FAKE_PACKAGE = textwrap.dedent(
    """
    import os
    import time

    class SentenceTransformer:
        def __init__(self, model_id):
            self.model_id = model_id

        def encode(self, texts, **kwargs):
            import numpy as np

            if "boom" in texts:
                raise ValueError("cannot embed boom")
            if "hang forever" in texts:
                with open(os.environ["HANG_PID_FILE"], "w") as f:
                    f.write(str(os.getpid()))
                time.sleep(3600)
            gpu = float(os.environ["CUDA_VISIBLE_DEVICES"])
            return np.array([[len(t), gpu] for t in texts], np.float32)
    """
)


@pytest.fixture
def worker_path(tmp_path, monkeypatch):
    """Put the fake `sentence_transformers` on the workers' PYTHONPATH."""
    package = tmp_path / "sentence_transformers"
    package.mkdir()
    (package / "__init__.py").write_text(_FAKE_PACKAGE)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))


def test_encode_in_worker_processes(worker_path):
    """Real workers: one per GPU, each on its own CUDA_VISIBLE_DEVICES."""
    texts = ["a", "bbbb", "cc", "ddddddd", "eee"]

    embeddings = _embed_dp.encode(texts, "m", ["3", "5"], ENCODE_KWARGS)

    assert embeddings[:, 0].tolist() == [len(t) for t in texts]
    # Sorted by length and dealt round-robin: 7, 3, 1 -> GPU 3; 4, 2 -> GPU 5.
    assert embeddings[:, 1].tolist() == [3, 5, 5, 3, 3]


def test_encode_in_worker_processes_reports_worker_error(worker_path):
    """A worker's exception comes back as a RuntimeError with its traceback."""
    with pytest.raises(RuntimeError, match="(?s)GPU 5 failed.*cannot embed boom"):
        _embed_dp.encode(["long text", "boom"], "m", ["3", "5"], ENCODE_KWARGS)


def test_encode_stops_a_busy_worker_when_another_fails(
    worker_path, tmp_path, monkeypatch
):
    """One worker fails while another is still embedding: raise now, stop both."""
    pid_file = tmp_path / "hang.pid"
    monkeypatch.setenv("HANG_PID_FILE", str(pid_file))
    start = time.time()

    with pytest.raises(RuntimeError, match="GPU 5 failed"):
        _embed_dp.encode(["hang forever", "boom"], "m", ["3", "5"], ENCODE_KWARGS)

    assert time.time() - start < 60
    if pid_file.exists():  # the worker got as far as embedding: it was stopped
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
