"""LocalModel loads cached local ids by path and frees its engine (fake vllm)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyforger.acquire import _local_dp
from pyforger.acquire.model import LocalModel, Model, room_for
from pyforger.alter.generate import self_taught_reasoner
from pyforger.filter.model_difficulty import by_comparative_difficulty
from pyforger.inspect import compare_to_reference
from pyforger.utils import resolve_model_path

LOCAL_ID = "local/Qwen3-4B-x-rl80"


class _Events(list):
    """("start"|"stop", model) events; ``cleanups``: live engines at each cleanup."""

    cleanups: list[int]


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def events(monkeypatch):
    """Install a fake ``vllm``; return ("start"|"stop", model) events in order.

    Every completion is ``"<think>t</think>7"``.
    """
    log = _Events()
    log.cleanups = []

    class FakeLLM:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            name = kwargs["model"]
            log.append(("start", name))
            core = SimpleNamespace(shutdown=lambda: log.append(("stop", name)))
            self.llm_engine = SimpleNamespace(engine_core=core)

        def chat(self, messages, sampling_params, **kwargs):
            n = sampling_params.kwargs["n"]
            out = SimpleNamespace(text="<think>t</think>7", token_ids=[1, 2])
            return [
                SimpleNamespace(prompt_token_ids=[1], outputs=[out] * n)
                for _ in messages
            ]

    module = types.ModuleType("vllm")
    module.LLM = FakeLLM
    module.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # one in-process engine
    monkeypatch.setattr(_local_dp, "_live_engines", 0)
    monkeypatch.setattr(
        _local_dp, "_cleanup_vllm", lambda: log.cleanups.append(_local_dp._live_engines)
    )
    return log


def _snapshot(repo: str, files: dict[str, str], revision: str = "abc123") -> Path:
    """Write ``repo`` into the (per-test) Hugging Face cache; return its snapshot."""
    from huggingface_hub import constants

    root = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text(revision)
    snap = root / "snapshots" / revision
    snap.mkdir(parents=True)
    for name, text in files.items():
        (snap / name).write_text(text)
    return snap


_WEIGHTS = {"config.json": "{}", "model.safetensors": "w", "tokenizer.json": "{}"}


# --------------------------------------------------------------------------- #
# resolve_model_path
# --------------------------------------------------------------------------- #


def test_cached_local_id_resolves_to_its_snapshot():
    """An id in the cache with config and weights gives its snapshot directory."""
    snap = _snapshot(LOCAL_ID, _WEIGHTS)
    assert resolve_model_path(LOCAL_ID) == str(snap)
    assert resolve_model_path(LOCAL_ID, part="tokenizer") == str(snap)


@pytest.mark.parametrize(
    "files, part",
    [
        ({"tokenizer.json": "{}", "config.json": "{}"}, "model"),  # no weights
        ({"model.safetensors": "w"}, "model"),  # no config
        ({"config.json": "{}", "model.safetensors": "w"}, "tokenizer"),
        (
            {
                "config.json": "{}",
                "model-00001-of-00002.safetensors": "w",
                "model.safetensors.index.json": json.dumps(
                    {
                        "weight_map": {
                            "a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
            },
            "model",
        ),  # a shard missing
    ],
)
def test_incomplete_snapshot_keeps_the_id(files, part):
    """A snapshot without what is loaded leaves the id, to download as before."""
    _snapshot(LOCAL_ID, files)
    assert resolve_model_path(LOCAL_ID, part=part) == LOCAL_ID


def test_uncached_ids_and_directories_are_unchanged(tmp_path):
    """Not cached, a directory, or another revision: the argument comes back."""
    assert resolve_model_path("Qwen/Qwen3-4B") == "Qwen/Qwen3-4B"
    assert resolve_model_path(str(tmp_path)) == str(tmp_path)
    _snapshot(LOCAL_ID, _WEIGHTS)
    assert resolve_model_path(LOCAL_ID, revision="other") == LOCAL_ID
    with pytest.raises(ValueError, match="part"):
        resolve_model_path(LOCAL_ID, part="weights")


def test_local_model_loads_snapshot_but_keeps_its_id(events):
    """vLLM gets the snapshot path; the cache key and history keep the id."""
    snap = _snapshot(LOCAL_ID, _WEIGHTS)
    tok = _snapshot("local/tok", {"tokenizer.json": "{}"})
    model = LocalModel(LOCAL_ID, tokenizer="local/tok")
    key = model.cache_key()
    model.generate("q")
    llm = model._llm
    assert llm.init_kwargs["model"] == str(snap)
    assert llm.init_kwargs["tokenizer"] == str(tok)
    assert model.cache_key() == key and key[1] == LOCAL_ID
    assert model._kwargs["tokenizer"] == "local/tok"


def test_compare_to_reference_tokenizer_id_uses_the_cache(monkeypatch):
    """A tokenizer id in the cache loads from its snapshot."""
    import transformers

    snap = _snapshot(LOCAL_ID, {"tokenizer_config.json": "{}"})
    loaded: list[str] = []

    def from_pretrained(name, *args, **kwargs):
        loaded.append(name)
        return lambda texts, **kw: {"input_ids": [t.split() for t in texts]}

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", from_pretrained)
    rows = [{"t": " ".join(["w"] * (50 + i))} for i in range(3)]
    compare_to_reference(rows, rows, tokenizer=LOCAL_ID, _no_cache=True)
    assert loaded == [str(snap)]


# --------------------------------------------------------------------------- #
# close() and room_for
# --------------------------------------------------------------------------- #


def test_close_shuts_down_in_process_engine_and_restarts(events):
    """close() stops the engine core and cleans up; the next call starts anew."""
    model = LocalModel("m")
    model.generate("q")
    model.close()
    assert model._llm is None
    assert events == [("start", "m"), ("stop", "m")]
    assert events.cleanups == [0]  # vLLM's cleanup ran with no engine left
    model.close()  # closing twice is fine
    model.generate("q")
    assert events[-1] == ("start", "m")


def test_close_keeps_distributed_state_while_another_engine_runs(events):
    """vLLM's process-wide cleanup waits until no in-process engine is left."""
    a, b = LocalModel("a"), LocalModel("b")
    a.generate("q")
    b.generate("q")
    a.close()
    assert events.cleanups == []
    b.close()
    assert events.cleanups == [0]


def test_close_survives_a_failing_shutdown(events, monkeypatch, caplog):
    """A shutdown error is logged, and the model still restarts."""
    model = LocalModel("m")
    model.generate("q")

    def boom():
        raise RuntimeError("engine dead")

    model._llm.llm_engine.engine_core.shutdown = boom
    model.close()
    assert "engine dead" in caplog.text
    model.generate("q")
    assert events[-1] == ("start", "m")


def test_room_for_closes_rivals_only_when_the_engine_starts(events):
    """A running model's rivals stay up; a starting one closes them first."""
    weak, strong = LocalModel("weak"), LocalModel("strong")
    weak.generate("q")
    strong.generate("q")
    with room_for(strong, weak):
        strong.generate("q")  # already running: nothing closes
    assert ("stop", "weak") not in events
    strong.close()
    events.clear()
    with room_for(strong, weak):
        strong.generate("q")
    assert events == [("stop", "weak"), ("start", "strong")]
    assert strong._rivals == []
    weak.generate("q")  # outside room_for: strong stays
    assert ("stop", "strong") not in events


def test_room_for_leaves_api_models_alone(events):
    """Only LocalModels are closed, and only for a LocalModel."""
    local = LocalModel("m")
    local.generate("q")
    api = Model("openai/gpt-4o-mini")
    with room_for(api, local):
        pass
    with room_for(local, api):
        local.generate("q")
    assert events == [("start", "m")]


def test_comparative_difficulty_frees_weak_before_strong(events):
    """The weak engine is shut down before the strong one starts."""
    rows = [{"task": "q", "answer": "7"}]
    by_comparative_difficulty(
        rows,
        weak_model=LocalModel("weak"),
        strong_model=LocalModel("strong"),
        input_field="task",
        ground_truth_field="answer",
    )
    assert events == [("start", "weak"), ("stop", "weak"), ("start", "strong")]


def test_star_alternates_student_and_local_judge(events):
    """Student and judge never run at once, through rationalization too."""
    items = [{"prompt": "q", "final_answer": "8"}]
    self_taught_reasoner(
        items,
        "{prompt}",
        LocalModel("student"),
        n=1,
        answer_judge=LocalModel("judge"),
    )
    running: set[str] = set()
    for kind, name in events:
        if kind == "start":
            running.add(name)
            assert len(running) == 1, events
        else:
            running.discard(name)
    assert events[:3] == [("start", "student"), ("stop", "student"), ("start", "judge")]
