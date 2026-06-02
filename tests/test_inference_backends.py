"""Tests for pyforger.inference.backends."""
from __future__ import annotations

import dataclasses
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from pyforger.acquire.model import Model
from pyforger.inference.backends import (
    InferenceBackend,
    LiteLLMBackend,
    SamplingConfig,
    VLLMBackend,
    _registry_key,
    clear_engine_cache,
)
from pyforger.op import _canonical


@pytest.fixture(autouse=True)
def _isolate_engine_registry():
    """Clear the engine registry around every test in this module.

    The registry is module-level state; tests would otherwise see each
    other's mock engines.
    """
    clear_engine_cache()
    yield
    clear_engine_cache()


class _FakeLLM:
    """Records constructor kwargs; ``.generate()`` returns canned outputs."""

    instances: list[_FakeLLM] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _FakeLLM.instances.append(self)

    def generate(self, prompts: list[str], params: Any) -> list[Any]:
        outputs = []
        for i, _ in enumerate(prompts):
            choice = types.SimpleNamespace(text=f"completion-{i}")
            outputs.append(types.SimpleNamespace(outputs=[choice]))
        return outputs


class _FakeSamplingParams:
    """Records construction kwargs for assertions."""

    instances: list[_FakeSamplingParams] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _FakeSamplingParams.instances.append(self)


@pytest.fixture
def fake_vllm(monkeypatch):
    """Install a fake ``vllm`` module exposing ``LLM`` and ``SamplingParams``."""
    _FakeLLM.instances = []
    _FakeSamplingParams.instances = []
    fake_module = types.ModuleType("vllm")
    fake_module.LLM = _FakeLLM  # type: ignore[attr-defined]
    fake_module.SamplingParams = _FakeSamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_module)
    yield fake_module


class TestSamplingConfig:
    """SamplingConfig is frozen, hashable, and produces a stable cache key."""

    def test_defaults(self):
        cfg = SamplingConfig()
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 1024
        assert cfg.top_p == 1.0
        assert cfg.stop == ()

    def test_frozen(self):
        cfg = SamplingConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.temperature = 0.5  # type: ignore[misc]

    def test_hashable(self):
        cfg1 = SamplingConfig(temperature=0.2, stop=("\n",))
        cfg2 = SamplingConfig(temperature=0.2, stop=("\n",))
        # Frozen + tuple fields = hashable + equal.
        assert hash(cfg1) == hash(cfg2)
        assert cfg1 == cfg2

    def test_cache_key_stable(self):
        cfg1 = SamplingConfig(temperature=0.2, max_tokens=64, stop=("\n",))
        cfg2 = SamplingConfig(temperature=0.2, max_tokens=64, stop=("\n",))
        assert cfg1.cache_key() == cfg2.cache_key()

    def test_cache_key_changes_with_fields(self):
        a = SamplingConfig().cache_key()
        b = SamplingConfig(temperature=0.5).cache_key()
        assert a != b


class TestVLLMBackendCacheKey:
    """VLLMBackend.cache_key is stable and excludes the live engine."""

    def test_excludes_engine_handle(self):
        backend = VLLMBackend(model="m")
        # ``_llm`` is not set until _engine() runs; the key shouldn't even
        # mention it.
        key = backend.cache_key()
        assert "_llm" not in key
        # Hash-ability via the canonical normalization op uses.
        _canonical(key)

    def test_stable_across_instances(self):
        b1 = VLLMBackend(model="m", seed=0)
        b2 = VLLMBackend(model="m", seed=0)
        assert b1.cache_key() == b2.cache_key()

    def test_prefix_caching_flag_changes_key(self):
        a = VLLMBackend(model="m", enable_prefix_caching=False).cache_key()
        b = VLLMBackend(model="m", enable_prefix_caching=True).cache_key()
        assert a != b
        assert _registry_key(a) != _registry_key(b)

    def test_model_change_changes_key(self):
        a = VLLMBackend(model="m1").cache_key()
        b = VLLMBackend(model="m2").cache_key()
        assert _registry_key(a) != _registry_key(b)


class TestEngineRegistry:
    """Engines are reused across instances with the same config."""

    def test_same_config_shares_engine(self, fake_vllm):
        b1 = VLLMBackend(model="m", seed=0)
        b2 = VLLMBackend(model="m", seed=0)
        b1.generate(["hi"], SamplingConfig())
        b2.generate(["hi"], SamplingConfig())
        assert len(_FakeLLM.instances) == 1, (
            "Expected one engine load for two backends sharing a config; "
            f"saw {len(_FakeLLM.instances)}"
        )

    def test_different_config_loads_separately(self, fake_vllm):
        b1 = VLLMBackend(model="m", seed=0)
        b2 = VLLMBackend(model="m", seed=1)
        b1.generate(["hi"], SamplingConfig())
        b2.generate(["hi"], SamplingConfig())
        assert len(_FakeLLM.instances) == 2

    def test_prefix_caching_makes_new_slot(self, fake_vllm):
        b1 = VLLMBackend(model="m", enable_prefix_caching=False)
        b2 = VLLMBackend(model="m", enable_prefix_caching=True)
        b1.generate(["hi"], SamplingConfig())
        b2.generate(["hi"], SamplingConfig())
        assert len(_FakeLLM.instances) == 2

    def test_clear_engine_cache_forces_reload(self, fake_vllm):
        b = VLLMBackend(model="m")
        b.generate(["hi"], SamplingConfig())
        clear_engine_cache()
        b.generate(["hi"], SamplingConfig())
        assert len(_FakeLLM.instances) == 2

    def test_engine_constructor_receives_prefix_caching_flag(self, fake_vllm):
        VLLMBackend(model="m", enable_prefix_caching=True).generate(
            ["hi"], SamplingConfig()
        )
        assert _FakeLLM.instances[0].kwargs["enable_prefix_caching"] is True

    def test_generate_passes_sampling(self, fake_vllm):
        cfg = SamplingConfig(
            temperature=0.3, max_tokens=42, top_p=0.9, stop=("\n",)
        )
        VLLMBackend(model="m", seed=7).generate(["hi"], cfg)
        params = _FakeSamplingParams.instances[0]
        assert params.kwargs["temperature"] == 0.3
        assert params.kwargs["max_tokens"] == 42
        assert params.kwargs["top_p"] == 0.9
        assert params.kwargs["stop"] == ["\n"]
        assert params.kwargs["seed"] == 7

    def test_generate_empty_stop_is_none(self, fake_vllm):
        VLLMBackend(model="m").generate(["hi"], SamplingConfig(stop=()))
        params = _FakeSamplingParams.instances[0]
        assert params.kwargs["stop"] is None

    def test_generate_returns_completions_in_order(self, fake_vllm):
        out = VLLMBackend(model="m").generate(["a", "b", "c"], SamplingConfig())
        assert out == ["completion-0", "completion-1", "completion-2"]


class TestVLLMBackendImportError:
    """Missing vLLM produces a helpful error pointing at the extra."""

    def test_constructor_does_not_import_vllm(self, monkeypatch):
        # Block any vllm import.
        monkeypatch.setitem(sys.modules, "vllm", None)
        # Construction must succeed.
        VLLMBackend(model="m")

    def test_engine_raises_helpful_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "vllm", None)
        backend = VLLMBackend(model="m")
        with pytest.raises(ImportError) as exc_info:
            backend.generate(["hi"], SamplingConfig())
        assert "pyforger[inference]" in str(exc_info.value)


class TestInferenceBackendProtocol:
    """Both backends satisfy the structural Protocol."""

    def test_vllm_satisfies_protocol(self):
        backend = VLLMBackend(model="m")
        assert isinstance(backend, InferenceBackend)

    def test_litellm_satisfies_protocol(self):
        backend = LiteLLMBackend(Model("gpt-4o"))
        assert isinstance(backend, InferenceBackend)


class TestLiteLLMBackend:
    """LiteLLMBackend correctly adapts pyforger.acquire.Model."""

    def test_exposes_model_id(self):
        backend = LiteLLMBackend(Model("gpt-4o"))
        assert backend.model == "gpt-4o"

    def test_cache_key_is_stable(self):
        b1 = LiteLLMBackend(Model("gpt-4o", temperature=0.5))
        b2 = LiteLLMBackend(Model("gpt-4o", temperature=0.5))
        # _canonical handles tuples + dicts identically; the canonical
        # forms must match.
        assert _canonical(b1.cache_key()) == _canonical(b2.cache_key())

    def test_cache_key_includes_model_state(self):
        a = LiteLLMBackend(Model("gpt-4o", temperature=0.5)).cache_key()
        b = LiteLLMBackend(Model("gpt-4o", temperature=0.9)).cache_key()
        assert _canonical(a) != _canonical(b)

    def test_generate_maps_sampling_to_generate_batch(self):
        model = Model("gpt-4o")
        # Substitute generate_batch so we can capture kwargs.
        captured: dict[str, Any] = {}

        def fake_batch(prompts, **kwargs):
            captured["prompts"] = prompts
            captured["kwargs"] = kwargs
            return [["resp-a"], ["resp-b"]]

        model.generate_batch = fake_batch  # type: ignore[method-assign]

        backend = LiteLLMBackend(model)
        out = backend.generate(
            ["p1", "p2"],
            SamplingConfig(
                temperature=0.4, max_tokens=128, top_p=0.7, stop=("\n",)
            ),
        )

        assert out == ["resp-a", "resp-b"]
        assert captured["prompts"] == ["p1", "p2"]
        assert captured["kwargs"]["n"] == 1
        assert captured["kwargs"]["temperature"] == 0.4
        assert captured["kwargs"]["max_tokens"] == 128
        assert captured["kwargs"]["stop"] == ["\n"]
        # top_p is not threaded through; documented as ignored.
        assert "top_p" not in captured["kwargs"]

    def test_generate_empty_stop_is_none(self):
        model = Model("gpt-4o")
        captured: dict[str, Any] = {}

        def fake_batch(prompts, **kwargs):
            captured["kwargs"] = kwargs
            return [["x"] for _ in prompts]

        model.generate_batch = fake_batch  # type: ignore[method-assign]
        LiteLLMBackend(model).generate(["p"], SamplingConfig(stop=()))
        assert captured["kwargs"]["stop"] is None

    def test_generate_unwraps_completion_list(self):
        model = MagicMock(spec=Model)
        model._model_id = "gpt-4o"
        model.generate_batch.return_value = [["one"], ["two"], ["three"]]
        # MagicMock doesn't carry the real cache_key signature; stub it.
        model.cache_key.return_value = ("Model", "gpt-4o", 0.7, None, ())

        backend = LiteLLMBackend(model)
        assert backend.generate(["a", "b", "c"], SamplingConfig()) == [
            "one",
            "two",
            "three",
        ]
