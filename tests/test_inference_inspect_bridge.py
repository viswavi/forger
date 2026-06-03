"""Tests for pyforger.inference.inspect_bridge.

``inspect_ai`` is an optional dependency and is not installed in CI, so these
tests inject a minimal fake ``inspect_ai`` package into ``sys.modules`` (the
same technique ``test_inference_backends.py`` uses for ``vllm``). The fake
provides just enough of the surface the bridge touches: ``modelapi``,
``ModelAPI``, ``GenerateConfig``, ``ModelOutput``, ``ChatCompletionChoice``, and
``ChatMessageAssistant``.
"""
from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from pyforger.acquire.model import Model
from pyforger.inference import inspect_bridge as bridge
from pyforger.inference.backends import LiteLLMBackend, SamplingConfig


class FakeBackend:
    """Records the prompts and sampling it was called with; returns canned text."""

    def __init__(self, completions: list[str] | None = None, *, identity: str = "fake"):
        self.model = identity
        self._completions = completions or ["the-completion"]
        self.calls: list[tuple[list[Any], SamplingConfig]] = []

    def cache_key(self) -> dict:
        return {"backend": "fake", "identity": self.model}

    def generate(self, prompts: list[Any], sampling: SamplingConfig) -> list[str]:
        self.calls.append((prompts, sampling))
        n = len(prompts)
        return [self._completions[i % len(self._completions)] for i in range(n)]


def _install_fake_inspect(monkeypatch) -> dict[str, Any]:
    """Install a fake ``inspect_ai`` + ``inspect_ai.model`` and return captures."""
    captures: dict[str, Any] = {"registered": []}

    def modelapi(name: str):
        def decorator(factory):
            captures["registered"].append((name, factory))
            return factory

        return decorator

    class ModelAPI:
        def __init__(self, model_name, base_url=None, api_key=None, config=None):
            self.model_name = model_name
            self.base_url = base_url
            self.api_key = api_key
            self.config = config

    class GenerateConfig:
        def __init__(self, temperature=None, max_tokens=None, top_p=None,
                     stop_seqs=None, num_choices=None):
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.top_p = top_p
            self.stop_seqs = stop_seqs
            self.num_choices = num_choices

    class ModelOutput:
        def __init__(self, model=None, choices=None, completion=None):
            self.model = model
            self.choices = choices or []
            self.completion = completion

        @staticmethod
        def from_content(model, content, stop_reason="stop", error=None):
            return ModelOutput(model=model, completion=content,
                               choices=[_Choice(_Assistant(content, model))])

    class _Assistant:
        def __init__(self, content, model=None, **kwargs):
            self.content = content
            self.text = content
            self.model = model

    class _Choice:
        def __init__(self, message, stop_reason="stop", **kwargs):
            self.message = message
            self.stop_reason = stop_reason

    model_mod = types.ModuleType("inspect_ai.model")
    model_mod.modelapi = modelapi
    model_mod.ModelAPI = ModelAPI
    model_mod.GenerateConfig = GenerateConfig
    model_mod.ModelOutput = ModelOutput
    model_mod.ChatMessageAssistant = _Assistant
    model_mod.ChatCompletionChoice = _Choice

    root = types.ModuleType("inspect_ai")
    root.model = model_mod

    monkeypatch.setitem(sys.modules, "inspect_ai", root)
    monkeypatch.setitem(sys.modules, "inspect_ai.model", model_mod)

    captures["GenerateConfig"] = GenerateConfig
    captures["ModelOutput"] = ModelOutput
    return captures


@pytest.fixture(autouse=True)
def _reset_bridge_state(monkeypatch):
    """Reset the bridge's module-level registries around each test."""
    bridge.clear_backend_registry()
    monkeypatch.setattr(bridge, "_provider_registered", False, raising=False)
    yield
    bridge.clear_backend_registry()
    monkeypatch.setattr(bridge, "_provider_registered", False, raising=False)


class TestRendering:
    """Default and passthrough message renderers."""

    def test_single_user_message_returned_verbatim(self):
        assert bridge.default_render([{"role": "user", "content": "hi"}]) == "hi"

    def test_multi_turn_is_role_tagged(self):
        out = bridge.default_render([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "2+2?"},
        ])
        assert out == "system: be terse\n\nuser: 2+2?\n\nassistant:"

    def test_passthrough_returns_messages(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert bridge.passthrough_render(msgs) is msgs

    def test_litellm_backend_gets_passthrough_default(self):
        backend = LiteLLMBackend(Model("gpt-4o"))
        assert bridge._default_render_for(backend) is bridge.passthrough_render

    def test_other_backend_gets_default_render(self):
        assert bridge._default_render_for(FakeBackend()) is bridge.default_render


class TestRegistration:
    """register_backend stores the backend and registers the Inspect provider."""

    def test_returns_model_string(self, monkeypatch):
        _install_fake_inspect(monkeypatch)
        model = bridge.register_backend(FakeBackend(), name="m1")
        assert model == "forger/m1"

    def test_provider_registered_once(self, monkeypatch):
        caps = _install_fake_inspect(monkeypatch)
        bridge.register_backend(FakeBackend(), name="a")
        bridge.register_backend(FakeBackend(), name="b")
        assert [n for n, _ in caps["registered"]] == ["forger"]

    def test_unknown_name_raises_with_hint(self, monkeypatch):
        _install_fake_inspect(monkeypatch)
        with pytest.raises(KeyError) as exc:
            bridge._lookup("nope")
        assert "register_backend" in str(exc.value)

    def test_missing_inspect_raises_install_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "inspect_ai", None)
        monkeypatch.setitem(sys.modules, "inspect_ai.model", None)
        with pytest.raises(ImportError) as exc:
            bridge.register_backend(FakeBackend())
        assert "pyforger[inspect]" in str(exc.value)


class TestSamplingMapping:
    """GenerateConfig -> SamplingConfig translation."""

    def test_full_mapping(self, monkeypatch):
        caps = _install_fake_inspect(monkeypatch)
        cfg = caps["GenerateConfig"](
            temperature=0.5, max_tokens=256, top_p=0.8, stop_seqs=["\n\n"]
        )
        s = bridge._sampling_from_config(cfg, default_max_tokens=1024)
        assert s == SamplingConfig(
            temperature=0.5, max_tokens=256, top_p=0.8, stop=("\n\n",)
        )

    def test_defaults_fill_in(self, monkeypatch):
        caps = _install_fake_inspect(monkeypatch)
        cfg = caps["GenerateConfig"]()
        s = bridge._sampling_from_config(cfg, default_max_tokens=512)
        assert s == SamplingConfig(
            temperature=0.0, max_tokens=512, top_p=1.0, stop=()
        )


class TestGenerate:
    """End-to-end through the provider class, with the fake inspect_ai."""

    def _make_api(self, monkeypatch, backend, name="default", **kw):
        caps = _install_fake_inspect(monkeypatch)
        bridge.register_backend(backend, name=name, **kw)
        api_cls = bridge._build_provider()
        api = api_cls(model_name=name)
        return api, caps

    def test_generate_wraps_completion(self, monkeypatch):
        backend = FakeBackend(["4"])
        api, caps = self._make_api(monkeypatch, backend)
        cfg = caps["GenerateConfig"](temperature=0.0, max_tokens=10)
        out = asyncio.run(api.generate("what is 2+2?", config=cfg))
        assert out.completion == "4"
        # backend saw a single rendered prompt + mapped sampling
        prompts, sampling = backend.calls[0]
        assert prompts == ["what is 2+2?"]
        assert sampling.max_tokens == 10

    def test_generate_normalizes_message_objects(self, monkeypatch):
        backend = FakeBackend(["ok"])
        api, caps = self._make_api(monkeypatch, backend)
        msg = types.SimpleNamespace(role="user", text="hello", content="hello")
        asyncio.run(api.generate([msg], config=caps["GenerateConfig"]()))
        prompts, _ = backend.calls[0]
        assert prompts == ["hello"]

    def test_num_choices_generates_multiple(self, monkeypatch):
        backend = FakeBackend(["a", "b", "c"])
        api, caps = self._make_api(monkeypatch, backend)
        cfg = caps["GenerateConfig"](num_choices=3)
        out = asyncio.run(api.generate("q", config=cfg))
        prompts, _ = backend.calls[0]
        assert len(prompts) == 3
        assert [c.message.content for c in out.choices] == ["a", "b", "c"]

    def test_custom_render_is_used(self, monkeypatch):
        backend = FakeBackend(["x"])
        api, caps = self._make_api(
            monkeypatch, backend, render=lambda msgs: "RENDERED"
        )
        asyncio.run(api.generate("ignored", config=caps["GenerateConfig"]()))
        prompts, _ = backend.calls[0]
        assert prompts == ["RENDERED"]


class TestMetricsExtraction:
    """_extract_metrics flattens scorer metrics across logs."""

    def test_flattens_scorer_metrics(self):
        metric = types.SimpleNamespace(value=0.74)
        score = types.SimpleNamespace(name="match", metrics={"accuracy": metric})
        results = types.SimpleNamespace(scores=[score])
        log = types.SimpleNamespace(results=results)
        assert bridge._extract_metrics([log]) == {"match/accuracy": 0.74}

    def test_handles_missing_results(self):
        log = types.SimpleNamespace(results=None)
        assert bridge._extract_metrics([log]) == {}


class TestEvaluateWithInspect:
    """evaluate_with_inspect wires registration + eval + extraction."""

    def test_runs_eval_and_returns_metrics(self, monkeypatch):
        _install_fake_inspect(monkeypatch)
        recorded: dict[str, Any] = {}

        def fake_eval(task, *, model, model_args, **kwargs):
            recorded["task"] = task
            recorded["model"] = model
            recorded["kwargs"] = kwargs
            metric = types.SimpleNamespace(value=0.5)
            score = types.SimpleNamespace(name="match", metrics={"accuracy": metric})
            results = types.SimpleNamespace(scores=[score], total_samples=12)
            return [types.SimpleNamespace(results=results)]

        caps_root = sys.modules["inspect_ai"]
        caps_root.eval = fake_eval

        backend = FakeBackend()
        result = bridge.evaluate_with_inspect(
            backend,
            "inspect_evals/gsm8k",
            limit=12,
            sampling=SamplingConfig(temperature=0.0, max_tokens=64),
        )

        assert recorded["model"] == "forger/default"
        assert recorded["task"] == "inspect_evals/gsm8k"
        assert recorded["kwargs"]["limit"] == 12
        assert recorded["kwargs"]["max_tokens"] == 64
        assert result.metrics == {"match/accuracy": 0.5}
        assert result.samples == 12
        assert result.task == "inspect_evals/gsm8k"
