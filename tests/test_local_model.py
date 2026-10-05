"""Tests for LocalModel (vLLM backend) and split_reasoning, using a fake vllm."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pyforger.acquire.model import GenerateResult, LocalModel, split_reasoning
from pyforger.alter.generate import self_taught_reasoner


class FakeSamplingParams:
    """Records the arguments LocalModel passes to ``vllm.SamplingParams``."""

    def __init__(self, **kwargs):
        """Store the sampling kwargs for later assertions."""
        self.kwargs = kwargs


@pytest.fixture
def engines(monkeypatch):
    """Install a fake ``vllm`` module and return the engines it creates.

    Completion ``j`` for prompt ``i`` is ``"<think>t{i}.{j}</think>a{i}.{j}"``
    with ``i + j + 1`` token ids; prompt ``i`` has ``i + 3`` prompt token ids.
    """
    created: list = []

    class FakeLLM:
        """Stand-in for ``vllm.LLM`` that records its calls."""

        def __init__(self, **kwargs):
            """Record the engine kwargs."""
            self.init_kwargs = kwargs
            self.chat_calls: list = []
            created.append(self)

        def chat(self, messages, sampling_params, **kwargs):
            """Record the call and return one fake RequestOutput per prompt."""
            self.chat_calls.append(
                SimpleNamespace(
                    messages=messages, sampling_params=sampling_params, kwargs=kwargs
                )
            )
            n = sampling_params.kwargs["n"]
            return [
                SimpleNamespace(
                    prompt_token_ids=list(range(i + 3)),
                    outputs=[
                        SimpleNamespace(
                            text=f"<think>t{i}.{j}</think>a{i}.{j}",
                            token_ids=list(range(i + j + 1)),
                        )
                        for j in range(n)
                    ],
                )
                for i in range(len(messages))
            ]

    module = types.ModuleType("vllm")
    module.LLM = FakeLLM
    module.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    return created


class TestLazyEngine:
    """The vLLM engine is only built on the first real generation call."""

    def test_construction_does_not_need_vllm(self, monkeypatch):
        """Constructing a LocalModel works without vllm installed."""
        monkeypatch.setitem(sys.modules, "vllm", None)  # makes `import vllm` fail
        model = LocalModel("Qwen/Qwen3-4B")
        assert model.history == []

    def test_construction_does_not_load_engine(self, engines):
        """No engine is created until generation."""
        LocalModel("Qwen/Qwen3-4B")
        assert engines == []

    def test_empty_batch_does_not_load_engine(self, engines):
        """An empty prompt list returns [] without building the engine."""
        model = LocalModel("Qwen/Qwen3-4B")
        assert model.generate_batch([]) == []
        assert engines == []
        assert model.history == []

    def test_engine_created_once_across_calls(self, engines):
        """Repeated calls reuse a single engine."""
        model = LocalModel("Qwen/Qwen3-4B")
        model.generate("a")
        model.generate_batch(["b", "c"])
        assert len(engines) == 1
        assert len(engines[0].chat_calls) == 2

    def test_engine_kwargs_forwarded(self, engines):
        """Engine settings and extra kwargs are passed to vllm.LLM."""
        model = LocalModel(
            "Qwen/Qwen3-4B",
            max_model_len=40960,
            gpu_memory_utilization=0.5,
            tensor_parallel_size=2,
            dtype="bfloat16",
            seed=7,
            max_num_seqs=8,
            revision="abc123",
        )
        model.generate("a")
        assert engines[0].init_kwargs == {
            "model": "Qwen/Qwen3-4B",
            "gpu_memory_utilization": 0.5,
            "tensor_parallel_size": 2,
            "dtype": "bfloat16",
            "seed": 7,
            "max_model_len": 40960,
            "max_num_seqs": 8,
            "revision": "abc123",
        }

    def test_max_model_len_defaults_to_none(self, engines):
        """Without max_model_len, vLLM gets None (its own default)."""
        LocalModel("Qwen/Qwen3-4B").generate("a")
        assert engines[0].init_kwargs["max_model_len"] is None


class TestMissingVllm:
    """Generating without vllm installed fails with an install hint."""

    def test_generation_explains_how_to_install(self, monkeypatch):
        """A missing vllm raises ImportError naming the pyforger[vllm] extra."""
        monkeypatch.setitem(sys.modules, "vllm", None)  # makes `import vllm` fail
        model = LocalModel("Qwen/Qwen3-4B")
        with pytest.raises(ImportError, match=r'pip install "pyforger\[vllm\]"'):
            model.generate("a")

    def test_empty_batch_needs_no_vllm(self, monkeypatch):
        """An empty batch returns [] even without vllm."""
        monkeypatch.setitem(sys.modules, "vllm", None)
        assert LocalModel("Qwen/Qwen3-4B").generate_batch([]) == []

    def test_broken_vllm_install_is_not_masked(self, monkeypatch, tmp_path):
        """If vllm is installed but one of its own imports fails, that surfaces."""
        package = tmp_path / "vllm"
        package.mkdir()
        (package / "__init__.py").write_text("import _pyforger_missing_dependency\n")
        monkeypatch.delitem(sys.modules, "vllm", raising=False)
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(ModuleNotFoundError) as excinfo:
            LocalModel("Qwen/Qwen3-4B").generate("a")
        assert excinfo.value.name == "_pyforger_missing_dependency"


class TestNoLiteLLMValidation:
    """LocalModel skips LiteLLM's parameter registry, which has no vLLM ids."""

    def test_construction_and_generation_skip_validation(self, engines):
        """Neither construction nor generation calls _validate_params."""
        with patch(
            "pyforger.acquire.model._validate_params", side_effect=AssertionError
        ) as validate:
            model = LocalModel("Qwen/Qwen3-4B", temperature=0.5)
            model.generate_batch(["a", "b"], n=4)
        validate.assert_not_called()


class TestChatCall:
    """Arguments passed to ``LLM.chat``."""

    def test_string_and_message_prompts(self, engines):
        """String prompts become user messages; message lists pass through."""
        chat = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "q2"},
        ]
        LocalModel("Qwen/Qwen3-4B").generate_batch(["q1", chat])
        assert engines[0].chat_calls[0].messages == [
            [{"role": "user", "content": "q1"}],
            chat,
        ]

    def test_enable_thinking_default(self, engines):
        """By default the chat template enables thinking; nothing else is sent."""
        LocalModel("Qwen/Qwen3-4B").generate("a")
        assert engines[0].chat_calls[0].kwargs == {
            "chat_template_kwargs": {"enable_thinking": True},
            "use_tqdm": False,
        }

    def test_enable_thinking_false(self, engines):
        """enable_thinking=False is forwarded to the chat template."""
        LocalModel("Qwen/Qwen3-4B", enable_thinking=False).generate("a")
        kwargs = engines[0].chat_calls[0].kwargs
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}

    def test_explicit_chat_template_kwargs_override(self, engines):
        """Explicit chat_template_kwargs are used verbatim."""
        template_kwargs = {"enable_thinking": False, "custom": 1}
        LocalModel(
            "Qwen/Qwen3-4B", enable_thinking=True, chat_template_kwargs=template_kwargs
        ).generate("a")
        kwargs = engines[0].chat_calls[0].kwargs
        assert kwargs["chat_template_kwargs"] == template_kwargs

    @pytest.mark.parametrize("show_progress", [False, True])
    def test_progress_bar_follows_show_progress(self, engines, show_progress):
        """use_tqdm mirrors show_progress."""
        LocalModel("Qwen/Qwen3-4B", show_progress=show_progress).generate("a")
        assert engines[0].chat_calls[0].kwargs["use_tqdm"] is show_progress

    def test_per_call_show_progress_overrides_instance(self, engines):
        """show_progress passed to generate_batch beats the instance setting."""
        model = LocalModel("Qwen/Qwen3-4B", show_progress=True)
        model.generate_batch(["a"], show_progress=False)
        assert engines[0].chat_calls[0].kwargs["use_tqdm"] is False


class TestSamplingParams:
    """Arguments passed to ``vllm.SamplingParams``."""

    def _params(self, engines) -> dict:
        """Sampling kwargs of the most recent chat call."""
        return engines[0].chat_calls[-1].sampling_params.kwargs

    def test_defaults(self, engines):
        """Defaults: n=1, instance temperature, generous max_tokens, seed 0."""
        LocalModel("Qwen/Qwen3-4B").generate("a")
        assert self._params(engines) == {
            "n": 1,
            "temperature": 0.7,
            "max_tokens": 32768,
            "stop": None,
            "seed": 0,
        }

    def test_instance_settings(self, engines):
        """Instance temperature, max_tokens and seed are used."""
        LocalModel("Qwen/Qwen3-4B", temperature=0.2, max_tokens=1024, seed=5).generate(
            "a"
        )
        params = self._params(engines)
        assert params["temperature"] == 0.2
        assert params["max_tokens"] == 1024
        assert params["seed"] == 5

    def test_per_call_overrides(self, engines):
        """Per-call n, temperature, max_tokens and stop win over defaults."""
        model = LocalModel("Qwen/Qwen3-4B", temperature=0.9, max_tokens=1024)
        model.generate("a", n=3, temperature=0.0, max_tokens=64, stop=["\n"])
        assert self._params(engines) == {
            "n": 3,
            "temperature": 0.0,
            "max_tokens": 64,
            "stop": ["\n"],
            "seed": 0,
        }


class TestGenerateResults:
    """Conversion of vLLM outputs into GenerateResult and history."""

    def test_batch_of_prompts_and_completions(self, engines):
        """One GenerateResult per prompt, n completions each, in order."""
        model = LocalModel("Qwen/Qwen3-4B")
        results = model.generate_batch(["p0", "p1", "p2"], n=2)
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, GenerateResult)
            assert result == [
                f"<think>t{i}.0</think>a{i}.0",
                f"<think>t{i}.1</think>a{i}.1",
            ]
            assert result.usage == {
                "prompt_tokens": i + 3,
                "completion_tokens": 2 * i + 3,  # (i + 1) + (i + 2) token ids
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
            }
            assert result.reasoning == [None, None]
            assert result.model_id == "Qwen/Qwen3-4B"
            assert result.raw is None

    def test_thinking_stays_inline(self, engines):
        """Completions keep the <think> block; split_reasoning separates it."""
        completion = LocalModel("Qwen/Qwen3-4B").generate("p")[0]
        assert split_reasoning(completion) == ("t0.0", "a0.0")

    def test_generate_returns_single_result(self, engines):
        """generate() returns the one prompt's GenerateResult."""
        result = LocalModel("Qwen/Qwen3-4B").generate("p", n=3)
        assert isinstance(result, GenerateResult)
        assert len(result) == 3

    def test_history_and_totals(self, engines):
        """Each prompt appends to history; totals sum usage at zero cost."""
        model = LocalModel("Qwen/Qwen3-4B")
        model.generate_batch(["p0", "p1"], n=2)
        model.generate("p0")
        assert len(model.history) == 3
        assert model.total_usage == {
            "prompt_tokens": 3 + 4 + 3,
            "completion_tokens": 3 + 5 + 1,
            "reasoning_tokens": 0,
        }
        assert model.total_cost == 0.0
        model.reset_history()
        assert model.history == []


class TestCacheKey:
    """cache_key separates configurations that can change outputs."""

    def test_equal_configs_share_key(self):
        """Two identically configured instances have the same key."""
        kwargs = {"dtype": "bfloat16", "revision": "abc", "seed": 3}
        assert (
            LocalModel("Qwen/Qwen3-4B", **kwargs).cache_key()
            == LocalModel("Qwen/Qwen3-4B", **kwargs).cache_key()
        )

    @pytest.mark.parametrize(
        "change",
        [
            {"temperature": 0.1},
            {"max_tokens": 128},
            {"max_model_len": 4096},
            {"dtype": "float16"},
            {"enable_thinking": False},
            {"chat_template_kwargs": {"enable_thinking": True, "x": 1}},
            {"seed": 1},
            {"revision": "def456"},
            {"quantization": "awq"},
        ],
    )
    def test_output_affecting_settings_change_key(self, change):
        """Changing any output-affecting setting changes the key."""
        base = LocalModel("Qwen/Qwen3-4B")
        assert LocalModel("Qwen/Qwen3-4B", **change).cache_key() != base.cache_key()

    @pytest.mark.parametrize(
        "change",
        [
            {"gpu_memory_utilization": 0.5},
            {"tensor_parallel_size": 4},
            {"show_progress": True},
        ],
    )
    def test_speed_only_settings_keep_key(self, change):
        """Settings that only affect speed or memory leave the key unchanged."""
        base = LocalModel("Qwen/Qwen3-4B")
        assert LocalModel("Qwen/Qwen3-4B", **change).cache_key() == base.cache_key()

    def test_empty_template_kwargs_differ_from_default(self):
        """chat_template_kwargs={} sends nothing, unlike the enable_thinking default."""
        default = LocalModel("Qwen/Qwen3-4B", enable_thinking=False)
        empty = LocalModel(
            "Qwen/Qwen3-4B", enable_thinking=False, chat_template_kwargs={}
        )
        assert default.cache_key() != empty.cache_key()

    def test_ignored_enable_thinking_keeps_key(self):
        """enable_thinking is ignored and unkeyed when template kwargs are explicit."""
        template_kwargs = {"enable_thinking": False}
        assert (
            LocalModel(
                "m", enable_thinking=True, chat_template_kwargs=template_kwargs
            ).cache_key()
            == LocalModel(
                "m", enable_thinking=False, chat_template_kwargs=template_kwargs
            ).cache_key()
        )

    @pytest.mark.parametrize(
        "change", [{"gpu_memory_utilization": 0.5}, {"tensor_parallel_size": 4}]
    )
    def test_memory_settings_keyed_when_auto_fitting(self, change):
        """With max_model_len=-1, memory settings decide the context length."""
        base = LocalModel("Qwen/Qwen3-4B", max_model_len=-1)
        changed = LocalModel("Qwen/Qwen3-4B", max_model_len=-1, **change)
        assert changed.cache_key() != base.cache_key()

    def test_op_cache_not_shared_across_engine_configs(self, engines):
        """An @op run with a different engine revision re-samples.

        An identical configuration hits the cache and never builds an engine.
        """
        items = [{"prompt": "q", "final_answer": "0"}]
        self_taught_reasoner(items, "{prompt}", LocalModel("m", revision="a"), n=1)
        self_taught_reasoner(items, "{prompt}", LocalModel("m", revision="a"), n=1)
        assert len(engines) == 1
        self_taught_reasoner(items, "{prompt}", LocalModel("m", revision="b"), n=1)
        assert len(engines) == 2


class TestSplitReasoning:
    """split_reasoning separates a <think> block from the answer."""

    def test_splits_think_block(self):
        """Thinking and answer are split and stripped."""
        assert split_reasoning("<think> step 1 </think>\n 42 ") == ("step 1", "42")

    def test_no_close_tag_is_all_answer(self):
        """Without </think> the whole text is the answer."""
        assert split_reasoning("  just 42 ") == ("", "just 42")

    def test_truncated_thought_is_all_answer(self):
        """An unclosed <think> (ran out of tokens) is returned as the answer."""
        assert split_reasoning("<think>still going") == ("", "<think>still going")

    def test_prefilled_open_tag(self):
        """Templates that pre-fill <think> emit only the closing tag."""
        assert split_reasoning("reasoning</think>42") == ("reasoning", "42")

    def test_splits_on_first_close_tag(self):
        """Only the first </think> ends the thinking block."""
        assert split_reasoning("<think>a</think>b</think>c") == ("a", "b</think>c")
