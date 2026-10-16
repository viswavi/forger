"""Tests for n > 1 in Model.generate_batch: fan-out and completion-count checks."""

from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

import pyforger.acquire.model as model_module
from pyforger.acquire.model import LocalModel, Model, _fans_out_n

FAN_OUT_IDS = [
    "anthropic/claude-sonnet-4-5",
    "litellm_proxy/claude-sonnet-4-5",
    "litellm_proxy/anthropic/claude-opus-4-1",
    "openrouter/anthropic/claude-sonnet-4-5",
    "bedrock/us.anthropic.claude-sonnet-4-5-v1:0",
    "xai/grok-4",
    "litellm_proxy/xai/grok-4",
    "litellm_proxy/grok-4-fast",
    "gemini/gemini-2.5-pro",
    "gemini/gemini-3.5-flash",
    "litellm_proxy/gemini/gemini-3.5-flash",
    "vertex_ai/gemini-2.0-flash",
    "openai/o1",
    "o3-mini",
    "litellm_proxy/openai/o4-mini",
    "gpt-5",
    "openai/gpt-5-mini",
    "azure_ai/gpt-5.6-terra",
    "litellm_proxy/gpt-5.6-sol",
]

SINGLE_REQUEST_IDS = [
    "gpt-4o",
    "openai/gpt-4.1",
    "litellm_proxy/gpt-4o-mini",
    "azure_ai/DeepSeek-V4-Flash",
    "deepseek/deepseek-chat",
    "openai/o44",
    "hosted_vllm/Qwen/Qwen3-8B",
]


def make_response(contents, prompt_tokens=10, completion_tokens=20, cost=0.001):
    """A litellm-shaped response with one choice per entry of ``contents``."""
    choices = []
    for text in contents:
        choice = MagicMock()
        choice.message.content = text
        choice.message.reasoning_content = f"why {text}"
        choices.append(choice)
    response = MagicMock()
    response.choices = choices
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.completion_tokens_details.reasoning_tokens = 1
    response._hidden_params = {"response_cost": cost}
    return response


def numbered():
    """Fake completion: the j-th request for prompt "P" answers "P#j"."""
    seen: dict[str, int] = {}
    lock = threading.Lock()

    def fake_completion(messages, **kwargs):
        text = messages[0]["content"]
        with lock:
            seen[text] = seen.get(text, 0) + 1
            j = seen[text] - 1
        return make_response([f"{text}#{j}"])

    return fake_completion


def sent_contents(mock_litellm):
    """First-message content of every completion request, in send order."""
    return [
        c.kwargs["messages"][0]["content"]
        for c in mock_litellm.completion.call_args_list
    ]


@pytest.fixture
def mock_litellm():
    """Patch litellm in pyforger.acquire.model; registry knows no model."""
    with patch("pyforger.acquire.model.litellm") as mock:
        mock.get_supported_openai_params.return_value = None
        yield mock


class TestFansOutN:
    """_fans_out_n picks the models that get n separate requests."""

    @pytest.mark.parametrize("model_id", FAN_OUT_IDS)
    def test_listed_families_fan_out(self, mock_litellm, model_id):
        """Claude, Grok, Gemini, o-series and gpt-5 fan out through any prefix."""
        assert _fans_out_n(model_id)

    @pytest.mark.parametrize("model_id", SINGLE_REQUEST_IDS)
    def test_other_models_send_n(self, mock_litellm, model_id):
        """Models outside the list keep a single request with n."""
        assert not _fans_out_n(model_id)

    def test_supported_params_without_n_fan_out(self, mock_litellm):
        """A known supported-params list that lacks "n" means fan-out."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "max_tokens",
        ]
        assert _fans_out_n("litellm_proxy/my-alias")

    def test_supported_params_with_n_do_not_fan_out(self, mock_litellm):
        """A supported-params list that includes "n" keeps a single request."""
        mock_litellm.get_supported_openai_params.return_value = ["temperature", "n"]
        assert not _fans_out_n("litellm_proxy/my-alias")


class TestFanOut:
    """n > 1 on a fan-out model sends n requests with n=1 per prompt."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "litellm_proxy/claude-sonnet-4-5",
            "anthropic/claude-sonnet-4-5",
            "xai/grok-4",
            "litellm_proxy/xai/grok-4",
            "litellm_proxy/gemini/gemini-3.5-flash",
            "gemini/gemini-2.5-pro",
            "azure_ai/gpt-5.6-terra",
            "openai/o3",
        ],
    )
    def test_requests_and_merge_order(self, mock_litellm, model_id):
        """Prompts [A, B] with n=3 send [A, A, A, B, B, B] and merge in order."""
        mock_litellm.completion.side_effect = numbered()
        model = Model(model_id, api_key="sk-test", temperature=1.0, max_concurrency=1)
        results = model.generate_batch(["A", "B"], n=3)
        assert results == [["A#0", "A#1", "A#2"], ["B#0", "B#1", "B#2"]]
        assert sent_contents(mock_litellm) == ["A"] * 3 + ["B"] * 3
        for c in mock_litellm.completion.call_args_list:
            assert c.kwargs["n"] == 1

    def test_single_prompt_is_one_concurrent_batch(self, mock_litellm):
        """One prompt with n=4 sends 4 requests."""
        mock_litellm.completion.side_effect = numbered()
        model = Model(
            "litellm_proxy/claude-sonnet-4-5", api_key="sk-test", max_concurrency=1
        )
        assert model.generate("Q", n=4) == ["Q#0", "Q#1", "Q#2", "Q#3"]
        assert mock_litellm.completion.call_count == 4

    def test_chat_messages_are_copied_per_request(self, mock_litellm):
        """Each request gets its own message dicts with the same content."""
        mock_litellm.completion.side_effect = numbered()
        messages = [{"role": "user", "content": "Q"}]
        Model("xai/grok-4").generate_batch([messages], n=2)
        sent = [c.kwargs["messages"] for c in mock_litellm.completion.call_args_list]
        assert sent == [messages, messages]
        assert sent[0][0] is not sent[1][0]

    def test_usage_cost_reasoning_and_raw_merged(self, mock_litellm):
        """Usage and cost are summed, reasoning concatenated, raw lists responses."""
        responses = [
            make_response(["a"], prompt_tokens=10, completion_tokens=1, cost=0.25),
            make_response(["b"], prompt_tokens=10, completion_tokens=2, cost=0.5),
        ]
        mock_litellm.completion.side_effect = responses
        result = Model("gemini/gemini-2.5-pro", max_concurrency=1).generate("Q", n=2)
        assert result == ["a", "b"]
        assert result.usage == {
            "prompt_tokens": 20,
            "completion_tokens": 3,
            "reasoning_tokens": 2,
            "cost_usd": 0.75,
        }
        assert result.reasoning == ["why a", "why b"]
        assert result.raw == responses
        assert result.model_id == "gemini/gemini-2.5-pro"

    def test_history_per_prompt_step_usage_per_request(self, mock_litellm):
        """History gets one merged result per prompt; add_usage sees each request."""
        mock_litellm.completion.side_effect = [
            make_response(["a0"], prompt_tokens=5, completion_tokens=1, cost=0.25),
            make_response(["a1"], prompt_tokens=5, completion_tokens=2, cost=0.25),
            make_response(["b0"], prompt_tokens=7, completion_tokens=3, cost=0.5),
            make_response(["b1"], prompt_tokens=7, completion_tokens=4, cost=0.5),
        ]
        model = Model("litellm_proxy/claude-sonnet-4-5", max_concurrency=1)
        with patch("pyforger.acquire.model._tracking.add_usage") as add_usage:
            results = model.generate_batch(["A", "B"], n=2)
        assert model.history == results
        assert len(model.history) == 2
        assert add_usage.call_args_list == [
            call(5, 1, 0.25, model_id="litellm_proxy/claude-sonnet-4-5"),
            call(5, 2, 0.25, model_id="litellm_proxy/claude-sonnet-4-5"),
            call(7, 3, 0.5, model_id="litellm_proxy/claude-sonnet-4-5"),
            call(7, 4, 0.5, model_id="litellm_proxy/claude-sonnet-4-5"),
        ]
        assert model.total_usage == {
            "prompt_tokens": 24,
            "completion_tokens": 10,
            "reasoning_tokens": 4,
        }
        assert model.total_cost == pytest.approx(1.5)

    def test_show_progress_keeps_order(self, mock_litellm):
        """The progress path fans out too and merges out-of-order completions."""
        b_done = threading.Event()
        b_count = []
        lock = threading.Lock()

        def fake_completion(messages, **kwargs):
            assert kwargs["n"] == 1
            text = messages[0]["content"]
            if text == "A":  # A's requests finish after both of B's
                assert b_done.wait(timeout=5)
            else:
                with lock:
                    b_count.append(1)
                    if len(b_count) == 2:
                        b_done.set()
            return make_response([text])

        mock_litellm.completion.side_effect = fake_completion
        model = Model("xai/grok-4", show_progress=True)
        assert model.generate_batch(["A", "B"], n=2) == [["A", "A"], ["B", "B"]]
        assert mock_litellm.completion.call_count == 4

    def test_sub_request_without_a_choice_raises(self, mock_litellm):
        """A fanned-out request with no choice raises instead of returning short."""
        mock_litellm.completion.side_effect = [
            make_response(["a"]),
            make_response([]),
        ]
        model = Model("gemini/gemini-2.5-pro", max_concurrency=1)
        with pytest.raises(ValueError, match="returned 1 completions .* n=2"):
            model.generate("Q", n=2)
        assert len(model.history) == 1  # the spend is still recorded

    def test_request_exception_is_raised(self, mock_litellm):
        """An exception from one request is raised as before."""
        error = RuntimeError("rate limited")
        mock_litellm.completion.side_effect = [make_response(["a"]), error]
        model = Model("xai/grok-4", max_concurrency=1)
        with pytest.raises(RuntimeError) as excinfo:
            model.generate("Q", n=2)
        assert excinfo.value is error
        assert model.history == []


class TestSingleRequest:
    """Models outside the fan-out list send n once and must return n choices."""

    def test_n_sent_once(self, mock_litellm):
        """n=3 on gpt-4o is one completion call with n=3."""
        mock_litellm.completion.return_value = make_response(["a", "b", "c"])
        result = Model("gpt-4o").generate("Q", n=3)
        assert result == ["a", "b", "c"]
        assert mock_litellm.completion.call_count == 1
        assert mock_litellm.completion.call_args.kwargs["n"] == 3


def ignores_n(empty_n1=False):
    """Fake completion that returns one choice whatever n is ("Q#0", "Q#1", ...).

    With empty_n1, n=1 requests return no choice at all.
    """
    seen: dict[str, int] = {}
    lock = threading.Lock()

    def fake_completion(messages, n=1, **kwargs):
        text = messages[0]["content"]
        if empty_n1 and n == 1:
            return make_response([], cost=0.25)
        with lock:
            k = seen.get(text, 0)
            seen[text] = k + 1
        return make_response([f"{text}#{k}"], cost=0.25)

    return fake_completion


class TestShortN:
    """A model that should support n but returns fewer is refilled with n=1."""

    def test_missing_completions_sent_as_n1(self, mock_litellm, caplog):
        """n=4 gets 1 choice: one warning, 3 n=1 requests, usage summed."""
        mock_litellm.completion.side_effect = ignores_n()
        model = Model("litellm_proxy/my-alias", max_concurrency=1)
        with caplog.at_level("WARNING", logger="pyforger.acquire.model"):
            result = model.generate("Q", n=4)
        assert result == ["Q#0", "Q#1", "Q#2", "Q#3"]
        sent_n = [c.kwargs["n"] for c in mock_litellm.completion.call_args_list]
        assert sent_n == [4, 1, 1, 1]
        assert result.usage["prompt_tokens"] == 40
        assert result.usage["cost_usd"] == pytest.approx(1.0)
        assert model.history == [result]
        warnings = [r.getMessage() for r in caplog.records]
        assert len(warnings) == 1 and "litellm_proxy/my-alias" in warnings[0]
        assert "litellm_proxy/my-alias" in model_module._LEARNED_NO_N

    def test_later_calls_fan_out_from_the_start(self, mock_litellm, caplog):
        """After the first short response, n=3 is three n=1 requests, no warning."""
        mock_litellm.completion.side_effect = ignores_n()
        model = Model("litellm_proxy/my-alias", max_concurrency=1)
        model.generate("Q", n=2)
        mock_litellm.completion.reset_mock()
        caplog.clear()
        with caplog.at_level("WARNING", logger="pyforger.acquire.model"):
            assert model.generate("R", n=3) == ["R#0", "R#1", "R#2"]
        sent_n = [c.kwargs["n"] for c in mock_litellm.completion.call_args_list]
        assert sent_n == [1, 1, 1]
        assert caplog.records == []
        assert _fans_out_n("litellm_proxy/my-alias")

    def test_only_short_prompts_are_refilled(self, mock_litellm):
        """Q1 got its 2 choices, Q2 got 1: one n=1 request, for Q2."""
        mock_litellm.completion.side_effect = [
            make_response(["a", "b"]),
            make_response(["c"]),
            make_response(["d"]),
        ]
        model = Model("gpt-4o", max_concurrency=1)
        assert model.generate_batch(["Q1", "Q2"], n=2) == [["a", "b"], ["c", "d"]]
        assert sent_contents(mock_litellm) == ["Q1", "Q2", "Q2"]

    def test_short_n1_requests_raise(self, mock_litellm):
        """If the n=1 requests come back short too, it raises; never pads."""
        mock_litellm.completion.side_effect = ignores_n(empty_n1=True)
        model = Model("litellm_proxy/my-alias")
        with pytest.raises(ValueError) as excinfo:
            model.generate("Q", n=4)
        message = str(excinfo.value)
        assert "litellm_proxy/my-alias returned 1 completions" in message
        assert "n=4" in message
        assert "separate n=1 requests" in message
        assert len(model.history) == 1  # the spend is still recorded


class TestNEqualsOne:
    """n=1 keeps today's request shape for every model."""

    @pytest.mark.parametrize("model_id", ["litellm_proxy/claude-sonnet-4-5", "gpt-4o"])
    def test_one_completion_call(self, mock_litellm, model_id):
        """One prompt, n=1: a single completion call with n=1, raw is the response."""
        response = make_response(["ok"])
        mock_litellm.completion.return_value = response
        result = Model(model_id).generate("Q")
        assert result == ["ok"]
        assert result.raw is response
        kwargs = mock_litellm.completion.call_args.kwargs
        assert kwargs["n"] == 1
        assert kwargs["messages"] == [{"role": "user", "content": "Q"}]
        assert mock_litellm.completion.call_count == 1

    def test_batch_is_one_request_per_prompt(self, mock_litellm):
        """Several prompts with n=1 send one request each."""
        mock_litellm.completion.side_effect = numbered()
        model = Model("xai/grok-4")
        assert model.generate_batch(["A", "B"]) == [["A#0"], ["B#0"]]
        assert sorted(sent_contents(mock_litellm)) == ["A", "B"]

    def test_supported_params_without_n_still_one_request(self, mock_litellm):
        """n=1 never consults the fan-out rule."""
        mock_litellm.get_supported_openai_params.return_value = ["temperature"]
        mock_litellm.completion.return_value = make_response(["ok"])
        assert Model("litellm_proxy/my-alias").generate("Q") == ["ok"]

    def test_cache_key_unchanged(self):
        """cache_key keeps its shape, so existing @op caches stay valid."""
        model = Model("litellm_proxy/claude-sonnet-4-5", temperature=1.0, foo="bar")
        assert model.cache_key() == (
            "Model",
            "litellm_proxy/claude-sonnet-4-5",
            1.0,
            None,
            2,
            (("foo", "bar"),),
        )


class TestLocalModelCount:
    """LocalModel sends n to vLLM and raises if it returns another count."""

    @pytest.fixture
    def fake_vllm(self, monkeypatch):
        """A fake vllm whose engine returns ``returned`` completions per prompt."""
        state = {"returned": None}

        class FakeLLM:
            def __init__(self, **kwargs):
                pass

            def chat(self, messages, sampling_params, **kwargs):
                count = state["returned"] or sampling_params.n
                return [
                    SimpleNamespace(
                        prompt_token_ids=[1, 2],
                        outputs=[
                            SimpleNamespace(text=f"c{j}", token_ids=[j])
                            for j in range(count)
                        ],
                    )
                    for _ in messages
                ]

        module = types.ModuleType("vllm")
        module.LLM = FakeLLM
        module.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        monkeypatch.setitem(sys.modules, "vllm", module)
        return state

    def test_n_completions_returned(self, fake_vllm):
        """The n completions from vLLM come back as one GenerateResult."""
        assert LocalModel("Qwen/Qwen3-4B").generate("p", n=3) == ["c0", "c1", "c2"]

    def test_fewer_completions_raise(self, fake_vllm):
        """A short vLLM output raises instead of returning fewer than n."""
        fake_vllm["returned"] = 2
        with pytest.raises(ValueError, match="returned 2 completions .* n=3"):
            LocalModel("Qwen/Qwen3-4B").generate("p", n=3)
