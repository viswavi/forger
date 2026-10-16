"""Tests for the Model class (LiteLLM wrapper)."""

from __future__ import annotations

import copy
import pickle
import threading
from unittest.mock import MagicMock, patch

import pytest

from pyforger.acquire.model import (
    _COST_WARNING_SEEN,
    _VALUE_CONSTRAINTS,
    GenerateResult,
    Model,
    _batch_completion_with_progress,
    _extract_cost,
    _extract_reasoning,
    _extract_usage,
    _family,
    _to_messages,
    _validate_params,
)


def make_litellm_response(
    contents: list[str | None],
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    response_cost: float = 0.001,
    include_usage: bool = True,
    include_hidden_params: bool = True,
):
    """Build a mock litellm completion response.

    Args:
        contents: List of response strings (or None), one per choice.
        prompt_tokens: Mock prompt token count.
        completion_tokens: Mock completion token count.
        response_cost: Mock response cost.
        include_usage: Whether to include usage data on the response.
        include_hidden_params: Whether to include _hidden_params dict.

    Returns:
        A MagicMock shaped like a litellm ModelResponse.
    """
    choices = []
    for text in contents:
        choice = MagicMock()
        choice.message.content = text
        choices.append(choice)
    response = MagicMock()
    response.choices = choices
    if include_usage:
        response.usage.prompt_tokens = prompt_tokens
        response.usage.completion_tokens = completion_tokens
    else:
        response.usage = None
    if include_hidden_params:
        response._hidden_params = {"response_cost": response_cost}
    else:
        response._hidden_params = {}
    return response


def by_content(responses: dict):
    """Fake completion answering each request by its first message's content.

    An exception value is raised for that request, e.g. {"Q2": RuntimeError()}.
    """

    def fake_completion(messages, **kwargs):
        response = responses[messages[0]["content"]]
        if isinstance(response, Exception):
            raise response
        return response

    return fake_completion


class TestConstructor:
    """Model constructor stores configuration."""

    def test_stores_model_id(self):
        """model_id is accessible after construction."""
        model = Model("gpt-4o")
        assert model._model_id == "gpt-4o"

    def test_default_temperature(self):
        """Default temperature is 1.0."""
        model = Model("gpt-4o")
        assert model._temperature == 1.0

    def test_custom_temperature(self):
        """Custom temperature is stored."""
        model = Model("gpt-4o", temperature=0.2)
        assert model._temperature == 0.2

    def test_default_max_tokens_is_none(self):
        """Default max_tokens is None."""
        model = Model("gpt-4o")
        assert model._max_tokens is None

    def test_custom_max_tokens(self):
        """Custom max_tokens is stored."""
        model = Model("gpt-4o", max_tokens=500)
        assert model._max_tokens == 500

    def test_default_num_retries(self):
        """Default num_retries is 2."""
        model = Model("gpt-4o")
        assert model._num_retries == 2

    def test_custom_num_retries(self):
        """Custom num_retries is stored."""
        model = Model("gpt-4o", num_retries=5)
        assert model._num_retries == 5

    def test_api_key_stored(self):
        """API key is stored when provided."""
        model = Model("gpt-4o", api_key="sk-test")
        assert model._api_key == "sk-test"

    def test_api_key_defaults_to_none(self):
        """API key defaults to None (falls back to env vars)."""
        model = Model("gpt-4o")
        assert model._api_key is None

    def test_kwargs_stored(self):
        """Extra kwargs are stored for passthrough to litellm."""
        model = Model("gpt-4o", api_base="http://localhost:8080")
        assert model._kwargs == {"api_base": "http://localhost:8080"}

    def test_history_initially_empty(self):
        """History is empty before any calls."""
        model = Model("gpt-4o")
        assert model.history == []

    def test_repr(self):
        """Repr shows model id."""
        model = Model("gpt-4o")
        assert repr(model) == "Model('gpt-4o')"


class TestToMessages:
    """_to_messages normalizes prompts to LiteLLM message format."""

    def test_string_prompt(self):
        """String prompt is wrapped as a user message."""
        result = _to_messages("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_chat_messages_passthrough(self):
        """Chat message list is returned as-is."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = _to_messages(messages)
        assert result == messages

    def test_single_user_message(self):
        """A single-element chat message list works."""
        messages = [{"role": "user", "content": "Hi"}]
        result = _to_messages(messages)
        assert result == messages


class TestGenerateBatch:
    """generate_batch sends one litellm.completion per prompt, concurrently."""

    @patch("pyforger.acquire.model.litellm")
    def test_single_prompt_n1(self, mock_litellm):
        """Single prompt with n=1 returns one response per prompt."""
        mock_litellm.completion.return_value = make_litellm_response(["hello"])
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate_batch(["Hi"], n=1)
        assert result == [["hello"]]

    @patch("pyforger.acquire.model.litellm")
    def test_single_prompt_n4(self, mock_litellm):
        """Single prompt with n=4 returns four responses."""
        mock_litellm.completion.return_value = make_litellm_response(
            ["a", "b", "c", "d"]
        )
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate_batch(["Hi"], n=4)
        assert result == [["a", "b", "c", "d"]]

    @patch("pyforger.acquire.model.litellm")
    def test_multiple_prompts(self, mock_litellm):
        """Multiple prompts each get their own response list."""
        mock_litellm.completion.side_effect = by_content(
            {"Q1": make_litellm_response(["r1"]), "Q2": make_litellm_response(["r2"])}
        )
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate_batch(["Q1", "Q2"], n=1)
        assert result == [["r1"], ["r2"]]

    @patch("pyforger.acquire.model.litellm")
    def test_chat_messages_input(self, mock_litellm):
        """Chat message lists are passed through to litellm."""
        mock_litellm.completion.return_value = make_litellm_response(["ahoy"])
        model = Model("gpt-4o", api_key="sk-test")
        messages = [{"role": "user", "content": "Hello"}]
        result = model.generate_batch([messages], n=1)
        assert result == [["ahoy"]]
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["messages"] == messages

    @patch("pyforger.acquire.model.litellm")
    def test_forwards_parameters(self, mock_litellm):
        """Model parameters are forwarded to litellm.completion."""
        mock_litellm.completion.return_value = make_litellm_response(["ok", "ok"])
        model = Model(
            "gpt-4o",
            api_key="sk-test",
            temperature=0.5,
            max_tokens=100,
            num_retries=3,
            api_base="http://localhost:8080",
        )
        model.generate_batch(["Hi"], n=2)
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["model"] == "gpt-4o"
        assert call_args.kwargs["api_key"] == "sk-test"
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["max_tokens"] == 100
        assert call_args.kwargs["num_retries"] == 3
        assert call_args.kwargs["n"] == 2
        assert call_args.kwargs["api_base"] == "http://localhost:8080"

    @patch("pyforger.acquire.model.litellm")
    def test_per_call_temperature_override(self, mock_litellm):
        """Per-call temperature overrides instance default."""
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", temperature=0.7)
        model.generate_batch(["Hi"], temperature=0.1)
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["temperature"] == 0.1

    @patch("pyforger.acquire.model.litellm")
    def test_per_call_max_tokens_override(self, mock_litellm):
        """Per-call max_tokens overrides instance default."""
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", max_tokens=200)
        model.generate_batch(["Hi"], max_tokens=50)
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["max_tokens"] == 50

    @patch("pyforger.acquire.model.litellm")
    def test_temperature_zero_is_not_replaced_by_default(self, mock_litellm):
        """Passing temperature=0.0 is a valid override, not replaced by default."""
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", temperature=0.7)
        model.generate_batch(["Hi"], temperature=0.0)
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["temperature"] == 0.0

    @patch("pyforger.acquire.model.litellm")
    def test_max_tokens_none_uses_instance_default(self, mock_litellm):
        """Omitting max_tokens falls back to instance default."""
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", max_tokens=200)
        model.generate_batch(["Hi"])
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["max_tokens"] == 200

    @patch("pyforger.acquire.model.litellm")
    def test_max_tokens_zero_is_not_replaced_by_default(self, mock_litellm):
        """Passing max_tokens=0 is a valid override, not replaced by default."""
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", max_tokens=200)
        model.generate_batch(["Hi"], max_tokens=0)
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["max_tokens"] == 0


class TestGenerate:
    """generate is a convenience wrapper for single prompts."""

    @patch("pyforger.acquire.model.litellm")
    def test_string_prompt_n1(self, mock_litellm):
        """String prompt with n=1 returns list with one response."""
        mock_litellm.completion.return_value = make_litellm_response(["hello"])
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate("Hi")
        assert result == ["hello"]

    @patch("pyforger.acquire.model.litellm")
    def test_string_prompt_n4(self, mock_litellm):
        """String prompt with n=4 returns list of four responses."""
        mock_litellm.completion.return_value = make_litellm_response(
            ["a", "b", "c", "d"]
        )
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate("Hi", n=4)
        assert result == ["a", "b", "c", "d"]

    @patch("pyforger.acquire.model.litellm")
    def test_chat_messages(self, mock_litellm):
        """Chat message list works as input."""
        mock_litellm.completion.return_value = make_litellm_response(["ahoy"])
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate(
            [
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "Hello"},
            ]
        )
        assert result == ["ahoy"]

    @patch("pyforger.acquire.model.litellm")
    def test_delegates_to_generate_batch(self, mock_litellm):
        """Generate forwards all parameters to litellm.completion."""
        mock_litellm.completion.return_value = make_litellm_response(["ok", "ok"])
        model = Model("gpt-4o", api_key="sk-test")
        model.generate("Hi", n=2, temperature=0.3, max_tokens=50)
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["n"] == 2
        assert call_args.kwargs["temperature"] == 0.3
        assert call_args.kwargs["max_tokens"] == 50

    @patch("pyforger.acquire.model.litellm")
    def test_default_n_is_one(self, mock_litellm):
        """Generate with no n argument returns a single-element list."""
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate("Hi")
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["n"] == 1
        assert len(result) == 1


class TestErrorHandling:
    """Model surfaces errors from litellm."""

    @patch("pyforger.acquire.model.litellm")
    def test_litellm_exception_propagates(self, mock_litellm):
        """Exceptions from litellm.completion propagate to the caller."""
        mock_litellm.completion.side_effect = RuntimeError("API rate limit")
        model = Model("gpt-4o", api_key="sk-test")
        with pytest.raises(RuntimeError, match="API rate limit"):
            model.generate("Hi")

    @patch("pyforger.acquire.model.litellm")
    def test_history_not_appended_on_first_call_error(self, mock_litellm):
        """A failing first call leaves history empty."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.side_effect = RuntimeError("API error")
        model = Model("gpt-4o", api_key="sk-test")
        with pytest.raises(RuntimeError):
            model.generate("Hi")
        assert model.history == []

    @patch("pyforger.acquire.model.litellm")
    def test_history_unchanged_on_subsequent_error(self, mock_litellm):
        """A failing call after a successful one leaves history at length 1."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", api_key="sk-test")
        model.generate("First call succeeds")
        assert len(model.history) == 1

        mock_litellm.completion.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError):
            model.generate("Second call fails")
        assert len(model.history) == 1

    @patch("pyforger.acquire.model.litellm")
    def test_batch_failure_leaves_history_empty(self, mock_litellm):
        """Batch failure leaves history empty."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.side_effect = RuntimeError("API error on batch")
        model = Model("gpt-4o", api_key="sk-test")
        with pytest.raises(RuntimeError, match="API error on batch"):
            model.generate_batch(["Q1", "Q2"])
        assert model.history == []

    @patch("pyforger.acquire.model.litellm")
    def test_batch_request_exception_is_raised(self, mock_litellm):
        """An exception returned for one batch request is raised as-is."""
        error = ValueError("context window exceeded: 200000 > 128000 tokens")
        mock_litellm.completion.side_effect = by_content(
            {"Q1": make_litellm_response(["r1"]), "Q2": error}
        )
        model = Model("gpt-4o", api_key="sk-test")
        with pytest.raises(ValueError) as excinfo:
            model.generate_batch(["Q1", "Q2"])
        assert excinfo.value is error
        # history gets only returned results, but Q1's spend still counts.
        assert model.history == []
        assert model.total_cost == pytest.approx(0.001)


class TestShowProgress:
    """Calls run in a thread pool; show_progress picks bar, log line, or off."""

    @patch("pyforger.acquire.model.litellm")
    def test_out_of_order_completions_keep_prompt_order(self, mock_litellm):
        """Q1 finishes after Q2, but results still follow prompt order."""
        q2_done = threading.Event()

        def fake_completion(messages, **kwargs):
            text = messages[0]["content"]
            if text == "Q1":
                assert q2_done.wait(timeout=5)
            else:
                q2_done.set()
            return make_litellm_response([f"r-{text}"])

        mock_litellm.completion.side_effect = fake_completion
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate_batch(["Q1", "Q2"], show_progress=True)
        assert result == [["r-Q1"], ["r-Q2"]]

    @patch("pyforger.acquire.model.litellm")
    def test_worker_exception_is_raised(self, mock_litellm):
        """An exception from one call is raised as-is."""
        error = ValueError("context window exceeded")

        def fake_completion(messages, **kwargs):
            if messages[0]["content"] == "Q2":
                raise error
            return make_litellm_response(["ok"])

        mock_litellm.completion.side_effect = fake_completion
        model = Model("gpt-4o", api_key="sk-test")
        with pytest.raises(ValueError) as excinfo:
            model.generate_batch(["Q1", "Q2"], show_progress=True)
        assert excinfo.value is error

    @patch("pyforger.acquire.model.litellm")
    def test_per_call_override_beats_instance_setting(self, mock_litellm):
        """show_progress=False on the call turns progress off."""
        mock_litellm.completion.side_effect = by_content(
            {"Q1": make_litellm_response(["r1"]), "Q2": make_litellm_response(["r2"])}
        )
        model = Model("gpt-4o", api_key="sk-test", show_progress=True)
        with patch(
            "pyforger.acquire.model._batch_completion_with_progress",
            wraps=_batch_completion_with_progress,
        ) as send:
            assert model.generate_batch(["Q1", "Q2"], show_progress=False) == [
                ["r1"],
                ["r2"],
            ]
        assert send.call_args.kwargs["progress"] == "off"

    @patch("pyforger.acquire.model.litellm")
    def test_instance_setting_used_when_call_omits_it(self, mock_litellm):
        """Model(show_progress=True) shows bar even if LLM call doesn't specify it."""
        mock_litellm.completion.side_effect = lambda messages, **kwargs: (
            make_litellm_response([f"r-{messages[0]['content']}"])
        )
        model = Model("gpt-4o", api_key="sk-test", show_progress=True)
        assert model.generate_batch(["Q1", "Q2"]) == [["r-Q1"], ["r-Q2"]]


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @patch("pyforger.acquire.model.litellm")
    def test_empty_prompt_list(self, mock_litellm):
        """Empty prompt list returns empty results and leaves history untouched."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate_batch([])
        assert result == []
        assert model.history == []
        mock_litellm.completion.assert_not_called()

    @patch("pyforger.acquire.model.litellm")
    def test_empty_prompt_list_skips_validation(self, mock_litellm):
        """Empty input is a pure no-op even when overrides would be invalid."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        model = Model("openai/o1", temperature=1.0, api_key="sk-test")
        # temperature=0.5 would normally raise on o1, but empty input is a no-op
        assert model.generate_batch([], temperature=0.5) == []

    @patch("pyforger.acquire.model.litellm")
    def test_empty_choices_from_api(self, mock_litellm):
        """API returning zero choices gives empty inner list."""
        mock_litellm.completion.return_value = make_litellm_response([])
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate_batch(["Hi"])
        assert result == [[]]

    @patch("pyforger.acquire.model.litellm")
    def test_empty_string_prompt(self, mock_litellm):
        """Empty string prompt is still sent to litellm."""
        mock_litellm.completion.return_value = make_litellm_response(["response"])
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate("")
        assert result == ["response"]
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["messages"] == [{"role": "user", "content": ""}]

    @patch("pyforger.acquire.model.litellm")
    def test_none_content_replaced_with_empty_string(self, mock_litellm):
        """None content from a choice is replaced with empty string."""
        mock_litellm.completion.return_value = make_litellm_response(
            [None, "valid", None]
        )
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate("Hi", n=3)
        assert result == ["", "valid", ""]


class TestGenerateResult:
    """GenerateResult subclasses list[str] and carries metadata."""

    def test_is_list_instance(self):
        """Passes isinstance(..., list) check for downstream callers."""
        result = GenerateResult(
            ["a", "b"], usage={}, reasoning=[None, None], model_id="gpt-4o", raw=None
        )
        assert isinstance(result, list)

    def test_iteration(self):
        """Iterates as a list of strings."""
        result = GenerateResult(
            ["a", "b"], usage={}, reasoning=[None, None], model_id="gpt-4o", raw=None
        )
        assert list(result) == ["a", "b"]

    def test_indexing(self):
        """Indexes like a list."""
        result = GenerateResult(
            ["a", "b"], usage={}, reasoning=[None, None], model_id="gpt-4o", raw=None
        )
        assert result[0] == "a"

    def test_equality_with_plain_list(self):
        """Compares equal to a plain list with same contents."""
        result = GenerateResult(
            ["a", "b"], usage={}, reasoning=[None, None], model_id="gpt-4o", raw=None
        )
        assert result == ["a", "b"]

    def test_attributes_accessible(self):
        """usage, reasoning, model_id, raw are accessible."""
        result = GenerateResult(
            ["a"],
            usage={"cost_usd": 0.01, "prompt_tokens": 5},
            reasoning=["thinking..."],
            model_id="gpt-5",
            raw={"id": "chatcmpl-123"},
        )
        assert result.usage["cost_usd"] == 0.01
        assert result.reasoning == ["thinking..."]
        assert result.model_id == "gpt-5"
        assert result.raw == {"id": "chatcmpl-123"}

    def test_pickle_roundtrip(self):
        """Pickles cleanly for @op caching."""
        result = GenerateResult(
            ["a"],
            usage={"cost_usd": 0.01},
            reasoning=[None],
            model_id="gpt-4o",
            raw=None,
        )
        restored = pickle.loads(pickle.dumps(result))
        assert restored == ["a"]
        assert restored.usage["cost_usd"] == 0.01
        assert restored.model_id == "gpt-4o"

    def test_deepcopy_preserves_attributes(self):
        """Deepcopy preserves both list contents and metadata."""
        result = GenerateResult(
            ["a"],
            usage={"cost_usd": 0.01},
            reasoning=[None],
            model_id="gpt-4o",
            raw=None,
        )
        copied = copy.deepcopy(result)
        assert copied == ["a"]
        assert copied.usage == {"cost_usd": 0.01}
        assert copied.model_id == "gpt-4o"


class TestFamily:
    """_family normalizes model_id to a constraint-table key prefix."""

    def test_strips_litellm_proxy_prefix(self):
        """Strips litellm proxy prefix."""
        assert _family("litellm_proxy/openai/o4-mini") == "o4-mini"

    def test_strips_provider_prefix_openai(self):
        """Strips provider prefix openai."""
        assert _family("openai/o1") == "o1"

    def test_strips_provider_prefix_gemini(self):
        """Strips provider prefix gemini."""
        assert _family("gemini/gemini-3.5-flash") == "gemini-3.5-flash"

    def test_strips_provider_prefix_xai(self):
        """Strips provider prefix xai."""
        assert _family("xai/grok-4") == "grok-4"

    def test_strips_provider_prefix_anthropic(self):
        """Strips provider prefix anthropic."""
        assert _family("anthropic/claude-sonnet-4-5") == "claude-sonnet-4-5"

    def test_no_prefix_passes_through(self):
        """No prefix passes through."""
        assert _family("gpt-4o") == "gpt-4o"

    def test_litellm_proxy_then_unknown_provider(self):
        """Litellm proxy then unknown provider."""
        assert (
            _family("litellm_proxy/gemini/gemini-3.1-flash-lite")
            == "gemini-3.1-flash-lite"
        )


class TestValueConstraints:
    """_VALUE_CONSTRAINTS table contents."""

    def test_o_series_require_temp_1(self):
        """O series require temp 1; n > 1 fans out instead of raising."""
        for key in ("o1", "o3", "o4", "gpt-5"):
            assert _VALUE_CONSTRAINTS[key] == {"temperature": 1.0}

    def test_gemini_3_requires_temp_1(self):
        """Gemini 3 requires temp 1; n > 1 fans out instead of raising."""
        assert _VALUE_CONSTRAINTS["gemini-3"] == {"temperature": 1.0}

    def test_no_constraint_for_gpt_4o(self):
        """No constraint for gpt 4o."""
        assert "gpt-4o" not in _VALUE_CONSTRAINTS

    def test_no_constraint_for_gemini_2(self):
        """No constraint for gemini 2."""
        assert not any(k.startswith("gemini-2") for k in _VALUE_CONSTRAINTS)


class TestValidateParams:
    """_validate_params raises ValueError on disallowed params/values."""

    @patch("pyforger.acquire.model.litellm")
    def test_value_constraint_violation_temperature(self, mock_litellm):
        """Value constraint violation temperature."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        with pytest.raises(ValueError, match="requires temperature=1.0"):
            _validate_params("openai/o1", temperature=0.5, n=1)

    @patch("pyforger.acquire.model.litellm")
    def test_n_is_not_constrained(self, mock_litellm):
        """No n constraint: generate_batch fans n > 1 out instead of raising."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        _validate_params("gemini/gemini-3.5-flash", temperature=1.0, n=2)

    @patch("pyforger.acquire.model.litellm")
    def test_value_constraint_satisfied_passes(self, mock_litellm):
        """Value constraint satisfied passes."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        _validate_params("openai/o1", temperature=1.0, n=1)  # no raise

    @patch("pyforger.acquire.model.litellm")
    def test_unsupported_param_name_raises(self, mock_litellm):
        """Unsupported param name raises."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
        ]
        with pytest.raises(
            ValueError, match="does not support parameter 'frequency_penalty'"
        ):
            _validate_params(
                "openai/gpt-5", temperature=1.0, n=1, frequency_penalty=0.5
            )

    @patch("pyforger.acquire.model.litellm")
    def test_unsupported_param_skips_none_values(self, mock_litellm):
        """Params with value None are not flagged as unsupported."""
        mock_litellm.get_supported_openai_params.return_value = ["temperature", "n"]
        _validate_params("openai/gpt-5", temperature=1.0, n=1, stop=None)  # no raise

    @patch("pyforger.acquire.model.litellm")
    def test_no_constraint_model_passes(self, mock_litellm):
        """No constraint model passes."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        _validate_params("gpt-4o", temperature=0.5, n=3)  # no raise

    @patch("pyforger.acquire.model.litellm")
    def test_thinking_enabled_requires_temperature_1(self, mock_litellm):
        """Claude with thinking enabled requires temperature=1.0."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "max_tokens",
            "stop",
            "thinking",
        ]
        with pytest.raises(ValueError, match="requires temperature=1.0"):
            _validate_params(
                "anthropic/claude-sonnet-4-5",
                temperature=0.5,
                thinking={"type": "enabled", "budget_tokens": 1024},
            )

    @patch("pyforger.acquire.model.litellm")
    def test_thinking_enabled_non_claude_no_temperature_constraint(self, mock_litellm):
        """Non-Claude model with thinking enabled accepts temperature != 1.0."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "max_tokens",
            "stop",
            "thinking",
        ]
        _validate_params(
            "deepseek/deepseek-reasoner",
            temperature=0.5,
            thinking={"type": "enabled", "budget_tokens": 1024},
        )  # no raise

    @patch("pyforger.acquire.model.litellm")
    def test_thinking_disabled_no_temperature_constraint(self, mock_litellm):
        """Thinking disabled no temperature constraint."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "max_tokens",
            "stop",
        ]
        _validate_params("anthropic/claude-sonnet-4-5", temperature=0.5)  # no raise

    @patch("pyforger.acquire.model.litellm")
    def test_registry_miss_does_not_block(self, mock_litellm):
        """If litellm returns None/empty for supported params, skip name check."""
        mock_litellm.get_supported_openai_params.return_value = None
        _validate_params("some/unreleased-model", temperature=0.5, n=2)  # no raise


def make_response_with_reasoning(
    contents: list[str],
    reasoning_contents: list[str | None] | None = None,
    openrouter_reasoning: list[str | None] | None = None,
):
    """Build a mock litellm response with optional reasoning data per choice."""
    response = MagicMock()
    choices = []
    for i, text in enumerate(contents):
        choice = MagicMock()
        choice.message.content = text
        if reasoning_contents is not None:
            choice.message.reasoning_content = reasoning_contents[i]
        else:
            choice.message.reasoning_content = None
        if openrouter_reasoning is not None:
            choice.message.reasoning = openrouter_reasoning[i]
        else:
            choice.message.reasoning = None
        choices.append(choice)
    response.choices = choices
    return response


class TestExtractReasoning:
    """_extract_reasoning pulls reasoning content per choice."""

    @patch("pyforger.acquire.model.litellm")
    def test_reasoning_content_populated(self, mock_litellm):
        """Reasoning content populated."""
        response = make_response_with_reasoning(
            ["answer"], reasoning_contents=["thought process"]
        )
        assert _extract_reasoning(response, "openai/o1") == ["thought process"]

    @patch("pyforger.acquire.model.litellm")
    def test_reasoning_absent_returns_none_per_choice(self, mock_litellm):
        """Reasoning absent returns none per choice."""
        response = make_response_with_reasoning(
            ["a", "b"], reasoning_contents=[None, None]
        )
        assert _extract_reasoning(response, "openai/o1") == [None, None]

    @patch("pyforger.acquire.model.litellm")
    def test_no_reasoning_field_returns_none(self, mock_litellm):
        """Choices without reasoning_content (or empty) yield None per choice.

        Does NOT rely on litellm.supports_reasoning — that registry has false
        negatives for unregistered gateway aliases. Always attempts extraction
        and falls back to None when the field is absent.
        """
        response = make_response_with_reasoning(["a", "b"])
        assert _extract_reasoning(response, "gpt-4o") == [None, None]

    def test_reasoning_extracted_even_when_supports_reasoning_false(self):
        """Do not gate on supports_reasoning (false negatives on proxy aliases)."""
        response = make_response_with_reasoning(
            ["answer"], reasoning_contents=["deep thought"]
        )
        assert _extract_reasoning(response, "openai/wine-deepseek-r1") == [
            "deep thought"
        ]

    @patch("pyforger.acquire.model.litellm")
    def test_openrouter_fallback(self, mock_litellm):
        """Fallback to message.reasoning for OpenRouter-routed responses."""
        response = make_response_with_reasoning(
            ["answer"],
            reasoning_contents=[None],
            openrouter_reasoning=["openrouter thought"],
        )
        assert _extract_reasoning(
            response, "openrouter/anthropic/claude-sonnet-4-5"
        ) == ["openrouter thought"]

    @patch("pyforger.acquire.model.litellm")
    def test_openrouter_fallback_via_litellm_proxy(self, mock_litellm):
        """Fallback works for ``litellm_proxy/openrouter/...`` paths too."""
        response = make_response_with_reasoning(
            ["answer"],
            reasoning_contents=[None],
            openrouter_reasoning=["openrouter thought"],
        )
        assert _extract_reasoning(
            response, "litellm_proxy/openrouter/anthropic/claude-sonnet-4-5"
        ) == ["openrouter thought"]


class TestExtractCost:
    """_extract_cost reads response cost; warns on registry miss."""

    def setup_method(self):
        """Clear the once-per-process warning set so each test starts fresh."""
        _COST_WARNING_SEEN.clear()

    @patch("litellm.completion_cost")
    def test_reads_hidden_params(self, mock_litellm):
        """Reads hidden params."""
        response = MagicMock()
        response._hidden_params = {"response_cost": 0.0042}
        assert _extract_cost(response, "gpt-4o") == 0.0042

    @patch("litellm.completion_cost")
    def test_falls_back_to_completion_cost(self, mock_litellm):
        """Falls back to completion cost."""
        response = MagicMock()
        response._hidden_params = {}
        mock_litellm.return_value = 0.005
        assert _extract_cost(response, "gpt-4o") == 0.005

    @patch("litellm.completion_cost")
    def test_completion_cost_raises_returns_zero(self, mock_litellm):
        """Completion cost raises returns zero."""
        response = MagicMock()
        response._hidden_params = {}
        mock_litellm.side_effect = RuntimeError("nope")
        with pytest.warns(UserWarning, match="pricing registry"):
            assert _extract_cost(response, "unknown/model") == 0.0

    @patch("litellm.completion_cost")
    def test_warning_emitted_once_per_model_id(self, mock_litellm):
        """Warning emitted once per model id."""
        response = MagicMock()
        response._hidden_params = {}
        mock_litellm.return_value = None
        with pytest.warns(UserWarning) as caught:
            _extract_cost(response, "litellm_proxy/x/y")
            _extract_cost(response, "litellm_proxy/x/y")
            _extract_cost(response, "litellm_proxy/x/y")
        assert len([w for w in caught if "pricing registry" in str(w.message)]) == 1

    @patch("litellm.completion_cost")
    def test_zero_cost_not_treated_as_missing(self, mock_litellm):
        """Free models (cost_usd == 0.0) must not trigger the registry-miss warning."""
        response = MagicMock()
        response._hidden_params = {"response_cost": 0.0}
        import warnings as warnings_mod

        with warnings_mod.catch_warnings():
            warnings_mod.simplefilter("error")
            assert _extract_cost(response, "free/model") == 0.0

    @patch("litellm.completion_cost")
    def test_no_warning_when_cost_present(self, mock_litellm):
        """No warning when cost present."""
        response = MagicMock()
        response._hidden_params = {"response_cost": 0.001}
        import warnings as warnings_mod

        with warnings_mod.catch_warnings():
            warnings_mod.simplefilter("error")
            assert _extract_cost(response, "gpt-4o") == 0.001


def make_response_with_usage(
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    reasoning_tokens: int | None = None,
    include_usage: bool = True,
):
    """Build a mock litellm response shaped for _extract_usage tests."""
    response = MagicMock()
    if include_usage:
        response.usage.prompt_tokens = prompt_tokens
        response.usage.completion_tokens = completion_tokens
        if reasoning_tokens is not None:
            response.usage.completion_tokens_details.reasoning_tokens = reasoning_tokens
        else:
            response.usage.completion_tokens_details = None
    else:
        response.usage = None
    return response


class TestExtractUsage:
    """_extract_usage returns a dict with token counts."""

    def test_basic_tokens(self):
        """Basic tokens."""
        response = make_response_with_usage(prompt_tokens=15, completion_tokens=25)
        usage = _extract_usage(response)
        assert usage["prompt_tokens"] == 15
        assert usage["completion_tokens"] == 25
        assert usage["reasoning_tokens"] == 0

    def test_reasoning_tokens_when_present(self):
        """Reasoning tokens when present."""
        response = make_response_with_usage(
            prompt_tokens=10, completion_tokens=50, reasoning_tokens=30
        )
        usage = _extract_usage(response)
        assert usage["reasoning_tokens"] == 30

    def test_no_usage_yields_zeros(self):
        """No usage yields zeros."""
        response = make_response_with_usage(include_usage=False)
        assert _extract_usage(response) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }


class TestConstructorValidation:
    """Model.__init__ rejects bad params at construction time."""

    @patch("pyforger.acquire.model.litellm")
    def test_o1_with_temperature_05_raises(self, mock_litellm):
        """O1 with temperature 05 raises."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        with pytest.raises(ValueError, match="requires temperature=1.0"):
            Model("openai/o1", temperature=0.5)

    @patch("pyforger.acquire.model.litellm")
    def test_o1_with_temperature_1_passes(self, mock_litellm):
        """O1 with temperature 1 passes."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        Model("openai/o1", temperature=1.0)  # no raise

    @patch("pyforger.acquire.model.litellm")
    def test_gpt_4o_unconstrained(self, mock_litellm):
        """Gpt 4o unconstrained."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        Model("gpt-4o", temperature=0.2)  # no raise

    @patch("pyforger.acquire.model.litellm")
    def test_kwargs_are_not_validated(self, mock_litellm):
        """Kwargs (api_base, timeout, etc.) are passthrough — not validated."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
        ]
        Model(
            "openai/gpt-5",
            temperature=1.0,
            api_base="http://proxy:8080",
            timeout=60,
        )  # no raise

    @patch("pyforger.acquire.model.litellm")
    def test_claude_thinking_kwarg_validated_at_construction(self, mock_litellm):
        """Claude with thinking enabled rejects bad temperature at construction."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "max_tokens",
            "stop",
            "thinking",
        ]
        with pytest.raises(ValueError, match="requires temperature=1.0"):
            Model(
                "anthropic/claude-sonnet-4-5",
                temperature=0.5,
                thinking={"type": "enabled", "budget_tokens": 1024},
            )

    @patch("pyforger.acquire.model.litellm")
    def test_o44_does_not_collide_with_o4(self, mock_litellm):
        """Future ``o44`` model must not inherit ``o4`` value constraints."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        Model("openai/o44", temperature=0.5)  # no raise


class TestGenerateBatchReturnType:
    """generate_batch returns list[GenerateResult] with metadata populated."""

    @patch("pyforger.acquire.model.litellm")
    def test_returns_generate_results(self, mock_litellm):
        """Returns generate results."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.side_effect = by_content(
            {"q1": make_litellm_response(["r1"]), "q2": make_litellm_response(["r2"])}
        )
        model = Model("gpt-4o", api_key="sk-test")
        results = model.generate_batch(["q1", "q2"])
        assert len(results) == 2
        assert all(isinstance(r, GenerateResult) for r in results)
        assert results[0] == ["r1"]
        assert results[1] == ["r2"]

    @patch("pyforger.acquire.model.litellm")
    def test_per_call_usage_attached(self, mock_litellm):
        """Per call usage attached."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.return_value = make_litellm_response(
            ["ok"], prompt_tokens=15, completion_tokens=25, response_cost=0.001
        )
        model = Model("gpt-4o", api_key="sk-test")
        results = model.generate_batch(["q"])
        assert results[0].usage["prompt_tokens"] == 15
        assert results[0].usage["completion_tokens"] == 25
        assert results[0].usage["cost_usd"] == 0.001
        assert results[0].model_id == "gpt-4o"

    @patch("pyforger.acquire.model.litellm")
    def test_reasoning_attached_when_supported(self, mock_litellm):
        """Reasoning attached when supported."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        response = make_response_with_reasoning(
            ["answer"], reasoning_contents=["thinking"]
        )
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 20
        response.usage.completion_tokens_details = None
        response._hidden_params = {"response_cost": 0.002}
        mock_litellm.completion.return_value = response
        model = Model("openai/o1", api_key="sk-test", temperature=1.0)
        results = model.generate_batch(["q"])
        assert results[0].reasoning == ["thinking"]

    @patch("pyforger.acquire.model.litellm")
    def test_history_appended(self, mock_litellm):
        """History appended."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", api_key="sk-test")
        assert model.history == []
        model.generate_batch(["q"])
        assert len(model.history) == 1
        assert model.history[0].model_id == "gpt-4o"

    @patch("pyforger.acquire.model.litellm")
    def test_history_grows_across_batches(self, mock_litellm):
        """History grows across batches."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.side_effect = by_content(
            {
                "q1": make_litellm_response(["r1"]),
                "q2": make_litellm_response(["r2"]),
                "q3": make_litellm_response(["r3"]),
            }
        )
        model = Model("gpt-4o", api_key="sk-test")
        model.generate_batch(["q1", "q2"])
        model.generate_batch(["q3"])
        assert len(model.history) == 3


class TestCumulativeTracking:
    """total_cost and total_usage accumulate across calls."""

    @patch("pyforger.acquire.model.litellm")
    def test_total_cost_sums_across_calls(self, mock_litellm):
        """Total cost sums across calls."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.side_effect = [
            make_litellm_response(["a"], response_cost=0.001),
            make_litellm_response(["b"], response_cost=0.002),
            make_litellm_response(["c"], response_cost=0.003),
        ]
        model = Model("gpt-4o", api_key="sk-test")
        model.generate("q1")
        model.generate("q2")
        model.generate("q3")
        assert model.total_cost == pytest.approx(0.006)

    @patch("pyforger.acquire.model.litellm")
    def test_total_usage_sums_tokens(self, mock_litellm):
        """Total usage sums tokens."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        r1 = make_litellm_response(["a"], prompt_tokens=10, completion_tokens=20)
        r1.usage.completion_tokens_details = None
        r2 = make_litellm_response(["b"], prompt_tokens=15, completion_tokens=25)
        r2.usage.completion_tokens_details = None
        mock_litellm.completion.side_effect = [r1, r2]
        model = Model("gpt-4o", api_key="sk-test")
        model.generate("q1")
        model.generate("q2")
        assert model.total_usage["prompt_tokens"] == 25
        assert model.total_usage["completion_tokens"] == 45
        assert model.total_usage["reasoning_tokens"] == 0

    @patch("pyforger.acquire.model.litellm")
    def test_reset_history_clears(self, mock_litellm):
        """Reset history clears."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.return_value = make_litellm_response(
            ["ok"], response_cost=0.005
        )
        model = Model("gpt-4o", api_key="sk-test")
        model.generate("q1")
        assert model.total_cost == pytest.approx(0.005)
        model.reset_history()
        assert model.history == []
        assert model.total_cost == 0.0
        assert model.total_usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }

    def test_total_cost_empty_history(self):
        """Total cost empty history."""
        model = Model("gpt-4o")
        assert model.total_cost == 0.0

    def test_total_usage_empty_history(self):
        """Total usage empty history."""
        model = Model("gpt-4o")
        assert model.total_usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }


class TestPerCallValidation:
    """generate/generate_batch validate per-call overrides."""

    @patch("pyforger.acquire.model.litellm")
    def test_per_call_temperature_violates_constraint(self, mock_litellm):
        """Per call temperature violates constraint."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        model = Model("openai/o1", api_key="sk-test", temperature=1.0)
        with pytest.raises(ValueError, match="requires temperature=1.0"):
            model.generate("Hi", temperature=0.5)

    @patch("pyforger.acquire.model.litellm")
    def test_per_call_n_fans_out(self, mock_litellm):
        """Per call n > 1 on gemini-3 fans out instead of raising."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.side_effect = [
            make_litellm_response([f"r{i}"]) for i in range(3)
        ]
        # One request in flight at a time, so they are answered in order.
        model = Model(
            "gemini/gemini-3.5-flash",
            api_key="sk-test",
            temperature=1.0,
            max_concurrency=1,
        )
        assert model.generate("Hi", n=3) == ["r0", "r1", "r2"]
        assert mock_litellm.completion.call_count == 3
        assert mock_litellm.completion.call_args.kwargs["n"] == 1

    @patch("pyforger.acquire.model.litellm")
    def test_per_call_unconstrained_passes(self, mock_litellm):
        """Per call unconstrained passes."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.return_value = make_litellm_response(["ok", "ok"])
        model = Model("gpt-4o", api_key="sk-test")
        model.generate("Hi", temperature=0.2, n=2)  # no raise


class TestUsageOnResult:
    """Usage data rides on each GenerateResult and on history aggregates."""

    @patch("pyforger.acquire.model.litellm")
    def test_per_call_usage_on_result(self, mock_litellm):
        """Individual call's usage rides on the returned GenerateResult."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.return_value = make_litellm_response(
            ["ok"], prompt_tokens=15, completion_tokens=25, response_cost=0.001
        )
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate("Hi")
        assert result.usage["prompt_tokens"] == 15
        assert result.usage["completion_tokens"] == 25
        assert result.usage["cost_usd"] == 0.001

    @patch("pyforger.acquire.model.litellm")
    def test_total_usage_sums_across_batch(self, mock_litellm):
        """Total usage sums across batch."""
        mock_litellm.get_supported_openai_params.return_value = [
            "temperature",
            "n",
            "max_tokens",
            "stop",
        ]
        mock_litellm.completion.side_effect = by_content(
            {
                "Q1": make_litellm_response(
                    ["r1"], prompt_tokens=10, completion_tokens=20
                ),
                "Q2": make_litellm_response(
                    ["r2"], prompt_tokens=30, completion_tokens=40
                ),
            }
        )
        model = Model("gpt-4o", api_key="sk-test")
        model.generate_batch(["Q1", "Q2"])
        assert model.total_usage["prompt_tokens"] == 40
        assert model.total_usage["completion_tokens"] == 60
        assert model.total_cost == pytest.approx(0.002)


class TestPublicExports:
    """GenerateResult is exported from pyforger.acquire."""

    def test_generate_result_importable_from_acquire(self):
        """Generate result importable from acquire."""
        from pyforger.acquire import GenerateResult as Exported

        assert Exported is GenerateResult


class TestReasoningEffort:
    """reasoning_effort is sent in the form each model family reads."""

    DEEPSEEK = "litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731"
    GLM = "litellm_proxy/azure_ai/FW-GLM-5.3"

    @pytest.mark.parametrize(
        "model_id,sent",
        [
            # The gateway's DeepSeek ignores reasoning_effort entirely.
            (DEEPSEEK, {"reasoning": {"effort": "high"}}),
            # GLM is served by Fireworks, which reads reasoning_effort.
            (GLM, {"reasoning_effort": "high"}),
            # Unmapped models get LiteLLM's standard parameter.
            ("gpt-4o", {"reasoning_effort": "high"}),
        ],
    )
    @patch("pyforger.acquire.model.litellm")
    def test_sent_in_family_form(self, mock_litellm, model_id, sent):
        """The effort reaches litellm under the name the provider honors."""
        mock_litellm.get_supported_openai_params.return_value = []
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        Model(model_id, reasoning_effort="high").generate("Hi")
        kwargs = mock_litellm.completion.call_args.kwargs
        for name in ("reasoning", "reasoning_effort"):
            assert kwargs.get(name) == sent.get(name)

    def test_overrides_kwargs_and_extra_body(self):
        """An explicit effort replaces reasoning settings passed any other way."""
        model = Model(
            self.DEEPSEEK,
            reasoning_effort="low",
            reasoning={"effort": "high"},
            extra_body={"reasoning_effort": "high", "top_k": 5},
        )
        assert model._kwargs == {
            "reasoning": {"effort": "low"},
            "extra_body": {"top_k": 5},
        }

    def test_without_effort_kwargs_pass_through(self):
        """With no explicit effort, caller kwargs are sent unchanged."""
        model = Model(self.DEEPSEEK, reasoning={"effort": "none"})
        assert model._kwargs == {"reasoning": {"effort": "none"}}


class TestReasoningEffortProperty:
    """Model.reasoning_effort reports the effort however it was passed."""

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({"reasoning_effort": "high"}, "high"),
            ({"reasoning": {"effort": "low"}}, "low"),
            ({}, None),
        ],
    )
    def test_reports_effort(self, kwargs, expected):
        """DeepSeek-V4 stores a dict, but the property reads it back."""
        model = Model("litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731", **kwargs)
        assert model.reasoning_effort == expected


class TestGatewayEnvironment:
    """litellm_proxy/ models read the LiteLLM gateway environment."""

    def test_reads_base_url_and_key(self, monkeypatch):
        """Without explicit values, the gateway URL and key come from env."""
        monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example")
        monkeypatch.setenv("LITELLM_API_KEY", "env-key")
        model = Model("litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731")
        assert model._kwargs["api_base"] == "https://gateway.example"
        assert model._api_key == "env-key"

    def test_explicit_values_win(self, monkeypatch):
        """Explicit api_base and api_key override the environment."""
        monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example")
        monkeypatch.setenv("LITELLM_API_KEY", "env-key")
        model = Model(
            "litellm_proxy/x", api_key="mine", api_base="https://other.example"
        )
        assert model._kwargs["api_base"] == "https://other.example"
        assert model._api_key == "mine"

    def test_other_providers_unchanged(self, monkeypatch):
        """Direct provider models keep litellm's own environment handling."""
        monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example")
        model = Model("gpt-4o")
        assert "api_base" not in model._kwargs
