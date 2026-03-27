"""Tests for the Model class (LiteLLM wrapper)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from pyforger.acquire.model import Model, _to_messages


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
        # Simulate providers that omit usage entirely
        response.usage = None
    if include_hidden_params:
        response._hidden_params = {"response_cost": response_cost}
    else:
        response._hidden_params = {}
    return response


class TestConstructor:
    """Model constructor stores configuration."""

    def test_stores_model_id(self):
        """model_id is accessible after construction."""
        model = Model("gpt-4o")
        assert model._model_id == "gpt-4o"

    def test_default_temperature(self):
        """Default temperature is 0.7."""
        model = Model("gpt-4o")
        assert model._temperature == 0.7

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

    def test_last_usage_initially_none(self):
        """last_usage is None before any calls."""
        model = Model("gpt-4o")
        assert model.last_usage is None

    def test_repr(self):
        """repr shows model id."""
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
    """generate_batch calls litellm.completion for each prompt."""

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
        mock_litellm.completion.side_effect = [
            make_litellm_response(["r1"]),
            make_litellm_response(["r2"]),
        ]
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
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
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
        """temperature=0.0 is a valid override and must not fall back to default."""
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", temperature=0.7)
        model.generate_batch(["Hi"], temperature=0.0)
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["temperature"] == 0.0

    @patch("pyforger.acquire.model.litellm")
    def test_max_tokens_none_uses_instance_default(self, mock_litellm):
        """max_tokens=None (the default) falls back to instance value."""
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", max_tokens=200)
        model.generate_batch(["Hi"])
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["max_tokens"] == 200


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
        result = model.generate([
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Hello"},
        ])
        assert result == ["ahoy"]

    @patch("pyforger.acquire.model.litellm")
    def test_delegates_to_generate_batch(self, mock_litellm):
        """generate forwards all parameters to litellm.completion."""
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", api_key="sk-test")
        model.generate("Hi", n=2, temperature=0.3, max_tokens=50)
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["n"] == 2
        assert call_args.kwargs["temperature"] == 0.3
        assert call_args.kwargs["max_tokens"] == 50

    @patch("pyforger.acquire.model.litellm")
    def test_default_n_is_one(self, mock_litellm):
        """generate with no n argument returns a single-element list."""
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate("Hi")
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["n"] == 1
        assert len(result) == 1


class TestLastUsage:
    """last_usage tracks token counts and cost."""

    @patch("pyforger.acquire.model.litellm")
    def test_tracks_single_call(self, mock_litellm):
        """last_usage reflects a single completion call."""
        mock_litellm.completion.return_value = make_litellm_response(
            ["ok"], prompt_tokens=15, completion_tokens=25,
        )
        model = Model("gpt-4o", api_key="sk-test")
        model.generate("Hi")
        assert model.last_usage["prompt_tokens"] == 15
        assert model.last_usage["completion_tokens"] == 25
        assert model.last_usage["total_cost"] == 0.001

    @patch("pyforger.acquire.model.litellm")
    def test_sums_across_batch(self, mock_litellm):
        """last_usage sums tokens across all prompts in a batch."""
        mock_litellm.completion.side_effect = [
            make_litellm_response(["r1"], prompt_tokens=10, completion_tokens=20),
            make_litellm_response(["r2"], prompt_tokens=30, completion_tokens=40),
        ]
        model = Model("gpt-4o", api_key="sk-test")
        model.generate_batch(["Q1", "Q2"])
        assert model.last_usage["prompt_tokens"] == 40
        assert model.last_usage["completion_tokens"] == 60
        assert model.last_usage["total_cost"] == 0.002

    @patch("pyforger.acquire.model.litellm")
    def test_resets_on_new_call(self, mock_litellm):
        """last_usage resets on each new generate call."""
        mock_litellm.completion.return_value = make_litellm_response(
            ["ok"], prompt_tokens=100, completion_tokens=200,
        )
        model = Model("gpt-4o", api_key="sk-test")
        model.generate("First call")
        mock_litellm.completion.return_value = make_litellm_response(
            ["ok"], prompt_tokens=5, completion_tokens=10,
        )
        model.generate("Second call")
        assert model.last_usage["prompt_tokens"] == 5
        assert model.last_usage["completion_tokens"] == 10

    @patch("pyforger.acquire.model.litellm")
    def test_no_usage_from_provider(self, mock_litellm):
        """Providers that omit usage data get zero tokens."""
        mock_litellm.completion.return_value = make_litellm_response(
            ["ok"], include_usage=False,
        )
        model = Model("gpt-4o", api_key="sk-test")
        model.generate("Hi")
        assert model.last_usage["prompt_tokens"] == 0
        assert model.last_usage["completion_tokens"] == 0

    @patch("pyforger.acquire.model.litellm")
    def test_no_response_cost(self, mock_litellm):
        """Missing response_cost in _hidden_params defaults to 0.0."""
        mock_litellm.completion.return_value = make_litellm_response(
            ["ok"], include_hidden_params=False,
        )
        model = Model("gpt-4o", api_key="sk-test")
        model.generate("Hi")
        assert model.last_usage["total_cost"] == 0.0


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
    def test_last_usage_none_on_first_call_error(self, mock_litellm):
        """last_usage is None if the very first call fails."""
        mock_litellm.completion.side_effect = RuntimeError("API error")
        model = Model("gpt-4o", api_key="sk-test")
        with pytest.raises(RuntimeError):
            model.generate("Hi")
        assert model.last_usage is None

    @patch("pyforger.acquire.model.litellm")
    def test_last_usage_cleared_on_batch_error(self, mock_litellm):
        """last_usage from a prior call is cleared when a new call fails."""
        mock_litellm.completion.return_value = make_litellm_response(["ok"])
        model = Model("gpt-4o", api_key="sk-test")
        model.generate("First call succeeds")
        assert model.last_usage is not None

        mock_litellm.completion.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError):
            model.generate("Second call fails")
        assert model.last_usage is None

    @patch("pyforger.acquire.model.litellm")
    def test_partial_batch_failure(self, mock_litellm):
        """If second prompt fails, exception propagates and usage is cleared."""
        mock_litellm.completion.side_effect = [
            make_litellm_response(["ok"]),
            RuntimeError("API error on second prompt"),
        ]
        model = Model("gpt-4o", api_key="sk-test")
        with pytest.raises(RuntimeError, match="API error on second prompt"):
            model.generate_batch(["Q1", "Q2"])
        assert model.last_usage is None


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @patch("pyforger.acquire.model.litellm")
    def test_empty_prompt_list(self, mock_litellm):
        """Empty prompt list returns empty results with zeroed usage."""
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate_batch([])
        assert result == []
        assert model.last_usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
        }
        mock_litellm.completion.assert_not_called()

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
            [None, "valid", None],
        )
        model = Model("gpt-4o", api_key="sk-test")
        result = model.generate("Hi", n=3)
        assert result == ["", "valid", ""]
