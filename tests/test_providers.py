"""Provider policy preserves caller configuration and endpoint compatibility."""

import pytest

from pyforger.acquire.providers import compatible_model


def test_compatibility_exact_endpoint_only():
    """Verify compatibility exact endpoint only."""
    for endpoint in [
        "https://example.org/v1",
        "https://llm-proxy.app.all-hands.dev.evil.org/v1",
    ]:
        model = compatible_model("gpt-6-luna", endpoint)
        assert model.config.api_mode == "auto"
        assert not model._kwargs.get("disable_prompt_cache_key")
    model = compatible_model("gpt-6-luna", "https://llm-proxy.app.all-hands.dev/v1")
    assert model.config.api_mode == "responses"
    assert model.config.capability_overrides == {}
    assert model._kwargs["disable_prompt_cache_key"] is True


@pytest.mark.parametrize(
    "endpoint",
    [
        "ftp://example.org",
        "https://user:secret@example.org",
        "https://example.org?key=secret",
    ],
)
def test_endpoint_rejects_embedded_credentials(endpoint):
    """Verify endpoint rejects embedded credentials."""
    with pytest.raises(ValueError):
        compatible_model("test", endpoint)


@pytest.mark.parametrize("endpoint", [None, "", "example.org/v1", "https:///v1"])
def test_endpoint_required(endpoint):
    """Missing or incomplete endpoints produce a clear validation error."""
    with pytest.raises(ValueError, match="endpoint"):
        compatible_model("test", endpoint)


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("test", "openai/test"),
        ("qwen/test", "openai/qwen/test"),
        ("meta/test", "openai/meta/test"),
        ("openai/test", "openai/test"),
        ("anthropic/test", "anthropic/test"),
    ],
)
def test_provider_preserves_settings_and_env_key(monkeypatch, identifier, expected):
    """Resolve the chosen key environment variable and preserve caller settings."""
    monkeypatch.setenv("TEST_PROVIDER_KEY", "example-provider-credential")
    model = compatible_model(
        identifier,
        "https://llm-proxy.app.all-hands.dev/v1",
        api_key_env="TEST_PROVIDER_KEY",
        extra_body={"custom": True},
        temperature=0.25,
        max_tokens=123,
        api_mode="chat",
        disable_prompt_cache_key=False,
    )
    assert model.config.model_id == expected
    assert model.config.api_key == "example-provider-credential"
    assert model.config.api_base == "https://llm-proxy.app.all-hands.dev/v1"
    assert model.config.extra_body == {"custom": True}
    assert model.config.temperature == 0.25
    assert model.config.max_tokens == 123
    assert model.config.api_mode == "chat"
    assert model._kwargs["disable_prompt_cache_key"] is False
