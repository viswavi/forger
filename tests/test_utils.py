"""Environment-credential redaction."""

from pyforger.utils import redact


def test_redact_environment_credentials(monkeypatch):
    """Remove credential values but retain ordinary and short environment values."""
    monkeypatch.setenv("TEST_API_KEY", "example-api-credential")
    monkeypatch.setenv("TEST_token", "example-token-credential")
    monkeypatch.setenv("TEST_SECRET", "example-secret-credential")
    monkeypatch.setenv("TEST_PASSWORD", "example-password-credential")
    monkeypatch.setenv("TEST_NAME", "example-ordinary-value")
    monkeypatch.setenv("TEST_SHORT_KEY", "short")
    assert (
        redact(
            "example-api-credential example-token-credential example-secret-credential "
            "example-password-credential example-ordinary-value short"
        )
        == "[REDACTED] [REDACTED] [REDACTED] [REDACTED] example-ordinary-value short"
    )
