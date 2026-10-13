"""Public resource retrieval shared with dynamic-step inspection."""
from pyforger.alter._dynamic_resources import (
    _public_https_address,
    fetch_public,
    public_address,
)


def validate_url(url):
    """Validate and resolve a public HTTPS resource without downloading it."""
    return _public_https_address(url)


__all__ = ["fetch_public", "public_address", "validate_url"]
