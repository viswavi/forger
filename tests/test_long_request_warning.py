"""Model warns once when max_tokens allows a response a gateway may cut off."""
import logging

from pyforger.acquire import model as model_module


def test_warns_once_above_the_limit(caplog, monkeypatch):
    monkeypatch.setattr(model_module, "_long_request_warned", set())
    with caplog.at_level(logging.WARNING):
        model_module._warn_long_request("m", 26_000)
        model_module._warn_long_request("m", None)
        assert not caplog.records
        model_module._warn_long_request("m", 32_000)
        model_module._warn_long_request("m", 32_000)
    assert len(caplog.records) == 1
    assert "max_tokens=32000 is above 26000" in caplog.records[0].getMessage()
