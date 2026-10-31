"""Generic callable sampling with explicit identities and standard op semantics."""
from __future__ import annotations

from collections import UserDict

import pytest

from pyforger import alter
from pyforger.op import clear_trace, trace


def deterministic_generate(item, sample_id):
    """Generate a plain value without an agent, model, or directory."""
    return item["value"] + int(sample_id)


def test_plain_sampling_identity_cache_and_trace(tmp_path, monkeypatch):
    """Identical samples reuse cache; distinct IDs request distinct values."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "cache"))
    clear_trace()
    rows = [{"value": 5}]
    first = alter.sample(rows, deterministic_generate, sample_ids=["0", "1"])
    assert first[0]["samples"] == [
        {"sample_id": "0", "value": 5},
        {"sample_id": "1", "value": 6},
    ]
    assert alter.sample(rows, deterministic_generate, sample_ids=["0", "1"]) == first
    assert trace()[-1].cache_hit
    assert (
        alter.sample(rows, deterministic_generate, sample_ids=["2"])[0]["samples"][0][
            "value"
        ]
        == 7
    )
    assert "samples" not in rows[0]


@pytest.mark.parametrize("sample_ids", [[], [""], ["a", "a"]])
def test_invalid_sample_identity(sample_ids):
    """Reject ambiguous identities before calling any generator."""
    with pytest.raises(ValueError):
        alter.sample([], deterministic_generate, sample_ids=sample_ids)


def test_sampling_mapping_preserves_entries():
    """Mapping rows receive samples by key without mutating the source."""
    row = UserDict({"value": 5})
    result = alter.sample.no_cache([row], deterministic_generate, sample_ids=["0"])
    assert result == [{"value": 5, "samples": [{"sample_id": "0", "value": 5}]}]
    assert "samples" not in row
