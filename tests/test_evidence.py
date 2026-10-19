"""Real filesystem evidence limits and cached public-resource receipts."""

import hashlib
import json

import pytest

from pyforger.evidence import PublicEvidence


def test_public_evidence_cached_receipt_and_pages(tmp_path):
    """Verify public evidence cached receipt and pages."""
    evidence = PublicEvidence(tmp_path, page_chars=3)
    url = "https://example.org/data"
    evidence.pages[url] = (url, b"abcdef", "text/plain")
    assert json.loads(evidence.fetch(url))["next_offset"] == 3
    assert json.loads(evidence.fetch(url, offset=3))["untrusted_web_content"] == "def"
    receipt = json.loads(evidence.fetch(url, filename="data.csv", license="CC0"))
    assert receipt["sha256"] == hashlib.sha256(b"abcdef").hexdigest()
    assert (tmp_path / "data.csv").read_bytes() == b"abcdef"
    assert (
        json.loads(evidence.fetch(url, filename="data.csv", license="CC0")) == receipt
    )
    with pytest.raises(ValueError):
        evidence.fetch(url, filename="../escape", license="CC0")
    with pytest.raises(ValueError):
        evidence.fetch("https://example.org/other", filename="data.csv", license="CC0")
