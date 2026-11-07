"""Cache keys stay what they were before steps recorded their key parts.

The keys below are the cache file names these calls wrote on 22d34cf (before
``StepRecord.key_parts``), so a change to how keys are built fails here
instead of silently invalidating every cache users already have. A key holds
the functions' bytecode, so each Python version has its own values.
"""

from __future__ import annotations

import sys

import pytest

from pyforger import _tracking
from pyforger.op import _cache_dir, op
from pyforger.steps import pipeline, step

ROWS = [{"id": "a", "text": "one two"}, {"id": "b", "text": "three"}]
PROMPT = "Shout: "

# Python version → (op key, step key, step key in a pipeline(limit=1) run).
KEYS: dict[tuple[int, int], tuple[str, str, str]] = {
    (3, 10): (
        "0d87696b3252ee49d3e8c571d7733bb48b7165525ad4f7614da23ce4903b0501",
        "c06f20d37b25c42c006c9164c0e6c112b0b2142d65372b72cc0a86684e49ff53",
        "79ee856c24fafa45ddf183e1dd46b597fad4466a5c55b78c9e689ce9a99c164c",
    ),
    (3, 12): (
        "4c23b81e76ddf1c4e826466b32aaa5b8dc07a9f8909055da54b61d1da42376ab",
        "3eef6a160c510e5022ccbebdbbdaa60a7e2bfc400c0744325f25b9096557d4a8",
        "fa48e56e32389b297ac0878d348afc2d4524ca49696502983def5e1e093ba73e",
    ),
    (3, 13): (
        "f461198b0f70cff16b12c628f3dcf1a3281626c0116123032064d288022cf0b9",
        "15de15b01bd42a17c65fc57ee4f0d8c15eb59c6b78e44b973f82a70dd0e2f760",
        "9aaa29a15fef7a0757936ff2c0b203843e5cfcedb1299f36ac3f110e02dd39f7",
    ),
}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FORGER_SPEND_LEDGER", "0")
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()


@op(name="test.keystable_upper")
def keystable_upper(rows: list[dict], suffix: str) -> list[dict]:
    """Upper-case each text and add suffix."""
    return [{**r, "text": PROMPT + r["text"].upper() + suffix} for r in rows]


@step
def keystable_shout(rows: list[dict], suffix: str = "!") -> list[dict]:
    """Shout every row."""
    return keystable_upper(rows, suffix)


@pipeline(limit=1)
def keystable_limited():
    """One row of keystable_shout."""
    return keystable_shout(ROWS, "?")


def _keys(folder: str) -> list[str]:
    return sorted(p.stem for p in (_cache_dir() / folder).glob("*.pkl"))


def test_op_and_step_keys_match_the_base_commit():
    """An op's and a step's cache file names, in a full and a limited run."""
    keystable_upper(ROWS, "!")
    [op_key] = _keys("test.keystable_upper")
    keystable_shout(ROWS)
    [step_key] = _keys("step.keystable_shout")
    keystable_limited()
    limited = [k for k in _keys("step.keystable_shout") if k != step_key]
    assert len(limited) == 1
    got = (op_key, step_key, limited[0])
    expected = KEYS.get(sys.version_info[:2])
    if expected is None:
        pytest.skip(f"no recorded keys for Python {sys.version_info[:2]}: {got}")
    assert got == expected


def test_steps_record_their_key():
    """The key a step records is the name of its cache file; parts name its pieces.

    PROMPT is read by keystable_upper, so it is inside that helper's part.
    """
    keystable_shout(ROWS)
    rec = _tracking.step_records()["keystable_shout"]
    assert rec.cache_key == _keys("step.keystable_shout")[0]
    assert set(rec.key_parts) == {
        "code",
        "global keystable_upper",
        "argument rows",
        "argument suffix",
    }
    assert all(len(digest) == 16 for digest in rec.key_parts.values())
