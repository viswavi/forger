"""Tests for inspect.compare_to_reference."""
from __future__ import annotations

import pytest

from pyforger.inspect import compare_to_reference
from pyforger.inspect.reference import render_reference_report

LONG = "word " * 50  # longer than a label


def _text(n: int) -> str:
    return "word " * n


def test_labels_lengths_and_flags():
    """A label share gap and a length shift are both flagged."""
    rows = [{"text": _text(200), "kind": "a"} for _ in range(9)] + [
        {"text": _text(200), "kind": "b"}
    ]
    ref = [{"text": _text(50 + i), "kind": "a" if i % 2 else "b"} for i in range(10)]
    report = compare_to_reference(rows, ref, _no_cache=True)
    by = {f.name: f for f in report.features}
    assert by["kind"].kind == "label"
    assert by["kind"].shares == {"a": [0.9, 0.5], "b": [0.1, 0.5]}
    assert by["kind"].tvd == pytest.approx(0.4) and by["kind"].flagged
    text = by["text"]
    assert (text.kind, text.unit, text.rows.p50) == ("length", "words", 200)
    assert text.outside_reference == 1.0 and text.flagged


def test_feature_pairs_functions_and_missing():
    """(rows_spec, reference_spec) pairs and functions; None counts as missing."""
    rows = [{"q": LONG, "gt": "IMPOSSIBLE" if i < 2 else LONG} for i in range(10)]
    ref = [{"task_text": LONG, "impossible": i < 5} for i in range(10)]
    report = compare_to_reference(
        rows,
        ref,
        features={
            "prompt": ("q", "task_text"),
            "abstains": (lambda r: r["gt"] == "IMPOSSIBLE", "impossible"),
            "target": ("gt", lambda r: None),
        },
        max_length=40,
        _no_cache=True,
    )
    by = {f.name: f for f in report.features}
    assert by["abstains"].shares["True"] == [0.2, 0.5]
    assert by["prompt"].outside_reference == 0.0 and not by["prompt"].flagged
    target = by["target"]
    assert target.reference is None and target.reference_missing == 10
    assert target.rows.over_max == 0.8  # 8 long targets of 50 words, 2 short ones
    assert "over 40" in render_reference_report(report)


def test_chat_messages_and_tokenizer():
    """Messages are measured by their contents; a tokenizer counts tokens."""

    class Chars:
        def encode(self, text):
            return list(text)

    rows = [{"p": [{"role": "user", "content": LONG}]}] * 3
    report = compare_to_reference(
        rows, [{"p": LONG}] * 3, tokenizer=Chars(), _no_cache=True
    )
    feature = report.features[0]
    assert (feature.unit, feature.rows.p50) == ("tokens", len(LONG))


def test_hugging_face_tokenizer_is_called_once():
    """A callable tokenizer gets every distinct text in one batch."""

    class Batch:
        calls: list[int] = []

        def __call__(self, texts, add_special_tokens=True):
            self.calls.append(len(texts))
            return {"input_ids": [t.split() for t in texts]}

    tok = Batch()
    rows = [{"t": _text(60)}, {"t": _text(70)}, {"t": _text(60)}]
    report = compare_to_reference(rows, rows, tokenizer=tok, _no_cache=True)
    assert tok.calls == [2] and report.features[0].rows.p50 == 60


def test_numbers_and_no_shared_field():
    """Numbers stay numbers; sets with no common field need features=."""
    report = compare_to_reference(
        [{"n": i} for i in range(10)], [{"n": i} for i in range(10)], _no_cache=True
    )
    assert report.features[0].kind == "number" and not report.features[0].flagged
    with pytest.raises(ValueError, match="features="):
        compare_to_reference([{"a": 1}], [{"b": 1}])
