"""Chat-message fields are read as their contents, not as Python reprs."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from pyforger import filter as ff
from pyforger import inspect as fi
from pyforger.inspect import cluster as cluster_mod
from pyforger.utils import as_text, field_text, is_chat, read_jsonl


def _chat(text: str, system: str | None = None) -> list[dict]:
    head = [{"role": "system", "content": system}] if system else []
    return head + [{"role": "user", "content": text}]


def _embedder(seen: list[str]):
    """An embed_fn that records its texts and embeds each by its md5."""

    def embed(texts):
        seen.extend(texts)
        return np.array(
            [[b / 255 for b in hashlib.md5(t.encode()).digest()[:8]] for t in texts]
        )

    return embed


# --------------------------------------------------------------------------- #
# as_text / is_chat / field_text
# --------------------------------------------------------------------------- #


def test_as_text_joins_message_contents_without_roles():
    """Messages give their contents one per line; strings and others as before."""
    assert as_text(_chat("Hi", system="Be brief.")) == "Be brief.\nHi"
    assert as_text("plain") == "plain"
    assert as_text(3) == "3"
    assert as_text({"a": 1}) == "{'a': 1}"
    assert as_text([]) == "[]"  # an empty list is not a chat
    assert as_text(["a", "b"]) == "['a', 'b']"


def test_as_text_reads_text_parts_and_empty_contents():
    """A list of content parts keeps its text; a tool-call message adds a blank."""
    parts = [
        {"type": "text", "text": "Look:"},
        {"type": "image_url", "image_url": {"url": "x"}},
    ]
    messages = [
        {"role": "user", "content": parts},
        {"role": "assistant", "content": None, "tool_calls": []},
    ]
    assert is_chat(messages)
    assert as_text(messages) == "Look:\n"


def test_is_chat_needs_message_dicts():
    """Rubric-like dicts without role or content are not messages."""
    assert not is_chat([{"criterion": "x", "weight": 1}])
    assert not is_chat("hi")
    assert is_chat(_chat("hi"))


def test_field_text_raises_get_fields_error():
    """A missing field names the caller and the fields there are."""
    assert field_text({"p": _chat("q")}, "p", "op") == "q"
    with pytest.raises(KeyError, match="op: item is missing required field 'x'"):
        field_text({"p": "q"}, "x", "op")


# --------------------------------------------------------------------------- #
# Ops that embed, compare, or count a field
# --------------------------------------------------------------------------- #


def test_embedding_dispersion_embeds_message_contents():
    """Rows and reference rows are embedded by their contents."""
    seen: list[str] = []
    rows = [{"prompt": _chat(f"question {i}")} for i in range(4)]
    ref = [{"prompt": _chat(f"reference {i}")} for i in range(4)]
    fi.embedding_dispersion(
        rows, "prompt", reference=ref, embed_fn=_embedder(seen), _no_cache=True
    )
    assert set(seen) == {f"question {i}" for i in range(4)} | {
        f"reference {i}" for i in range(4)
    }


def test_by_embedding_similarity_embeds_message_contents():
    """Generated rows and reference rows are compared by their contents."""
    seen: list[str] = []
    rows = [{"prompt": _chat(f"q{i}", system="sys")} for i in range(3)]
    ref = [{"prompt": _chat("other")}]
    kept = ff.by_embedding_similarity(
        rows, field="prompt", reference=ref, embed_fn=_embedder(seen)
    )
    assert kept == rows
    assert set(seen) == {"sys\nq0", "sys\nq1", "sys\nq2", "other"}


def test_rouge_and_n_gram_filters_read_messages():
    """These filters used to fail on lists; equal contents are duplicates now."""
    rows = [
        {"prompt": _chat("the cat sat on the mat")},
        {"prompt": _chat("the cat sat on the mat")},
        {"prompt": _chat("quantum fields obey gauge symmetry")},
    ]
    assert ff.by_rouge_l(rows, field="prompt") == [rows[0], rows[2]]
    assert ff.by_n_gram_overlap(rows, field="prompt") == [rows[0], rows[2]]
    assert ff.by_exact_match(rows, field="prompt") == [rows[0], rows[2]]
    pairs = [{"a": _chat("same words here"), "b": _chat("same words here")}]
    assert ff.by_rouge_l_agreement(pairs, field_a="a", field_b="b") == pairs


def test_phrase_frequency_counts_message_contents():
    """A message field is counted by its contents, and found by default."""
    rows = [
        {"prompt": _chat(f"Please help me plan item {w} for tomorrow.")}
        for w in "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    ]
    report = fi.phrase_frequency(rows)
    by_field = {f.field: f for f in report.fields}
    assert "prompt" in by_field
    texts = [p.text for p in by_field["prompt"].phrases]
    assert any("please help me" in t for t in texts)
    assert not any("role" in t or "content" in t for t in texts)


def test_topic_based_clustering_sends_contents(monkeypatch):
    """The cached op gets message contents; string rows pass through as given."""
    calls: list[list] = []

    def fake(items, **kwargs):
        calls.append(items)
        return ["t"] * len(items)

    monkeypatch.setattr(cluster_mod, "_topic_based_clustering", fake)
    chats = [{"prompt": _chat("a b", system="s"), "id": 1}]
    assert fi.topic_based_clustering(chats, field="prompt", model=None) == ["t"]
    assert calls[-1] == [{"prompt": "s\na b", "id": 1}]
    assert chats[0]["prompt"] == _chat("a b", system="s")  # rows untouched
    strings = [{"prompt": "a b"}]
    fi.topic_based_clustering(strings, field="prompt", model=None)
    assert calls[-1] is strings


def test_compare_to_reference_measures_messages_and_rubric_lists():
    """Messages are measured by their contents; a long list by its text."""
    long = " ".join(["word"] * 30)
    rows = [{"prompt": _chat(long), "rubric": [f"c{i} " * 20]} for i in range(25)]
    ref = [
        {"prompt": _chat(long + " more"), "rubric": [f"d{i} " * 20]} for i in range(25)
    ]
    report = fi.compare_to_reference(rows, ref, _no_cache=True)
    by_name = {f.name: f for f in report.features}
    assert by_name["prompt"].kind == "length"
    assert by_name["prompt"].rows.p50 == 30
    assert by_name["prompt"].reference.p50 == 31
    assert by_name["rubric"].kind == "length"


# --------------------------------------------------------------------------- #
# read_jsonl
# --------------------------------------------------------------------------- #


def test_read_jsonl_reads_rows_and_skips_blank_lines(tmp_path):
    """One dict per non-blank line."""
    path = tmp_path / "rows.jsonl"
    rows = [{"prompt": _chat("q"), "rubric": []}, {"prompt": _chat("r")}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    assert read_jsonl(path) == rows
    assert read_jsonl(str(path)) == rows


@pytest.mark.parametrize(
    "line, message", [("{bad", "not valid JSON"), ("[1, 2]", "got list")]
)
def test_read_jsonl_names_the_bad_line(tmp_path, line, message):
    """A bad line's error names the file and line number."""
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n' + line + "\n")
    with pytest.raises(ValueError, match=f"rows.jsonl:2: .*{message}"):
        read_jsonl(path)
