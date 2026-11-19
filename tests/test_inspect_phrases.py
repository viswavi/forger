"""Tests for inspect.phrase_frequency and the step report's repeated: line."""

from __future__ import annotations

import copy
import random

import pytest

from pyforger import _phrases, _tracking, op
from pyforger.inspect import PhraseReport, phrase_frequency
from pyforger.inspect.phrases import render_phrase_report
from pyforger.steps import render_step, step


@pytest.fixture(autouse=True)
def _quiet_and_fresh(monkeypatch):
    """Silence reports unless a test opts in, and forget earlier step records."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()
    yield
    _tracking.clear_step_records()


_WORDS = (
    "river lantern copper meadow violin harbor pepper glacier saddle orchard "
    "marble falcon thimble canyon lettuce compass ribbon walnut furnace tunnel "
    "pillow beacon cactus anchor biscuit tornado velvet garnet ladder puzzle "
    "kettle sparrow banjo quarry nectar oyster plaster rocket shovel tulip"
).split()
_VERBS = "found painted carried mended sold hid traded folded cleaned weighed".split()
_NAMES = "Ana Ben Chloe Dev Elif Femi Gus Hana Ivo Jun Kofi Lena Milo Nia Omar".split()
_PREPS = "near under beside behind past inside across beyond against around".split()


def _sentence(rng: random.Random) -> str:
    a, b = rng.sample(_WORDS, 2)
    verb, prep = rng.choice(_VERBS), rng.choice(_PREPS)
    return f"{rng.choice(_NAMES)} {verb} {rng.choice(_WORDS)} {a} {prep} {b}."


def _diverse(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(_sentence(rng) for _ in range(3)) for _ in range(n)]


def _collapsed(n: int) -> list[dict]:
    """Rewrites that all say "I need", contexts where the writer is mostly Maya."""
    texts = _diverse(n)
    return [
        {
            "rewrite": f"I need a break. {t}",
            "context": (f"Then {'Maya' if i % 3 else 'Jon'} wrote to a friend. " + t),
            "topic": "family" if i % 2 else "work",
        }
        for i, t in enumerate(texts)
    ]


# --------------------------------------------------------------------------- #
# inspect.phrase_frequency
# --------------------------------------------------------------------------- #


def test_phrase_frequency_reports_phrases_and_names(capsys):
    """Per field: the phrases and names most rows share, by share of rows."""
    rows = _collapsed(30)
    before = copy.deepcopy(rows)
    report = phrase_frequency(rows)
    assert isinstance(report, PhraseReport) and rows == before
    by_field = {f.field: f for f in report.fields}
    top = by_field["rewrite"].phrases[0]
    assert (top.text, top.rows, top.share) == ("i need a break", 30, 1.0)
    # "i need" is inside it in as many rows: not listed again.
    assert "i need" not in [p.text for p in by_field["rewrite"].phrases]
    assert by_field["context"].names[0].text == "Maya"
    assert by_field["context"].names[0].rows == 20
    assert by_field["topic"].skipped == "categorical (2 distinct of 30)"
    out = capsys.readouterr().out
    assert '"i need a break" 30/30 (100%)' in out
    assert '"Maya" 20/30 (67%)' in out
    assert render_phrase_report(report) in out


def test_phrase_frequency_options_and_step_report():
    """fields= picks fields, (1, 4) counts single words; the op shows in a step."""
    rows = _collapsed(30)

    @step
    def look(rows):
        phrase_frequency(rows, "rewrite", ngram_range=(1, 4), top_k=2)
        return rows

    look(rows)
    rec = _tracking.step_records()["look"]
    assert [c.op_name for c in rec.ops] == ["inspect.phrase_frequency"]
    report = phrase_frequency(rows, ["rewrite"], ngram_range=(1, 1), top_k=3)
    assert [f.field for f in report.fields] == ["rewrite"]
    assert [p.text for p in report.fields[0].phrases][:1] == ["break"]
    with pytest.raises(ValueError, match="ngram_range"):
        phrase_frequency(rows, ngram_range=(3, 2))


def test_phrase_frequency_reads_at_most_max_rows():
    """max_rows= reads evenly spaced rows; the report says how many."""
    report = phrase_frequency(_collapsed(40), max_rows=10)
    assert (report.n_rows, report.scanned) == (40, 10)
    assert report.fields[0].n_rows == 10


def test_diverse_text_has_no_repeated_phrases():
    """Varied sentences: no phrase or name reaches the step-report bar."""
    found = _phrases.field_phrases(_diverse(60), "text", min_share=0.3)
    assert found.phrases == [] and found.names == []


def test_template_markup_and_code_are_not_counted():
    """Headings, shared sections, boilerplate lines, and code are template text."""
    rng = random.Random(1)
    statements = [
        "Solve the problem below in Python 3 and print the answer.\n\n"
        f"# The {rng.choice(_WORDS)} {i}\n\n{_sentence(rng)} {_sentence(rng)}\n\n"
        "## Input\nThe first line contains an integer n.\n\n"
        "Output format:\nPrint one integer.\n\n"
        "**Example 1**\n```\n3\n1 2 3\n```\n"
        for i in range(30)
    ]
    found = _phrases.field_phrases(statements, "statement", min_share=0.3)
    assert found.phrases == [] and found.skipped is None
    code = [
        f"import sys\n\ndef solve():\n    n = int(sys.stdin.readline())\n"
        f"    print(n * {i})\n\nsolve()\n"
        for i in range(30)
    ]
    assert _phrases.field_phrases(code, "solution").skipped == "code"
    ids = [f"imp_s1_{i:03d}" for i in range(30)]
    assert _phrases.field_phrases(ids, "pid").skipped == "identifiers"


def test_names_are_mid_sentence_capitals_without_weekdays():
    """Maya's counts as Maya; Monday, at a sentence start, is no name."""
    texts = [
        f"On Monday we met Maya's sister. {t}" if i < 20 else t
        for i, t in enumerate(_diverse(30, seed=3))
    ]
    names = _phrases.field_phrases(texts, "t", min_share=0.3).names
    assert [(p.text, p.rows) for p in names] == [("Maya", 20)]


# --------------------------------------------------------------------------- #
# The step report's repeated: line
# --------------------------------------------------------------------------- #


@op(name="alter.test_rewrite")
def _rewrite(rows: list[dict]) -> list[dict]:
    return [{**r, **row} for r, row in zip(rows, _collapsed(len(rows)))]


@op(name="alter.test_varied")
def _varied(rows: list[dict]) -> list[dict]:
    return [{**r, "note": t} for r, t in zip(rows, _diverse(len(rows), seed=7))]


def _seeds(n: int) -> list[dict]:
    return [{"seed_id": i, "seed": t} for i, t in enumerate(_diverse(n, seed=5))]


def test_step_report_flags_a_phrase_and_a_name(capsys, monkeypatch):
    """At most 3 items, by share; the name gets a place."""
    monkeypatch.delenv("FORGER_STEP_REPORT")

    @step
    def rewrite(rows):
        return _rewrite(rows)

    rewrite(_seeds(30))
    rec = _tracking.step_records()["rewrite"]
    assert [(r.field, r.text, r.kind, r.rows, r.n_rows) for r in rec.repeated] == [
        ("rewrite", "i need a break", "phrase", 30, 30),
        ("context", "wrote to a friend", "phrase", 30, 30),
        ("context", "Maya", "name", 20, 30),
    ]
    out = " ".join(capsys.readouterr().out.split())  # the line wraps at 120
    assert (
        'repeated: rewrite "i need a break" 30/30 (100%) · context "wrote to a '
        'friend" 30/30 (100%) · context "Maya" 20/30 (67%)'
    ) in out


def test_step_report_is_quiet_when_nothing_repeats(capsys, monkeypatch):
    """Varied new text, too few rows, or unchanged fields: no repeated: line."""
    monkeypatch.delenv("FORGER_STEP_REPORT")

    @step
    def varied(rows):
        return _varied(rows)

    varied(_seeds(40))
    assert _tracking.step_records()["varied"].repeated == []
    assert "repeated:" not in render_step(_tracking.step_records()["varied"])

    @step
    def few(rows):
        return _rewrite(rows)

    few(_seeds(_phrases.MIN_STEP_ROWS - 1))
    assert _tracking.step_records()["few"].repeated == []

    # Rows already collapsed upstream, passed on unchanged: not this step's.
    upstream = _rewrite(_seeds(30))

    @step
    def keep(rows):
        return _varied(rows)

    keep(upstream)
    assert _tracking.step_records()["keep"].repeated == []


def test_phrases_the_input_already_had_are_not_flagged(capsys, monkeypatch):
    """A template the input rows carry (here in "seed") is not the step's doing."""
    monkeypatch.delenv("FORGER_STEP_REPORT")
    seeds = [{**r, "seed": "I need a break. " + r["seed"]} for r in _seeds(30)]

    @op(name="alter.test_copy")
    def copy_seed(rows):
        return [{**r, "draft": r["seed"] + " Thanks."} for r in rows]

    @step
    def draft(rows):
        return copy_seed(rows)

    draft(seeds)
    assert _tracking.step_records()["draft"].repeated == []


def test_report_line_is_skipped_when_reports_are_off(monkeypatch):
    """prod=True (or FORGER_STEP_REPORT=0) does not pay for the scan."""
    calls = []
    monkeypatch.setattr(_phrases, "step_repeats", lambda *a: calls.append(1) or [])

    @step
    def silent(rows):
        return _rewrite(rows)

    silent(_seeds(20))
    assert calls == []
    monkeypatch.delenv("FORGER_STEP_REPORT")
    silent(_seeds(21))
    assert calls == [1]


def test_repeat_bar_rises_with_length():
    """Long values share phrases by chance, so they need a larger share."""
    assert _phrases.repeat_bar(40) == 0.3
    assert _phrases.repeat_bar(200) == pytest.approx(0.6)
    assert _phrases.repeat_bar(5000) == 0.9
