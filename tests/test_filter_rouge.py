"""ROUGE-L in Forger gives the same scores as rouge_score, without its logging."""
import logging
import os
import random
import subprocess
import sys

import pytest

import pyforger
from pyforger import _tracking
from pyforger.filter import by_rouge_l, by_rouge_l_agreement
from pyforger.filter._rouge import RougePool, RougeText, lcs_length, rouge_l


@pytest.fixture
def reference():
    """rouge_score's scorer, with the root logger handler it adds removed again."""
    rouge_scorer = pytest.importorskip("rouge_score.rouge_scorer")
    root = logging.getLogger()
    handlers = root.handlers[:]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    root.handlers[:] = handlers
    return lambda target, prediction: scorer.score(target, prediction)[
        "rougeL"
    ].fmeasure


def _lcs_dp(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def _pairs() -> list[tuple[str, str]]:
    rng = random.Random(0)
    small = ["a", "b", "c", "the", "cat", "A", "B!", "1", "x2"]
    vocab = (
        small + [f"w{i}" for i in range(50)] + ["é", "Ünïcode", "İstanbul", "K", "ß"]
    )
    seps = [" ", "  ", ", ", ".", "-", "\n", "\t", "'", "—", "…"]

    def text(words, n):
        return "".join(rng.choice(words) + rng.choice(seps) for _ in range(n))

    pairs = [
        ("", ""),
        ("", "hello world"),
        ("hello world", ""),
        ("!!! ... ???", "--- ,,,"),
        ("!!!", "hello"),
        ("a a a a a", "a"),
        ("a", "a a a a a"),
        ("a b a b a b", "b a b a b a"),
        ("Hello, World!", "hello world"),
        ("don't stop", "don t stop"),
        ("café naïve", "caf na ve"),
        ("日本語 テキスト", "中文 文本"),
        ("İstanbul", "i stanbul"),
        ("K", "k"),
        ("x" * 1000, "x" * 1000),
        ("the cat sat on the mat", "the cat sat on a mat"),
    ]
    long_a = text(vocab, 1200)
    pairs += [
        (long_a, long_a),
        (long_a, text(vocab, 1000)),
        (text(small, 800), text(small, 800)),
    ]
    for _ in range(400):
        words = small if rng.random() < 0.5 else vocab
        pairs.append((text(words, rng.randint(0, 80)), text(words, rng.randint(0, 80))))
    return pairs


def test_rouge_l_matches_rouge_score(reference):
    """Random and adversarial pairs score as rouge_score does, both ways round."""
    for target, prediction in _pairs():
        assert rouge_l(target, prediction) == pytest.approx(
            reference(target, prediction), abs=1e-9
        ), (target, prediction)
        assert rouge_l(prediction, target) == pytest.approx(
            reference(prediction, target), abs=1e-9
        )


def test_lcs_length_matches_dynamic_programming():
    """The bit-parallel LCS equals the table LCS, with few distinct tokens."""
    rng = random.Random(1)
    for _ in range(300):
        a = [rng.choice("abc") for _ in range(rng.randint(0, 70))]
        b = [rng.choice("abcd") for _ in range(rng.randint(0, 70))]
        lcs = lcs_length(RougeText(" ".join(a)), RougeText(" ".join(b)))
        assert lcs == _lcs_dp(a, b)


@pytest.mark.parametrize("block_bits", [1, 64, 8192])
def test_pool_max_equals_scoring_every_pair(reference, block_bits):
    """Packed texts (one per block, a few, or all in one) score as each pair does."""
    pairs = _pairs()
    rng = random.Random(2)
    for _ in range(40):
        others = [rng.choice(pairs)[0] for _ in range(rng.randint(0, 12))]
        text = rng.choice(pairs)[1]
        pool = RougePool(block_bits=block_bits)
        for other in others:
            pool.add(RougeText(other))
        expected = max((reference(other, text) for other in others), default=0.0)
        assert pool.max_f_measure(RougeText(text)) == pytest.approx(expected, abs=1e-9)


def test_by_rouge_l_matches_rouge_score_kept_rows_and_scores(reference, monkeypatch):
    """by_rouge_l keeps the rows, and records the scores, that rouge_score gives."""
    rng = random.Random(3)
    base = ["the cat sat on the mat", "a quick brown fox jumps", "hello there world"]
    texts = [
        " ".join(rng.choice(base).split()[: rng.randint(1, 6)] + rng.sample(base, 1))
        for _ in range(60)
    ]
    rows = [{"text": t} for t in texts]
    for threshold in (0.3, 0.5, 0.7):
        kept_texts, expected_scores = [], []
        for text in texts:
            score = max((reference(o, text) for o in kept_texts), default=0.0)
            expected_scores.append(score)
            if not kept_texts or score < threshold:
                kept_texts.append(text)
        recorded = []
        monkeypatch.setattr(
            _tracking,
            "record_scores",
            lambda label, values, **kw: recorded.append(values),
        )
        out = by_rouge_l(rows, field="text", threshold=threshold, _no_cache=True)
        assert [r["text"] for r in out] == kept_texts
        assert recorded[0] == pytest.approx(expected_scores, abs=1e-9)


def test_rouge_l_ops_leave_root_logger_handlers_alone(monkeypatch):
    """Neither op adds a handler to the root logger, so Forger logs print once."""
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    rows = [
        {"a": "the cat sat", "b": "the cat sat"},
        {"a": "a dog ran", "b": "the cat"},
    ]
    by_rouge_l(rows, field="a", _no_cache=True)
    by_rouge_l_agreement(rows, field_a="a", field_b="b", _no_cache=True)
    assert root.handlers == []


def test_pyforger_does_not_import_rouge_score():
    """rouge-score is a test-only dependency: no Forger module imports it."""
    code = (
        "import sys, pyforger, pyforger.filter.model_difficulty\n"
        "assert 'rouge_score' not in sys.modules, 'rouge_score imported'\n"
    )
    # This checkout's src, not whichever pyforger is installed.
    src = os.path.dirname(os.path.dirname(pyforger.__file__))
    env = {**os.environ, "PYTHONPATH": src}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
