"""ROUGE-L F1, with the same scores as ``rouge_score``'s ``RougeScorer(["rougeL"])``.

``rouge_score`` fills an O(n*m) table in Python per pair, and building its
scorer adds an absl ``StreamHandler`` to the root logger (so every later log
line prints twice). This module tokenizes the same way (no stemmer) and
computes LCS lengths bit-parallel on Python ints: a few int ops per token of
one text, instead of one Python step per cell of the table.
"""
from __future__ import annotations

import re

# rouge_score's tokenizer: lowercase, then every run of characters outside
# [a-z0-9] is a separator, e.g. "Don't stop!" -> ["don", "t", "stop"].
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class RougeText:
    """One text's tokens, and its per-token position masks (built once)."""

    __slots__ = ("tokens", "n", "_masks")

    def __init__(self, text: str):
        """Tokenize ``text``."""
        self.tokens: list[str] = _TOKEN_RE.findall(text.lower())
        self.n = len(self.tokens)
        self._masks: dict[str, int] | None = None

    @property
    def masks(self) -> dict[str, int]:
        """Token -> int with bit i set where ``tokens[i]`` is that token."""
        if self._masks is None:
            masks: dict[str, int] = {}
            for i, token in enumerate(self.tokens):
                masks[token] = masks.get(token, 0) | (1 << i)
            self._masks = masks
        return self._masks


def lcs_length(a: RougeText, b: RougeText) -> int:
    """Length of the longest common token subsequence of ``a`` and ``b``.

    Bit-parallel LCS (Allison-Dix, Hyyro): after each token of ``b``, bit i of
    ``v`` is 0 where the LCS with ``a``'s first i+1 tokens steps up, so the LCS
    is the number of zero bits. ``m`` marks where that token occurs in ``a``.
    """
    if a.n < b.n:
        a, b = b, a
    # v = -1 is all ones, and its high bits stay ones, so no masking is needed.
    v = -1
    for m in filter(None, map(a.masks.get, b.tokens)):
        u = v & m
        v = (v + u) | (v - u)
    return (~v).bit_count()


def f_measure(lcs: int, n_target: int, n_prediction: int) -> float:
    """ROUGE-L F1 from the LCS length, computed as rouge_score does."""
    if not lcs:
        return 0.0
    precision = lcs / n_prediction
    recall = lcs / n_target
    return 2 * precision * recall / (precision + recall)


def rouge_l(target: str, prediction: str) -> float:
    """ROUGE-L F1 of ``prediction`` against ``target``."""
    t, p = RougeText(target), RougeText(prediction)
    return f_measure(lcs_length(t, p), t.n, p.n)


class RougePool:
    """Texts packed side by side into one int per token, to score a text against all.

    Each text takes ``n`` bits, then a zero guard bit that absorbs the carry
    out of the text below it, so one pass over the new text's tokens computes
    the LCS with every packed text. Blocks of about ``block_bits`` bits keep the
    ints (and the memory for rare tokens) small.
    """

    def __init__(self, block_bits: int = 8192):
        """Start an empty pool."""
        self.block_bits = block_bits
        # Per block: (token -> mask, ones at every text bit, [(offset, n)], bits used).
        self._blocks: list[tuple[dict[str, int], int, list[tuple[int, int]], int]] = []

    def add(self, text: RougeText) -> None:
        """Pack ``text``. Empty texts score 0.0 against everything: skip them."""
        if not text.n:
            return
        if not self._blocks or self._blocks[-1][3] + text.n > self.block_bits:
            self._blocks.append(({}, 0, [], 0))
        masks, ones, spans, offset = self._blocks[-1]
        for token, m in text.masks.items():
            masks[token] = masks.get(token, 0) | (m << offset)
        ones |= ((1 << text.n) - 1) << offset
        spans.append((offset, text.n))
        self._blocks[-1] = (masks, ones, spans, offset + text.n + 1)

    def max_f_measure(self, prediction: RougeText) -> float:
        """Max ROUGE-L F1 of ``prediction`` against each packed text; 0.0 if none."""
        n = prediction.n
        best = 0.0
        if not n:
            return best
        for masks, ones, spans, _ in self._blocks:
            v = ones
            for m in filter(None, map(masks.get, prediction.tokens)):
                u = v & m
                # & ones: clear the guard bits (and the carry out of the top).
                v = ((v + u) | (v - u)) & ones
            # "1" where v is 0 (a step up), lowest bit first, e.g. 0b110 -> "011".
            zeros = bin(ones ^ v)[:1:-1]
            for offset, n_target in spans:
                lcs = zeros.count("1", offset, offset + n_target)
                best = max(best, f_measure(lcs, n_target, n))
        return best
