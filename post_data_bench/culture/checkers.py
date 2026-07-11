"""Shared parsing, scoring, and row IO helpers for the NormAd evaluation pipeline.

Used by both make_splits.py and evaluate_normad.py.
"""
from __future__ import annotations

import base64
import json
import pickle
import re
import zlib
from pathlib import Path

import numpy as np

LABELS = ("yes", "no", "neutral")


def is_none_explanation(e) -> bool:
    """Return whether an Explanation value counts as "empty".

    Preserved verbatim from post_train_bench_evaluate_normad.py: an empty Explanation
    marks a Gold Label == "neutral" row as likely mislabeled (see make_splits.py's
    filter).
    """
    return e is None or (
        isinstance(e, str) and e.strip().lower() in ("", "none", "nan")
    )


def parse_prediction(text: str | None) -> str:
    """Extract a normalized {yes,no,neutral} label from a completion.

    Ignores any <think>...</think> reasoning, prefers the text after the final
    "Answer:", and falls back to the last label keyword anywhere. Returns "" when
    nothing parseable is found (counts as an incorrect prediction).
    """
    if not text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    matches = list(re.finditer(r"answer\s*:\s*", text, flags=re.IGNORECASE))
    tail = text[matches[-1].end() :] if matches else text
    tail = tail.strip().lower()

    m = re.match(r"[^a-z]*([a-z]+)", tail)
    if m and m.group(1) in LABELS:
        return m.group(1)
    # Fallback for outputs that don't follow the "Answer:" format: take the LAST label
    # keyword (the conclusion usually comes at the end).
    kws = re.findall(r"\b(yes|no|neutral)\b", tail)
    return kws[-1] if kws else ""


def _f1_from_arrays(
    g: np.ndarray, p: np.ndarray, class_ids
) -> tuple[float, float, list[float]]:
    """Compute vectorized micro-F1, macro-F1, and per-class F1 from label id arrays.

    Predictions outside the class set (encoded distinctly from any gold id) are never
    true/false positives for any class -- they only become false negatives for the gold
    class.
    """
    tp = fp = fn = 0
    per = []
    for c in class_ids:
        c_tp = int(np.sum((g == c) & (p == c)))
        c_fp = int(np.sum((p == c) & (g != c)))
        c_fn = int(np.sum((g == c) & (p != c)))
        tp += c_tp
        fp += c_fp
        fn += c_fn
        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0.0
        rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0.0
        per.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    macro = sum(per) / len(per) if per else 0.0
    return micro, macro, per


def score(rows: list[dict], classes=LABELS, n_boot: int = 1000, seed: int = 0) -> dict:
    """Score predictions: accuracy, micro-F1, macro-F1, and bootstrap stderr.

    Rows must have "gold"/"pred" keys. Macro-F1 is this benchmark's headline metric --
    the only number used to compare models; the rest are secondary, developer-facing
    context only.
    """
    n = len(rows)
    label_to_id = {c: i for i, c in enumerate(classes)}
    g = np.array([label_to_id.get(r["gold"], -1) for r in rows], dtype=int)
    p = np.array([label_to_id.get(r["pred"], -2) for r in rows], dtype=int)
    class_ids = list(range(len(classes)))

    correct = int(np.sum(g == p))
    acc = correct / n if n else 0.0
    stderr = (acc * (1 - acc) / n) ** 0.5 if n else 0.0
    micro, macro, per = _f1_from_arrays(g, p, class_ids)

    out = {
        "n": n,
        "accuracy": acc,
        "accuracy_stderr": stderr,
        "correct": correct,
        "micro_f1": micro,
        "macro_f1": macro,
        "per_class_f1": {c: per[i] for i, c in enumerate(classes)},
    }
    if n_boot and n > 0:
        rng = np.random.default_rng(seed)
        micros = np.empty(n_boot)
        macros = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            mi, ma, _ = _f1_from_arrays(g[idx], p[idx], class_ids)
            micros[b] = mi
            macros[b] = ma
        out["micro_f1_stderr"] = float(micros.std(ddof=1))
        out["macro_f1_stderr"] = float(macros.std(ddof=1))
    return out


def encode_answer(value: str) -> str:
    """Obfuscate a gold label at rest.

    Uses base64(zlib(pickle(value))) -- the same recipe
    post_data_bench/coding/checkers.py uses for gold_response, applied uniformly to
    every row.
    """
    return base64.b64encode(zlib.compress(pickle.dumps(value))).decode("utf-8")


def decode_answer(blob: str):
    """Inverse of encode_answer."""
    return pickle.loads(zlib.decompress(base64.b64decode(blob.encode("utf-8"))))


def read_jsonl(path) -> list[dict]:
    """Read a jsonl file into a list of dicts. Returns [] if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows: list[dict]) -> None:
    """Write rows to path as newline-delimited JSON."""
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
