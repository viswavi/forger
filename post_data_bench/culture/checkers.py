"""Shared NormAd parsing, scoring, and row I/O."""
from __future__ import annotations

import base64
import json
import re
import zlib
from pathlib import Path

import numpy as np

LABELS = ("yes", "no", "neutral")


def is_none_explanation(e) -> bool:
    """Detect empty explanations used to filter likely mislabeled neutral rows."""
    return e is None or (
        isinstance(e, str) and e.strip().lower() in ("", "none", "nan")
    )


def parse_prediction(text: str | None) -> str:
    """Parse a standalone yes/no/neutral label or the label after the final Answer:.

    Ignore thinking and incidental prose; return an empty string for invalid output."""
    if not text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    if "<think>" in text:
        return ""  # An unclosed reasoning trace is not an answer.

    matches = list(re.finditer(r"answer\s*:\s*", text, flags=re.IGNORECASE))
    tail = text[matches[-1].end() :] if matches else text
    tail = tail.strip().lower()

    m = re.fullmatch(r"[\s*]*(yes|no|neutral)[\s*.!]*", tail)
    if m and m.group(1) in LABELS:
        return m.group(1)
    return ""


def _f1_from_arrays(
    g: np.ndarray, p: np.ndarray, class_ids
) -> tuple[float, float, list[float]]:
    """Compute micro, macro, and per-class F1.

    Invalid predictions count only as false negatives for their gold classes."""
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
    """Score gold/pred rows with headline macro-F1 and secondary metrics/stderr."""
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
    """Obfuscate labels with base64(zlib(value)); this is not a security boundary."""
    return base64.b64encode(zlib.compress(value.encode("utf-8"))).decode("utf-8")


def decode_answer(blob: str) -> str:
    """Inverse of encode_answer."""
    return zlib.decompress(base64.b64decode(blob.encode("utf-8"))).decode("utf-8")


def read_jsonl(path) -> list[dict]:
    """Read a jsonl file into a list of dicts. Returns [] if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows: list[dict]) -> None:
    """Write `rows` to `path`, one JSON object per line."""
    import os
    import tempfile
    import shutil
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row) + "\n" for row in rows)
    if path.exists() and path.read_text() == content:
        return
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
