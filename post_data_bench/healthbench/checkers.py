"""Final-answer extraction, scoring summaries and row I/O for clinician chats."""
from __future__ import annotations

import json
import math
from pathlib import Path

# Gemma 4 thinks in "<|channel>thought\n...<channel|>"; a server keeps these
# special tokens in the text only with skip_special_tokens off.
_REASONING_MARKERS = (("<|channel>thought", "<think>"), ("<channel|>", "</think>"))
_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<turn|>", "<end_of_turn>", "<eos>")


def normalize_reasoning(completion: str | None) -> str:
    """Write another model's reasoning delimiters as ``<think>...</think>``.

    E.g. Gemma 4's "<|channel>thought plan...<channel|>Answer" becomes
    "<think> plan...</think>Answer". Qwen3 text is unchanged.
    """
    text = completion or ""
    for marker, think in _REASONING_MARKERS:
        text = text.replace(marker, think)
    return text


def final_answer(completion: str | None) -> str:
    """The graded answer: the text after the last ``</think>``, stripped.

    A response whose reasoning never closes (cut off by the token limit) has no
    answer, so it is graded as an empty response. Trailing chat stop tokens, which
    a server keeps with special tokens on, are dropped, e.g. "Answer.<turn|>".
    """
    text = normalize_reasoning(completion)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return ""
    text = text.strip()
    while True:
        stop = next((s for s in _STOP_TOKENS if text.endswith(s)), None)
        if stop is None:
            return text
        text = text[: -len(stop)].rstrip()


def mean_and_stderr(values: list[float]) -> tuple[float, float | None]:
    """Mean and standard error of the mean (None for fewer than two values)."""
    n = len(values)
    if not n:
        return 0.0, None
    mean = sum(values) / n
    if n < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file; a missing file reads as no rows."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write rows as JSONL, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
