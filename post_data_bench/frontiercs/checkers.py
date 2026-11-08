"""Answer extraction, scoring summaries, and row I/O for C++ problem evaluation."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

_CPP_BLOCK = re.compile(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", re.DOTALL)
# Gemma 4 thinks in "<|channel>thought\n...<channel|>"; a server keeps these
# special tokens in the text only with skip_special_tokens off.
_REASONING_MARKERS = (("<|channel>thought", "<think>"), ("<channel|>", "</think>"))


def normalize_reasoning(completion: str | None) -> str:
    """Write another model's reasoning delimiters as ``<think>...</think>``.

    E.g. Gemma 4's "<|channel>thought plan...<channel|>```cpp ...```" becomes
    "<think> plan...</think>```cpp ...```". Qwen3 text is unchanged.
    """
    text = completion or ""
    for marker, think in _REASONING_MARKERS:
        text = text.replace(marker, think)
    return text


def extract_code(completion: str | None, lenient: bool = False) -> str:
    """The C++ program in a completion's answer, or "" when there is none.

    The answer is the text after the last ``</think>``; an unclosed reasoning trace
    is not an answer. Within it, the benchmark's own extraction rule: the longest
    ```cpp (or ```c++ / bare ```) block, else the text with fences stripped.
    With ``lenient``, an unclosed trace is searched as the answer: a response cut
    off by the token limit is judged on the longest draft block in its reasoning.
    """
    completion = normalize_reasoning(completion)
    if not completion:
        return ""
    if "</think>" in completion:
        completion = completion.rsplit("</think>", 1)[-1]
    unclosed = "<think>" in completion
    if unclosed and not lenient:
        return ""
    text = completion.strip()
    blocks = _CPP_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    if unclosed:  # lenient: reasoning prose with no draft block is not a program
        return ""
    for fence in ("```cpp", "```c++", "```"):
        if text.startswith(fence):
            text = text[len(fence) :].strip()
            break
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


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
