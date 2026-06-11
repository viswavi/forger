#!/usr/bin/env python3
"""Offline, robust re-scoring of saved AIME Inspect logs.

The upstream aime_scorer only inspects the LAST line of the completion and
numeric-matches it, so a correct answer wrapped in `$$\\text{ANSWER: } 70$$`
(last line == "$$") scores as wrong. This re-scorer reads the model's final
answer text from the saved logs (no inference needed) and extracts the integer
answer robustly, then reports accuracy alongside the originally-reported value.

Usage:
    python rescore_aime.py [aime2025_outputs/<dir> ...]
With no args it scans all subdirectories of aime2025_outputs/.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys


def output_parts(sample: dict) -> tuple[str, str]:
    """Return (final_text, reasoning_text) for a sample.

    Qwen3 thinking sometimes leaves only "$$" as the post-</think> text and
    puts the actual \\boxed{} answer inside the reasoning trace, so we need
    both parts to score fairly.
    """
    out = sample.get("output", {}) or {}
    choices = out.get("choices") or []
    text, reasoning = "", ""
    if choices:
        content = (choices[0].get("message", {}) or {}).get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(it.get("text", "") for it in content
                             if isinstance(it, dict) and it.get("type") == "text")
            reasoning = "\n".join(it.get("reasoning", "") for it in content
                                  if isinstance(it, dict) and it.get("type") == "reasoning")
    comp = out.get("completion")
    if (not text.strip()) and isinstance(comp, str):
        text = comp
    return text, reasoning


def clean(tok: str) -> str:
    tok = tok.strip()
    tok = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", tok)
    tok = re.sub(r"\\text\{([^{}]*)\}", r"\1", tok)
    tok = tok.replace("$", "").replace("\\(", "").replace("\\)", "")
    tok = tok.replace(",", "").replace("\\,", "").strip()
    tok = tok.strip(" .:*`{}")
    return tok


def _last_answer_line(text: str):
    for cand in reversed(re.findall(r"ANSWER\s*:?\s*([^\n]+)", text, flags=re.IGNORECASE)):
        m = re.search(r"-?\d+", clean(cand))
        if m:
            return int(m.group())
    return None


def _last_boxed(text: str):
    for cand in reversed(re.findall(r"\\boxed\{([^{}]*)\}", text)):
        m = re.search(r"-?\d+", clean(cand))
        if m:
            return int(m.group())
    return None


def extract_answer(text: str, reasoning: str = ""):
    """Recover the model's integer answer, consulting the reasoning trace too.

    Priority (conservative -> avoid picking random numbers out of 70k-char
    reasoning): explicit "ANSWER:" in the final text, then the last \\boxed{}
    anywhere (final text wins over reasoning), then "ANSWER:" in reasoning,
    then the last integer in the (short) final text only.
    """
    if text:
        a = _last_answer_line(text)
        if a is not None:
            return a
    a = _last_boxed(text) if text else None
    if a is not None:
        return a
    a = _last_boxed(reasoning) if reasoning else None
    if a is not None:
        return a
    if reasoning:
        a = _last_answer_line(reasoning)
        if a is not None:
            return a
    if text and len(text) < 40:  # only trust a bare trailing int in a SHORT final text
        ints = re.findall(r"-?\d+", text)
        if ints:
            return int(ints[-1])
    return None


def score_dir(d: str) -> None:
    logs = sorted(glob.glob(os.path.join(d, "*aime2025*.json")))
    if not logs:
        print(f"{d}: (no aime json log)")
        return
    data = json.load(open(logs[-1]))
    samples = data.get("samples", [])
    orig = None
    res = data.get("results") or {}
    for sc in (res.get("scores") or []):
        m = sc.get("metrics", {})
        if "accuracy" in m:
            orig = m["accuracy"]["value"]
    n = len(samples)
    correct = 0
    recovered = 0
    for s in samples:
        target = s.get("target")
        try:
            tgt = int(str(target).strip())
        except (TypeError, ValueError):
            continue
        text, reasoning = output_parts(s)
        ans = extract_answer(text, reasoning)
        if ans is not None:
            recovered += 1
        if ans == tgt:
            correct += 1
    acc = correct / n if n else 0.0
    orig_s = f"{orig:.3f}" if isinstance(orig, (int, float)) else "n/a"
    name = os.path.basename(d.rstrip("/"))
    print(f"{name:42s} n={n:3d}  orig={orig_s:>6s}  rescored={acc:.3f}  "
          f"(answer recovered for {recovered}/{n})")


def main() -> None:
    dirs = sys.argv[1:]
    if not dirs:
        dirs = sorted(glob.glob("aime2025_outputs/*/"))
    dirs = [d for d in dirs if os.path.isdir(d)]
    print(f"{'RUN':42s} {'':3s}   {'ORIG':>6s}     {'RESCORED':>8s}")
    print("-" * 86)
    for d in dirs:
        score_dir(d)


if __name__ == "__main__":
    main()
