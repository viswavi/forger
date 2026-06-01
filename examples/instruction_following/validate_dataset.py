"""Validate and summarize the generated instruction-following dataset.

Usage: python validate_dataset.py [path.jsonl]
Reports count, schema validity, duplicate prompts, length stats, and prints a
random sample of (prompt, answer) pairs for manual inspection.
"""
from __future__ import annotations

import json
import os
import random
import sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "instruction_following.jsonl"
)

rows = []
bad = 0
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("prompt"), str)
            and isinstance(obj.get("answer"), str)
            and obj["prompt"].strip()
            and obj["answer"].strip()
        ):
            rows.append(obj)
        else:
            bad += 1

prompts = [r["prompt"] for r in rows]
dupes = len(prompts) - len(set(prompts))
plen = [len(r["prompt"]) for r in rows]
alen = [len(r["answer"]) for r in rows]


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    return (min(xs), xs[n // 2], max(xs)) if xs else (0, 0, 0)


print(f"file: {path}")
print(f"valid rows: {len(rows)}   malformed/skipped: {bad}")
print(f"duplicate prompts: {dupes}")
print(f"prompt chars (min/median/max): {stats(plen)}")
print(f"answer chars (min/median/max): {stats(alen)}")
print("\n--- 5 random samples ---")
random.seed(0)
for r in random.sample(rows, min(5, len(rows))):
    print("=" * 80)
    print("PROMPT:", r["prompt"][:600])
    print("-" * 40)
    print("ANSWER:", r["answer"][:800])
