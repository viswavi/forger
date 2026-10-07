"""Remove <think> prefixes from converted TRL assistant targets.

Write bare answers, letting Qwen3’s default template supply the empty think
wrapper. No raw source rows or length re-filtering are needed: text only shrinks.
Usage: python strip_thinking_from_trl.py <input.jsonl>
Output: <stem>_no_thinking.jsonl beside the input."""
import json
import re
import sys
from pathlib import Path

THINK_PREFIX = re.compile(r"^<think>\n.*?\n</think>\n\n", re.S)

assert len(sys.argv) == 2, "usage: strip_thinking_from_trl.py <generated_raw_trl_format.jsonl>"
SRC = Path(sys.argv[1])
DST = SRC.with_name(f"{SRC.stem}_no_thinking.jsonl")


def main() -> None:
    n = 0
    n_unwrapped = 0
    with SRC.open() as fin, DST.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assistant = row["messages"][-1]
            assert assistant["role"] == "assistant"
            stripped, count = THINK_PREFIX.subn("", assistant["content"], count=1)
            if count == 0:
                n_unwrapped += 1
            assistant["content"] = stripped
            fout.write(json.dumps(row) + "\n")
            n += 1
    print(f"Wrote {n} rows to {DST}")
    if n_unwrapped:
        print(
            f"!!! warning: {n_unwrapped} row(s) had no <think>...</think> prefix to strip "
            "(left unchanged) -- verify these weren't already non-thinking rows mixed in"
        )


if __name__ == "__main__":
    main()
