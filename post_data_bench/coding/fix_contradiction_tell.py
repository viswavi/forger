"""Repair recognized examples in legacy conversational training records.

Benchmark builds use harmonize.py. Keep this host tool out of agent exports;
report unparsed examples without changing them."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BLOCK_MARKER = "### Public Test Cases"

# Statement-side worked example. The statement uses "Example 1:\n\nInput: ...\nOutput: ...",
# the block uses a single newline; \n+ covers both so the same pattern reads either side.
STMT_EXAMPLE = re.compile(r"(Example\s*(\d+)\s*:?\s*\n+Input:[^\n]*\n+Output:)([^\n]*)")
BLOCK_EXAMPLE = re.compile(r"Example\s*(\d+):\nInput:[^\n]*\nOutput:\s*([^\n]*)")

# An Explanation runs to the next example or section heading. Terminators vary: some
# problems separate examples with a blank line and some with a single newline, and the
# trailing section is Constraints:/Follow-up:/the "### Format:" heading, or end of half.
EXPLANATION = re.compile(
    r"\n[ \t]*Explanation:.*?(?=\n\s*Example\s*\d+\s*:|\n\s*Constraints:|"
    r"\n\s*Follow-up:|\n\s*### |\Z)",
    re.S,
)


def fix_prompt(user_content: str) -> tuple[str, str]:
    """Return (new_content, outcome) for one rendered prompt."""
    question, sep, block = user_content.partition(BLOCK_MARKER)
    if not sep:
        return user_content, "no_test_case_block"

    block_outputs = {int(k): v.strip() for k, v in BLOCK_EXAMPLE.findall(block)}
    stmt = STMT_EXAMPLE.findall(question)
    if not stmt or not block_outputs:
        return user_content, "unparsed"
    if len({int(n) for _, n, _ in stmt}) != len(block_outputs):
        # Statement and block enumerate different numbers of examples, so index alignment
        # is not trustworthy; rewriting by index could corrupt a correct row.
        return user_content, "index_misaligned"

    changed = False

    def sync(m: re.Match) -> str:
        nonlocal changed
        head, num, old = m.group(1), int(m.group(2)), m.group(3)
        if num not in block_outputs:
            return m.group(0)
        new = block_outputs[num]
        if old.strip() != new:
            changed = True
            return f"{head} {new}"
        return m.group(0)

    question = STMT_EXAMPLE.sub(sync, question)
    question, n_expl = EXPLANATION.subn("", question)
    return question + sep + block, ("synced" if changed else "already_consistent") + (
        "+explanations_removed" if n_expl else ""
    )


def main() -> None:
    assert len(sys.argv) >= 2, __doc__
    for arg in sys.argv[1:]:
        src = Path(arg)
        dst = src.with_name(f"{src.stem}_fixed.jsonl")
        counts: dict[str, int] = {}
        n = 0
        with src.open() as fin, dst.open("w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                user = row["messages"][1]
                assert user["role"] == "user", "expected [system, user, assistant]"
                user["content"], outcome = fix_prompt(user["content"])
                counts[outcome] = counts.get(outcome, 0) + 1
                fout.write(json.dumps(row) + "\n")
                n += 1
        print(f"{src.name} -> {dst.name}  ({n} rows)")
        for k in sorted(counts, key=lambda k: -counts[k]):
            print(f"    {k:34} {counts[k]:5}")


if __name__ == "__main__":
    main()
