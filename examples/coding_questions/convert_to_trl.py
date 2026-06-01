"""Convert coding_questions.jsonl into a TRL-friendly dataset.

Source rows look like: {"prompt": "...", "answer": "..."}
TRL's conversational format wants a single `messages` column:
    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ]}
"""

import json
from pathlib import Path

SRC = Path(__file__).with_name("coding_questions.jsonl")
DST = Path(__file__).with_name("coding_questions_trl_format.jsonl")


def main() -> None:
    n = 0
    with SRC.open() as fin, DST.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out = {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["canonical_solution"]},
                ]
            }
            fout.write(json.dumps(out) + "\n")
            n += 1
    print(f"Wrote {n} rows to {DST}")


if __name__ == "__main__":
    main()
