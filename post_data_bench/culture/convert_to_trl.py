"""Convert country/situation/question/thinking/answer rows to TRL conversations.

Reuse evaluation prompt builders; assistant targets contain <think> reasoning
followed by the formatted answer.
"""
import argparse
import json
import re
from pathlib import Path

import prompt_building as pb
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("model_name", nargs="?")
parser.add_argument("--tokenizer-model")
parser.add_argument("--max-length", type=int, default=8192)
args = parser.parse_args()
SRC = args.source
suffix = (
    "_" + re.sub(r"[^0-9A-Za-z._-]+", "_", args.model_name) if args.model_name else ""
)
DST = SRC.with_name(f"{SRC.stem}_trl_format{suffix}.jsonl")
_tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_model or args.model_name or "Qwen/Qwen3-4B"
)


def main() -> None:
    """Run the command-line utility described in the module documentation."""
    n = dropped = 0
    with SRC.open() as fin, DST.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if len((row.get("thinking") or "").split()) == 0:
                continue
            # Same transform make_splits.py applies before calling build_user_content,
            # so a freshly-generated row renders identically to a real eval row.
            country = row["country"].replace("_", " ").title()
            user_content = pb.build_user_content(country, row["situation"])
            answer_formatted = "Answer: " + row["answer"]
            answer_thinking = (
                f"<think>\n{row['thinking']}\n</think>\n\n{answer_formatted}"
            )
            out = {
                "messages": [
                    {"role": "system", "content": pb.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer_thinking},
                ]
            }
            if (
                len(
                    _tokenizer.apply_chat_template(
                        out["messages"], tokenize=True, return_dict=False
                    )
                )
                > args.max_length
            ):
                dropped += 1
                continue
            fout.write(json.dumps(out) + "\n")
            n += 1
    print(f"Wrote {n} rows to {DST}; dropped {dropped} overlength targets")


if __name__ == "__main__":
    main()
