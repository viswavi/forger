"""Convert generated coding rows to TRL conversations using evaluation prompts.

Inputs: question_content, public_test_cases (list or JSON string), thinking,
and gold_response. Rows are stdin/stdout problems without starter code.
The abstention marker stays verbatim; other answers become fenced Python.

By default, assistant targets contain <think> reasoning followed by the answer.
--no-thinking writes bare answers; Qwen3’s default template supplies the empty
think wrapper used at evaluation.

Drop conversations over --max-length using --tokenizer-model’s chat template.
Match the training length limit to avoid truncated completion targets."""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

import prompt_building as pb

# Arguments: input JSONL, optional model-name suffix, and length/tokenizer flags.
argv = sys.argv[1:]


def _pop_flag(name: str, default: str) -> str:
    if name in argv:
        i = argv.index(name)
        value = argv[i + 1]
        del argv[i : i + 2]
        return value
    return default


def _pop_bool_flag(name: str) -> bool:
    if name in argv:
        argv.remove(name)
        return True
    return False


# Must match whatever --max-length train_model.sh is passed (default 18800,
# below the 28672-token MAX_TOKENS evaluate_model.sh allows at inference time --
# see that file's comment on why: longer sequences don't fit our hardware).
MAX_LENGTH = int(_pop_flag("--max-length", "18800"))
# Only used to count tokens for the length filter above -- independent of
# model_name below (which only affects the output file name).
TOKENIZER_MODEL = _pop_flag("--tokenizer-model", None)
NO_THINKING = _pop_bool_flag("--no-thinking")

assert len(argv) >= 1, (
    "usage: convert_to_trl.py <input.jsonl> [model_name] "
    "[--max-length N] [--tokenizer-model NAME] [--no-thinking]"
)
SRC = Path(argv[0])

model_name = argv[1] if len(argv) > 1 else None
suffix = f"_{re.sub(r'[^0-9A-Za-z._-]+', '_', model_name)}" if model_name else ""
think_tag = "_no_thinking" if NO_THINKING else ""
DST = SRC.with_name(f"{SRC.stem}_trl_format{think_tag}{suffix}.jsonl")

_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL or model_name or "Qwen/Qwen3-4B")


def _as_problem(row: dict) -> SimpleNamespace:
    """Duck-typed stand-in for checkers.CodeGenerationProblem, just for prompt
    rendering -- prompt_building.build_candidate_prompt only reads .question_content,
    .starter_code, and .public_test_cases[i].input/.output off whatever it's given."""
    tests = row["public_test_cases"]
    if isinstance(tests, str):
        tests = json.loads(tests)
    return SimpleNamespace(
        question_content=row["question_content"],
        starter_code=row.get("starter_code") or "",
        public_test_cases=[
            SimpleNamespace(input=t["input"], output=t["output"]) for t in tests
        ],
    )


def main() -> None:
    n = 0
    n_too_long = 0
    with SRC.open() as fin, DST.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Drop degenerate empty-reasoning rows even when omitting thinking from targets.
            if len((row.get("thinking") or "").split()) == 0:
                continue
            messages = pb.build_candidate_prompt(
                _as_problem(row), include_abstention_instruction=True
            )
            if row["gold_response"] == pb.ABSTENTION_MARKER:
                answer = pb.ABSTENTION_MARKER
            else:
                answer = f"```python\n{row['gold_response']}\n```"
            content = answer if NO_THINKING else f"<think>\n{row['thinking']}\n</think>\n\n{answer}"
            messages.append({"role": "assistant", "content": content})

            # BatchEncoding, not a plain list, unless return_dict=False -- len()
            # on the former counts dict keys (2), not tokens.
            n_tokens = len(
                _tokenizer.apply_chat_template(
                    messages, tokenize=True, return_dict=False
                )
            )
            if n_tokens > MAX_LENGTH:
                n_too_long += 1
                continue

            fout.write(json.dumps({"messages": messages}) + "\n")
            n += 1
    print(f"Wrote {n} rows to {DST}")
    print(
        f"Dropped {n_too_long} row(s) exceeding {MAX_LENGTH} tokens "
        f"(tokenized with {TOKENIZER_MODEL}'s chat template)"
    )


if __name__ == "__main__":
    main()
