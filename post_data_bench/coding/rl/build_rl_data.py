"""Convert SFT source rows to Slime prompt/label/metadata JSONL.

Reuse evaluation prompts without an assistant turn. Metadata contains the
encoded gold_response (the sole impossibility signal) and raw problem fields,
with empty private tests. Drop perturbation bookkeeping. Possible rows are
graded on visible public tests; see impossible_lcb_rm’s hardcoding guard.

--drop-unverifiable removes rows whose gold solutions fail their own tests,
changing the class mix. Self-test both train and val and combine results
first; infrastructure errors are never grounds for dropping rows.

Usage: python rl/build_rl_data.py <source.jsonl> --out-dir rl/data/full --val-frac 0.05"""
from __future__ import annotations

import argparse
import json
import hashlib
import random
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import checkers  # noqa: E402
import prompt_building as pb  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=str, help="raw training rows jsonl (pre-TRL)")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.05,
        help="Fraction held out as slime's --eval-prompt-data. This is an "
        "in-distribution reward tracker over *generated* problems; it is not the "
        "study's benchmark, which stays evaluate_model.sh on test_tasks.jsonl.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--impossible-frac",
        type=float,
        default=None,
        help="subsample impossible rows so they are this fraction of the output "
             "(e.g. 0.204, the validation-set prior). GRPO learns the abstain "
             "threshold that is reward-optimal for whatever prior the prompt mix "
             "encodes, and per-class reward weighting cannot fix it -- every "
             "sample in a group shares one prompt and therefore one class, so "
             "group-relative advantage cancels any per-class scaling. The mix is "
             "the only lever. Possible rows are always kept in full; only "
             "impossible rows are dropped, so this trades dataset size for "
             "calibration.",
    )
    p.add_argument(
        "--drop-unverifiable",
        type=str,
        default=None,
        help="LCB_RM_DUMP jsonl from `impossible_lcb_rm.py --self-test`; drop every "
        "row whose label scored 0 there, i.e. whose gold response fails its own "
        "tests.",
    )
    p.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen3-4B")
    p.add_argument(
        "--max-prompt-len",
        type=int,
        default=6144,
        help="Drop rows whose rendered prompt exceeds this many tokens. slime "
        "filters at --rollout-max-prompt-len anyway; doing it here means the "
        "reported row counts are the ones actually trained on.",
    )
    return p.parse_args()


def to_problem_fields(row: dict) -> dict:
    """Extract raw LCB fields, supplying empty private tests for generated rows."""
    fields = {k: row[k] for k in checkers.PROBLEM_FIELDS if k != "private_test_cases"}
    fields["private_test_cases"] = "[]"
    return fields


def build_row(row: dict) -> dict:
    problem = checkers.row_to_problem(to_problem_fields(row))
    messages = pb.build_candidate_prompt(problem, include_abstention_instruction=True)
    return {
        "prompt": messages,
        "label": row["question_id"],
        "metadata": {
            "gold_response": checkers.encode_answer(row["gold_response"]),
            "problem": to_problem_fields(row),
            "source_id": row.get("source_question_id") or row["question_id"].split("__perturbed")[0],
        },
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.val_frac < 1 or (args.impossible_frac is not None and not 0 <= args.impossible_frac < 1):
        raise SystemExit("Fractions must lie in (0,1); impossible fraction may also be zero")
    source_rows = checkers.read_jsonl(args.source)
    if not source_rows:
        raise SystemExit("Missing or empty generated-data source")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unverifiable: set[str] = set()
    if args.drop_unverifiable:
        unverifiable = {
            g["label"]
            for g in checkers.read_jsonl(args.drop_unverifiable)
            if g.get("category") == "failed"
        }
        print(f"dropping {len(unverifiable)} row(s) that fail their own gold response")

    rows: list[dict] = []
    n_too_long = 0
    n_impossible = 0
    n_unverifiable = 0
    for src in source_rows:
        if src["question_id"] in unverifiable:
            n_unverifiable += 1
            continue
        out = build_row(src)
        n_tokens = len(
            tokenizer.apply_chat_template(
                out["prompt"], tokenize=True, add_generation_prompt=True,
                return_dict=False,
            )
        )
        if n_tokens > args.max_prompt_len:
            n_too_long += 1
            continue
        if src["gold_response"] == pb.ABSTENTION_MARKER:
            n_impossible += 1
        rows.append(out)

    rng = random.Random(args.seed)
    if args.impossible_frac is not None:
        def _impossible(r: dict) -> bool:
            # Same test impossible_lcb_rm applies at grading time.
            return checkers.decode_answer(r["metadata"]["gold_response"]) == pb.ABSTENTION_MARKER

        imp = [r for r in rows if _impossible(r)]
        pos = [r for r in rows if not _impossible(r)]
        f = args.impossible_frac
        # keep every possible row; solve n_imp/(n_pos+n_imp) == f for n_imp
        keep = min(len(imp), int(round(f * len(pos) / (1 - f)))) if f < 1 else len(imp)
        rng.shuffle(imp)
        dropped = len(imp) - keep
        rows = pos + imp[:keep]
        n_impossible = keep
        print(f"rebalancing to {f:.1%} impossible: kept {keep}/{len(imp) + 0} "
              f"impossible (dropped {dropped}), all {len(pos)} possible")

    def held_out(row):
        key = f"{args.seed}:{row['metadata']['source_id']}"
        return int(hashlib.sha256(key.encode()).hexdigest(),16) / 2**256 < args.val_frac
    val = [row for row in rows if held_out(row)]
    train = [row for row in rows if not held_out(row)]
    if not train or not val:
        raise SystemExit("Both generated-data splits must be nonempty; add rows or adjust --val-frac")

    for name, split in (("train", train), ("val", val)):
        path = out_dir / f"{name}.jsonl"
        checkers.write_jsonl(path, split)
        print(f"wrote {len(split):5d} rows -> {path}")

    print(
        f"source rows kept: {len(rows)} ({n_impossible} impossible, "
        f"{n_impossible / len(rows):.1%}); dropped {n_too_long} over "
        f"{args.max_prompt_len} prompt tokens, {n_unverifiable} unverifiable"
    )


if __name__ == "__main__":
    main()
