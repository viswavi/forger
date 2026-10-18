"""Stage B: rejection-sample a verified gold response for every lcb_native problem.

LiveCodeBench ships no canonical solutions, so for each of the (up to) 1055 problems
in test.jsonl..test6.jsonl we sample candidate solutions from a reference model
until one passes every public+private test case, or we give up. Samples are
collected in escalating stages so easy problems (most of them) resolve cheaply and
only genuinely hard problems pay for the full budget:

    stage    1  2  3  4  5
    new      1  1  2  4  8
    cumulative 1  2  4  8 16

Each stage is only run for problems that haven't already found a passing sample in
an earlier stage. Every (problem, stage) sampling call is also @op-cached via
llm_async.sample_sync.

Determinism / no repeated LLM calls: results are stored in a lean, checked-into-git
verdict file (gold_responses.jsonl -- just
question_id/gold_response/verified_solvable, not a duplicate of the full problem)
with gold_response zlib->base64 obfuscated. A run only calls the LLM for
problems *not already present* in that file, so once it's fully populated and
committed, running this script (and therefore set_up_data.sh) does zero network
calls -- pure disk load.

Usage:
    python gold_generation.py [--limit N] [--concurrency 20] [--refresh]
                               [--frozen-file gold_responses.jsonl]

Requires LITELLM_API_KEY and LITELLM_BASE_URL in the environment unless all results
are already completed/cached.
LITELLM_MODEL_ID is optional (defaults to litellm_proxy/anthropic/claude-sonnet-5).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import checkers
import llm_async
import prompt_building as pb
from dotenv import load_dotenv

load_dotenv()

# This can be overrided by the environment variable LITELLM_MODEL_ID
if "LITELLM_MODEL_ID" in os.environ:
    MODEL_ID = os.environ["LITELLM_MODEL_ID"]
else:
    print(
        "LITELLM_MODEL_ID not set; default to litellm_proxy/anthropic/claude-sonnet-5"
    )
    MODEL_ID = "litellm_proxy/anthropic/claude-sonnet-5"
# claude-sonnet-5 rejects an explicit temperature param
# outright, so we don't set one -- diversity across rejection-sampling stages comes
# from the model's own default stochastic decoding plus each stage getting a
# genuinely new network call (see stage_tag in llm_async.sample_sync).
TEMPERATURE = None
MAX_TOKENS = 4096
GRADE_TIMEOUT = 6

# New samples collected at each stage; cumulative totals are 1, 2, 4, 8, 16.
STAGE_SCHEDULE = [1, 1, 2, 4, 8]

DATA_DIR = Path(__file__).resolve().parent
SOURCE_FILES = [
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
]
DEFAULT_FROZEN_FILE = DATA_DIR / "gold_responses.jsonl"


def load_lcb_native_rows(limit: int | None = None) -> list[dict]:
    """Load raw rows from test.jsonl..test6.jsonl, optionally capped at `limit`."""
    rows = [
        row for name in SOURCE_FILES for row in checkers.read_jsonl(DATA_DIR / name)
    ]
    return rows[:limit] if limit is not None else rows


def load_frozen_verdicts(path) -> dict[str, dict]:
    """question_id -> {gold_response (decoded, or None), verified_solvable}."""
    verdicts = {}
    for row in checkers.read_jsonl(path):
        gold_response = (
            checkers.decode_answer(row["gold_response"])
            if row["gold_response"]
            else None
        )
        verdicts[row["question_id"]] = {
            "gold_response": gold_response,
            "verified_solvable": row["verified_solvable"],
        }
    return verdicts


def _job_for(row: dict, api_key: str, base_url: str, n: int, stage_idx: int) -> dict:
    problem = checkers.row_to_problem(row)
    messages = pb.build_candidate_prompt(problem, include_abstention_instruction=False)
    return {
        "model_id": MODEL_ID,
        "api_key": api_key,
        "api_base": base_url,
        "messages": messages,
        "n": n,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stage_tag": f"gold_stage_{stage_idx}",
    }


def run_gold_generation(
    rows: list[dict], *, api_key: str, base_url: str, concurrency: int = 20
) -> list[dict]:
    """Rejection-sample every row in `rows` (assumed not yet resolved).

    Returns [{question_id, gold_response, verified_solvable}, ...].
    """
    n_rows = len(rows)
    gold_response: list[str | None] = [None] * n_rows
    solved: list[bool] = [False] * n_rows

    for stage_idx, n in enumerate(STAGE_SCHEDULE, start=1):
        pending_indices = [i for i in range(n_rows) if not solved[i]]
        if not pending_indices:
            break
        print(
            f"[gold] stage {stage_idx}/{len(STAGE_SCHEDULE)} (n={n} new samples): "
            f"{len(pending_indices)} problem(s) still unsolved"
        )
        t0 = time.time()
        jobs = [
            _job_for(rows[i], api_key, base_url, n, stage_idx) for i in pending_indices
        ]
        results = llm_async.sample_many(jobs, max_workers=concurrency)
        n_newly_solved = 0
        for idx, completions in zip(pending_indices, results):
            if completions is None:
                continue
            problem = checkers.row_to_problem(rows[idx])
            for completion in completions:
                code = checkers.extract_code(completion)
                if not code.strip():
                    continue
                if checkers.check_lcb_solution(problem, code, timeout=GRADE_TIMEOUT):
                    gold_response[idx] = code
                    solved[idx] = True
                    n_newly_solved += 1
                    break
        print(
            f"[gold] stage {stage_idx} done in {time.time() - t0:.0f}s: "
            f"{n_newly_solved} newly solved, {sum(solved)}/{n_rows} solved total"
        )

    return [
        {
            "question_id": row["question_id"],
            "gold_response": gold,
            "verified_solvable": ok,
        }
        for row, gold, ok in zip(rows, gold_response, solved)
    ]


def main() -> None:
    """Parse CLI args and populate/extend the gold_responses.jsonl frozen file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N rows."
    )
    parser.add_argument(
        "--concurrency", type=int, default=20, help="Max concurrent LLM calls."
    )
    parser.add_argument(
        "--frozen-file",
        type=str,
        default=str(DEFAULT_FROZEN_FILE),
        help="Checked-in verdict file: loaded for already-resolved problems, "
        "updated with any newly-resolved ones.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the frozen file's existing coverage and re-resolve every row "
        "(still updates the frozen file with the fresh results).",
    )
    args = parser.parse_args()

    rows = load_lcb_native_rows(limit=args.limit)
    print(f"[gold] loaded {len(rows)} lcb_native problem(s)")

    verdicts = {} if args.refresh else load_frozen_verdicts(args.frozen_file)
    pending_rows = [row for row in rows if row["question_id"] not in verdicts]
    print(
        f"[gold] {len(rows) - len(pending_rows)} already resolved in "
        f"{args.frozen_file}, {len(pending_rows)} pending"
    )

    if pending_rows:
        api_key = checkers.require_litellm_api_key()
        base_url = checkers.require_litellm_base_url()
        new_verdicts = run_gold_generation(
            pending_rows,
            api_key=api_key,
            base_url=base_url,
            concurrency=args.concurrency,
        )
        for v in new_verdicts:
            verdicts[v["question_id"]] = {
                "gold_response": v["gold_response"],
                "verified_solvable": v["verified_solvable"],
            }

    n_solved = sum(1 for v in verdicts.values() if v["verified_solvable"])
    print(
        f"[gold] final: {n_solved}/{len(verdicts)} problems have a verified gold "
        "response"
    )

    frozen_rows = [
        {
            "question_id": qid,
            "gold_response": (
                checkers.encode_answer(v["gold_response"])
                if v["gold_response"]
                else None
            ),
            "verified_solvable": v["verified_solvable"],
        }
        for qid, v in sorted(verdicts.items())
    ]
    checkers.write_jsonl(args.frozen_file, frozen_rows)
    print(f"[gold] wrote {args.frozen_file}")


if __name__ == "__main__":
    main()
