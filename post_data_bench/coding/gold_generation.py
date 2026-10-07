"""Rejection-sample verified gold solutions for LiveCodeBench problems.

Unresolved problems receive 1, 1, 2, 4, then 8 new samples, cached per stage.
Persist question_id, obfuscated gold_response, and verified_solvable in
gold_responses.jsonl; completed rows need no further network calls.

Usage: python gold_generation.py [--limit N] [--concurrency 20] [--refresh]
                               [--frozen-file gold_responses.jsonl]
Uncached calls require LITELLM_API_KEY and LITELLM_BASE_URL; LITELLM_MODEL_ID
defaults to litellm_proxy/anthropic/claude-sonnet-5."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import checkers
from source_policy import eligible_sources
import prompt_building as pb
from dotenv import load_dotenv

MODEL_ID = os.environ.get("LITELLM_MODEL_ID", "litellm_proxy/anthropic/claude-sonnet-5")
# Sonnet rejects explicit temperature; use defaults and distinct stage cache keys.
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
    rows = eligible_sources(rows)
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


def _job_for(row: dict, n: int, stage_idx: int) -> dict:
    problem = checkers.row_to_problem(row)
    messages = pb.build_candidate_prompt(problem, include_abstention_instruction=False)
    return {
        "model_id": MODEL_ID,
        "messages": messages,
        "n": n,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stage_tag": f"gold_stage_{stage_idx}",
    }


def run_gold_generation(
    rows: list[dict], *, concurrency: int = 20, refresh: bool = False, on_progress=None
) -> list[dict]:
    """Sample unresolved rows; return question_id, gold_response, verified_solvable."""
    # Frozen split construction does not need the generation/framework stack.
    import llm_async

    n_rows = len(rows)
    gold_response: list[str | None] = [None] * n_rows
    solved: list[bool] = [False] * n_rows
    failed = set()

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
            _job_for(rows[i], n, stage_idx) for i in pending_indices
        ]
        if refresh:
            for job in jobs:
                job["_no_cache"] = True
        results = llm_async.sample_many(jobs, max_workers=concurrency)
        n_newly_solved = 0
        for idx, completions in zip(pending_indices, results):
            if completions is None or any(c is None for c in completions):
                failed.add(idx)
            if completions is None:
                continue
            problem = checkers.row_to_problem(rows[idx])
            for completion in completions:
                if completion is None:
                    continue
                code = checkers.extract_code(completion)
                if not code.strip():
                    continue
                if checkers.check_lcb_solution(problem, code, timeout=GRADE_TIMEOUT):
                    gold_response[idx] = code
                    solved[idx] = True
                    n_newly_solved += 1
                    break
        if on_progress:
            on_progress([{"question_id": rows[i]["question_id"], "gold_response": gold_response[i], "verified_solvable": True} for i in range(n_rows) if solved[i]])
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
        for i, (row, gold, ok) in enumerate(zip(rows, gold_response, solved))
        if ok or i not in failed
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
        help="Externally supplied verdict file: loaded for already-resolved problems, "
        "updated with any newly-resolved ones.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the frozen file's existing coverage and re-resolve every row "
        "(still updates the frozen file with the fresh results).",
    )
    parser.add_argument("--no-generation", action="store_true", help="Fail if any paid generation is needed")
    args = parser.parse_args()

    load_dotenv()
    rows = load_lcb_native_rows(limit=args.limit)
    print(f"[gold] loaded {len(rows)} lcb_native problem(s)")

    verdicts = load_frozen_verdicts(args.frozen_file)
    def persist_progress(resolved):
        for item in resolved:
            verdicts[item["question_id"]] = {"gold_response": item["gold_response"], "verified_solvable": item["verified_solvable"]}
        checkers.write_jsonl(args.frozen_file, [
            {"question_id": qid, "gold_response": checkers.encode_answer(v["gold_response"]) if v["gold_response"] else None, "verified_solvable": v["verified_solvable"]}
            for qid, v in sorted(verdicts.items())
        ])
    pending_rows = [row for row in rows if args.refresh or row["question_id"] not in verdicts]
    print(
        f"[gold] {len(rows) - len(pending_rows)} already resolved in "
        f"{args.frozen_file}, {len(pending_rows)} pending"
    )

    if pending_rows and args.no_generation:
        parser.error("Frozen artifacts are incomplete; explicitly allow generation or restore the missing records")
    if pending_rows:
        checkers.require_litellm_api_key()
        checkers.require_litellm_base_url()
        new_verdicts = run_gold_generation(
            pending_rows,
            concurrency=args.concurrency, refresh=args.refresh, on_progress=persist_progress,
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
    if pending_rows and len(new_verdicts) != len(pending_rows):
        raise SystemExit("Generation failures left unresolved rows pending; successful verdicts were saved.")


if __name__ == "__main__":
    main()
