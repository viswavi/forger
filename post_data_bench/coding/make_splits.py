"""Stage D: join lcb_native rows with the frozen gold/perturbation files.

Produces validation_tasks.jsonl / test_tasks.jsonl.

Reads raw problems fresh from test.jsonl..test6.jsonl and joins them against two
checked-in, LLM-call-free artifacts:
  - gold_responses.jsonl (gold_generation.py): question_id ->
    gold_response/verified_solvable.
  - perturbed_cases.jsonl (perturb.py): source_question_id ->
    flipped_test_case_index/wrong_output.
Given those two files, this whole stage is a pure, deterministic, no-network
join+split -- no LLM calls happen here, ever.

Split logic:
  - Unperturbed rows: partitioned into solved/unsolved by the gold verdict's
    `verified_solvable`. Solved rows are shuffled (fixed seed) and split 10/90 into
    val/test. Unsolved rows always go to test (never validation -- a "validation
    set with gold responses" should only contain problems we've actually verified
    are solvable).
  - Perturbed rows: each one's split assignment is *inherited* from its own
    unperturbed twin's assignment (whatever the shuffle, or the unsolved-fallback
    rule, decided for that twin) -- so a problem's real and perturbed versions
    always land in the same file, for direct before/after comparison.

Sensitive fields (gold_response and a perturbed row's true_output_for_flipped_case)
are zlib->base64 obfuscated in the written files -- applied uniformly to every row
regardless of content, so the mere presence of obfuscation can't leak is_possible.

private_test_cases is NOT stored inline: it's often thousands of characters (some
problems' stress-test inputs run into tens of MB) and a perturbed row's is always
byte-identical to its unperturbed twin's, so embedding it in every split-file row
both bloats validation_tasks.jsonl/test_tasks.jsonl and duplicates the same bytes
across twins. Instead every row's blob is written once, in raw-row order, to
private_test_cases.txt (flat text, one obfuscated blob per line -- see
checkers.write_private_test_cases/load_private_test_cases), and every split-file row
(perturbed or not) carries a `private_test_case_idx` int: a 0-based line number into
that file. A perturbed row shares its twin's idx.

Usage:
    python make_splits.py [--gold gold_responses.jsonl]
                           [--perturbed perturbed_cases.jsonl]
                           [--val-out validation_tasks.jsonl]
                           [--test-out test_tasks.jsonl]
                           [--private-test-cases-out private_test_cases.txt]
"""
from __future__ import annotations

import argparse
import random

import checkers
import prompt_building as pb
from gold_generation import DATA_DIR
from gold_generation import DEFAULT_FROZEN_FILE as DEFAULT_GOLD_FILE
from gold_generation import load_frozen_verdicts, load_lcb_native_rows
from perturb import DEFAULT_FROZEN_FILE as DEFAULT_PERTURBED_FILE
from perturb import load_frozen_perturbations

SEED = 20260706
VAL_FRACTION = 0.1


def _split_solved(
    solved_question_ids: list[str], seed: int, val_fraction: float
) -> dict[str, str]:
    """Shuffle+split solved question_ids into {qid: 'val'|'test'}, fixed seed."""
    ids = list(solved_question_ids)
    random.Random(seed).shuffle(ids)
    n_val = round(len(ids) * val_fraction)
    assignment = {qid: "val" for qid in ids[:n_val]}
    assignment.update({qid: "test" for qid in ids[n_val:]})
    return assignment


def _render_task_text(row: dict) -> str:
    """Render the human-facing task text a candidate would see.

    Question + starter code + explicit public test cases, without the system
    message or the abstention-instruction suffix (that's added at eval time by
    evaluate_lcb_v6_plus_impossible.py, not baked into the stored task).
    """
    problem = checkers.row_to_problem(row)
    messages = pb.build_candidate_prompt(problem, include_abstention_instruction=False)
    return messages[1]["content"]


def _finalize_row(
    row: dict,
    *,
    gold_response: str | None,
    is_perturbed: bool,
    private_test_case_idx: int,
) -> dict:
    out = {
        "question_id": row["question_id"],
        "question_title": row["question_title"],
        "contest_id": row["contest_id"],
        "is_perturbed": is_perturbed,
        "is_possible": not is_perturbed,
        "difficulty": row["difficulty"],
        "contest_date": row["contest_date"],
        "platform": row["platform"],
        "question_content": row["question_content"],
        "starter_code": row["starter_code"],
        "public_test_cases": row["public_test_cases"],
        "metadata": row["metadata"],
        "task_text": _render_task_text(row),
        "private_test_case_idx": private_test_case_idx,
        "gold_response": (
            checkers.encode_answer(gold_response) if gold_response is not None else None
        ),
    }
    if is_perturbed:
        out["source_question_id"] = row["source_question_id"]
        out["flipped_test_case_index"] = row["flipped_test_case_index"]
        out["true_output_for_flipped_case"] = checkers.encode_answer(
            row["true_output_for_flipped_case"]
        )
    return out


def build_private_test_cases_blobs(raw_rows: list[dict]) -> list[str]:
    """Obfuscated private_test_cases, one per raw problem, in raw_rows order.

    Line i of the written file corresponds to raw_rows[i]'s question_id -- the
    caller (build_split_files) is responsible for recording that same index (i) on
    every split-file row derived from raw_rows[i] (including its perturbed twin, if
    any).

    checkers.normalize_private_test_cases runs first to collapse away LiveCodeBench's
    own pre-obfuscation on the rows that ship with it, so every blob here is
    single-layer zlib->base64 (checkers.encode_answer) -- decode_answer alone always
    recovers the plain JSON-text private_test_cases string directly, never a pickle
    blob or extra JSON-string wrapping.
    """
    return [
        checkers.encode_answer(
            checkers.normalize_private_test_cases(row["private_test_cases"])
        )
        for row in raw_rows
    ]


def build_split_files(
    raw_rows: list[dict], verdicts: dict[str, dict], perturbations: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    """Join raw rows with gold verdicts and perturbations into (val_rows, test_rows)."""
    missing = [r["question_id"] for r in raw_rows if r["question_id"] not in verdicts]
    if missing:
        raise SystemExit(
            f"{len(missing)} problem(s) have no gold_generation.py verdict yet "
            f"(e.g. {missing[:5]}). Run gold_generation.py first (it will skip "
            "anything already in gold_responses.jsonl, so this is cheap if most "
            "problems are already covered)."
        )

    solved_ids = [
        r["question_id"]
        for r in raw_rows
        if verdicts[r["question_id"]]["verified_solvable"]
    ]
    assignment = _split_solved(solved_ids, SEED, VAL_FRACTION)
    raw_by_id = {r["question_id"]: r for r in raw_rows}
    idx_by_id = {r["question_id"]: i for i, r in enumerate(raw_rows)}

    def split_for(qid: str) -> str:
        # Unsolved problems always go to test, never validation (see module
        # docstring); perturbed rows call this with their twin's qid so they
        # inherit the twin's assignment.
        return assignment[qid] if verdicts[qid]["verified_solvable"] else "test"

    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for row in raw_rows:
        qid = row["question_id"]
        finalized = _finalize_row(
            row,
            gold_response=verdicts[qid]["gold_response"],
            is_perturbed=False,
            private_test_case_idx=idx_by_id[qid],
        )
        (val_rows if split_for(qid) == "val" else test_rows).append(finalized)

    for qid, perturbation in perturbations.items():
        twin = raw_by_id.get(qid)
        if twin is None:
            print(
                f"[splits] WARNING: perturbation for {qid} has no matching "
                "lcb_native row; skipping."
            )
            continue
        split = split_for(qid)
        perturbed_row = checkers.build_perturbed_row(
            twin, perturbation["flipped_test_case_index"], perturbation["wrong_output"]
        )
        if (
            perturbed_row["true_output_for_flipped_case"]
            != perturbation["true_output_for_flipped_case"]
        ):
            print(
                f"[splits] WARNING: {qid}'s recomputed true output doesn't match "
                "perturbed_cases.jsonl's stored value -- source data may have "
                "changed since perturb.py ran; using the freshly recomputed value."
            )
        finalized = _finalize_row(
            perturbed_row,
            gold_response=pb.ABSTENTION_MARKER,
            is_perturbed=True,
            private_test_case_idx=idx_by_id[qid],
        )
        (val_rows if split == "val" else test_rows).append(finalized)

    # Shuffle final row order (distinct seed) so possible/impossible rows aren't
    # trivially grouped by file position -- doesn't affect grading (each row is
    # scored independently), just tidiness for anyone skimming the file.
    random.Random(SEED + 1).shuffle(val_rows)
    random.Random(SEED + 2).shuffle(test_rows)
    return val_rows, test_rows


def main() -> None:
    """Parse CLI args and write validation_tasks.jsonl / test_tasks.jsonl."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=str, default=str(DEFAULT_GOLD_FILE))
    parser.add_argument("--perturbed", type=str, default=str(DEFAULT_PERTURBED_FILE))
    parser.add_argument(
        "--val-out", type=str, default=str(DATA_DIR / "validation_tasks.jsonl")
    )
    parser.add_argument(
        "--test-out", type=str, default=str(DATA_DIR / "test_tasks.jsonl")
    )
    parser.add_argument(
        "--private-test-cases-out",
        type=str,
        default=str(DATA_DIR / "private_test_cases.txt"),
    )
    args = parser.parse_args()

    raw_rows = load_lcb_native_rows()
    verdicts = load_frozen_verdicts(args.gold)
    perturbations = load_frozen_perturbations(args.perturbed)
    if not verdicts:
        raise SystemExit(
            f"No verdicts found at {args.gold} -- run gold_generation.py first."
        )

    val_rows, test_rows = build_split_files(raw_rows, verdicts, perturbations)

    n_solved = sum(1 for v in verdicts.values() if v["verified_solvable"])
    print(
        f"[splits] {len(raw_rows)} lcb_native rows ({n_solved} solved, "
        f"{len(raw_rows) - n_solved} unsolved); {len(perturbations)} perturbed rows"
    )
    n_val_perturbed = sum(r["is_perturbed"] for r in val_rows)
    n_test_perturbed = sum(r["is_perturbed"] for r in test_rows)
    print(f"[splits] val: {len(val_rows)} rows ({n_val_perturbed} perturbed)")
    print(f"[splits] test: {len(test_rows)} rows ({n_test_perturbed} perturbed)")

    checkers.write_jsonl(args.val_out, val_rows)
    checkers.write_jsonl(args.test_out, test_rows)
    checkers.write_private_test_cases(
        args.private_test_cases_out, build_private_test_cases_blobs(raw_rows)
    )
    print(
        f"[splits] wrote {args.val_out}, {args.test_out}, and "
        f"{args.private_test_cases_out}"
    )


if __name__ == "__main__":
    main()
