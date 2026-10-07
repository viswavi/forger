"""Build a versioned benchmark from frozen inputs with a 10/90 val/test split.

Keep originals and impossible twins together; stratify paired/unpaired quotas
by difficulty and verified solvability. Exclude non-unique-output sources.
Existing test files require --force to overwrite."""
from __future__ import annotations

import argparse
import random
import hashlib
import json
from harmonize import harmonize_statement
from split_contract import PUBLIC_FIELDS, sha256, write_manifest, staged_build
from pathlib import Path

import checkers
from source_policy import exclusion_reason, POLICY_VERSION
import prompt_building as pb
from gold_generation import DATA_DIR
from gold_generation import DEFAULT_FROZEN_FILE as DEFAULT_GOLD_FILE
from gold_generation import load_frozen_verdicts, load_lcb_native_rows
from perturb import DEFAULT_FROZEN_FILE as DEFAULT_PERTURBED_FILE
from perturb import load_frozen_perturbations

SEED = 20260706
VAL_FRACTION = 0.1


def _split_sources(rows, verdicts, perturbations, *, seed=SEED, val_fraction=VAL_FRACTION):
    """Choose reproducible, stratified source groups near the requested row quota.

    Minimize row-count error, then paired/unpaired quota error. Use largest
    remainders across strata and seeded hashes within them. Membership is not
    append-stable, and whole pairs can prevent exact row quotas."""
    import math
    from collections import defaultdict
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    if len({row['question_id'] for row in rows}) != len(rows):
        raise ValueError("Duplicate source IDs in split inputs")
    groups = {True: [], False: []}
    for row in rows:
        groups[row['question_id'] in perturbations].append(row)
    paired, unpaired = len(groups[True]), len(groups[False])
    total_rows = 2 * paired + unpaired
    target_rows = max(1, min(total_rows - 1, math.floor(total_rows * val_fraction + 0.5)))
    # One paired source contributes two rows; one unpaired source contributes one.
    choices = []
    for n_pairs in range(paired + 1):
        n_single = min(unpaired, max(0, target_rows - 2 * n_pairs))
        n_rows = 2 * n_pairs + n_single
        if not 0 < n_rows < total_rows:
            continue
        error = ((n_pairs - paired * val_fraction) ** 2
                 + (n_single - unpaired * val_fraction) ** 2)
        choices.append((abs(n_rows - target_rows), error, n_pairs, n_single))
    if not choices:
        raise ValueError("At least two source groups are required for nonempty splits")
    _, _, pair_quota, single_quota = min(choices)
    assignment = {row['question_id']: 'test' for row in rows}
    for is_paired, quota in ((True, pair_quota), (False, single_quota)):
        strata = defaultdict(list)
        for row in groups[is_paired]:
            strata[(str(row['difficulty']).lower(), bool(verdicts[row['question_id']]['verified_solvable']))].append(row['question_id'])
        group_size = len(groups[is_paired])
        if not group_size:
            continue
        # Integer arithmetic avoids floating-point ties in the remainder ranking.
        counts = {key: quota * len(ids) // group_size for key, ids in strata.items()}
        remaining = quota - sum(counts.values())
        ranked = sorted(strata, key=lambda key: (-(quota * len(strata[key]) % group_size), key))
        for key in ranked[:remaining]:
            counts[key] += 1
        for key, ids in strata.items():
            ids = sorted(ids, key=lambda qid: (hashlib.sha256(f'{seed}:{qid}'.encode()).digest(), qid))
            for qid in ids[:counts[key]]:
                assignment[qid] = 'val'
    return assignment


def _render_task_text(row: dict) -> str:
    """Render question, starter code, and public tests without system/abstention text."""
    problem = checkers.row_to_problem({**row, "private_test_cases": "[]"})
    messages = pb.build_candidate_prompt(problem, include_abstention_instruction=False)
    return messages[1]["content"]


def _finalize_row(
    row: dict,
    *,
    gold_response: str | None,
    is_perturbed: bool,
    private_test_case_idx: int,
) -> dict:
    row = dict(row)
    if json.loads(row["metadata"]).get("func_name"):
        tests = json.loads(row["public_test_cases"])
        for test in tests:
            test["output"] = json.dumps(json.loads(test["output"]), ensure_ascii=False, separators=(",", ":"))
        row["public_test_cases"] = json.dumps(tests)
    row["question_content"] = harmonize_statement(row)
    out = {
        "question_id": hashlib.sha256(f"{SEED}:public:{row['question_id']}".encode()).hexdigest()[:20],
        "question_title": row["question_title"],
        "contest_id": row["contest_id"],
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
    return out


def build_private_test_cases_blobs(raw_rows: list[dict]) -> list[str]:
    """Write normalized, obfuscated private tests in raw_rows order.

    Callers assign the same line index to each source and its twin. Normalize
    LCB’s encoding first so decode_answer alone recovers plain JSON."""
    return [
        checkers.encode_answer(
            "[]" if exclusion_reason(row) else checkers.normalize_private_test_cases(row["private_test_cases"])
        )
        for row in raw_rows
    ]


def build_split_files(
    raw_rows: list[dict], verdicts: dict[str, dict], perturbations: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Join raw rows with gold verdicts and perturbations into (val_rows, test_rows)."""
    eligible = [row for row in raw_rows if exclusion_reason(row) is None]
    missing = [r["question_id"] for r in eligible if r["question_id"] not in verdicts]
    if missing:
        raise SystemExit(
            f"{len(missing)} problem(s) have no gold_generation.py verdict yet "
            f"(e.g. {missing[:5]}). Run gold_generation.py first (it will skip "
            "anything already in gold_responses.jsonl, so this is cheap if most "
            "problems are already covered)."
        )

    assignment = _split_sources(eligible, verdicts, perturbations)
    raw_by_id = {r["question_id"]: r for r in eligible}
    idx_by_id = {r["question_id"]: i for i, r in enumerate(raw_rows)}

    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for row in eligible:
        qid = row["question_id"]
        finalized = _finalize_row(
            row,
            gold_response=verdicts[qid]["gold_response"],
            is_perturbed=False,
            private_test_case_idx=idx_by_id[qid],
        )
        (val_rows if assignment[qid] == "val" else test_rows).append(finalized)

    for qid, perturbation in perturbations.items():
        twin = raw_by_id.get(qid)
        if twin is None and any(r["question_id"] == qid and exclusion_reason(r) for r in raw_rows):
            continue
        if twin is None:
            print(
                f"[splits] WARNING: perturbation for {qid} has no matching "
                "lcb_native row; skipping."
            )
            continue
        split = assignment[qid]
        perturbed_row = checkers.build_perturbed_row(
            twin, perturbation["flipped_test_case_index"], perturbation["wrong_output"]
        )
        if (
            perturbed_row["true_output_for_flipped_case"]
            != perturbation["true_output_for_flipped_case"]
        ):
            raise ValueError(f"{qid}: frozen perturbation no longer matches the source output")
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
        "--test-out", type=str, default=str(DATA_DIR / ".hidden" / "test_tasks.jsonl")
    )
    parser.add_argument(
        "--private-test-cases-out",
        type=str,
        default=str(DATA_DIR / "private_test_cases.txt"),
    )
    parser.add_argument("--force", action="store_true", help="Explicitly replace an existing secret test split.")
    parser.add_argument("--source-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, help="Write all versioned artifacts below this new directory")
    args = parser.parse_args()
    if args.output_dir:
        args.val_out = str(args.output_dir / "validation_tasks.jsonl")
        args.test_out = str(args.output_dir / ".hidden/test_tasks.jsonl")
        args.private_test_cases_out = str(args.output_dir / "private_test_cases.txt")
    if Path(args.test_out).exists() and not args.force:
        parser.error("test split already exists; use a new output path or --force")

    from gold_generation import SOURCE_FILES
    source_files = [args.source_dir / name for name in SOURCE_FILES]
    for path in [*source_files, Path(args.gold), Path(args.perturbed)]:
        if not path.is_file() or not path.stat().st_size:
            raise SystemExit(f"Missing or empty build input: {path}")
    raw_rows = [row for path in source_files for row in checkers.read_jsonl(path)]
    verdicts = load_frozen_verdicts(args.gold)
    perturbations = load_frozen_perturbations(args.perturbed)
    if not verdicts:
        raise SystemExit(
            f"No verdicts found at {args.gold} -- run gold_generation.py first."
        )

    val_rows, test_rows = build_split_files(raw_rows, verdicts, perturbations)

    eligible = [row for row in raw_rows if exclusion_reason(row) is None]
    n_solved = sum(bool(verdicts[row["question_id"]]["verified_solvable"]) for row in eligible)
    print(
        f"[splits] {len(raw_rows)} lcb_native rows ({n_solved} solved, "
        f"{len(eligible) - n_solved} unsolved, {len(raw_rows) - len(eligible)} excluded)"
    )
    n_val_perturbed = sum(r["gold_response"] == checkers.encode_answer(pb.ABSTENTION_MARKER) for r in val_rows)
    n_test_perturbed = sum(r["gold_response"] == checkers.encode_answer(pb.ABSTENTION_MARKER) for r in test_rows)
    print(f"[splits] val: {len(val_rows)} rows ({n_val_perturbed} perturbed)")
    print(f"[splits] test: {len(test_rows)} rows ({n_test_perturbed} perturbed)")

    root = Path(args.val_out).resolve().parent
    val_out_relative = Path(args.val_out).resolve().relative_to(root)
    test_out_relative = Path(args.test_out).resolve().relative_to(root)
    private_test_cases_out_relative = Path(args.private_test_cases_out).resolve().relative_to(root)
    with staged_build(root) as stage:
        checkers.write_jsonl((stage / val_out_relative), val_rows)
        checkers.write_jsonl((stage / test_out_relative), test_rows)
        checkers.write_private_test_cases(
            (stage / private_test_cases_out_relative), build_private_test_cases_blobs(raw_rows)
        )
        deobfuscated = stage / "validation_tasks_de_obfuscated.jsonl"
        decoded = [{**row, "gold_response": checkers.decode_answer(row["gold_response"]) if row["gold_response"] else None} for row in val_rows]
        checkers.write_jsonl(deobfuscated, decoded)
        exclusions = stage / ".hidden/source_exclusions.json"
        exclusions.parent.mkdir(parents=True, exist_ok=True)
        exclusions.write_text(json.dumps({"policy_version": POLICY_VERSION, "excluded": [
            {"question_id": row["question_id"], "private_test_case_idx": i, "reason": exclusion_reason(row)}
            for i, row in enumerate(raw_rows) if exclusion_reason(row)
        ]}, indent=2) + "\n")
        write_manifest(stage, [exclusions, (stage / val_out_relative), (stage / test_out_relative), (stage / private_test_cases_out_relative), deobfuscated], val_rows, test_rows,
                       seed=SEED, inputs={str(path): sha256(path) for path in [*source_files, Path(args.gold), Path(args.perturbed), Path(__file__).with_name("source_policy.py"), Path(__file__)]})
    print(
        f"[splits] wrote {args.val_out}, {args.test_out}, and "
        f"{args.private_test_cases_out}"
    )


if __name__ == "__main__":
    main()
