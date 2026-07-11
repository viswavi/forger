"""Build validation_tasks.jsonl / test_tasks.jsonl for the NormAd benchmark.

NormAd (akhilayerukola/NormAd on HF) ships a single "train" split (2633 rows) -- no
built-in val/test split, the same situation as LiveCodeBench. This mirrors
post_data_bench/coding's make_splits.py: filter, fixed-seed shuffle, 10/90 (val/test)
split, write two jsonl files with the gold label obfuscated. Validation is the small
(10%) inspectable slice; test is the large (90%) held-out slice used for final
comparison.

Filter (preserved verbatim from the original post_train_bench_evaluate_normad.py's
"filtered" setting): drop rows where Gold Label == "neutral" AND Explanation is None --
these are the rows most likely to be mislabeled. The filter runs *before* the split, so
both validation_tasks.jsonl and test_tasks.jsonl are already filtered -- a downstream
user never needs to re-apply this logic.

Usage:
    python make_splits.py [--val-out PATH] [--test-out PATH]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import checkers
import prompt_building as pb
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent
SEED = 20260706
VAL_FRACTION = 0.1
DATASET_NAME = "akhilayerukola/NormAd"


def load_normad_rows() -> list[dict]:
    """Load and lightly normalize every row of the NormAd "train" split."""
    ds = load_dataset(DATASET_NAME, split="train")
    rows = []
    for r in ds:
        country = r["Country"].replace("_", " ").title()
        rows.append(
            {
                "id": r["ID"],
                "country": country,
                "axis": r["Axis"],
                "subaxis": r["Subaxis"],
                "value": r["Value"],
                "rule_of_thumb": r["Rule-of-Thumb"],
                "situation": r["Story"],
                "gold": (r["Gold Label"] or "").strip().lower(),
                "explanation_is_none": checkers.is_none_explanation(r["Explanation"]),
            }
        )
    return rows


def _finalize_row(row: dict) -> dict:
    return {
        "id": row["id"],
        "country": row["country"],
        "axis": row["axis"],
        "subaxis": row["subaxis"],
        "value": row["value"],
        "rule_of_thumb": row["rule_of_thumb"],
        "situation": row["situation"],
        "task_text": pb.build_user_content(row["country"], row["situation"]),
        "explanation_is_none": row["explanation_is_none"],
        "gold_label": checkers.encode_answer(row["gold"]),
    }


def build_split_files(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Filter, fixed-seed shuffle, and split rows into (val_rows, test_rows)."""
    filtered = [
        r for r in rows if not (r["gold"] == "neutral" and r["explanation_is_none"])
    ]

    ids = [r["id"] for r in filtered]
    random.Random(SEED).shuffle(ids)
    n_val = round(len(ids) * VAL_FRACTION)
    val_ids = set(ids[:n_val])

    val_rows: list[dict] = []
    test_rows: list[dict] = []
    for row in filtered:
        finalized = _finalize_row(row)
        (val_rows if row["id"] in val_ids else test_rows).append(finalized)

    # Distinct seed from the split assignment, purely so possible in-file ordering isn't
    # tied to the shuffle above -- doesn't affect grading, just tidiness for anyone
    # skimming the file.
    random.Random(SEED + 1).shuffle(val_rows)
    random.Random(SEED + 2).shuffle(test_rows)
    return val_rows, test_rows


def main() -> None:
    """Build and write validation_tasks.jsonl / test_tasks.jsonl."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-out", type=str, default=str(DATA_DIR / "validation_tasks.jsonl")
    )
    parser.add_argument(
        "--test-out", type=str, default=str(DATA_DIR / "test_tasks.jsonl")
    )
    args = parser.parse_args()

    rows = load_normad_rows()
    n_dropped = sum(
        1 for r in rows if r["gold"] == "neutral" and r["explanation_is_none"]
    )
    print(
        f"[splits] {len(rows)} NormAd rows loaded; "
        f"dropping {n_dropped} neutral rows with no Explanation"
    )

    val_rows, test_rows = build_split_files(rows)
    print(f"[splits] val: {len(val_rows)} rows, test: {len(test_rows)} rows")

    checkers.write_jsonl(args.val_out, val_rows)
    checkers.write_jsonl(args.test_out, test_rows)
    print(f"[splits] wrote {args.val_out} and {args.test_out}")


if __name__ == "__main__":
    main()
