"""Build NormAd’s 10/90 validation/secret-test split by stable situation hashing.

Drop neutral rows with empty explanations before splitting and obfuscate gold
labels in both outputs.
Usage: python make_splits.py [--val-out PATH] [--test-out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import checkers
import prompt_building as pb
from datasets import load_dataset
from split_contract import staged_build, write_manifest

DATA_DIR = Path(__file__).resolve().parent
SEED = 20260706
VAL_FRACTION = 0.1
DATASET_NAME = "akhilayerukola/NormAd"


def load_normad_rows(revision: str) -> list[dict]:
    """Load and lightly normalize every row of the NormAd "train" split."""
    ds = load_dataset(DATASET_NAME, split="train", revision=revision)
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
        "gold_label": checkers.encode_answer(row["gold"]),
    }


def build_split_files(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Filter, fixed-seed shuffle, and split rows into (val_rows, test_rows)."""
    if any(row["gold"] not in checkers.LABELS for row in rows):
        raise ValueError("NormAd contains an unknown gold label")
    filtered = [
        r for r in rows if not (r["gold"] == "neutral" and r["explanation_is_none"])
    ]

    val_ids = {
        r["id"]
        for r in filtered
        if int(
            hashlib.sha256(f"{SEED}:{r['situation'].strip()}".encode()).hexdigest(), 16
        )
        / 2**256
        < VAL_FRACTION
    }

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
        "--test-out", type=str, default=str(DATA_DIR / ".hidden" / "test_tasks.jsonl")
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing secret test split.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--revision",
        default="main",
        help=(
            "HF revision; resolved to an immutable commit and recorded "
            "in the manifest"
        ),
    )
    args = parser.parse_args()
    if args.output_dir:
        args.val_out = str(args.output_dir / "validation_tasks.jsonl")
        args.test_out = str(args.output_dir / ".hidden/test_tasks.jsonl")
    if Path(args.test_out).exists() and not args.force:
        parser.error("test split already exists; use a new output path or --force")

    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(DATASET_NAME, revision=args.revision).sha
    print(f"[splits] {DATASET_NAME}@{revision}")
    rows = load_normad_rows(revision)
    n_dropped = sum(
        1 for r in rows if r["gold"] == "neutral" and r["explanation_is_none"]
    )
    print(
        f"[splits] {len(rows)} NormAd rows loaded; "
        f"dropping {n_dropped} neutral rows with no Explanation"
    )

    val_rows, test_rows = build_split_files(rows)
    print(f"[splits] val: {len(val_rows)} rows, test: {len(test_rows)} rows")

    root = Path(args.val_out).resolve().parent
    val_out_relative = Path(args.val_out).resolve().relative_to(root)
    test_out_relative = Path(args.test_out).resolve().relative_to(root)
    with staged_build(root) as stage:
        checkers.write_jsonl((stage / val_out_relative), val_rows)
        checkers.write_jsonl((stage / test_out_relative), test_rows)
        deobfuscated = stage / "validation_tasks_de_obfuscated.jsonl"
        checkers.write_jsonl(
            deobfuscated,
            [
                {**row, "gold_label": checkers.decode_answer(row["gold_label"])}
                for row in val_rows
            ],
        )
        write_manifest(
            stage,
            [(stage / val_out_relative), (stage / test_out_relative), deobfuscated],
            val_rows,
            test_rows,
            seed=SEED,
            inputs={
                "dataset_revision": revision,
                DATASET_NAME: hashlib.sha256(
                    json.dumps(rows, sort_keys=True).encode()
                ).hexdigest(),
            },
        )
    print(f"[splits] wrote {args.val_out} and {args.test_out}")


if __name__ == "__main__":
    main()
