"""Build the clinician-chat benchmark's validation/secret-test split.

Rows come from HealthBench Professional (openai/healthbench-professional on the
Hugging Face Hub, file healthbench_professional_eval.jsonl, pinned by revision and
sha256): 525 physician-written chats, each with physician-written rubric criteria
and a physician's own answer. Only the 426 English rows are used (see
language_filter.py and language_labels.json).

The validation set is 86 rows (~20%), stratified by specialty: each specialty gets
its share by largest remainder (at least one test row each), and among 20,000
fixed-seed draws with those counts the one whose use case, type, difficulty and
single/multi-turn mix is closest to the whole set's is kept. The rest is the secret
test set (340 rows).

    validation_tasks.jsonl   .hidden/test_tasks.jsonl   .hidden/split_manifest.json

Each row keeps "id", "messages" (the chat, ending in a user turn), "rubric_items"
([{"criterion_text", "points"}]), "physician_response", and the dataset's labels
"use_case", "type", "difficulty" and "specialty" for analysis. The agent's copy of
validation (rl/agent_validation.py) drops the labels and renames the ids.

judge.py's grader prompt (GRADER_TEMPLATE, verbatim), scoring and length
adjustment are from OpenAI's MIT-licensed simple-evals reference implementation
(healthbench_eval.py at commit 652c89d: GRADER_TEMPLATE, calculate_score,
calculate_length_adjusted_score) and the benchmark paper (Section 4.1: center 2,000
characters, 2.94e-5 per character). This credit lives here, not in judge.py, because
agents get a copy of judge.py and must not learn the benchmark's name.

Usage: python make_splits.py --data /path/to/healthbench_professional_eval.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import checkers
from language_filter import is_english
from split_contract import staged_build, write_manifest

DATA_DIR = Path(__file__).resolve().parent
DATASET = "openai/healthbench-professional"
REVISION = "349962fd46dd02343a0d8a606491baf59154ea1a"
DATA_SHA256 = "d44b08e6e952e04c945e2c406f02533d9e7a989a84e35820ee7efdff20c9e4e2"
SEED = 20261111
N_VALIDATION = 86
DRAWS = 20_000
LABELS = ("use_case", "type", "difficulty", "specialty")


def _row(raw: dict) -> dict:
    return {
        "id": raw["id"],
        "messages": raw["conversation"]["messages"],
        "rubric_items": raw["rubric_items"],
        "physician_response": raw["physician_response"],
        **{name: raw[name] for name in LABELS},
    }


def _features(row: dict) -> list[tuple[str, object]]:
    return [
        ("use_case", row["use_case"]),
        ("type", row["type"]),
        ("difficulty", row["difficulty"]),
        ("multi_turn", len(row["messages"]) > 1),
    ]


def specialty_counts(rows: list[dict], n_val: int) -> dict[str, int]:
    """Validation rows per specialty: largest-remainder shares, one test row kept."""
    by_specialty = collections.Counter(r["specialty"] for r in rows)
    share = {s: n * n_val / len(rows) for s, n in by_specialty.items()}
    counts = {s: min(int(q), by_specialty[s] - 1) for s, q in share.items()}
    order = sorted(share, key=lambda s: share[s] - int(share[s]), reverse=True)
    for s in order[: n_val - sum(counts.values())]:
        counts[s] += 1
    assert sum(counts.values()) == n_val
    return counts


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The stratified validation sample and the remaining test rows, by id order."""
    rows = sorted(rows, key=lambda r: r["id"])
    by_specialty: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_specialty[row["specialty"]].append(row)
    counts = specialty_counts(rows, N_VALIDATION)
    target = collections.Counter(f for r in rows for f in _features(r))

    def distance(sample: list[dict]) -> float:
        got = collections.Counter(f for r in sample for f in _features(r))
        return sum((got[f] - target[f] * N_VALIDATION / len(rows)) ** 2 for f in target)

    rng = random.Random(SEED)
    best = None
    for _ in range(DRAWS):
        sample = [
            r
            for s, group in sorted(by_specialty.items())
            for r in rng.sample(group, counts[s])
        ]
        d = distance(sample)
        if best is None or d < best[0]:
            best = (d, sample)
    assert best is not None
    chosen = {r["id"] for r in best[1]}
    return (
        [r for r in rows if r["id"] in chosen],
        [r for r in rows if r["id"] not in chosen],
    )


def main() -> None:
    """Check the pinned file and labels, split, and publish the build."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing secret test split.",
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (out / ".hidden/test_tasks.jsonl").exists() and not args.force:
        parser.error("test split already exists; use a new --output-dir or --force")
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise SystemExit(f"{args.data} has sha256 {digest}, not {DATA_SHA256}")
    raw = checkers.read_jsonl(args.data)
    record = json.loads((DATA_DIR / "language_labels.json").read_text())
    english = {i for i, label in record["labels"].items() if is_english(label)}
    if english != set(record["english_ids"]) or not english <= {r["id"] for r in raw}:
        raise SystemExit("language_labels.json disagrees with its own rule or the data")
    rows = [_row(r) for r in raw if r["id"] in english]
    validation, test = split_rows(rows)
    print(
        f"[splits] {len(raw)} rows, {len(rows)} English; "
        f"validation {len(validation)}, test {len(test)}"
    )
    with staged_build(out) as stage:
        checkers.write_jsonl(stage / "validation_tasks.jsonl", validation)
        checkers.write_jsonl(stage / ".hidden/test_tasks.jsonl", test)
        write_manifest(
            stage,
            [stage / "validation_tasks.jsonl", stage / ".hidden/test_tasks.jsonl"],
            validation,
            test,
            seed=SEED,
            inputs={
                "dataset": DATASET,
                "revision": REVISION,
                "sha256": DATA_SHA256,
                "english_rows": len(rows),
                "n_validation": N_VALIDATION,
            },
        )
    print(f"[splits] wrote {out}")


if __name__ == "__main__":
    main()
