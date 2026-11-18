"""Prepare a release that starts agents from a post-trained checkpoint.

Two steps, both on the host:

1. stage-model: put the checkpoint into a Hugging Face cache under a local id,
   e.g. local/Qwen3-4B-clinical-rl80, so that snapshot_download(id,
   local_files_only=True) finds it the way it finds Qwen/Qwen3-4B:
     python make_starting_point.py stage-model \
       --checkpoint <bundle>/initial_checkpoint \
       --hf-home <dir> --model-id local/Qwen3-4B-clinical-rl80
   and upload <dir>/hub/models--local--<name> to GCS for PDB_START_MODEL_URI.
2. build: add the checkpoint's validation results and starting_point/ to a
   benchmark build (PDB_BENCHMARK_DIR):
     python make_starting_point.py build --build <build> \
       --model-id local/Qwen3-4B-clinical-rl80 \
       --dataset <run>/rl_trial_1.data/submitted.jsonl --val-dir <val eval dir> \
       --test-before 18.78 --test-after 24.01 --new-rows 2000 --min-new-rows 400

Then run the agent with PDB_LOCAL_BASE_MODEL=<id> and PDB_START_MODEL_URI=<the
uploaded cache entry): orchard_agent_run.sbatch copies it into the run's cache,
and launch_agent.py adds starting_point/prompt.txt to the task prompt, copies the
dataset into the task directory, and shows both models' validation results.

--dataset is the rows the checkpoint trained on (its RL runs' compiled
submitted.jsonl), which can differ from the agent's final file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ORIGINAL = "Qwen__Qwen3-4B"
# The checkpoint's model and tokenizer files; receipts and logs stay behind.
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)

PROMPT = (
    'This is not a fresh start. The student model, {{STUDENT_MODEL}} ("the '
    'starting model" below), is Qwen3-4B after one round of RL post-training: '
    "{steps} RL steps of the recipe under Training and Validation, on an earlier "
    "dataset of {rows:,} rows in the same format as yours, which is in "
    "{{TASK_DIR}}/starting_point/dataset.jsonl.{history} That round changed the "
    "secret test score from {test_before:.2f} (Qwen3-4B) to {test_after:.2f}, and "
    "the validation score from {val_before:.2f} to {val_after:.2f}.{score_note} "
    "Validation results for both models are in {{TASK_DIR}}/model_outputs/: "
    "{original}/val/ before that round and {{BASELINE_MODEL}}/val/ after it. Your "
    "job is to improve this model further. Your RL runs start from it, and your "
    "submitted checkpoint may add at most 80 RL steps to it. The starting model "
    "has already trained on the earlier dataset, so we recommend generating new "
    "data rather than reusing those rows; use the earlier dataset as a reference. "
    "The model's weights are in {{START_MODEL_PATH}}: pass that directory to any "
    "tool that needs the model's files, because {{STUDENT_MODEL}} is not on the "
    "Hugging Face Hub."
)


def digest(path):
    """sha256 of a file, e.g. a build's validation_tasks.jsonl."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage_model(args):
    """Copy the checkpoint into <hf-home>/hub/models--<org>--<name>/snapshots/<rev>."""
    org, name = args.model_id.split("/")
    if org != "local":
        raise SystemExit(
            "use a local/<name> id: the evaluator serves local/* ids from the cache"
        )
    source = Path(args.checkpoint).resolve(strict=True)
    weights = sorted(
        p.name
        for p in source.iterdir()
        if p.name.startswith(("model-", "model.safetensors")) and p.is_file()
    )
    if not weights:
        raise SystemExit(f"no safetensors weights in {source}")
    # The first shard's hash: index files are identical across checkpoints.
    rev = digest(source / weights[0])[:40]
    entry = Path(args.hf_home) / "hub" / f"models--{org}--{name}"
    snapshot = entry / "snapshots" / rev
    if entry.exists():
        raise SystemExit(f"{entry} exists; remove it first to restage")
    snapshot.mkdir(parents=True)
    for file in [*weights, *MODEL_FILES]:
        if (source / file).is_file():
            shutil.copyfile(source / file, snapshot / file)
    (entry / "refs").mkdir()
    (entry / "refs" / "main").write_text(rev)
    print(f"staged {args.model_id} at {snapshot} ({len(weights)} weight files)")


def write(path, text=None, source=None):
    """Write a new file at path.

    Builds are hard-linked copies of a release, so writing into an existing file
    in place would change the release too.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    if source is not None:
        shutil.copyfile(source, path)
    else:
        path.write_text(text)


def build(args):
    """Add the starting model's validation results and starting_point/ to a build."""
    root = Path(args.build).resolve(strict=True)
    tasks_sha = digest(root / "validation_tasks.jsonl")
    if args.base_val_dir:
        # Qwen3-4B re-scored the same way as the checkpoint, e.g. on the same cluster.
        base = json.loads((Path(args.base_val_dir) / "metrics.json").read_text())
        base["model"] = "Qwen/Qwen3-4B"
        base.pop("source", None)  # a host path
        val = root / "model_outputs" / ORIGINAL / "val"
        write(val / "metrics.json", json.dumps(base, indent=2) + "\n")
        write(
            val / "predictions.jsonl",
            source=Path(args.base_val_dir) / "predictions.jsonl",
        )
    before = json.loads(
        (root / "model_outputs" / ORIGINAL / "val" / "metrics.json").read_text()
    )
    after = json.loads((Path(args.val_dir) / "metrics.json").read_text())
    for name, metrics in ((ORIGINAL, before), (args.model_id, after)):
        if metrics.get("split") != "val" or metrics.get("tasks_sha256") != tasks_sha:
            raise SystemExit(
                f"{name}: validation results are for other tasks than this build's"
            )
    if after.get("evaluation_contract") != before.get("evaluation_contract"):
        raise SystemExit(
            "the two validation results use different evaluation contracts"
        )
    if after.get("n_samples") != before.get("n_samples") or after.get(
        "judge"
    ) != before.get("judge"):
        raise SystemExit(
            "validation protocols differ: "
            f"samples {before.get('n_samples')} vs {after.get('n_samples')}, "
            f"judge {before.get('judge')} vs {after.get('judge')}"
        )
    out = root / "model_outputs" / args.model_id.replace("/", "__") / "val"
    after["model"] = args.model_id
    after.pop("source", None)  # a host path
    write(out / "metrics.json", json.dumps(after, indent=2) + "\n")
    write(out / "predictions.jsonl", source=Path(args.val_dir) / "predictions.jsonl")

    rows = [
        line for line in Path(args.dataset).read_text().splitlines() if line.strip()
    ]
    for line in rows:
        row = json.loads(line)
        if (
            set(row) - {"prompt", "rubric", "ground_truth"}
            or "prompt" not in row
            or "rubric" not in row
        ):
            raise SystemExit(
                "the dataset must hold agent-format rows (prompt, rubric, ground_truth)"
            )
    point = root / "starting_point"
    write(point / "dataset.jsonl", "\n".join(rows) + "\n")
    # The task prompt's row target and the low-row nudge count new rows.
    write(
        point / "config.json",
        json.dumps({"new_rows": args.new_rows, "min_new_rows": args.min_new_rows})
        + "\n",
    )
    write(
        point / "prompt.txt",
        PROMPT.format(
            steps=args.steps,
            rows=len(rows),
            history=f" {args.history}" if args.history else "",
            score_note=f" {args.score_note}" if args.score_note else "",
            test_before=args.test_before,
            test_after=args.test_after,
            val_before=before["score"],
            val_after=after["score"],
            original=ORIGINAL,
        )
        + "\n",
    )
    print(
        f"{root}: {len(rows)} starting rows; "
        f"validation {before['score']:.2f} -> {after['score']:.2f}; "
        f"test {args.test_before:.2f} -> {args.test_after:.2f}"
    )


def main():
    """Run stage-model or build."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser(
        "stage-model", help="Put a checkpoint into an HF cache under a local id"
    )
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--hf-home", required=True)
    s.add_argument(
        "--model-id", required=True, help="e.g. local/Qwen3-4B-clinical-rl80"
    )
    s.set_defaults(run=stage_model)
    b = sub.add_parser(
        "build", help="Add starting_point/ and validation results to a build"
    )
    b.add_argument(
        "--build", required=True, help="the benchmark build (PDB_BENCHMARK_DIR)"
    )
    b.add_argument("--model-id", required=True)
    b.add_argument(
        "--dataset", required=True, help="the rows that trained the checkpoint"
    )
    b.add_argument(
        "--val-dir",
        required=True,
        help="its validation metrics.json and predictions.jsonl",
    )
    b.add_argument(
        "--test-before", type=float, required=True, help="Qwen3-4B's secret test score"
    )
    b.add_argument(
        "--test-after",
        type=float,
        required=True,
        help="the checkpoint's secret test score",
    )
    b.add_argument(
        "--new-rows", type=int, required=True, help="recommended new rows, e.g. 200"
    )
    b.add_argument(
        "--min-new-rows",
        type=int,
        required=True,
        help="new rows below which an agent that finishes early is nudged once",
    )
    b.add_argument("--steps", type=int, default=80)
    b.add_argument(
        "--history", default="", help="an extra sentence on how the rows were used"
    )
    b.add_argument(
        "--score-note",
        default="",
        help="sentences after the scores, e.g. why validation fell while test rose",
    )
    b.add_argument(
        "--base-val-dir",
        help="Qwen3-4B validation results to ship instead of the build's, "
        "scored like --val-dir",
    )
    b.set_defaults(run=build)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
