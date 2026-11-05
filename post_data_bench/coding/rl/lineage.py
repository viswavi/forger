"""Which data trained a checkpoint: its RL runs, back to where they began.

They begin from the base model or, on Tinker, from an SFT trial.

The final submission allows any number of experiments, but the selected checkpoint's
own training (every RL run on the way to it, and its SFT trial if any) must use at
most max_rows() rows per run (2,000; 800 for algorithmic), all with prompts from the
final dataset. Each RL run records the checkpoint it started from in `lineage.json`,
inside its compiled data directory (`<run>.data/`); SFT and RL receipts name that
directory.
"""
from __future__ import annotations

import json
from pathlib import Path

from .rubric_data import max_rows, prompt_id, read_rows  # noqa: ABS101

LINEAGE = "lineage.json"


def record_start(data_directory, checkpoint):
    """Record, in an RL run's compiled data, the checkpoint the run starts from."""
    (Path(data_directory) / LINEAGE).write_text(
        json.dumps({"init_checkpoint": str(Path(checkpoint).resolve())}, indent=2)
        + "\n"
    )


def stages(checkpoint):
    """The training runs behind `checkpoint`, newest first.

    e.g. [{"stage": "rl", ...rl_trial_2}, {"stage": "rl", ...rl_trial_1},
    {"stage": "sft", ...sft_trial_3}]; empty for the base model.
    """
    found = []
    checkpoint = Path(checkpoint).resolve()
    while len(found) < 64:
        rl_receipt = checkpoint / "rl_submission.json"
        sft_receipt = checkpoint / "sft_submission.json"
        if rl_receipt.is_file():
            data = Path(json.loads(rl_receipt.read_text())["data_directory"])
            found.append(
                {
                    "stage": "rl",
                    "checkpoint": str(checkpoint),
                    "data_directory": str(data),
                }
            )
            if not (data / LINEAGE).is_file():
                raise ValueError(
                    f"{checkpoint}: its RL run did not record its starting checkpoint "
                    f"({data / LINEAGE}); it predates this check"
                )
            start = json.loads((data / LINEAGE).read_text())["init_checkpoint"]
            checkpoint = Path(start).resolve()
        elif sft_receipt.is_file():
            data = Path(json.loads(sft_receipt.read_text())["data_directory"])
            found.append(
                {
                    "stage": "sft",
                    "checkpoint": str(checkpoint),
                    "data_directory": str(data),
                }
            )
            return found
        else:
            return found  # the base model
    raise ValueError(f"{checkpoint}: more than 64 training runs in its lineage")


def check(checkpoint, final_data):
    """Raise unless each run behind `checkpoint` used <= max_rows() final prompts."""
    allowed = {prompt_id(row) for row in read_rows(final_data)}
    runs = stages(checkpoint)
    for run in runs:
        rows = read_rows(Path(run["data_directory"]) / "submitted.jsonl")
        name = f"{run['stage'].upper()} run for {run['checkpoint']}"
        if len(rows) > max_rows():
            raise ValueError(f"{name} used {len(rows)} rows; the limit is {max_rows()}")
        missing = {prompt_id(row) for row in rows} - allowed
        if missing:
            raise ValueError(
                f"{name} used {len(missing)} prompts not in the final dataset; "
                "the submitted checkpoint's training data must come from its prompts"
            )
    return runs
