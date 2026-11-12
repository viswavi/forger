"""Which data trained a checkpoint: its RL runs, back to where they began.

They begin from the base model or, on Tinker, from an SFT trial.

The final submission allows any number of experiments, but the selected checkpoint's
own training (every RL run on the way to it, and its SFT trial if any) must use at
most max_rows() rows per run (2,000; 800 for algorithmic), all with prompts from the
final dataset, and on a backend with an RL_STEP_BUDGET at most that many RL steps in
total. Each RL run records the checkpoint it started from in `lineage.json`, inside
its compiled data directory (`<run>.data/`); SFT and RL receipts name that directory.
"""
from __future__ import annotations

import json
from pathlib import Path

from .backends import RL_STEP_BUDGET  # noqa: ABS101
from .rubric_data import max_rows, prompt_id, read_rows  # noqa: ABS101

LINEAGE = "lineage.json"


def record_start(data_directory, checkpoint):
    """Record, in an RL run's compiled data, the checkpoint the run starts from."""
    (Path(data_directory) / LINEAGE).write_text(
        json.dumps({"init_checkpoint": str(Path(checkpoint).resolve())}, indent=2)
        + "\n"
    )


def is_base_snapshot(checkpoint):
    """Whether `checkpoint` is a Hugging Face cache snapshot of a base model.

    e.g. .../hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a72..., where local RL
    starts. It is the only checkpoint allowed to have no receipt.
    """
    checkpoint = Path(checkpoint).resolve()
    return (
        checkpoint.parent.name == "snapshots"
        and checkpoint.parent.parent.name.startswith("models--")
    )


def stages(checkpoint):
    """The training runs behind `checkpoint`, newest first.

    e.g. [{"stage": "rl", ...rl_trial_2}, {"stage": "rl", ...rl_trial_1},
    {"stage": "sft", ...sft_trial_3}]; empty for the base model. Any other
    checkpoint without a receipt, e.g. rl_trial_1/hf_rollout19 with no
    rl_submission.json, raises: its steps and data cannot be counted.
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
        elif is_base_snapshot(checkpoint):
            return found
        else:
            raise ValueError(
                f"{checkpoint} has no rl_submission.json, so the RL steps and data "
                "behind it cannot be counted; only the base model's Hugging Face "
                "snapshot may have none"
            )
    raise ValueError(f"{checkpoint}: more than 64 training runs in its lineage")


def rl_steps(checkpoint):
    """Completed RL steps behind `checkpoint`, summed over every RL run in its lineage.

    e.g. 100 for rl_trial_2/hf_rollout39 when rl_trial_2 continued from
    rl_trial_1/hf_rollout59; 0 for the base model.
    """
    total = 0
    for run in stages(checkpoint):
        if run["stage"] == "rl":
            receipt = Path(run["checkpoint"]) / "rl_submission.json"
            total += int(json.loads(receipt.read_text()).get("completed_steps", 0))
    return total


def check_step_budget(checkpoint):
    """Raise if a checkpoint has more than its backend's RL step budget behind it.

    Fewer steps are allowed, e.g. hf_rollout49 of an 80-step run, or the last
    checkpoint of a run the judge quota stopped early.
    """
    receipt = Path(checkpoint).resolve() / "rl_submission.json"
    if not receipt.is_file():
        return
    budget = RL_STEP_BUDGET.get(json.loads(receipt.read_text()).get("backend"))
    if budget is None:
        return
    steps = rl_steps(checkpoint)
    if steps > budget:
        raise ValueError(
            f"{checkpoint} has {steps} RL steps in total (counting the runs it "
            f"continues from); a submitted checkpoint may have at most {budget}"
        )


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
    check_step_budget(checkpoint)
    return runs
