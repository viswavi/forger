"""Messages the launcher may send an agent that finished early.

The launcher calls these after the agent's session ends on its own. Each
returns None when there is nothing to say, so the session stays finished.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Both RL recipes draw this many prompts (80 steps x 32 prompts), from the
# 90% of rows the harness keeps for training.
RL_PROMPT_DRAWS = 80 * 32
TRAIN_SHARE = 0.9
# Adding rows means regenerating data and retraining; with less time left, a
# half-finished attempt could break a valid submission. The default leaves
# time for a 5-hour RL run; tasks with longer RL pass their own.
MIN_SECONDS_LEFT = 7 * 60 * 60


def count_rows(path: str | Path) -> int | None:
    """Non-blank lines in a JSONL dataset; None when the file does not exist."""
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return None


def new_prompt_count(path: str | Path, known: set[str]) -> int | None:
    """Rows whose prompt is not in ``known`` (rl.rubric_data.prompt_id hashes).

    e.g. a dataset of the 472 starting rows plus 150 new ones counts 150; None
    when the file does not exist. Unreadable lines count as new.
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
    except FileNotFoundError:
        return None
    new = 0
    for line in lines:
        try:
            prompt = json.loads(line)["prompt"]
        except (ValueError, KeyError, TypeError):
            new += 1
            continue
        digest = hashlib.sha256(json.dumps(prompt, sort_keys=True).encode())
        new += digest.hexdigest() not in known
    return new


def low_rows_message(
    path: str | Path,
    minimum: int,
    seconds_left: float,
    target: int = 1250,
    min_seconds_left: float = MIN_SECONDS_LEFT,
    prompt_draws: int = RL_PROMPT_DRAWS,
    starting_prompts: set[str] | None = None,
) -> str | None:
    """A note for an agent that finished with fewer than ``minimum`` rows.

    e.g. 321 rows with ``minimum=1000``: the note says RL would see each
    training prompt about 11 times, and that finishing again keeps the
    dataset as it is. None when ``minimum`` is 0, the dataset has enough rows,
    or under ``min_seconds_left`` are left (leave time for training).
    ``prompt_draws`` is the default RL schedule's steps x prompts per step.
    With ``starting_prompts`` (a release's starting dataset), ``minimum`` and
    ``target`` count only new rows, whose prompts are not among them.
    """
    if minimum <= 0 or seconds_left < min_seconds_left:
        return None
    rows = count_rows(path)
    if starting_prompts is not None:
        new = new_prompt_count(path, starting_prompts)
        if new is not None and new >= minimum:
            return None
    elif rows is not None and rows >= minimum:
        return None
    hours = seconds_left / 3600
    left = (
        f"about {hours:.1f} hours"
        if hours >= 1
        else f"about {seconds_left / 60:.0f} minutes"
    )
    if rows is None:
        found = f"There is no dataset at {path} yet."
    elif starting_prompts is not None:
        found = (
            f"Your dataset at {path} has {rows} rows, {new} of them new (not in "
            f"the earlier dataset the starting model trained on); about {target:,} "
            f"new rows is a good target, and at least {minimum:,} is what we "
            "expect realistically."
        )
    else:
        visits = prompt_draws / max(1.0, rows * TRAIN_SHARE)
        found = (
            f"Your dataset at {path} has {rows} rows. The default RL recipe draws "
            f"{prompt_draws:,} prompts from your training rows, so each would be "
            f"seen about {visits:.0f} times; about {target:,} rows is a good target, "
            "and "
            f"at least {minimum:,} is what we expect realistically."
        )
    return (
        f"{found} You have {left} left. If you want to add rows, continue now: "
        "build a new dataset version and run additional training experiments if "
        "you expect them to help. Keep submission.json pointing at your current "
        "completed RL checkpoint until a better RL checkpoint is ready. "
        "You need not retrain merely to match the latest data file. If you are "
        "satisfied, finish again; you will not be asked a second time."
    )
