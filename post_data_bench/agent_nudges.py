"""Messages the launcher may send an agent that finished early.

The launcher calls these after the agent's session ends on its own. Each
returns None when there is nothing to say, so the session stays finished.
"""

from __future__ import annotations

from pathlib import Path

# The final RL run draws this many prompts (25 rollouts x 128 prompts), from
# the 90% of rows the harness keeps for training.
RL_PROMPT_DRAWS = 25 * 128
TRAIN_SHARE = 0.9
# Adding rows means regenerating data and retraining SFT; with less time
# left, a half-finished attempt could break a valid submission.
MIN_SECONDS_LEFT = 30 * 60


def count_rows(path: str | Path) -> int | None:
    """Non-blank lines in a JSONL dataset; None when the file does not exist."""
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return None


def low_rows_message(path: str | Path, minimum: int, seconds_left: float) -> str | None:
    """A note for an agent that finished with fewer than ``minimum`` rows.

    e.g. 321 rows with ``minimum=1000``: the note says RL would see each
    training prompt about 11 times, and that finishing again keeps the
    dataset as it is. None when ``minimum`` is 0, the dataset has enough rows,
    or under 30 minutes are left (too little to add rows and retrain SFT).
    """
    if minimum <= 0 or seconds_left < MIN_SECONDS_LEFT:
        return None
    rows = count_rows(path)
    if rows is not None and rows >= minimum:
        return None
    hours = seconds_left / 3600
    left = (
        f"about {hours:.1f} hours"
        if hours >= 1
        else f"about {seconds_left / 60:.0f} minutes"
    )
    if rows is None:
        found = f"There is no dataset at {path} yet."
    else:
        visits = RL_PROMPT_DRAWS / max(1.0, rows * TRAIN_SHARE)
        found = (
            f"Your dataset at {path} has {rows} rows. The final RL run draws "
            f"{RL_PROMPT_DRAWS:,} prompts from your training rows, so each would be "
            f"seen about {visits:.0f} times; about 1,600 rows is a good target, and "
            f"at least {minimum:,} is what we expect realistically."
        )
    return (
        f"{found} You have {left} left. If you want to add rows, continue now: "
        "build the larger dataset in a new file, run SFT on it, and only when "
        "that trial has finished copy the file over your dataset and point "
        "submission.json at the new trial (the submitted checkpoint must be "
        "trained on exactly the submitted rows, so keep the current pair intact "
        "until then). If you are satisfied with the dataset as it is, simply "
        "finish again; you will not be asked a second time."
    )
