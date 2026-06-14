"""Dataset loaders and seeded splits for the ICL data-quality eval.

Loads HumanEval (validation; HF ``openai_humaneval``, ungated) and the Forger
TRL-format JSONL (candidate demos), and provides seeded subsample/holdout
helpers so reruns are reproducible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def load_humaneval() -> list[dict]:
    """Load all 164 HumanEval problems with fields ``task_id``, ``prompt``,
    ``canonical_solution``, ``test``, ``entry_point``."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - datasets is a hard dep
        raise ImportError(
            "load_humaneval requires the 'datasets' library, which is a "
            "core pyforger dependency. Install with: pip install datasets"
        ) from exc

    # Use the fully-qualified ``namespace/name`` repo id. Newer
    # huggingface_hub rejects the bare canonical name "openai_humaneval".
    ds = load_dataset("openai/openai_humaneval", split="test")
    return [
        {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        }
        for row in ds
    ]


def load_forger_jsonl(path: str | Path) -> list[dict]:
    """Load the Forger TRL-format JSONL into ``[{"prompt", "solution"}, ...]``.

    Each line is ``{"messages": [user, assistant]}``; the user content
    becomes ``prompt`` and the assistant content becomes ``solution``.
    Malformed lines raise :class:`ValueError` with the 1-based line number.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Forger JSONL not found: {p}")

    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{p}:{lineno}: invalid JSON ({exc.msg})"
                ) from exc
            messages = obj.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(
                    f"{p}:{lineno}: expected 'messages' to be a list of "
                    f"length >= 2, got {type(messages).__name__}"
                )
            user, assistant = messages[0], messages[1]
            if user.get("role") != "user" or assistant.get("role") != "assistant":
                raise ValueError(
                    f"{p}:{lineno}: expected first two messages to be "
                    f"user then assistant, got "
                    f"{user.get('role')!r}/{assistant.get('role')!r}"
                )
            out.append(
                {"prompt": user.get("content", ""), "solution": assistant.get("content", "")}
            )
    return out


def subsample_pool(demos: list[dict], n: int = 20, seed: int = 0) -> list[dict]:
    """Seeded ``n``-example subsample (without replacement) of ``demos``.

    Returns the input unchanged when ``n >= len(demos)``.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n >= len(demos):
        return list(demos)
    rng = random.Random(seed)
    return rng.sample(demos, n)


def holdout_split(data: list[dict], n: int = 30, seed: int = 0) -> list[dict]:
    """Seeded ``n``-example holdout slice of ``data`` (the validation set).

    Distinct from :func:`subsample_pool` so it can later use a different
    policy (e.g. stratified sampling).
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n >= len(data):
        return list(data)
    rng = random.Random(seed)
    return rng.sample(data, n)
