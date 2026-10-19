"""Fresh independent panel trials composed from generic Forger operations."""

import json
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from pyforger.alter.sample import sample
from pyforger.filter.success_rate import by_success_rate
from pyforger.inspect.success_rate import success_rates


def validate_panel(
    models: Sequence[str],
    *,
    attempts: int = 5,
    concurrency: int | None = None,
    min_rate: float = 0.2,
    max_rate: float = 0.8,
    min_valid: int | None = None,
) -> None:
    """Validate panel configuration without executing trials or creating files."""
    if (
        isinstance(models, (str, bytes))
        or not models
        or any(not isinstance(model, str) or not model.strip() for model in models)
    ):
        raise ValueError("Require nonempty model names")
    if len(set(models)) != len(models):
        raise ValueError("Model names must be unique")
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    workers = 1 if concurrency is None else concurrency
    if type(workers) is not int or workers < 1:
        raise ValueError("concurrency must be a positive integer")
    by_success_rate.no_cache(
        [], min_rate=min_rate, max_rate=max_rate, min_valid=min_valid
    )


def evaluate_panel(
    models: Sequence[str],
    evaluate: Callable[[str, str], dict[str, Any]],
    *,
    attempts: int = 5,
    concurrency: int | None = None,
    min_rate: float = 0.2,
    max_rate: float = 0.8,
    min_valid: int | None = None,
) -> dict[str, Any]:
    """Evaluate fresh uniquely identified trials and summarize each model.

    Callback results contain a boolean or null ``passed`` field. Null grades
    are excluded from rates; false grades count as failures. By default every
    requested trial must have a valid grade. Set min_valid explicitly to allow
    exclusions. Each model must meet that minimum and the inclusive rate bounds.
    With concurrency=None all model-by-attempt trials may run simultaneously.
    """
    validate_panel(
        models,
        attempts=attempts,
        concurrency=concurrency,
        min_rate=min_rate,
        max_rate=max_rate,
        min_valid=min_valid,
    )
    workers = len(models) * attempts if concurrency is None else concurrency
    jobs = [
        {"model": model, "trial_id": f"model-{index}-trial-{trial}"}
        for index, model in enumerate(models)
        for trial in range(attempts)
    ]

    by_identity = {job["trial_id"]: job["model"] for job in jobs}

    def invoke(item: dict[str, Any], identity: str) -> dict[str, Any]:
        result = evaluate(by_identity[identity], identity)
        if not isinstance(result, dict) or "passed" not in result:
            raise ValueError("Panel callback must return a result with passed")
        return result

    # no_cache still hashes callables; partial keeps opaque callback state out.
    sampled = sample.no_cache(
        [{}], partial(invoke), sample_ids=list(by_identity), concurrency=workers
    )
    trials = [
        {
            "model": by_identity[entry["sample_id"]],
            "trial_id": entry["sample_id"],
            "result": entry["value"],
        }
        for entry in sampled[0]["samples"]
    ]
    rows: list[dict[str, Any]] = [{"evaluations": {model: [] for model in models}}]
    for trial in trials:
        rows[0]["evaluations"][trial["model"]].append(trial["result"]["passed"])
    counts = success_rates.no_cache(rows)[0]
    accepted = bool(
        by_success_rate.no_cache(
            rows, min_rate=min_rate, max_rate=max_rate, min_valid=min_valid
        )
    )
    report = {"accepted": accepted, "models": counts, "trials": trials}
    normalized: dict[str, Any] = json.loads(json.dumps(report, allow_nan=False))
    return normalized
