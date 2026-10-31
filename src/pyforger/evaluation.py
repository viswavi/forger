"""Fresh independent panel trials composed from generic Forger operations."""

import json
from collections.abc import Callable, Sequence
from functools import partial
from threading import Lock
from typing import Any

from pyforger.alter.sample import sample
from pyforger.filter.success_rate import by_success_rate
from pyforger.inspect.success_rate import success_rates
from pyforger.utils import run_concurrently


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
    replacement_predicate: Callable[[dict[str, Any]], bool] | None = None,
    max_replacements: int = 0,
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
    if type(max_replacements) is not int or max_replacements < 0:
        raise ValueError("max_replacements must be a nonnegative integer")
    if max_replacements and replacement_predicate is None:
        raise ValueError("Replacements require a predicate")
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
        if result["passed"] is not None and type(result["passed"]) is not bool:
            raise ValueError("Panel callback passed must be a boolean or null")
        return json.loads(json.dumps(result, allow_nan=False))

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
    history = [
        {
            "model": trial["model"],
            "trial_id": trial["trial_id"],
            "execution_id": trial["trial_id"],
            "result": trial["result"],
        }
        for trial in trials
    ]
    # Deterministic waves allocate each model's budget in canonical slot order.
    used = dict.fromkeys(models, 0)
    while max_replacements:
        selected = []
        for index, trial in enumerate(trials):
            if used[trial["model"]] >= max_replacements:
                continue
            retry = replacement_predicate(trial["result"])
            if type(retry) is not bool:
                raise ValueError("Replacement predicate must return bool")
            if retry:
                used[trial["model"]] += 1
                identity = f"{trial["trial_id"]}-replacement-{used[trial["model"]]}"
                by_identity[identity] = trial["model"]
                selected.append((index, identity))
        if not selected:
            break
        completed = {}
        completion_lock = Lock()

        def replacement(identity: str) -> dict[str, Any]:
            result = invoke({}, identity)
            with completion_lock:
                completed[identity] = result
            return result

        try:
            replacements = run_concurrently(
                [partial(replacement, identity) for _, identity in selected],
                concurrency=workers,
            )
        except Exception as exc:
            diagnostics = dict(getattr(exc, "diagnostics", {}) or {})
            with completion_lock:
                diagnostics["panel_execution_history"] = history + [
                    {
                        "model": trials[index]["model"],
                        "trial_id": trials[index]["trial_id"],
                        "execution_id": identity,
                        "result": completed[identity],
                    }
                    for index, identity in selected
                    if identity in completed
                ]
            exc.diagnostics = diagnostics
            raise
        for (index, identity), result in zip(selected, replacements, strict=True):
            previous = trials[index]
            history.append(
                {
                    "model": previous["model"],
                    "trial_id": previous["trial_id"],
                    "execution_id": identity,
                    "result": result,
                }
            )
            trials[index] = {**previous, "result": result}
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
    if max_replacements:
        report["execution_history"] = history
    normalized: dict[str, Any] = json.loads(json.dumps(report, allow_nan=False))
    return normalized
