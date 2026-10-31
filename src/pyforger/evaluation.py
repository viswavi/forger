"""Fresh independent panel trials composed from generic Forger operations."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from functools import partial
from threading import Lock
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

from pyforger.alter.sample import sample
from pyforger.filter.success_rate import by_success_rate
from pyforger.inspect.success_rate import success_rates
from pyforger.measurements import Measurement
from pyforger.utils import get_field, run_concurrently


class PanelTrial(BaseModel):
    """One executed panel trial, independent of the caller's result shape."""

    model_config = ConfigDict(frozen=True)

    model: str
    trial_id: str
    result: dict[str, Any]


class PanelReport(BaseModel):
    """Typed panel summary: admission verdict, per-model counts, and trials."""

    model_config = ConfigDict(frozen=True)

    accepted: bool
    models: dict[str, dict[str, Any]]
    trials: tuple[PanelTrial, ...]


class ReplacementPanelReport(PanelReport):
    """Opt-in execution history, separate from canonical scientific trials."""

    execution_history: tuple[dict[str, Any], ...]


ResultT = TypeVar("ResultT")


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
    evaluate: Callable[[str, str], ResultT],
    *,
    attempts: int = 5,
    concurrency: int | None = None,
    min_rate: float = 0.2,
    max_rate: float = 0.8,
    min_valid: int | None = None,
    replacement_predicate: Callable[[dict[str, Any]], bool] | None = None,
    max_replacements: int = 0,
) -> PanelReport:
    """Evaluate fresh uniquely identified trials and summarize each model.

    Callback results are mappings or data models with a boolean or null
    ``passed`` field. Null grades are excluded from rates; false grades count
    as failures. By default every requested trial must have a valid grade. Set
    min_valid explicitly to allow exclusions. Each model must meet that
    minimum and the inclusive rate bounds. With concurrency=None all
    model-by-attempt trials may run simultaneously. Trial results are
    normalized to JSON-compatible dicts in the returned report. Opt-in replacements
    consume at most max_replacements extra executions per model, in deterministic
    waves. Callback IDs are execution IDs; report trial IDs remain canonical.
    Execution history retains originals separately; exceptions propagate unchanged.
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
        try:
            passed = get_field(result, "passed", "evaluate_panel")
        except KeyError as exc:
            raise ValueError("Panel callback must return a result with passed") from exc
        if passed is not None and not isinstance(passed, bool):
            raise ValueError("Panel callback passed must be a boolean or null")
        return _as_result_dict(result)

    # no_cache still hashes callables; partial keeps opaque callback state out.
    sampled: list[dict[str, Any]] = sample.no_cache(
        [{}], partial(invoke), sample_ids=list(by_identity), concurrency=workers
    )
    trials = [
        PanelTrial(
            model=by_identity[entry["sample_id"]],
            trial_id=entry["sample_id"],
            result=entry["value"],
        )
        for entry in sampled[0]["samples"]
    ]
    history = [
        {
            "model": trial.model,
            "trial_id": trial.trial_id,
            "execution_id": trial.trial_id,
            "result": trial.result,
        }
        for trial in trials
    ]
    # Deterministic waves allocate each model's budget in canonical slot order.
    used = dict.fromkeys(models, 0)
    while max_replacements:
        selected = []
        for index, trial in enumerate(trials):
            if used[trial.model] >= max_replacements:
                continue
            retry = replacement_predicate(trial.result)
            if type(retry) is not bool:
                raise ValueError("Replacement predicate must return bool")
            if retry:
                used[trial.model] += 1
                identity = f"{trial.trial_id}-replacement-{used[trial.model]}"
                by_identity[identity] = trial.model
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
                        "model": trials[index].model,
                        "trial_id": trials[index].trial_id,
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
                    "model": previous.model,
                    "trial_id": previous.trial_id,
                    "execution_id": identity,
                    "result": result,
                }
            )
            trials[index] = previous.model_copy(update={"result": result})
    rows: list[dict[str, Any]] = [{"evaluations": {model: [] for model in models}}]
    for trial in trials:
        rows[0]["evaluations"][trial.model].append(trial.result["passed"])
    counts = success_rates.no_cache(rows)[0]
    accepted = bool(
        by_success_rate.no_cache(
            rows, min_rate=min_rate, max_rate=max_rate, min_valid=min_valid
        )
    )
    if max_replacements:
        return ReplacementPanelReport(
            accepted=accepted,
            models=counts,
            trials=tuple(trials),
            execution_history=tuple(history),
        )
    return PanelReport(accepted=accepted, models=counts, trials=tuple(trials))


def evaluate_panels(
    panels: Mapping[str, Sequence[str]],
    evaluate_panel: Callable[[str], PanelReport],
    *,
    concurrency: int | None = 1,
) -> dict[str, PanelReport]:
    """Run several independent task panels, one typed report per panel name.

    ``panels`` maps an opaque caller-owned panel identifier to that panel's
    model names; identifying names are not interpreted and never have to be
    path-safe.  ``evaluate_panel`` receives the identifier and returns the
    already-composed panel report for that panel (normally a closure over
    ``pyforger.evaluation.evaluate_panel``), so this operation only owns the
    set-level fan-out: order of reports follows ``panels``, every panel is
    evaluated even when others are rejected, and callback exceptions
    propagate rather than becoming difficulty evidence.  A model listed by
    more than one panel is validated by that panel's own evaluation.  The
    same model may legitimately be evaluated in several panels.
    """
    if concurrency is not None and (type(concurrency) is not int or concurrency < 1):
        raise ValueError("concurrency must be a positive integer or None")
    names = list(panels)
    for name in names:
        models = panels[name]
        if (
            isinstance(models, (str, bytes))
            or not models
            or any(not isinstance(model, str) or not model.strip() for model in models)
        ):
            raise ValueError(f"Panel {name!r} requires nonempty model names")
        if len(set(models)) != len(models):
            raise ValueError(f"Panel {name!r} model names must be unique")

    def run(name: str) -> PanelReport:
        report = evaluate_panel(name)
        if not isinstance(report, PanelReport):
            raise TypeError("evaluate_panel callback must return a PanelReport")
        return report

    reports = run_concurrently(
        [partial(run, name) for name in names], concurrency=concurrency
    )
    return dict(zip(names, reports, strict=True))


def _as_result_dict(result: Any) -> dict[str, Any]:
    """Normalize a mapping or data-model panel result to a JSON-safe dict."""
    from pyforger.evidence_journal import has_private_evidence

    if isinstance(result, BaseModel) and has_private_evidence(result.__dict__):
        raise ValueError("Publish private evidence before returning a panel result")
    if isinstance(result, Measurement):
        payload: Any = result.model_dump(mode="json")
    elif isinstance(result, BaseModel):
        payload = result.model_dump(mode="json")
    elif isinstance(result, Mapping):
        payload = dict(result)
    elif is_dataclass(result) and not isinstance(result, type):
        payload = asdict(result)
    else:
        payload = {"passed": get_field(result, "passed", "evaluate_panel")}
        for name in ("status", "reason", "model", "trial_id"):
            try:
                payload[name] = get_field(result, name, "evaluate_panel")
            except KeyError:
                pass
    if has_private_evidence(payload):
        raise ValueError("Publish private evidence before returning a panel result")
    normalized: dict[str, Any] = json.loads(json.dumps(payload, allow_nan=False))
    return normalized
