"""End-to-end ICL data-quality pipeline.

:func:`estimate_quality` returns a :class:`QualityReport` quantifying how
much using a candidate dataset as few-shot demos raises HumanEval pass@1 over
the zero-shot baseline (``metric = conditioned - baseline``). All prompts are
batched into a single ``backend.generate`` call (wrapped in a thin ``@op``
cache so reruns hit disk) and scored via the subprocess executor.
"""
from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from pyforger.evaluate.execute import score_program
from pyforger.evaluate.icl import build_messages, extract_code, sample_subsets
from pyforger.inference.backends import InferenceBackend, SamplingConfig
from pyforger.op import op
from pyforger.utils import get_formatted_logger

logger = get_formatted_logger("pyforger.evaluate.quality")


Render = Callable[[list[dict]], str]


@dataclass(frozen=True)
class QualityReport:
    """Summary of an ICL data-quality run.

    Attributes:
        metric: ICL uplift = ``conditioned_pass_at_1 - baseline_pass_at_1``.
        baseline_pass_at_1: zero-shot pass@1 averaged over validation items.
        conditioned_pass_at_1: pass@1 averaged over all (problem, subset).
        bootstrap_ci: 95% percentile bootstrap CI on ``metric``.
        per_item: per-validation-item dicts with ``task_id``,
            ``baseline_passed``, ``subset_passes``, ``conditioned_pass``,
            ``uplift``.
        config: ``k``, ``n_subsets``, ``seed``, model id, sampling, dataset
            sizes — so JSON dumps are self-describing.
    """

    metric: float
    baseline_pass_at_1: float
    conditioned_pass_at_1: float
    bootstrap_ci: tuple[float, float]
    per_item: list[dict]
    config: dict

    def to_dict(self) -> dict:
        """JSON-serializable dict representation."""
        return asdict(self)


@op(name="evaluate.generate_completions")
def _generate_completions(
    backend: InferenceBackend,
    prompts: list[str],
    sampling: SamplingConfig,
) -> list[str]:
    """``@op``-cached ``backend.generate`` (cache key includes all three args)."""
    return backend.generate(prompts, sampling)


def _bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Seeded percentile bootstrap CI on the mean of ``values``."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(alpha * n_resamples))
    hi_idx = min(n_resamples - 1, int((1.0 - alpha) * n_resamples))
    return (means[lo_idx], means[hi_idx])


def _score_one(
    item: dict,
    completion: str,
    *,
    timeout_s: int,
    allow_code_execution: bool,
) -> bool:
    body = extract_code(item["prompt"], completion)
    outcome = score_program(
        item["prompt"],
        body,
        item["test"],
        item["entry_point"],
        timeout_s=timeout_s,
        allow_code_execution=allow_code_execution,
    )
    return outcome.passed


def estimate_quality(
    backend: InferenceBackend,
    forger_data: list[dict],
    eval_data: list[dict],
    *,
    render: Render,
    k: int = 4,
    n_subsets: int = 5,
    sampling: SamplingConfig = SamplingConfig(temperature=0.0, max_tokens=1024),
    seed: int = 0,
    timeout_s: int = 10,
    allow_code_execution: bool = False,
) -> QualityReport:
    """Estimate the ICL uplift of ``forger_data`` on ``eval_data`` for ``backend``.

    Args:
        backend: any :class:`InferenceBackend`; one batched generate call.
        forger_data: candidate demos with ``"prompt"`` / ``"solution"``.
        eval_data: HumanEval-shaped validation items.
        render: chat-messages-to-prompt-string (model chat template).
        k: demos per few-shot prompt.
        n_subsets: few-shot prompts per validation item.
        sampling: decoding parameters; greedy recommended for pass@1.
        seed: master seed (per-problem subset seeds derived from it).
        timeout_s: per-program wall-clock limit.
        allow_code_execution: must be ``True`` to run scoring.

    Raises:
        ValueError: if ``eval_data`` is empty or ``forger_data`` is empty
            when ``k > 0``.
        RuntimeError: if ``allow_code_execution`` is ``False``.
    """
    if not eval_data:
        raise ValueError("eval_data is empty; nothing to evaluate against")
    if k > 0 and not forger_data:
        raise ValueError(
            f"k={k} demos requested but forger_data is empty; "
            "either pass demos or call with k=0 for a pure baseline"
        )

    P = len(eval_data)
    S = n_subsets

    # Build all prompts up front for one batched generate. Layout: P
    # zero-shot prompts, then P*S few-shot prompts grouped by problem;
    # (problem_idx, subset_idx) tracks each, with subset_idx == -1 for
    # zero-shot.
    layout: list[tuple[int, int]] = []
    rendered_prompts: list[str] = []

    for p_idx, item in enumerate(eval_data):
        rendered_prompts.append(render(build_messages([], item["prompt"])))
        layout.append((p_idx, -1))

    # Per-problem seeded subsets so different problems draw independently.
    per_problem_subsets: list[list[list[dict]]] = []
    for p_idx, item in enumerate(eval_data):
        if k == 0 or S == 0:
            per_problem_subsets.append([])
            continue
        problem_seed = seed * 1_000_003 + p_idx  # large prime to decorrelate
        subsets = sample_subsets(
            forger_data, k=k, n_subsets=S, seed=problem_seed
        )
        per_problem_subsets.append(subsets)
        for s_idx, demos in enumerate(subsets):
            rendered_prompts.append(render(build_messages(demos, item["prompt"])))
            layout.append((p_idx, s_idx))

    logger.info(
        "estimate_quality: generating %d completions (P=%d, S=%d, k=%d)",
        len(rendered_prompts),
        P,
        S,
        k,
    )
    completions = _generate_completions(backend, rendered_prompts, sampling)
    if len(completions) != len(rendered_prompts):
        raise RuntimeError(
            f"backend returned {len(completions)} completions for "
            f"{len(rendered_prompts)} prompts"
        )

    baseline_passes: list[bool] = [False] * P
    subset_passes: list[list[bool]] = [[False] * S for _ in range(P)]

    for completion, (p_idx, s_idx) in zip(completions, layout):
        item = eval_data[p_idx]
        passed = _score_one(
            item,
            completion,
            timeout_s=timeout_s,
            allow_code_execution=allow_code_execution,
        )
        if s_idx == -1:
            baseline_passes[p_idx] = passed
        else:
            subset_passes[p_idx][s_idx] = passed

    per_item: list[dict] = []
    item_uplifts: list[float] = []
    for p_idx, item in enumerate(eval_data):
        b = 1.0 if baseline_passes[p_idx] else 0.0
        if S == 0 or k == 0:
            cond = b  # no demos -> conditioned == baseline by construction
        else:
            cond = sum(1.0 for x in subset_passes[p_idx] if x) / S
        uplift = cond - b
        item_uplifts.append(uplift)
        per_item.append(
            {
                "task_id": item.get("task_id"),
                "baseline_passed": bool(baseline_passes[p_idx]),
                "subset_passes": list(subset_passes[p_idx]),
                "conditioned_pass": cond,
                "uplift": uplift,
            }
        )

    baseline = statistics.fmean(1.0 if x else 0.0 for x in baseline_passes)
    conditioned = (
        statistics.fmean(item["conditioned_pass"] for item in per_item)
        if per_item
        else 0.0
    )
    metric = conditioned - baseline
    ci = _bootstrap_ci(item_uplifts, seed=seed)

    config = {
        "model": getattr(backend, "model", None),
        "k": k,
        "n_subsets": n_subsets,
        "seed": seed,
        "n_val": P,
        "n_demos_pool": len(forger_data),
        "sampling": sampling.cache_key(),
        "timeout_s": timeout_s,
    }

    return QualityReport(
        metric=metric,
        baseline_pass_at_1=baseline,
        conditioned_pass_at_1=conditioned,
        bootstrap_ci=ci,
        per_item=per_item,
        config=config,
    )
