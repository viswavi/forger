"""End-to-end evaluation of an :class:`InferenceBackend` on GPQA.

The core entrypoint is :func:`evaluate_gpqa`, which is ``@op``-cached on
the canonical hash of ``(backend, few_shot, sampling, n, subset)``.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyforger.inference.backends import InferenceBackend, SamplingConfig
from pyforger.inference.benchmarks.gpqa import (
    GPQAExample,
    format_prompt,
    load_gpqa,
    parse_answer,
)
from pyforger.op import op


@dataclass(frozen=True)
class EvalResult:
    """Outcome of an evaluation run.

    Attributes:
        accuracy: ``n_correct / n_total``.
        n_total: Number of examples evaluated.
        n_correct: Number of examples the backend answered correctly.
        n_unparseable: Number of completions :func:`parse_answer`
            could not extract a letter from. These count as wrong, not
            correct.
    """

    accuracy: float
    n_total: int
    n_correct: int
    n_unparseable: int


@op
def evaluate_gpqa(
    backend: InferenceBackend,
    *,
    n: int = 200,
    few_shot: tuple[GPQAExample, ...] = (),
    sampling: SamplingConfig = SamplingConfig(),
    subset: str = "gpqa_main",
) -> EvalResult:
    """Score ``backend`` on GPQA.

    Cached on disk by the canonical hash of
    ``(backend.cache_key(), few_shot, sampling.cache_key(), n, subset)``.

    Args:
        backend: Any :class:`InferenceBackend` (e.g.
            :class:`VLLMBackend`, :class:`LiteLLMBackend`).
        n: **Eval-set size** — the number of GPQA examples to score
            against. This is NOT the per-prompt sample count;
            per-prompt sampling is fixed at one completion (the eval
            scores accuracy over examples, not over samples).
        few_shot: Optional tuple of exemplars prepended to every prompt.
            Tuple (not list) so it hashes cleanly into the ``@op`` key.
        sampling: Decoding parameters. Defaults to greedy
            (``temperature=0.0``, ``max_tokens=1024``).
        subset: GPQA config to evaluate on; see :func:`load_gpqa`.

    Returns:
        An :class:`EvalResult` with accuracy and counts.
    """
    examples = load_gpqa(subset=subset, n=n)
    prompts = [format_prompt(ex, few_shot=few_shot) for ex in examples]
    completions = backend.generate(prompts, sampling)

    correct = 0
    unparseable = 0
    for ex, completion in zip(examples, completions, strict=True):
        pred = parse_answer(completion)
        if pred is None:
            unparseable += 1
        elif pred == ex.gold:
            correct += 1

    n_total = len(examples)
    return EvalResult(
        accuracy=correct / n_total if n_total else 0.0,
        n_total=n_total,
        n_correct=correct,
        n_unparseable=unparseable,
    )
