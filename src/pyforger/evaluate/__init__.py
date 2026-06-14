"""Data-quality evaluation via in-context-learning (ICL) uplift.

:func:`estimate_quality` measures how much using a candidate dataset as
few-shot demos raises HumanEval pass@1 over a zero-shot baseline.
"""
from pyforger.evaluate.datasets import (
    holdout_split,
    load_forger_jsonl,
    load_humaneval,
    subsample_pool,
)
from pyforger.evaluate.execute import ScoreOutcome, score_program
from pyforger.evaluate.icl import build_messages, extract_code, sample_subsets
from pyforger.evaluate.quality import QualityReport, estimate_quality

__all__ = [
    "QualityReport",
    "ScoreOutcome",
    "build_messages",
    "estimate_quality",
    "extract_code",
    "holdout_split",
    "load_forger_jsonl",
    "load_humaneval",
    "sample_subsets",
    "score_program",
    "subsample_pool",
]
