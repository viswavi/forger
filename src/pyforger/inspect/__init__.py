"""Inspect operations: analyze existing data."""
from pyforger.inspect.cluster import topic_based_clustering
from pyforger.inspect.compare import compare_pipelines
from pyforger.inspect.dispersion import DispersionReport, embedding_dispersion
from pyforger.inspect.eval_tree import eval_tree
from pyforger.inspect.likelihood import (
    LikelihoodReport,
    RowLikelihood,
    student_log_likelihood,
)
from pyforger.inspect.phrases import PhraseReport, phrase_frequency
from pyforger.inspect.reference import ReferenceReport, compare_to_reference
from pyforger.inspect.report import inspect_step
from pyforger.inspect.runs import runs
from pyforger.inspect.scores import RubricScoreReport, rubric_scores
from pyforger.inspect.spend import SpendReport, SpendTotal, spend

__all__ = [
    "DispersionReport",
    "LikelihoodReport",
    "RowLikelihood",
    "PhraseReport",
    "ReferenceReport",
    "RubricScoreReport",
    "SpendReport",
    "SpendTotal",
    "embedding_dispersion",
    "compare_to_reference",
    "compare_pipelines",
    "eval_tree",
    "inspect_step",
    "phrase_frequency",
    "rubric_scores",
    "runs",
    "spend",
    "student_log_likelihood",
    "topic_based_clustering",
]
