"""Inspect operations: analyze existing data."""
from pyforger.inspect.cluster import topic_based_clustering
from pyforger.inspect.dispersion import DispersionReport, embedding_dispersion
from pyforger.inspect.eval_tree import eval_tree
from pyforger.inspect.phrases import PhraseReport, phrase_frequency
from pyforger.inspect.report import inspect_step
from pyforger.inspect.runs import runs
from pyforger.inspect.spend import SpendReport, SpendTotal, spend

__all__ = [
    "DispersionReport",
    "PhraseReport",
    "SpendReport",
    "SpendTotal",
    "embedding_dispersion",
    "eval_tree",
    "inspect_step",
    "phrase_frequency",
    "runs",
    "spend",
    "topic_based_clustering",
]
