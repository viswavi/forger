"""Inspect operations: analyze existing data."""
from pyforger.inspect.cluster import topic_based_clustering
from pyforger.inspect.phrases import PhraseReport, phrase_frequency
from pyforger.inspect.report import inspect_step
from pyforger.inspect.spend import SpendReport, SpendTotal, spend

__all__ = [
    "PhraseReport",
    "SpendReport",
    "SpendTotal",
    "inspect_step",
    "phrase_frequency",
    "spend",
    "topic_based_clustering",
]
