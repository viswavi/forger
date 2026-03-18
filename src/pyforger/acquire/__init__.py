"""Acquire operations for Forger.

Acquire operations generate or retrieve new data.
"""
from pyforger.acquire.model import Model, model
from pyforger.acquire.sample import sample_from_LLM

__all__ = [
    "Model",
    "model",
    "sample_from_LLM",
]
