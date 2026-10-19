"""Acquire operations: generate or retrieve new data."""

from pyforger.acquire.model import (
    GenerateResult,
    LocalModel,
    Model,
    ModelSettings,
    Usage,
    split_reasoning,
)
from pyforger.acquire.retrieve import (
    retrieve,
    retrieve_citations_from_s2,
    retrieve_from_arxiv,
    retrieve_from_hf,
    retrieve_from_s2,
    retrieve_from_web,
    retrieve_from_wiki,
)
from pyforger.acquire.sample import sample_from_LLM
from pyforger.acquire.sources import (
    ArxivRetrievalSource,
    HFRetrievalSource,
    RetrievalSource,
    RetrieveMode,
    S2RetrievalSource,
    WebRetrievalSource,
    WikiRetrievalSource,
)

__all__ = [
    "GenerateResult",
    "ModelSettings",
    "Usage",
    "Model",
    "LocalModel",
    "split_reasoning",
    "retrieve",
    "retrieve_citations_from_s2",
    "retrieve_from_web",
    "retrieve_from_wiki",
    "retrieve_from_hf",
    "retrieve_from_arxiv",
    "retrieve_from_s2",
    "sample_from_LLM",
    "RetrievalSource",
    "RetrieveMode",
    "WebRetrievalSource",
    "WikiRetrievalSource",
    "HFRetrievalSource",
    "ArxivRetrievalSource",
    "S2RetrievalSource",
]
