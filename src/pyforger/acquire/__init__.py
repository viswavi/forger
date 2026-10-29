"""Acquire operations: generate or retrieve new data."""

from pyforger.acquire.model import (
    GenerateResult,
    LocalModel,
    Model,
    PromptTooLongError,
    TargetLogprobs,
    split_reasoning,
)
from pyforger.acquire.retrieve import (
    retrieve,
    retrieve_from_arxiv,
    retrieve_from_hf,
    retrieve_from_s2,
    retrieve_from_web,
    retrieve_from_wiki,
)
from pyforger.acquire.rubric import (
    convert_rubric_criteria,
    generate_custom_rubric,
    generate_custom_rubric_criteria,
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
from pyforger.acquire.tinker_model import TinkerModel
from pyforger.tinker_usage import TinkerBudgetExceeded

__all__ = [
    "GenerateResult",
    "Model",
    "LocalModel",
    "TinkerModel",
    "TinkerBudgetExceeded",
    "PromptTooLongError",
    "TargetLogprobs",
    "split_reasoning",
    "retrieve",
    "retrieve_from_web",
    "retrieve_from_wiki",
    "retrieve_from_hf",
    "retrieve_from_arxiv",
    "retrieve_from_s2",
    "sample_from_LLM",
    "generate_custom_rubric",
    "convert_rubric_criteria",
    "generate_custom_rubric_criteria",
    "RetrievalSource",
    "RetrieveMode",
    "WebRetrievalSource",
    "WikiRetrievalSource",
    "HFRetrievalSource",
    "ArxivRetrievalSource",
    "S2RetrievalSource",
]
