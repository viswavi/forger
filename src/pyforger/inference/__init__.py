"""Inference and evaluation: run language models against benchmarks."""
from pyforger.inference.backends import (
    InferenceBackend,
    LiteLLMBackend,
    SamplingConfig,
    VLLMBackend,
    clear_engine_cache,
)
from pyforger.inference.benchmarks.gpqa import (
    GPQAExample,
    format_prompt,
    load_gpqa,
    parse_answer,
)
from pyforger.inference.evaluate import EvalResult, evaluate_gpqa

__all__ = [
    "EvalResult",
    "GPQAExample",
    "InferenceBackend",
    "LiteLLMBackend",
    "SamplingConfig",
    "VLLMBackend",
    "clear_engine_cache",
    "evaluate_gpqa",
    "format_prompt",
    "load_gpqa",
    "parse_answer",
]
