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
from pyforger.inference.inspect_bridge import (
    InspectEvalResult,
    clear_backend_registry,
    default_render,
    evaluate_with_inspect,
    passthrough_render,
    register_backend,
)

__all__ = [
    "EvalResult",
    "GPQAExample",
    "InferenceBackend",
    "InspectEvalResult",
    "LiteLLMBackend",
    "SamplingConfig",
    "VLLMBackend",
    "clear_backend_registry",
    "clear_engine_cache",
    "default_render",
    "evaluate_gpqa",
    "evaluate_with_inspect",
    "format_prompt",
    "load_gpqa",
    "parse_answer",
    "passthrough_render",
    "register_backend",
]
