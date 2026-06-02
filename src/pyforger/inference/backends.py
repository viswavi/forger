"""Inference backends for evaluating language models on benchmarks.

Two backends are provided:

- :class:`VLLMBackend` — offline vLLM engine for local GPU inference. The
  engine itself is lazy-loaded (vLLM is an optional extra) and reused across
  ``VLLMBackend`` instances that share a config via a module-level registry,
  so a candidate-evaluation loop does not reload model weights into VRAM on
  every iteration.
- :class:`LiteLLMBackend` — adapter that wraps :class:`pyforger.acquire.Model`
  so the same evaluation code path works against hosted APIs.

Both satisfy the :class:`InferenceBackend` structural Protocol. Backends are
hashable for ``@op`` caching via :meth:`cache_key`; see
:mod:`pyforger.op` for how the canonical hash is computed.
"""
from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pyforger.acquire.model import Model
from pyforger.op import _canonical


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameters for a single generation pass.

    Frozen so instances are hashable and safe to pass as ``@op`` arguments.
    ``stop`` is a tuple (not a list) so the dataclass remains immutable.

    Attributes:
        temperature: Sampling temperature. ``0.0`` is greedy.
        max_tokens: Maximum tokens to generate per prompt.
        top_p: Nucleus sampling cutoff. Ignored by
            :class:`LiteLLMBackend` (see its docstring).
        stop: Stop sequences. Generation halts when any is produced.
    """

    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    stop: tuple[str, ...] = ()

    def cache_key(self) -> dict:
        """Stable hash key consumed by ``@op``'s ``_canonical``."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": list(self.stop),
        }


@runtime_checkable
class InferenceBackend(Protocol):
    """Structural protocol for a batched text-completion backend.

    Backends do not inherit from this; they satisfy it by exposing the
    three members below.

    Attributes:
        model: Human-readable identifier (e.g. ``"meta-llama/Llama-3-8B"``)
            used for logging and debugging. NOT used for hashing — the
            canonical cache key comes from :meth:`cache_key`.
    """

    model: str

    def generate(
        self, prompts: list[str], sampling: SamplingConfig
    ) -> list[str]: ...

    def cache_key(self) -> dict: ...


_ENGINE_REGISTRY: dict[bytes, Any] = {}
_REGISTRY_LOCK = threading.Lock()


def _registry_key(ck: dict) -> bytes:
    """Canonical bytes key for the engine registry.

    Uses the same ``_canonical`` normalization as ``@op``'s on-disk cache,
    so configs that hash identically under ``@op`` also share an engine
    slot. This avoids the failure mode of ``sorted(dict.items())`` when
    values are nested or not directly comparable.
    """
    return pickle.dumps(_canonical(ck), protocol=4)


def clear_engine_cache() -> None:
    """Drop all cached vLLM engines. Use in tests or for explicit teardown.

    Does not call ``.shutdown()`` on the engines; the engines are simply
    dereferenced and become eligible for garbage collection.
    """
    with _REGISTRY_LOCK:
        _ENGINE_REGISTRY.clear()


class VLLMBackend:
    """vLLM offline inference backend.

    The underlying ``vllm.LLM`` engine is loaded lazily on the first call
    to :meth:`generate` and shared across all ``VLLMBackend`` instances
    whose :meth:`cache_key` is canonically equal. This is important for
    candidate-evaluation loops where many backend instances are
    instantiated but the model config is constant.

    vLLM is an optional dependency. Importing this class, constructing
    it, and reading attributes never imports vLLM. The first
    :meth:`generate` call triggers the import; if vLLM is not installed,
    an :class:`ImportError` is raised pointing users at the
    ``pyforger[inference]`` extra.

    Attributes:
        model: HF model identifier or local path.
        tensor_parallel_size: Number of GPUs to shard across.
        dtype: vLLM dtype string (``"auto"``, ``"float16"``, …).
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        max_model_len: Maximum context length; ``None`` uses the model's
            default.
        seed: Engine seed.
        enable_prefix_caching: Whether to enable vLLM's prefix-caching.
            Default ``False``; flip to ``True`` for in-context optimization
            workloads where many candidates share a prompt prefix.
    """

    def __init__(
        self,
        model: str,
        *,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        seed: int = 0,
        enable_prefix_caching: bool = False,
    ) -> None:
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.seed = seed
        self.enable_prefix_caching = enable_prefix_caching

    def cache_key(self) -> dict:
        """Stable hash key. Excludes any live engine handle."""
        return {
            "backend": "vllm",
            "model": self.model,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "seed": self.seed,
            "enable_prefix_caching": self.enable_prefix_caching,
        }

    def _engine(self) -> Any:
        """Return the live vLLM engine, loading or reusing as needed."""
        key = _registry_key(self.cache_key())
        with _REGISTRY_LOCK:
            engine = _ENGINE_REGISTRY.get(key)
            if engine is None:
                try:
                    from vllm import LLM
                except ImportError as exc:
                    raise ImportError(
                        "VLLMBackend requires the 'inference' extra. "
                        "Install with: pip install 'pyforger[inference]'"
                    ) from exc
                engine = LLM(
                    model=self.model,
                    tensor_parallel_size=self.tensor_parallel_size,
                    dtype=self.dtype,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    seed=self.seed,
                    enable_prefix_caching=self.enable_prefix_caching,
                )
                _ENGINE_REGISTRY[key] = engine
            return engine

    def generate(
        self, prompts: list[str], sampling: SamplingConfig
    ) -> list[str]:
        """Generate one completion per prompt.

        vLLM preserves input order in its output list, so the returned
        completions align positionally with ``prompts``.
        """
        engine = self._engine()
        # Safe to import here: ``_engine()`` succeeded, so vLLM is available.
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=sampling.temperature,
            max_tokens=sampling.max_tokens,
            top_p=sampling.top_p,
            stop=list(sampling.stop) or None,
            seed=self.seed,
        )
        outputs = engine.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]


class LiteLLMBackend:
    """Adapter that exposes a :class:`pyforger.acquire.Model` as an
    :class:`InferenceBackend`.

    Lets evaluation code (``evaluate_gpqa`` etc.) target any LiteLLM
    provider — OpenAI, Anthropic, Together, local OpenAI-compatible
    servers — through the same protocol as :class:`VLLMBackend`.

    ``SamplingConfig.top_p`` is **ignored** by this backend because the
    underlying :meth:`Model.generate_batch` has no ``top_p`` parameter.
    If you need ``top_p`` for hosted models, extend
    :class:`pyforger.acquire.Model` rather than papering over it here.

    Attributes:
        model: The wrapped model's identifier (mirrored from
            :class:`Model` so the Protocol is satisfied).
    """

    def __init__(self, model: Model) -> None:
        self._model = model
        self.model = model._model_id

    def cache_key(self) -> dict:
        """Stable hash key delegating to the wrapped Model."""
        return {"backend": "litellm", "model_ck": self._model.cache_key()}

    def generate(
        self, prompts: list[str], sampling: SamplingConfig
    ) -> list[str]:
        """Generate one completion per prompt via LiteLLM.

        Maps :class:`SamplingConfig` to :meth:`Model.generate_batch`
        kwargs. ``n=1`` is hardcoded because the eval loop scores one
        completion per example.
        """
        batched = self._model.generate_batch(
            prompts,
            n=1,
            temperature=sampling.temperature,
            max_tokens=sampling.max_tokens,
            stop=list(sampling.stop) or None,
        )
        return [completions[0] for completions in batched]
