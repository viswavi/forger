"""Compatibility bridge between Forger backends and the Inspect AI eval harness.

`inspect_ai <https://inspect.aisi.org.uk>`_ (and the companion ``inspect_evals``
task suite) ships maintained, canonical implementations of many benchmarks —
GSM8K, HumanEval (sandboxed), GPQA, MMLU, MATH, … — together with scorers,
sample limits, concurrency control, and logging. Rather than re-implement each
benchmark's loader / prompt format / answer parser inside Forger, this module
lets any :class:`~pyforger.inference.backends.InferenceBackend` be evaluated by
Inspect on *any* of those tasks.

It does this by registering a custom Inspect model provider named ``forger``.
After :func:`register_backend` returns a model string like ``"forger/default"``,
you can hand that string to ``inspect_ai.eval(...)`` exactly as you would
``"vllm/..."`` or ``"openai/..."``::

    from pyforger.inference import VLLMBackend
    from pyforger.inference.inspect_bridge import register_backend
    from inspect_ai import eval as inspect_eval
    import inspect_evals.gsm8k  # registers the task

    backend = VLLMBackend(model="meta-llama/Llama-3.1-8B-Instruct")
    model = register_backend(backend)  # -> "forger/default"
    logs = inspect_eval("inspect_evals/gsm8k", model=model, limit=150)

Or use :func:`evaluate_with_inspect`, which wires registration + ``eval`` +
metric extraction together (mirroring the typical eval script).

Design notes
------------
* **Engine reuse is preserved.** Going through the Forger backend means
  :class:`~pyforger.inference.backends.VLLMBackend`'s shared-engine registry
  still applies; we do *not* spin up Inspect's own ``vllm`` provider.
* **Prompt rendering is explicit.** :class:`VLLMBackend` is a *text-completion*
  backend with no chat template, so Inspect's chat messages must be flattened
  to a single string. The default :func:`default_render` is a generic role-
  tagged join. **For instruct models you should pass a ``render`` that applies
  the model's actual chat template**, otherwise scores will be depressed.
  :class:`~pyforger.inference.backends.LiteLLMBackend` natively accepts message
  lists, so for it the default passes messages straight through and the hosted
  provider applies its own template.
* **Optional dependency.** Inspect is not a hard dependency; importing this
  module never imports ``inspect_ai``. Install with
  ``pip install 'pyforger[inspect]'`` (the ``inspect_evals`` task suite is a
  separate install — see its README).
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pyforger.inference.backends import InferenceBackend, SamplingConfig

Message = dict[str, str]
Render = Callable[[list[Message]], "str | list[Message]"]

_PROVIDER_NAME = "forger"
_INSTALL_HINT = (
    "Inspect AI integration requires the 'inspect' extra. "
    "Install with: pip install 'pyforger[inspect]' "
    "(the inspect_evals task suite is a separate install)."
)


def default_render(messages: list[Message]) -> str:
    """Flatten chat messages into a single completion prompt string.

    A single user message is returned verbatim (this matches how Forger's
    own evals build plain completion prompts). Multi-turn or system-prefixed
    conversations are rendered as a generic ``role: content`` transcript with
    a trailing ``assistant:`` cue.

    .. warning::
        This is **not** a model-specific chat template. For instruct models,
        register the backend with a ``render`` that applies the model's real
        chat template (e.g. via the vLLM engine tokenizer's
        ``apply_chat_template``); otherwise eval scores will be lower than the
        model's true capability.
    """
    if len(messages) == 1 and messages[0].get("role") == "user":
        return messages[0].get("content", "")
    parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    parts.append("assistant:")
    return "\n\n".join(parts)


def passthrough_render(messages: list[Message]) -> list[Message]:
    """Return messages unchanged, for backends that accept chat message lists.

    Used as the default for :class:`LiteLLMBackend`, whose underlying
    :class:`~pyforger.acquire.model.Model` accepts OpenAI-format message lists
    directly and lets the hosted provider apply its own chat template.
    """
    return messages


@dataclass(frozen=True)
class _Registered:
    backend: InferenceBackend
    render: Render
    default_max_tokens: int


_REGISTRY: dict[str, _Registered] = {}
_REGISTRY_LOCK = threading.Lock()

_PROVIDER_LOCK = threading.Lock()
_provider_registered = False


def _default_render_for(backend: InferenceBackend) -> Render:
    """Pick a sensible default renderer based on the backend type.

    LiteLLM-style backends accept message lists; everything else is treated as
    a text-completion backend and gets :func:`default_render`.
    """
    # Imported lazily-ish; LiteLLMBackend is always importable (no heavy deps).
    from pyforger.inference.backends import LiteLLMBackend

    if isinstance(backend, LiteLLMBackend):
        return passthrough_render
    return default_render


def register_backend(
    backend: InferenceBackend,
    name: str = "default",
    *,
    render: Render | None = None,
    default_max_tokens: int = 1024,
) -> str:
    """Register ``backend`` so Inspect AI can target it, and register the provider.

    Args:
        backend: Any :class:`~pyforger.inference.backends.InferenceBackend`.
        name: Registry key. The returned model string is ``"forger/<name>"``.
            Re-registering the same name overwrites the previous entry.
        render: How to turn Inspect's chat messages into the backend's input.
            Defaults to :func:`passthrough_render` for
            :class:`LiteLLMBackend` and :func:`default_render` otherwise. Pass a
            custom callable to apply a model-specific chat template.
        default_max_tokens: ``max_tokens`` to use when Inspect's
            ``GenerateConfig`` does not specify one.

    Returns:
        The Inspect model string (e.g. ``"forger/default"``) to pass as
        ``model=`` to ``inspect_ai.eval``.

    Raises:
        ImportError: If ``inspect_ai`` is not installed.
    """
    _ensure_provider_registered()
    chosen_render = render if render is not None else _default_render_for(backend)
    with _REGISTRY_LOCK:
        _REGISTRY[name] = _Registered(
            backend=backend,
            render=chosen_render,
            default_max_tokens=default_max_tokens,
        )
    return f"{_PROVIDER_NAME}/{name}"


def clear_backend_registry() -> None:
    """Drop all registered backends. Use in tests or for explicit teardown."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


def _lookup(name: str) -> _Registered:
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(name)
    if entry is None:
        known = sorted(_REGISTRY)
        raise KeyError(
            f"No Forger backend registered under {name!r}. "
            f"Call register_backend(backend, name={name!r}) first. "
            f"Currently registered: {known}."
        )
    return entry


def _normalize_messages(input: Any) -> list[Message]:
    """Convert Inspect's ``str | list[ChatMessage]`` input to OpenAI-style dicts."""
    if isinstance(input, str):
        return [{"role": "user", "content": input}]
    messages: list[Message] = []
    for m in input:
        role = getattr(m, "role", "user")
        # ChatMessage exposes a ``text`` property that flattens content parts;
        # fall back to ``content`` for plain-string messages / test doubles.
        content = getattr(m, "text", None)
        if content is None:
            content = getattr(m, "content", "")
        if not isinstance(content, str):
            content = str(content)
        messages.append({"role": str(role), "content": content})
    return messages


def _sampling_from_config(config: Any, default_max_tokens: int) -> SamplingConfig:
    """Map an Inspect ``GenerateConfig`` onto a Forger :class:`SamplingConfig`."""
    temperature = getattr(config, "temperature", None)
    max_tokens = getattr(config, "max_tokens", None)
    top_p = getattr(config, "top_p", None)
    stop_seqs = getattr(config, "stop_seqs", None)
    return SamplingConfig(
        temperature=temperature if temperature is not None else 0.0,
        max_tokens=max_tokens if max_tokens is not None else default_max_tokens,
        top_p=top_p if top_p is not None else 1.0,
        stop=tuple(stop_seqs) if stop_seqs else (),
    )


def _to_model_output(model_name: str, completions: list[str]) -> Any:
    """Wrap one or more completion strings in an Inspect ``ModelOutput``."""
    from inspect_ai.model import ModelOutput

    if len(completions) == 1:
        return ModelOutput.from_content(model=model_name, content=completions[0])

    # Multiple choices (GenerateConfig.num_choices > 1): build choices by hand
    # since ModelOutput.from_content only produces a single choice.
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant

    choices = [
        ChatCompletionChoice(
            message=ChatMessageAssistant(content=text, model=model_name),
            stop_reason="stop",
        )
        for text in completions
    ]
    return ModelOutput(model=model_name, choices=choices)


def _build_provider() -> type:
    """Return the ``ForgerAPI`` Inspect model-provider class.

    Defined lazily inside this factory because it subclasses ``ModelAPI``,
    which only exists once ``inspect_ai`` is importable. This is the same
    pattern Inspect's own built-in providers use.
    """
    from inspect_ai.model import GenerateConfig, ModelAPI

    class ForgerAPI(ModelAPI):
        """Inspect ``ModelAPI`` that delegates generation to a Forger backend."""

        def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            api_key: str | None = None,
            config: GenerateConfig | None = None,
            **model_args: Any,
        ) -> None:
            super().__init__(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                config=config or GenerateConfig(),
            )
            self._entry = _lookup(model_name)

        async def generate(
            self,
            input: Any,
            tools: Any = None,
            tool_choice: Any = None,
            config: Any = None,
        ) -> Any:
            entry = self._entry
            messages = _normalize_messages(input)
            rendered = entry.render(messages)
            sampling = _sampling_from_config(
                config if config is not None else GenerateConfig(),
                entry.default_max_tokens,
            )
            n = getattr(config, "num_choices", None) or 1
            prompts = [rendered] * n
            # Forger backends are synchronous; run off the event loop so we do
            # not block Inspect's concurrency scheduler.
            completions = await asyncio.to_thread(
                entry.backend.generate, prompts, sampling
            )
            return _to_model_output(self.model_name, completions)

    return ForgerAPI


def _ensure_provider_registered() -> None:
    """Register the ``forger`` provider with Inspect exactly once."""
    global _provider_registered
    with _PROVIDER_LOCK:
        if _provider_registered:
            return
        try:
            from inspect_ai.model import modelapi
        except ImportError as exc:  # pragma: no cover - exercised via message
            raise ImportError(_INSTALL_HINT) from exc
        modelapi(name=_PROVIDER_NAME)(_build_provider)
        _provider_registered = True


@dataclass(frozen=True)
class InspectEvalResult:
    """Metrics extracted from an Inspect eval run.

    Attributes:
        task: The task identifier that was evaluated.
        model: The Forger model string used (``"forger/<name>"``).
        metrics: Flat ``{"<scorer>/<metric>": value}`` mapping (e.g.
            ``{"match/accuracy": 0.74}``).
        samples: Number of samples evaluated, if reported by Inspect.
    """

    task: str
    model: str
    metrics: dict[str, float]
    samples: int | None = None


def _extract_metrics(logs: Any) -> dict[str, float]:
    """Flatten scorer metrics from one or more Inspect ``EvalLog`` objects."""
    metrics: dict[str, float] = {}
    for log in logs:
        results = getattr(log, "results", None)
        if results is None:
            continue
        for score in getattr(results, "scores", None) or []:
            scorer = getattr(score, "name", "score")
            for key, metric in (getattr(score, "metrics", None) or {}).items():
                metrics[f"{scorer}/{key}"] = getattr(metric, "value", metric)
    return metrics


def evaluate_with_inspect(
    backend: InferenceBackend,
    task: Any,
    *,
    name: str = "default",
    limit: int | None = None,
    sampling: SamplingConfig | None = None,
    render: Render | None = None,
    default_max_tokens: int = 1024,
    model_args: dict[str, Any] | None = None,
    **eval_kwargs: Any,
) -> InspectEvalResult:
    """Evaluate ``backend`` on an Inspect ``task`` and return flattened metrics.

    This mirrors the typical eval script: register the backend as an Inspect
    model, run ``inspect_ai.eval``, then pull scorer metrics out of the log.

    Args:
        backend: The Forger backend to evaluate.
        task: An Inspect task — either a registered task string (e.g.
            ``"inspect_evals/gsm8k"``) or a ``Task`` object.
        name: Registry name for the backend (see :func:`register_backend`).
        limit: Optional cap on the number of samples to evaluate.
        sampling: Decoding parameters; mapped onto Inspect's ``GenerateConfig``
            (``temperature`` / ``max_tokens`` / ``top_p`` / ``stop_seqs``).
        render: Optional message-to-prompt renderer (see
            :func:`register_backend`).
        default_max_tokens: Fallback ``max_tokens`` (see
            :func:`register_backend`).
        model_args: Extra ``model_args`` forwarded to ``inspect_ai.eval``.
        **eval_kwargs: Additional keyword arguments forwarded verbatim to
            ``inspect_ai.eval`` (e.g. ``max_connections``, ``log_dir``).

    Returns:
        An :class:`InspectEvalResult` with flattened scorer metrics.

    Raises:
        ImportError: If ``inspect_ai`` is not installed.
    """
    model = register_backend(
        backend,
        name=name,
        render=render,
        default_max_tokens=default_max_tokens,
    )
    try:
        from inspect_ai import eval as inspect_eval
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    kwargs: dict[str, Any] = dict(eval_kwargs)
    if limit is not None:
        kwargs["limit"] = limit
    if sampling is not None:
        kwargs.setdefault("temperature", sampling.temperature)
        kwargs.setdefault("max_tokens", sampling.max_tokens)
        kwargs.setdefault("top_p", sampling.top_p)
        if sampling.stop:
            kwargs.setdefault("stop_seqs", list(sampling.stop))

    logs = inspect_eval(task, model=model, model_args=model_args or {}, **kwargs)

    samples: int | None = None
    for log in logs:
        n = getattr(getattr(log, "results", None), "total_samples", None)
        if n is not None:
            samples = n
            break

    return InspectEvalResult(
        task=str(task),
        model=model,
        metrics=_extract_metrics(logs),
        samples=samples,
    )
