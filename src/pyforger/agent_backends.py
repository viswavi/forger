"""Backend capability policy owned by Forger."""

from pyforger.agent_backend import AgentBackend, AgentSpec


def supports_inner_loop_validation(backend: AgentBackend) -> bool:
    """Report host validation capability without interpreting backend names."""
    return bool(getattr(backend, "supports_inner_loop_validation", False))


def require_inner_loop_validation(backend: AgentBackend) -> None:
    """Reject executors without host validation in a persistent conversation."""
    if not supports_inner_loop_validation(backend):
        raise NotImplementedError(
            f"{backend.name} does not support same-session validated generation"
        )


def require_learner_runtime(backend: AgentBackend, agent: AgentSpec) -> None:
    """Fail before execution when a requested runtime cannot be honored."""
    runtime = agent.runtime
    if runtime is None:
        return
    if runtime.reasoning_replay:
        raise NotImplementedError("Reasoning replay requires a shared learner runner")
    if (
        runtime.condenser_max_tokens is not None
        or runtime.condenser_max_events is not None
    ) and not getattr(backend, "supports_sdk_condensation", False):
        raise NotImplementedError(f"{backend.name} cannot honor SDK condensation")
    if runtime.tools == "science-rl" and not getattr(
        backend, "supports_science_rl_tools", False
    ):
        raise NotImplementedError(
            f"{backend.name} cannot honor the science-RL tool surface"
        )

    if runtime.tools == "science-rl" and agent.kwargs.keys() & {
        "tools",
        "include_default_tools",
        "agent_context",
    }:
        raise ValueError("Agent kwargs conflict with the explicit tool profile")
    if (
        runtime.condenser_max_tokens is not None
        or runtime.condenser_max_events is not None
    ) and "condenser" in agent.kwargs:
        raise ValueError("Agent kwargs conflict with the explicit condenser settings")
