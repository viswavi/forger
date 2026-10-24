"""Backend capability policy owned by Forger."""

from pyforger.agent_backend import AgentBackend


def supports_inner_loop_validation(backend: AgentBackend) -> bool:
    """Report host validation capability without interpreting backend names."""
    return bool(getattr(backend, "supports_inner_loop_validation", False))


def require_inner_loop_validation(backend: AgentBackend) -> None:
    """Reject executors without host validation in a persistent conversation."""
    if not supports_inner_loop_validation(backend):
        raise NotImplementedError(
            f"{backend.name} does not support same-session validated generation"
        )
