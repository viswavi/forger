"""Backend construction and capability policy owned by Forger."""

from pathlib import Path

from pyforger.agent_backend import AgentBackend

DEFAULT_BACKEND = "local"
BACKEND_CHOICES = ("local", "sail", "harbor-sail", "docker", "modal")


def get_backend(
    name: str | AgentBackend | None = None, retain: str | Path | None = None
) -> AgentBackend:
    """Construct an executor without importing optional SDKs until needed."""
    if name is not None and not isinstance(name, str):
        if not isinstance(name, AgentBackend):
            raise TypeError("backend must be a name or an AgentBackend")
        return name
    name = DEFAULT_BACKEND if name is None else name
    if name == "local":
        from pyforger.local_backend import LocalBackend

        return LocalBackend()
    if name == "sail":
        from pyforger.sail_backend import SailBackend

        return SailBackend()
    if name in {"harbor-sail", "docker", "modal"}:
        from pyforger.harbor_backend import HarborBackend

        return HarborBackend(
            {"type": "sail" if name == "harbor-sail" else name},
            retain_trials=retain,
        )
    raise ValueError(f"Unknown backend {name!r}; choose from {BACKEND_CHOICES}")


def supports_inner_loop_validation(backend: AgentBackend) -> bool:
    """Report host validation capability without interpreting backend names."""
    return bool(getattr(backend, "supports_inner_loop_validation", False))


def require_inner_loop_validation(backend: AgentBackend) -> None:
    """Reject executors without host validation in a persistent conversation."""
    if not supports_inner_loop_validation(backend):
        raise NotImplementedError(
            f"{backend.name} does not support same-session validated generation"
        )
