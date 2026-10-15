"""Backend contract for source-producing dynamic alter operations."""
from __future__ import annotations

from typing import Any, Protocol


class DynamicStepRunner(Protocol):
    """Synthesize source and execute it with the same row-to-strings contract.

    Implementations own their model, limits, and execution environment. Cache
    identity must include every functional setting, but not credentials or
    transient execution paths. Execution must not sample another program.
    """

    def cache_key(self) -> object:
        """Identify the backend and its functional configuration."""
        ...

    def generate_source(self, goal: str, item: dict[str, Any]) -> str:
        """Return a validated function accepting a row and returning strings."""
        ...

    def execute(self, code: str, item: dict[str, Any]) -> list[str]:
        """Execute the supplied source afresh without generating another sample."""
        ...
