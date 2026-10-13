"""Session-scoped read-only paginated evidence for agent conversations."""

from __future__ import annotations

import json
import re

from openhands.sdk import Action, Observation, ToolDefinition
from openhands.sdk.tool import ToolExecutor, register_tool
from pydantic import Field

PAGE_SIZE = 12000
SEARCH_MATCHES = 8
SEARCH_CONTEXT = 600
SESSIONS: dict[str, EvidenceReader] = {}


class EvidenceAction(Action):
    """Select a bounded page or search from an explicit evidence artifact."""

    artifact: str = Field(description="Exact artifact ID, or empty to list IDs")
    offset: int = Field(default=0, ge=0)
    query: str | None = Field(default=None, min_length=1, max_length=512)
    # Requests above the page size are clamped, not rejected: the reviewer is
    # asking for "more" and a page is the honest answer, whereas a validation
    # error yields no evidence and burns the reviewer's iteration budget.
    limit: int = Field(default=PAGE_SIZE, ge=1)


class EvidenceObservation(Observation):
    """Return read-only evidence as model-visible text."""

    pass


class EvidenceReader(ToolExecutor):
    """Read only the registered evidence snapshot."""

    def __init__(self, artifacts):
        """Snapshot trusted inputs and execution configuration."""
        self.artifacts = dict(artifacts)

    def __call__(self, action, conversation=None):
        """Return a bounded evidence page or literal search matches."""
        if not action.artifact:
            if action.query is not None:
                return EvidenceObservation.from_text(
                    text="An artifact ID is required for search"
                )
            text = json.dumps({k: len(v) for k, v in self.artifacts.items()})
        elif action.artifact in self.artifacts:
            text = self.artifacts[action.artifact]
        else:
            return EvidenceObservation.from_text(text="Unknown artifact ID")
        if action.query is not None:
            matches = []
            total_matches = 0
            pattern = re.compile(re.escape(action.query), re.IGNORECASE)
            for match in pattern.finditer(text):
                total_matches += 1
                if len(matches) >= SEARCH_MATCHES:
                    continue
                index = match.start()
                start = max(0, index - SEARCH_CONTEXT)
                end = min(len(text), match.end() + SEARCH_CONTEXT)
                matches.append({"offset": index, "content": text[start:end]})
            return EvidenceObservation.from_text(
                text=json.dumps(
                    {
                        "matches": matches,
                        "total_matches": total_matches,
                        "truncated": total_matches > len(matches),
                    }
                )
            )
        end = min(action.offset + min(action.limit, PAGE_SIZE), len(text))
        return EvidenceObservation.from_text(
            text=json.dumps(
                {
                    "content": text[action.offset : end],
                    "total_characters": len(text),
                    "next_offset": end if end < len(text) else None,
                }
            )
        )


class EvidenceTool(ToolDefinition[EvidenceAction, EvidenceObservation]):
    """Expose bounded evidence reads without filesystem capabilities."""

    @classmethod
    def create(cls, conv_state, session_id):
        """Bind the evidence tool to its registered session."""
        return [
            cls(
                description=(
                    f"Read immutable evidence pages of at most {PAGE_SIZE} chars, or "
                    f"search for up to {SEARCH_MATCHES} bounded matches. "
                    "Limits are clamped to one page. Empty artifact lists IDs. "
                    "No filesystem access."
                ),
                action_type=EvidenceAction,
                observation_type=EvidenceObservation,
                executor=SESSIONS[session_id],
            )
        ]


register_tool(EvidenceTool.name, EvidenceTool)
