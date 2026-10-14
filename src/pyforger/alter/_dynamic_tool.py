"""Read-only resource and restricted Python tools for OpenHands dynamic steps."""

from __future__ import annotations

import json
import threading
import uuid
from typing import TYPE_CHECKING, Any, Literal

from openhands.sdk import Action, Observation, TextContent, ToolDefinition
from openhands.sdk.tool import ToolExecutor, register_tool

from pyforger.alter._dynamic_resources import prepare_input
from pyforger.alter.dynamic import _execute

if TYPE_CHECKING:
    from openhands.sdk.conversation import ConversationState, LocalConversation

_EXECUTORS: dict[str, PythonExecutor] = {}
_LOCK = threading.Lock()


def register_executor(executor: PythonExecutor) -> str:
    """Register a session executor and return its opaque identifier."""
    session_id = uuid.uuid4().hex
    with _LOCK:
        _EXECUTORS[session_id] = executor
    return session_id


def unregister_executor(session_id: str) -> None:
    """Remove a session executor from the process-local registry."""
    with _LOCK:
        _EXECUTORS.pop(session_id, None)


def _session(session_id: str) -> PythonExecutor:
    with _LOCK:
        return _EXECUTORS[session_id]


class PythonAction(Action):
    """Code to test and optionally submit as a pipeline."""

    code: str
    submit: bool = False


class ResourceAction(Action):
    """Request a summary of the current prepared input."""

    pass


class GitAction(Action):
    """List or read files from the current Git snapshot."""

    operation: Literal["list", "read"]
    path: str = ""


class ResultObservation(Observation):
    """Text returned by a restricted OpenHands tool."""

    result_text: str

    @property
    def to_llm_content(self) -> list[TextContent]:
        """Expose the observation as model-readable text."""
        return [TextContent(text=self.result_text)]


class PythonExecutor(ToolExecutor[PythonAction, ResultObservation]):
    """Test generated Python and retain a successful submission."""

    def __init__(self, item: dict[str, Any], timeout: float) -> None:
        """Store the input and the generated-code timeout."""
        self.item = item
        self.timeout = timeout
        self.accepted: list[str] = []
        self._prepared: dict[str, Any] | None = None

    def prepared_input(self) -> dict[str, Any]:
        """Materialize the input once for all tests in this session."""
        if self._prepared is None:
            self._prepared = prepare_input(self.item)
        return self._prepared

    def __call__(
        self, action: PythonAction, conversation: LocalConversation | None = None
    ) -> ResultObservation:
        """Run generated code against the prepared input."""
        try:
            output = _execute(
                action.code,
                self.item,
                self.timeout,
                prepared_input=self.prepared_input(),
            )
        except Exception as exc:
            return ResultObservation(result_text=f"{type(exc).__name__}: {exc}")
        if action.submit:
            self.accepted[:] = [action.code]
        return ResultObservation(
            result_text=json.dumps(
                {"output": output, "submitted": action.submit}, ensure_ascii=False
            )
        )


class ResourceExecutor(ToolExecutor[ResourceAction, ResultObservation]):
    """Summarize the session's prepared input for the agent."""

    def __init__(self, session: PythonExecutor) -> None:
        """Bind the resource tool to one session."""
        self.session = session

    def __call__(
        self, action: ResourceAction, conversation: LocalConversation | None = None
    ) -> ResultObservation:
        """Return a bounded summary of the prepared input."""
        try:
            summary: dict[str, Any] | str
            prepared = self.session.prepared_input()
            if prepared.get("kind") == "git" and isinstance(
                prepared.get("files"), dict
            ):
                summary = {
                    "kind": "git",
                    "url": prepared["url"],
                    "files": sorted(prepared["files"]),
                }
            else:
                summary = json.dumps(prepared, ensure_ascii=False)[:20000]
            return ResultObservation(
                result_text=(
                    summary
                    if isinstance(summary, str)
                    else json.dumps(summary, ensure_ascii=False)
                )
            )
        except Exception as exc:
            return ResultObservation(result_text=f"{type(exc).__name__}: {exc}")


class GitExecutor(ToolExecutor[GitAction, ResultObservation]):
    """Expose read-only list and read operations on a Git snapshot."""

    def __init__(self, session: PythonExecutor) -> None:
        """Bind the Git tool to one session."""
        self.session = session

    def __call__(
        self, action: GitAction, conversation: LocalConversation | None = None
    ) -> ResultObservation:
        """List or read text files from the cloned repository snapshot."""
        try:
            resource = self.session.prepared_input()
            if resource.get("kind") != "git" or not isinstance(
                resource.get("files"), dict
            ):
                raise ValueError("The input is not a Git repository")
            files = resource["files"]
            result: dict[str, str | list[str]]
            if action.operation == "list":
                result = {"files": sorted(files)}
            elif action.path in files:
                result = {"path": action.path, "content": files[action.path][:20000]}
            else:
                raise FileNotFoundError(
                    "File is not in the bounded repository snapshot"
                )
            return ResultObservation(result_text=json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            return ResultObservation(result_text=f"{type(exc).__name__}: {exc}")


class PythonTool(ToolDefinition[PythonAction, ResultObservation]):
    """Register the restricted Python test and submit tool."""

    @classmethod
    def create(
        cls, conv_state: ConversationState, session_id: str, **params: Any
    ) -> list[PythonTool]:
        """Create the Python tool for a registered session."""
        return [
            cls(
                description=(
                    "Test Python code on the prepared input. Set submit=true to "
                    "save a working pipeline. Define the entry function last, "
                    "with one dict argument and a list of string results "
                    "(which may be empty). URL-only HTTPS records gain "
                    "kind/content or kind/files; other records retain their fields. "
                    "Imports: json, re, math, "
                    "html only. No files, network, shell, or environment "
                    "access."
                ),
                action_type=PythonAction,
                observation_type=ResultObservation,
                executor=_session(session_id),
            )
        ]


class ResourceTool(ToolDefinition[ResourceAction, ResultObservation]):
    """Register read-only resource inspection."""

    @classmethod
    def create(
        cls, conv_state: ConversationState, session_id: str, **params: Any
    ) -> list[ResourceTool]:
        """Create the resource tool for a registered session."""
        return [
            cls(
                description=(
                    "Inspect the prepared input record. URL-only HTML and PDF "
                    "records gain extracted text; Git returns a bounded file "
                    "listing. The Python tool "
                    "receives the same prepared input."
                ),
                action_type=ResourceAction,
                observation_type=ResultObservation,
                executor=ResourceExecutor(_session(session_id)),
            )
        ]


class GitTool(ToolDefinition[GitAction, ResultObservation]):
    """Register read-only Git inspection."""

    @classmethod
    def create(
        cls, conv_state: ConversationState, session_id: str, **params: Any
    ) -> list[GitTool]:
        """Create the Git tool for a registered session."""
        return [
            cls(
                description=(
                    "Read-only Git repository inspection for a URL-only HTTPS "
                    ".git record. The repository is shallow-cloned and a bounded "
                    "text snapshot is kept. Use operation='list' to see files, then "
                    "operation='read' and path to inspect a source file. The "
                    "Python tool receives these files in the input dict."
                ),
                action_type=GitAction,
                observation_type=ResultObservation,
                executor=GitExecutor(_session(session_id)),
            )
        ]


register_tool("ForgerPythonTool", PythonTool)
register_tool("ForgerResourceTool", ResourceTool)
register_tool("ForgerGitTool", GitTool)
