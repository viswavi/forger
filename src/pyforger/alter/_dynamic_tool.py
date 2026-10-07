"""Read-only resource and restricted Python tools for OpenHands dynamic steps."""

import json
import threading
import uuid
from typing import Literal

from openhands.sdk import Action, Observation, TextContent, ToolDefinition
from openhands.sdk.tool import ToolExecutor, register_tool

from pyforger.alter._dynamic_resources import prepare_input
from pyforger.alter.dynamic import _execute

_EXECUTORS = {}
_LOCK = threading.Lock()


def register_executor(executor):
    """Register a session executor and return its opaque identifier."""
    session_id = uuid.uuid4().hex
    with _LOCK:
        _EXECUTORS[session_id] = executor
    return session_id


def unregister_executor(session_id):
    """Remove a session executor from the process-local registry."""
    with _LOCK:
        _EXECUTORS.pop(session_id, None)


def _session(session_id):
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
    def to_llm_content(self):
        """Expose the observation as model-readable text."""
        return [TextContent(text=self.result_text)]


class PythonExecutor(ToolExecutor):
    """Test generated Python and retain a successful submission."""

    def __init__(self, item, timeout):
        """Store the input and the generated-code timeout."""
        self.item = item
        self.timeout = timeout
        self.accepted = []
        self._prepared = None

    def prepared_input(self):
        """Materialize the input once for all tests in this session."""
        if self._prepared is None:
            self._prepared = prepare_input(self.item)
        return self._prepared

    def __call__(self, action, conversation=None):
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
            result_text=json.dumps({"output": output, "submitted": action.submit})
        )


class ResourceExecutor(ToolExecutor):
    """Summarize the session's prepared input for the agent."""

    def __init__(self, session):
        """Bind the resource tool to one session."""
        self.session = session

    def __call__(self, action, conversation=None):
        """Return a bounded summary of the prepared input."""
        try:
            prepared = self.session.prepared_input()
            if not self.session.item.startswith("https://"):
                summary = {"kind": "text", "content": prepared[:20000]}
            else:
                resource = json.loads(prepared)
                if resource["kind"] == "git":
                    summary = {
                        "kind": "git",
                        "url": resource["url"],
                        "files": sorted(resource["files"]),
                    }
                else:
                    summary = {
                        "kind": resource["kind"],
                        "url": resource["url"],
                        "content": resource["content"][:20000],
                    }
            return ResultObservation(result_text=json.dumps(summary))
        except Exception as exc:
            return ResultObservation(result_text=f"{type(exc).__name__}: {exc}")


class GitExecutor(ToolExecutor):
    """Expose read-only list and read operations on a Git snapshot."""

    def __init__(self, session):
        """Bind the Git tool to one session."""
        self.session = session

    def __call__(self, action, conversation=None):
        """List or read text files from the cloned repository snapshot."""
        try:
            resource = json.loads(self.session.prepared_input())
            if resource["kind"] != "git":
                raise ValueError("The input is not a Git repository")
            files = resource["files"]
            if action.operation == "list":
                result = {"files": sorted(files)}
            elif action.path in files:
                result = {"path": action.path, "content": files[action.path][:20000]}
            else:
                raise FileNotFoundError(
                    "File is not in the bounded repository snapshot"
                )
            return ResultObservation(result_text=json.dumps(result))
        except Exception as exc:
            return ResultObservation(result_text=f"{type(exc).__name__}: {exc}")


class PythonTool(ToolDefinition):
    """Register the restricted Python test and submit tool."""

    @classmethod
    def create(cls, conv_state, session_id, **params):
        """Create the Python tool for a registered session."""
        return [
            cls(
                description=(
                    "Test Python code on the prepared input. Set submit=true to "
                    "save a working pipeline. Define the entry function last, "
                    "with one string argument and a string return value. HTTPS "
                    "inputs are JSON text with kind/url/content or kind/url/files; "
                    "other inputs are raw strings. Imports: json, re, math, "
                    "string, html only. No files, network, shell, or environment "
                    "access."
                ),
                action_type=PythonAction,
                observation_type=ResultObservation,
                executor=_session(session_id),
            )
        ]


class ResourceTool(ToolDefinition):
    """Register read-only resource inspection."""

    @classmethod
    def create(cls, conv_state, session_id, **params):
        """Create the resource tool for a registered session."""
        return [
            cls(
                description=(
                    "Inspect the prepared input. HTML and PDF return extracted "
                    "text; Git returns a bounded file listing. The Python tool "
                    "receives the same prepared input."
                ),
                action_type=ResourceAction,
                observation_type=ResultObservation,
                executor=ResourceExecutor(_session(session_id)),
            )
        ]


class GitTool(ToolDefinition):
    """Register read-only Git inspection."""

    @classmethod
    def create(cls, conv_state, session_id, **params):
        """Create the Git tool for a registered session."""
        return [
            cls(
                description=(
                    "Read-only Git repository inspection for the current HTTPS "
                    ".git input. The repository is cloned once with bounded size "
                    "and no credentials. Use operation='list' to see files, then "
                    "operation='read' and path to inspect a source file. The "
                    "Python tool receives these files as JSON text."
                ),
                action_type=GitAction,
                observation_type=ResultObservation,
                executor=GitExecutor(_session(session_id)),
            )
        ]


register_tool("ForgerPythonTool", PythonTool)
register_tool("ForgerResourceTool", ResourceTool)
register_tool("ForgerGitTool", GitTool)
