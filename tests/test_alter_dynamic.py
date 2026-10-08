"""Tests for dynamic pipeline operations."""

import importlib
import importlib.metadata
import json
import threading
import time
from types import SimpleNamespace

import pytest

from pyforger.alter import dynamic_step, dynamic_step_source
from pyforger.op import clear_trace, trace

dynamic = importlib.import_module("pyforger.alter.dynamic")


@pytest.fixture(autouse=True)
def isolated_op_cache(tmp_path, monkeypatch):
    """Isolated op cache."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "op-cache"))


def test_source_cache_order_and_overlap(tmp_path, monkeypatch):
    """Test source cache order and overlap."""
    calls = []
    executions = []

    def generate(goal, item, *args):
        calls.append(item)
        return "def run(item):\n    return item.upper()"

    def execute(code, item, timeout):
        executions.append(item)
        return item.upper()

    monkeypatch.setattr(dynamic, "_generate", generate)
    monkeypatch.setattr(dynamic, "_execute", execute)
    model = SimpleNamespace(_model_id="test")
    assert dynamic_step(["a", "b", "a"], "uppercase", model) == ["A", "B", "A"]
    assert dynamic_step(["b", "c"], "uppercase", model) == ["B", "C"]
    assert sorted(calls) == ["a", "b", "c"]
    assert sorted(executions) == ["a", "b", "b", "c"]
    assert (
        len(list((tmp_path / "op-cache" / "alter.dynamic_step_source").glob("*.pkl")))
        == 3
    )


def test_parallel_limit(tmp_path, monkeypatch):
    """Test parallel limit."""
    lock = threading.Lock()
    active = peak = 0

    def generate(*args):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return "code"

    monkeypatch.setattr(dynamic, "_generate", generate)
    monkeypatch.setattr(dynamic, "_execute", lambda code, item, timeout: item)
    assert dynamic_step(
        list("abcd"), "goal", SimpleNamespace(_model_id="test"), max_parallel=2
    ) == list("abcd")
    assert peak == 2


def test_worker_restrictions_and_timeout():
    """Test worker restrictions and timeout."""
    version = importlib.metadata.version("RestrictedPython")
    if tuple(map(int, version.split(".")[:2])) < (8, 4):
        pytest.skip("Dynamic worker requires RestrictedPython >=8.4")
    try:
        dynamic._require_dynamic_platform()
    except NotImplementedError:
        pytest.skip("Dynamic worker requires Unix signal timers")
    assert (
        dynamic._execute("def run(item):\n    return item.upper()", "hello", 5)
        == "HELLO"
    )
    with pytest.raises(ValueError, match="not allowed"):
        dynamic._execute("import os\ndef run(item):\n    return os.environ", "hello", 5)
    with pytest.raises(ValueError, match="string"):
        dynamic._execute("def run(item):\n    return 3", "hello", 5)
    with pytest.raises(ValueError, match="SandboxTimeout"):
        dynamic._execute("def run(item):\n    while True:\n        pass", "hello", 0.5)


def test_empty_and_validation(tmp_path):
    """Test empty and validation."""
    assert dynamic_step([], "goal", SimpleNamespace(_model_id="test")) == []
    with pytest.raises(ValueError):
        dynamic_step([], "goal", None, max_parallel=0)


def test_failed_generation_is_not_cached(tmp_path, monkeypatch):
    """Test failed generation is not cached."""

    def fail(*args):
        raise RuntimeError("no submission")

    monkeypatch.setattr(dynamic, "_generate", fail)
    with pytest.raises(RuntimeError, match="no submission"):
        dynamic_step(["a"], "goal", SimpleNamespace(_model_id="test"))
    assert not list((tmp_path / "op-cache" / "alter.dynamic_step_source").glob("*.pkl"))


def test_source_is_returned_with_standard_op_cache_hit(tmp_path, monkeypatch):
    """Test source is returned with standard op cache hit."""
    monkeypatch.setattr(
        dynamic, "_generate", lambda *args: "def run(item):\n    return item"
    )
    model = SimpleNamespace(_model_id="test")
    clear_trace()
    first = dynamic_step_source("a", "goal", model)
    second = dynamic_step_source("a", "goal", model)
    assert first == second == "def run(item):\n    return item"
    hits = [
        record.cache_hit
        for record in trace()
        if record.op_name == "alter.dynamic_step_source"
    ]
    assert hits == [False, True]


def test_source_version_refreshes_generated_program(monkeypatch):
    """A new source version bypasses a stale cached program."""
    calls = []

    def generate(*args):
        calls.append(None)
        return f"def run(item): return {len(calls)!r}"

    monkeypatch.setattr(dynamic, "_generate", generate)
    model = SimpleNamespace(_model_id="test")
    first = dynamic_step_source("a", "goal", model)
    assert dynamic_step_source("a", "goal", model) == first
    refreshed = dynamic_step_source("a", "goal", model, source_version="refresh-1")
    assert refreshed != first
    assert len(calls) == 2


def test_source_inspection_never_generates_when_disabled(tmp_path, monkeypatch):
    """Test source inspection never generates when disabled."""
    monkeypatch.setattr(
        dynamic, "_generate", lambda *args: pytest.fail("agent launched")
    )
    with pytest.raises(FileNotFoundError, match="No cached result"):
        dynamic_step_source(
            "a", "goal", SimpleNamespace(_model_id="test"), _cache_only=True
        )


def test_execution_timeout_reaches_worker(monkeypatch):
    """Pass the requested sandbox timeout with a subprocess cleanup margin."""

    def run(*args, **kwargs):
        """Inspect the worker request without launching the process."""
        assert json.loads(kwargs["input"])["timeout_s"] == 30
        assert kwargs["timeout"] == 32
        return SimpleNamespace(returncode=0, stdout='"ok"', stderr="")

    monkeypatch.setattr(dynamic.subprocess, "run", run)
    assert dynamic._execute("def run(item): return 'ok'", "plain", 30) == "ok"


def test_agent_retries_text_and_keeps_successful_submission(monkeypatch):
    """Retry a text-only turn and retain code after a later SDK error."""
    sdk = pytest.importorskip("openhands.sdk")
    from pyforger.alter import _dynamic_tool as tool

    calls = []

    class FakeConversation:
        def __init__(self, *, agent, **kwargs):
            self.agent = agent
            self.state = SimpleNamespace(events=[])
            assert kwargs["max_iteration_per_run"] == 3
            assert "git" in agent.filter_tools_regex
            assert "TerminalTool" not in agent.filter_tools_regex

        def send_message(self, text):
            calls.append(("message", text))

        def run(self):
            calls.append(("run", None))
            self.state.events.append(
                SimpleNamespace(
                    source="agent", llm_response_id=str(len(self.state.events))
                )
            )
            if len([x for x in calls if x[0] == "run"]) == 2:
                session_id = self.agent.tools[0].params["session_id"]
                tool._session(session_id).accepted[:] = ["def run(item): return item"]
                raise RuntimeError("SDK failed after submission")

        def close(self):
            calls.append(("close", None))

    monkeypatch.setattr(sdk, "Conversation", FakeConversation)
    monkeypatch.setattr(sdk, "LLM", lambda **kwargs: object())
    monkeypatch.setattr(sdk, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(dynamic, "prepare_input", lambda item: item)
    monkeypatch.setattr(tool, "prepare_input", lambda item: item)
    model = SimpleNamespace(
        _model_id="test", _temperature=0, _kwargs={}, _api_key=None, _max_tokens=None
    )
    assert dynamic._generate("goal", "input", model, 3, 5).startswith("def run")
    assert len([x for x in calls if x[0] == "run"]) == 2
    assert len([x for x in calls if x[0] == "message"]) == 2


def test_bad_resource_fails_before_agent_creation(monkeypatch):
    """Invalid input should not incur LLM turns."""
    sdk = pytest.importorskip("openhands.sdk")
    from pyforger.alter import _dynamic_tool as tool

    monkeypatch.setattr(
        tool,
        "prepare_input",
        lambda item: (_ for _ in ()).throw(ValueError("bad resource")),
    )
    monkeypatch.setattr(
        sdk,
        "LLM",
        lambda **kwargs: pytest.fail("LLM must not be created"),
    )
    model = SimpleNamespace(_model_id="test")
    with pytest.raises(ValueError, match="bad resource"):
        dynamic._generate("goal", "https://example.org/missing", model, 3, 5)


def test_unsupported_platform_fails_before_agent(monkeypatch):
    """Reject missing Unix timers before any SDK or resource work."""
    monkeypatch.delattr(dynamic.signal, "setitimer", raising=False)
    with pytest.raises(NotImplementedError, match="setitimer"):
        dynamic._generate("goal", "input", None, 3, 5)


def test_real_agent_retains_restricted_tools(monkeypatch):
    """Check resolved SDK tool names survive the agent's actual filter."""
    pytest.importorskip("openhands.sdk")
    from openhands.sdk import Agent
    from openhands.sdk.testing import TestLLM
    from openhands.sdk.tool import Tool

    from pyforger.alter import _dynamic_tool as tool

    monkeypatch.setattr(
        "openhands.sdk.agent.base.has_vision_profile_available", lambda: False
    )
    session_id = tool.register_executor(tool.PythonExecutor("plain input", 5))
    try:
        agent = Agent(
            llm=TestLLM.from_messages([]),
            tools=[
                Tool(name=name, params={"session_id": session_id})
                for name in ("ForgerPythonTool", "ForgerResourceTool", "ForgerGitTool")
            ],
            include_default_tools=["FinishTool"],
            filter_tools_regex=r"^(python|resource|git)$",
        )
        agent._initialize(None)
        assert {"python", "resource", "git", "finish"} <= set(agent._tools)
    finally:
        tool.unregister_executor(session_id)
