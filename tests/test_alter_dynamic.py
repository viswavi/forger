"""Tests for dynamic pipeline operations."""

import importlib
import importlib.metadata
import json
import threading
import time
from types import SimpleNamespace

import pytest
from completion import completion_server

from pyforger import local_agent
from pyforger.acquire import Model
from pyforger.acquire.model import ModelSettings
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
        calls.append(item["text"])
        return "def run(item):\n    return [item['text'].upper(), item['text'].lower()]"

    def execute(code, item, timeout):
        executions.append(item["text"])
        return [item["text"].upper(), item["text"].lower()]

    monkeypatch.setattr(
        local_agent.LocalAgentRunner,
        "generate_source",
        lambda self, goal, item: generate(
            goal, item, self.model, self.max_iterations, self.execution_timeout_s
        ),
    )
    monkeypatch.setattr(local_agent, "_execute", execute)
    model = Model("openai/test")
    first = dynamic_step(
        [{"text": "a"}, {"text": "b"}, {"text": "a"}], "uppercase", model
    )
    assert first == [["A", "a"], ["B", "b"], ["A", "a"]]
    assert first[0] is not first[2]
    assert dynamic_step([{"text": "b"}, {"text": "c"}], "uppercase", model) == [
        ["B", "b"],
        ["C", "c"],
    ]
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

    monkeypatch.setattr(
        local_agent.LocalAgentRunner,
        "generate_source",
        lambda self, goal, item: generate(
            goal, item, self.model, self.max_iterations, self.execution_timeout_s
        ),
    )
    monkeypatch.setattr(
        local_agent, "_execute", lambda code, item, timeout: [item["text"]]
    )
    assert dynamic_step(
        [{"text": x} for x in "abcd"],
        "goal",
        Model("openai/test"),
        max_parallel=2,
    ) == [[item] for item in "abcd"]
    assert peak == 2


def test_worker_restrictions_and_timeout():
    """Test worker restrictions and timeout."""
    version = importlib.metadata.version("RestrictedPython")
    if tuple(map(int, version.split(".")[:2])) < (8, 4):
        pytest.skip("Dynamic worker requires RestrictedPython >=8.4")
    try:
        local_agent._require_dynamic_platform()
    except NotImplementedError:
        pytest.skip("Dynamic worker requires Unix signal timers")
    assert local_agent._execute(
        "def run(item):\n    return [item['text'].lower(), item['text'].upper()]",
        {"text": "hello"},
        5,
    ) == ["hello", "HELLO"]
    record = {
        "text": "# Python Challenges\nIs Disarium Number?",
        "id": "https://programiz.pro/community-challenges/python",
        "title": "Python Coding Challenges",
        "url": "https://programiz.pro/community-challenges/python",
        "query": "Very hard programming challenges in Python",
    }
    assert local_agent._execute(
        "def run(item):\n    return [item['title'], item['text'], item['query']]",
        record,
        5,
    ) == [record["title"], record["text"], record["query"]]
    assert (
        local_agent._execute("def run(item):\n    return []", {"text": "hello"}, 5)
        == []
    )
    with pytest.raises(ValueError, match="not allowed"):
        local_agent._execute(
            "import os\ndef run(item):\n    return os.environ", {"text": "hello"}, 5
        )
    with pytest.raises(ValueError, match="list of strings"):
        local_agent._execute("def run(item):\n    return 3", {"text": "hello"}, 5)
    with pytest.raises(ValueError, match="list of strings"):
        local_agent._execute(
            "def run(item):\n    return ['ok', 3]", {"text": "hello"}, 5
        )
    with pytest.raises(ValueError, match="SandboxTimeout"):
        local_agent._execute(
            "def run(item):\n    while True:\n        pass", {"text": "hello"}, 0.5
        )


def test_empty_and_validation(tmp_path):
    """Test empty and validation."""
    assert dynamic_step([], "goal", Model("openai/test")) == []
    with pytest.raises(ValueError):
        dynamic_step([], "goal", None, max_parallel=0)
    with pytest.raises(TypeError, match="each input must be a dict"):
        dynamic_step(["plain text"], "goal", None)
    with pytest.raises(TypeError, match="JSON-compatible"):
        dynamic_step([{"bad": {1, 2}}], "goal", None)


def test_records_with_reordered_keys_share_generation(monkeypatch):
    """Nested record key order must not launch a duplicate agent."""
    calls = []
    monkeypatch.setattr(
        local_agent.LocalAgentRunner,
        "generate_source",
        lambda *args: calls.append(1) or "code",
    )
    monkeypatch.setattr(
        local_agent, "_execute", lambda code, item, timeout: [item["text"]]
    )
    first = {"text": "challenge", "meta": {"a": 1, "b": 2}}
    second = {"meta": {"b": 2, "a": 1}, "text": "challenge"}
    assert dynamic_step([first, second], "goal", Model("openai/test")) == [
        ["challenge"],
        ["challenge"],
    ]
    assert len(calls) == 1


def test_failed_generation_is_not_cached(tmp_path, monkeypatch):
    """Test failed generation is not cached."""

    def fail(*args):
        raise RuntimeError("no submission")

    monkeypatch.setattr(
        local_agent.LocalAgentRunner,
        "generate_source",
        lambda self, goal, item: fail(
            goal, item, self.model, self.max_iterations, self.execution_timeout_s
        ),
    )
    with pytest.raises(RuntimeError, match="no submission"):
        dynamic_step([{"text": "a"}], "goal", Model("openai/test"))
    assert not list((tmp_path / "op-cache" / "alter.dynamic_step_source").glob("*.pkl"))


def test_source_is_returned_with_standard_op_cache_hit(tmp_path, monkeypatch):
    """Test source is returned with standard op cache hit."""
    monkeypatch.setattr(
        local_agent.LocalAgentRunner,
        "generate_source",
        lambda *args: "def run(item):\n    return [item]",
    )
    model = Model("openai/test")
    clear_trace()
    first = dynamic_step_source(
        {"text": "a"}, "goal", local_agent.LocalAgentRunner(model)
    )
    second = dynamic_step_source(
        {"text": "a"}, "goal", local_agent.LocalAgentRunner(model)
    )
    assert first == second == "def run(item):\n    return [item]"
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
        return f"def run(item): return [{str(len(calls))!r}]"

    monkeypatch.setattr(
        local_agent.LocalAgentRunner,
        "generate_source",
        lambda self, goal, item: generate(
            goal, item, self.model, self.max_iterations, self.execution_timeout_s
        ),
    )
    model = Model("openai/test")
    first = dynamic_step_source(
        {"text": "a"}, "goal", local_agent.LocalAgentRunner(model)
    )
    assert (
        dynamic_step_source({"text": "a"}, "goal", local_agent.LocalAgentRunner(model))
        == first
    )
    refreshed = dynamic_step_source(
        {"text": "a"},
        "goal",
        local_agent.LocalAgentRunner(model),
        source_version="refresh-1",
    )
    assert refreshed != first
    assert len(calls) == 2


def test_source_inspection_never_generates_when_disabled(tmp_path, monkeypatch):
    """Test source inspection never generates when disabled."""
    monkeypatch.setattr(
        local_agent.LocalAgentRunner,
        "generate_source",
        lambda *args: pytest.fail("agent launched"),
    )
    with pytest.raises(FileNotFoundError, match="No cached result"):
        dynamic_step_source(
            {"text": "a"},
            "goal",
            local_agent.LocalAgentRunner(Model("openai/test")),
            _cache_only=True,
        )


def test_execution_timeout_reaches_worker(monkeypatch):
    """Pass the requested sandbox timeout with a subprocess cleanup margin."""

    def run(*args, **kwargs):
        """Inspect the worker request without launching the process."""
        request = json.loads(kwargs["input"])
        assert request["timeout_s"] == 30
        assert request["item"] == {"text": "plain"}
        assert kwargs["timeout"] == 32
        return SimpleNamespace(returncode=0, stdout='["ok"]', stderr="")

    monkeypatch.setattr(local_agent.subprocess, "run", run)
    assert local_agent._execute(
        "def run(item): return ['ok']", {"text": "plain"}, 30
    ) == ["ok"]


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
                tool._session(session_id).accepted[:] = [
                    "def run(item): return [item['text']]"
                ]
                raise RuntimeError("SDK failed after submission")

        def close(self):
            calls.append(("close", None))

    monkeypatch.setattr(sdk, "Conversation", FakeConversation)
    monkeypatch.setattr(local_agent, "agent_llm", lambda model: object())
    monkeypatch.setattr(sdk, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(local_agent, "prepare_input", lambda item: item)
    monkeypatch.setattr(tool, "prepare_input", lambda item: item)
    model = SimpleNamespace(config=ModelSettings(model_id="test", temperature=0))
    assert (
        local_agent.LocalAgentRunner(model, max_iterations=3, execution_timeout_s=5)
        .generate_source("goal", {"text": "input"})
        .startswith("def run")
    )
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
    model = Model("openai/test")
    with pytest.raises(ValueError, match="bad resource"):
        local_agent.LocalAgentRunner(
            model, max_iterations=3, execution_timeout_s=5
        ).generate_source("goal", {"url": "https://example.org/missing"})


def test_unsupported_platform_fails_before_agent(monkeypatch):
    """Reject missing Unix timers before any SDK or resource work."""
    monkeypatch.delattr(local_agent.signal, "setitimer", raising=False)
    with pytest.raises(NotImplementedError, match="setitimer"):
        local_agent.LocalAgentRunner(
            None, max_iterations=3, execution_timeout_s=5
        ).generate_source("goal", {"text": "input"})


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
    session_id = tool.register_executor(tool.PythonExecutor({"text": "plain input"}, 5))
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


def test_real_sdk_corrects_code_and_reexecutes_cached_source(tmp_path, monkeypatch):
    """Only the paid API is scripted; tools, workers, and caching are real."""
    invalid = "def run(item): return 3"
    corrected = "def run(item): return [item['text'].upper()]"

    def submit(code, call_id):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": code, "submit": True}),
                    },
                }
            ],
        }

    executions = []
    real_run = local_agent.subprocess.run

    def record_run(command, **kwargs):
        if command[-1].endswith("_dynamic_worker.py"):
            executions.append(json.loads(kwargs["input"])["code"])
        return real_run(command, **kwargs)

    monkeypatch.setattr(local_agent.subprocess, "run", record_run)
    clear_trace()
    with completion_server(
        [
            submit(invalid, "invalid"),
            submit(corrected, "corrected"),
            {"role": "assistant", "content": "Finished."},
        ]
    ) as (url, requests):
        model = Model(
            "openai/test",
            api_key="test-not-a-real-secret",
            api_base=url,
            num_retries=0,
            timeout=5,
            temperature=None,
            capability_overrides={"supports_vision": True},
        )
        options = {"max_iterations": 3, "execution_timeout_s": 5}
        assert dynamic_step(
            [{"text": "hello"}], "Uppercase text", model, **options
        ) == [["HELLO"]]
        assert len(requests) == 3
        assert "list of strings" in json.dumps(requests[1]["messages"])
        assert executions == [invalid, corrected, corrected]
        tools = {entry["function"]["name"] for entry in requests[0]["tools"]}
        assert tools == {"python", "resource", "git", "finish"}

    # The API is closed: a cache miss cannot silently generate another program.
    assert dynamic_step([{"text": "hello"}], "Uppercase text", model, **options) == [
        ["HELLO"]
    ]
    assert executions == [invalid, corrected, corrected, corrected]
    assert (
        dynamic_step_source(
            {"text": "hello"},
            "Uppercase text",
            local_agent.LocalAgentRunner(model, **options),
            _cache_only=True,
        )
        == corrected
    )
    assert [
        entry.cache_hit
        for entry in trace()
        if entry.op_name == "alter.dynamic_step_source"
    ] == [False, True, True]


@pytest.mark.parametrize(
    "options",
    [
        {},
        {
            "timeout": None,
            "reasoning_effort": None,
            "extra_body": {"custom": 1},
            "api_mode": "chat",
            "capability_overrides": {"vision": False},
        },
    ],
)
def test_agent_llm_real_settings_translation(options, monkeypatch, tmp_path):
    """Validate the adapter against the actual SDK, without making API calls."""
    from openhands.sdk import LLM

    monkeypatch.chdir(tmp_path)
    model = Model(
        "gpt-4o",
        api_key="test-key",
        api_base="http://localhost:1234",
        temperature=0,
        max_tokens=123,
        num_retries=0,
        **options,
    )
    llm = local_agent.agent_llm(model)
    assert llm.model == model.config.model_id
    assert llm.temperature == 0
    assert llm.max_output_tokens == 123
    assert llm.num_retries == 1
    assert llm.base_url == model.config.api_base
    assert llm.api_key.get_secret_value() == "test-key"
    assert llm.timeout == (
        model.config.timeout if model.config.timeout is not None else 300
    )
    assert llm.reasoning_effort == (model.config.reasoning_effort or "high")
    assert llm.litellm_extra_body == model.config.extra_body
    assert llm.api_mode == model.config.api_mode
    assert llm.capability_overrides == model.config.capability_overrides
    if not options:
        assert llm.timeout == LLM.model_fields["timeout"].default
        assert llm.reasoning_effort == LLM.model_fields["reasoning_effort"].default


def test_agent_llm_env_fallback(monkeypatch, tmp_path):
    """Agent llm env fallback."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LITELLM_API_KEY", "env-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:9876")
    llm = local_agent.agent_llm(Model("gpt-4o"))
    assert llm.api_key.get_secret_value() == "env-key"
    assert llm.base_url == "http://localhost:9876"


def test_local_runner_executes_saved_source_without_model():
    """A retained program can be executed without starting another conversation."""
    runner = local_agent.LocalAgentRunner(Model("openai/test"), execution_timeout_s=5)
    assert runner.execute(
        "def run(item): return [item['text'].upper()]", {"text": "hello"}
    ) == ["HELLO"]


def test_explicit_local_backend_shares_dynamic_step_contract(tmp_path):
    """An explicit backend preserves source inspection, caching, and execution."""

    class ProgramRunner:
        def __init__(self, identity):
            self.identity = identity
            self.generated = 0
            self.executed = 0

        def cache_key(self):
            return self.identity

        def generate_source(self, goal, item):
            self.generated += 1
            return "def run(item): return [item['text'].upper()]"

        def execute(self, code, item):
            self.executed += 1
            return local_agent._execute(code, item, 5)

    first = ProgramRunner("backend-one")
    rows = [{"text": "hello"}]
    assert dynamic_step(rows, "uppercase", runner=first) == [["HELLO"]]
    assert dynamic_step(rows, "uppercase", runner=first) == [["HELLO"]]
    assert first.generated == 1 and first.executed == 2
    assert dynamic_step_source.cache_only(
        rows[0], "uppercase", runner=first
    ).startswith("def")
    other = ProgramRunner("backend-two")
    assert dynamic_step(rows, "uppercase", runner=other) == [["HELLO"]]
    assert other.generated == 1


def test_explicit_backend_takes_precedence_over_default_configuration(monkeypatch):
    """A supplied backend bypasses construction of the default local runner."""
    runner = local_agent.LocalAgentRunner(Model("openai/test"))
    monkeypatch.setattr(runner, "generate_source", lambda goal, item: "retained source")
    monkeypatch.setattr(
        dynamic, "_default_runner", lambda *args: pytest.fail("default constructed")
    )
    assert dynamic_step_source.no_cache({"x": 1}, "goal", runner) == "retained source"
