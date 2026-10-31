"""Tests for dynamic pipeline operations."""

import importlib
import importlib.metadata
import json
import threading
import time
from types import SimpleNamespace

import pytest

from pyforger import agents, local_agent
from pyforger.acquire import Model
from pyforger.agent_backend import AgentRunResult, AgentSpec
from pyforger.alter import dynamic_step, dynamic_step_source
from pyforger.local_backend import LocalBackend
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
        agents,
        "generate_source",
        lambda goal, item, **kw: generate(goal, item),
    )
    monkeypatch.setattr("pyforger.local_backend._execute", execute)
    model = Model("openai/test")
    first = dynamic_step(
        [{"text": "a"}, {"text": "b"}, {"text": "a"}],
        "uppercase",
        AgentSpec(model),
        LocalBackend(),
    )
    assert first == [["A", "a"], ["B", "b"], ["A", "a"]]
    assert first[0] is not first[2]
    assert dynamic_step(
        [{"text": "b"}, {"text": "c"}], "uppercase", AgentSpec(model), LocalBackend()
    ) == [
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
        agents,
        "generate_source",
        lambda goal, item, **kw: generate(goal, item),
    )
    monkeypatch.setattr(
        "pyforger.local_backend._execute", lambda code, item, timeout: [item["text"]]
    )
    assert dynamic_step(
        [{"text": x} for x in "abcd"],
        "goal",
        AgentSpec(Model("openai/test")),
        LocalBackend(),
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
    agent, backend = AgentSpec(Model("openai/test")), LocalBackend()
    assert dynamic_step([], "goal", agent, backend) == []
    with pytest.raises(ValueError):
        dynamic_step([], "goal", agent, backend, max_parallel=0)
    with pytest.raises(TypeError, match="each input must be a dict"):
        dynamic_step(["plain text"], "goal", agent, backend)
    with pytest.raises(TypeError, match="JSON-compatible"):
        dynamic_step([{"bad": {1, 2}}], "goal", agent, backend)


def test_records_with_reordered_keys_share_generation(monkeypatch):
    """Nested record key order must not launch a duplicate agent."""
    calls = []
    monkeypatch.setattr(
        agents, "generate_source", lambda *args, **kw: calls.append(1) or "code"
    )
    monkeypatch.setattr(
        "pyforger.local_backend._execute", lambda code, item, timeout: [item["text"]]
    )
    first = {"text": "challenge", "meta": {"a": 1, "b": 2}}
    second = {"meta": {"b": 2, "a": 1}, "text": "challenge"}
    assert dynamic_step(
        [first, second], "goal", AgentSpec(Model("openai/test")), LocalBackend()
    ) == [
        ["challenge"],
        ["challenge"],
    ]
    assert len(calls) == 1


def test_failed_generation_is_not_cached(tmp_path, monkeypatch):
    """Test failed generation is not cached."""

    def fail(*args):
        raise RuntimeError("no submission")

    monkeypatch.setattr(
        agents, "generate_source", lambda goal, item, **kw: fail(goal, item)
    )
    with pytest.raises(RuntimeError, match="no submission"):
        dynamic_step(
            [{"text": "a"}], "goal", AgentSpec(Model("openai/test")), LocalBackend()
        )
    assert not list((tmp_path / "op-cache" / "alter.dynamic_step_source").glob("*.pkl"))


def test_source_is_returned_with_standard_op_cache_hit(tmp_path, monkeypatch):
    """Test source is returned with standard op cache hit."""
    monkeypatch.setattr(
        agents,
        "generate_source",
        lambda *args, **kw: "def run(item):\n    return [item]",
    )
    model = Model("openai/test")
    clear_trace()
    first = dynamic_step_source({"text": "a"}, "goal", AgentSpec(model), LocalBackend())
    second = dynamic_step_source(
        {"text": "a"}, "goal", AgentSpec(model), LocalBackend()
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
        agents,
        "generate_source",
        lambda goal, item, **kw: generate(goal, item),
    )
    model = Model("openai/test")
    first = dynamic_step_source({"text": "a"}, "goal", AgentSpec(model), LocalBackend())
    assert (
        dynamic_step_source({"text": "a"}, "goal", AgentSpec(model), LocalBackend())
        == first
    )
    refreshed = dynamic_step_source(
        {"text": "a"},
        "goal",
        AgentSpec(model),
        LocalBackend(),
        source_version="refresh-1",
    )
    assert refreshed != first
    assert len(calls) == 2


def test_source_inspection_never_generates_when_disabled(tmp_path, monkeypatch):
    """Test source inspection never generates when disabled."""
    monkeypatch.setattr(
        agents, "generate_source", lambda *args, **kw: pytest.fail("agent launched")
    )
    with pytest.raises(FileNotFoundError, match="No cached result"):
        dynamic_step_source(
            {"text": "a"},
            "goal",
            AgentSpec(Model("openai/test")),
            LocalBackend(),
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
    assert llm.reasoning_effort == model.config.reasoning_effort
    assert llm.litellm_extra_body == model.config.extra_body
    assert llm.api_mode == model.config.api_mode
    assert llm.capability_overrides == model.config.capability_overrides
    if not options:
        assert llm.timeout == LLM.model_fields["timeout"].default
        assert llm.reasoning_effort is None


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
    assert LocalBackend().execute_program(
        "def run(item): return [item['text'].upper()]", {"text": "hello"}, 5
    ) == ["HELLO"]


def test_any_backend_shares_dynamic_step_contract(tmp_path):
    """A stub backend preserves source caching/inspection and execution."""

    class StubBackend:
        name = "stub"
        generated = 0
        executed = 0

        def cache_key(self):
            return "stub-v1"

        def run_agent(self, task, agent):
            type(self).generated += 1
            return AgentRunResult(
                status="completed",
                files={"program.py": b"def run(item): return [item['text'].upper()]"},
            )

        def execute_program(self, code, item, timeout_s):
            type(self).executed += 1
            return LocalBackend().execute_program(code, item, timeout_s)

    backend = StubBackend()
    agent = AgentSpec(kind="oracle", solution="true")
    rows = [{"text": "hello"}]
    assert dynamic_step(rows, "uppercase", agent, backend) == [["HELLO"]]
    assert dynamic_step(rows, "uppercase", agent, backend) == [["HELLO"]]
    # One generate (validated by one replay) plus two dynamic executions.
    assert backend.generated == 1 and backend.executed == 3
    assert dynamic_step_source.cache_only(
        rows[0], "uppercase", agent, backend
    ).startswith("def")


def test_dynamic_step_requires_agent_and_backend():
    """The operation does not hide agent or backend configuration."""
    with pytest.raises(TypeError):
        dynamic_step([{"text": "hello"}], "uppercase")
