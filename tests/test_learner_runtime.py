"""Explicit runtime requests never silently degrade to backend defaults."""

import tempfile
from dataclasses import replace

import pytest

from pyforger.acquire.model import Model
from pyforger.agent_backend import AgentSpec, LearnerRuntime, WorkspaceTask
from pyforger.agent_backends import require_learner_runtime
from pyforger.harbor_backend import HarborBackend
from pyforger.local_backend import LocalBackend
from pyforger.sail_backend import SailBackend


def spec(**settings):
    """Build an SDK agent with explicit runtime settings."""
    return AgentSpec(
        model=Model("openai/test", api_base="http://localhost:1/v1"),
        runtime=LearnerRuntime(**settings),
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"tools": "unknown"},
        {"reasoning_replay": 1},
        {"condenser_max_tokens": True},
        {"condenser_max_tokens": 0},
        {"condenser_max_tokens": -1},
        {"condenser_max_tokens": 1.5},
        {"condenser_max_tokens": "104857"},
        {"condenser_max_events": True},
        {"condenser_max_events": 0},
        {"condenser_max_events": -1},
        {"condenser_max_events": 1.5},
        {"condenser_max_events": "160"},
    ],
)
def test_runtime_validation(settings):
    """Reject invalid runtime settings before execution."""
    with pytest.raises(ValueError):
        LearnerRuntime(**settings)


def test_runtime_identity():
    """Include runtime settings in agent identity and reject deterministic kinds."""
    assert spec().cache_key() != spec(tools="science-rl").cache_key()
    assert spec().cache_key() != spec(condenser_max_tokens=104857).cache_key()
    with pytest.raises(ValueError, match="Only SDK"):
        AgentSpec(kind="nop", runtime=LearnerRuntime())


@pytest.mark.parametrize("backend", [LocalBackend(), SailBackend(), HarborBackend()])
@pytest.mark.parametrize("settings", [{"reasoning_replay": True}])
def test_unsupported_history_fails_before_execution(backend, settings):
    """Reject unsupported reasoning replay before starting a backend."""
    agent = spec(**settings)
    with pytest.raises(NotImplementedError, match="shared learner runner"):
        backend.run_agent(WorkspaceTask(instruction="test"), agent)


@pytest.mark.parametrize("backend", [SailBackend(), HarborBackend()])
def test_unsupported_tools_fail_before_execution(backend):
    """Reject unavailable tool profiles before starting a backend."""
    with pytest.raises(NotImplementedError, match="tool surface"):
        backend.run_agent(WorkspaceTask(instruction="test"), spec(tools="science-rl"))


def test_local_science_tools_session(tmp_path):
    """Resolve the requested science tool surface without ambient skills."""
    session = LocalBackend()._sdk_session(
        tmp_path, WorkspaceTask(instruction="test"), spec(tools="science-rl")
    )
    agent = session.agent
    assert [tool.name for tool in agent.tools] == [
        "terminal",
        "file_editor",
        "task_tracker",
    ]
    assert agent.include_default_tools == ["FinishTool", "ThinkTool"]
    assert agent.agent_context.load_user_skills is False
    assert agent.agent_context.load_public_skills is False
    assert agent.agent_context.load_project_skills is False
    require_learner_runtime(LocalBackend(), spec(tools="science-rl"))


def test_local_tool_profile_rejects_conflicting_kwargs(tmp_path):
    """Reject kwargs that override the explicit tool profile."""
    from dataclasses import replace

    agent = replace(spec(tools="science-rl"), kwargs={"agent_context": None})
    with pytest.raises(ValueError, match="conflict"):
        LocalBackend()._sdk_session(tmp_path, WorkspaceTask(instruction="test"), agent)


def test_validated_local_rejects_history_before_workspace(tmp_path):
    """Apply history capability checks to validated sessions too."""
    with pytest.raises(NotImplementedError, match="shared learner runner"):
        LocalBackend().run_agent_validated(
            WorkspaceTask(instruction="test"),
            spec(reasoning_replay=True),
            validate=lambda files: None,
        )


@pytest.mark.parametrize("backend", [LocalBackend(), SailBackend()])
def test_sdk_backends_accept_condensation(backend):
    """Accept explicit condensation on native SDK backends."""
    require_learner_runtime(backend, spec(condenser_max_tokens=104857))


def test_harbor_accepts_and_serializes_condensation(monkeypatch):
    """Validate native configuration without invoking remote model transport."""
    monkeypatch.setenv("LITELLM_API_KEY", "fixture-local-only")
    backend = HarborBackend()
    agent = spec(condenser_max_tokens=104857)
    require_learner_runtime(backend, agent)
    config = backend._agent_config(agent)
    assert config.env["FORGER_CONDENSER_MAX_TOKENS"] == "104857"
    assert config.env["LLM_API_KEY"] == "${LITELLM_API_KEY}"


def test_local_condenser_uses_agent_model(tmp_path):
    """Share the agent model with its task-preserving condenser."""
    session = LocalBackend()._sdk_session(
        tmp_path, WorkspaceTask(instruction="test"), spec(condenser_max_tokens=104857)
    )
    try:
        condenser = session.agent.condenser
        assert condenser.max_tokens == 104857
        assert condenser.max_size > 10**6
        assert condenser.keep_first == 2
        assert condenser.llm is session.agent.llm
    finally:
        session.close()


def test_event_cap_serializes_for_harbor(monkeypatch):
    """Pass the event cap to the Harbor runner environment."""
    monkeypatch.setenv("LITELLM_API_KEY", "fixture-local-only")
    backend = HarborBackend()
    agent = spec(condenser_max_events=160)
    require_learner_runtime(backend, agent)
    config = backend._agent_config(agent)
    assert config.env["FORGER_CONDENSER_MAX_EVENTS"] == "160"
    assert "FORGER_CONDENSER_MAX_TOKENS" not in config.env


def test_local_event_cap_condenser(tmp_path):
    """Build the condenser from the event cap alone."""
    import sys

    session = LocalBackend()._sdk_session(
        tmp_path, WorkspaceTask(instruction="test"), spec(condenser_max_events=160)
    )
    try:
        condenser = session.agent.condenser
        assert condenser.max_tokens is None
        assert condenser.max_size == 160
        assert condenser.keep_first == 2
        assert condenser.llm is session.agent.llm
    finally:
        session.close()


def test_local_event_cap_http_condensation(tmp_path):
    """Cross the event cap in the native SDK loop without a token cap."""
    import json

    from completion import completion_server
    from openhands.sdk.event.condenser import Condensation

    def tool(name, arguments):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_" + name,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }

    responses = [
        tool("terminal", {"command": "marker old output"}),
        tool("terminal", {"command": "marker old output"}),
        tool("terminal", {"command": "marker old output"}),
        tool("terminal", {"command": "marker old output"}),
        {"role": "assistant", "content": "The command printed marker old output."},
        tool("finish", {"message": "done"}),
    ]
    with completion_server(responses) as (url, requests):
        agent = AgentSpec(
            model=Model("openai/gpt-4o", api_base=url, api_key="test"),
            max_iterations=8,
            runtime=LearnerRuntime(condenser_max_events=8),
        )
        session = LocalBackend()._sdk_session(
            tmp_path, WorkspaceTask(instruction="test"), agent
        )
        try:
            session.send_message("ORIGINAL_TASK: print marker output then finish")
            session.run()
            events = list(session.state.events)
            condensations = [e for e in events if isinstance(e, Condensation)]
            assert len(condensations) == 1
            condensation = condensations[0]
            assert condensation.summary_offset == 2
            assert condensation.forgotten_event_ids
            assert events[0].id not in condensation.forgotten_event_ids
            assert events[1].id not in condensation.forgotten_event_ids
            final_messages = json.dumps(requests[-1]["messages"])
            assert "ORIGINAL_TASK" in final_messages
        finally:
            session.close()


def test_local_http_condensation_preserves_task(tmp_path):
    """Execute real tools and cross the token threshold in the native SDK loop."""
    import json

    from completion import completion_server
    from openhands.sdk.context.condenser.utils import get_total_token_count
    from openhands.sdk.context.view import View
    from openhands.sdk.event.condenser import Condensation

    def tool(name, arguments):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_" + name,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }

    responses = [
        tool("terminal", {"command": "true # " + "oldword " * 15000}),
        {"role": "assistant", "content": "The command printed old diagnostic data."},
        tool("finish", {"message": "done"}),
    ]
    with completion_server(responses) as (url, requests):
        agent = AgentSpec(
            model=Model("openai/gpt-4o", api_base=url, api_key="test"),
            max_iterations=8,
            runtime=LearnerRuntime(condenser_max_tokens=10000),
        )
        session = LocalBackend()._sdk_session(
            tmp_path, WorkspaceTask(instruction="test"), agent
        )
        try:
            session.send_message("ORIGINAL_TASK: print diagnostic data then finish")
            session.run()
            events = list(session.state.events)
            condensations = [e for e in events if isinstance(e, Condensation)]
            assert len(condensations) == 1
            condensation = condensations[0]
            assert condensation.summary_offset == 2
            assert condensation.forgotten_event_ids
            assert events[0].id not in condensation.forgotten_event_ids
            assert events[1].id not in condensation.forgotten_event_ids
            assert len(requests) == 3
            final_messages = json.dumps(requests[-1]["messages"])
            assert "ORIGINAL_TASK" in final_messages
            assert "The command printed old diagnostic data." in final_messages
            assert "oldword " * 100 not in final_messages
            assert (
                get_total_token_count(
                    View.from_events(events).events, session.agent.llm
                )
                < 12000
            )
        finally:
            session.close()


def test_explicit_condenser_conflict(tmp_path):
    """Reject kwargs that override typed condenser settings."""
    from dataclasses import replace

    agent = replace(spec(condenser_max_tokens=1000), kwargs={"condenser": None})
    with pytest.raises(ValueError, match="conflict"):
        LocalBackend()._sdk_session(tmp_path, WorkspaceTask(instruction="test"), agent)


def test_condenser_selects_oldest_turns_towards_half_threshold():
    """Use real SDK token estimates and atomic event selection."""
    from openhands.sdk import LLM
    from openhands.sdk.context.condenser.utils import get_total_token_count
    from openhands.sdk.context.view import View
    from openhands.sdk.event import MessageEvent, SystemPromptEvent
    from openhands.sdk.llm import Message, TextContent

    from pyforger._sdk_runtime import sdk_history_kwargs

    llm = LLM(model="openai/gpt-4o", api_key="test")
    events = [SystemPromptEvent(system_prompt=TextContent(text="system"), tools=[])]
    for i in range(13):
        role = "user" if i % 2 == 0 else "assistant"
        events.append(
            MessageEvent(
                source="user" if role == "user" else "agent",
                llm_message=Message(
                    role=role,
                    content=[TextContent(text=f"turn {i} " + "data " * 500)],
                ),
            )
        )
    view = View.from_events(events)
    condenser = sdk_history_kwargs(spec(condenser_max_tokens=5000), llm)["condenser"]
    forgotten, offset = condenser._get_forgotten_events(view, agent_llm=llm)
    assert offset == 2
    assert forgotten[0].id == events[2].id
    assert forgotten[-1].id != events[-1].id
    remainder = [event for event in events if event not in forgotten]
    assert get_total_token_count(remainder, llm) <= 2500


@pytest.mark.parametrize("asynchronous", [False, True])
def test_hard_reset_preserves_task(asynchronous):
    """Native fallback must never summarize away the original task."""
    import asyncio

    from completion import completion_server
    from openhands.sdk import LLM
    from openhands.sdk.context.view import View
    from openhands.sdk.event import MessageEvent, SystemPromptEvent
    from openhands.sdk.llm import Message, TextContent

    from pyforger._sdk_runtime import sdk_history_kwargs

    events = [SystemPromptEvent(system_prompt=TextContent(text="system"), tools=[])]
    for text in ["protected task", "old turn"]:
        events.append(
            MessageEvent(
                source="user",
                llm_message=Message(role="user", content=[TextContent(text=text)]),
            )
        )
    with completion_server({"role": "assistant", "content": "summary"}) as (url, _):
        llm = LLM(model="openai/gpt-4o", base_url=url, api_key="test")
        condenser = sdk_history_kwargs(spec(condenser_max_tokens=1000), llm)[
            "condenser"
        ]
        view = View.from_events(events)
        result = (
            asyncio.run(condenser.ahard_context_reset(view, llm))
            if asynchronous
            else condenser.hard_context_reset(view, llm)
        )
        assert result.summary_offset == 2
        assert result.forgotten_event_ids == {events[2].id}
        with pytest.raises(ValueError, match="protected"):
            if asynchronous:
                asyncio.run(condenser.ahard_context_reset(View.from_events(events[:2])))
            else:
                condenser.hard_context_reset(View.from_events(events[:2]))


def test_unspecified_history_leaves_sdk_default_unchanged():
    """Leave SDK history behavior unchanged without an explicit request."""
    from openhands.sdk import LLM

    from pyforger._sdk_runtime import sdk_history_kwargs

    assert sdk_history_kwargs(spec(), LLM(model="openai/gpt-4o")) == {}


def test_local_unlimited_budget_adapts_only_at_sdk_boundary(tmp_path):
    """Local unlimited budgets adapt only at the SDK boundary."""
    import sys
    from dataclasses import replace

    from pyforger.session_budget import SessionBudget

    agent = replace(spec(), max_iterations=None)
    budget = SessionBudget(None, 30)
    session = LocalBackend()._sdk_session(
        tmp_path, WorkspaceTask(instruction="test"), agent, budget=budget
    )
    try:
        assert session.max_iteration_per_run == sys.maxsize
        assert session.agent.budget is budget
        assert budget.max_iterations is None
        assert agent.max_iterations is None
    finally:
        session.close()


@pytest.mark.parametrize("entry", ["run_agent", "run_agent_validated"])
@pytest.mark.parametrize(
    "settings,kwargs",
    [
        ({"tools": "science-rl"}, {"agent_context": None}),
        ({"tools": "science-rl"}, {"tools": []}),
        ({"tools": "science-rl"}, {"include_default_tools": []}),
        ({"condenser_max_tokens": 1000}, {"condenser": None}),
    ],
)
def test_conflicting_runtime_rejected_before_setup(
    tmp_path, monkeypatch, entry, settings, kwargs
):
    """Both public paths reject conflicts before workspace or package work."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    agent = replace(spec(**settings), kwargs=kwargs)
    task = WorkspaceTask(instruction="test", packages=("forger-invalid-test-package",))
    options = {"validate": lambda files: None} if entry == "run_agent_validated" else {}

    class SetupGuard(LocalBackend):
        def _install(self, root, task):
            raise AssertionError("Package setup started before runtime validation")

    with pytest.raises(ValueError, match="conflict"):
        getattr(SetupGuard(), entry)(task, agent, **options)
    assert list(tmp_path.iterdir()) == []


def test_full_prompt_budget_counts_real_tool_schemas(tmp_path):
    """Count the serialized system, task, tool schemas and security fields."""
    from openhands.sdk.llm import Message, TextContent
    from openhands.sdk.llm.exceptions import LLMContextWindowExceedError

    from pyforger._sdk_runtime import PromptBudgetLLM

    session = LocalBackend()._sdk_session(
        tmp_path, WorkspaceTask(instruction="test"), spec()
    )
    try:
        session.agent._initialize(session.state)
        tools = list(session.agent.tools_map.values())
        llm = PromptBudgetLLM(
            model="openai/gpt-4o",
            api_key="test",
            prompt_token_limit=1000,
            max_input_tokens=131066,
            max_output_tokens=49152,
        )
        messages = [
            Message(role="system", content=[TextContent(text="system")]),
            Message(role="user", content=[TextContent(text="original task")]),
        ]
        assert llm.get_token_count(messages) < 1000
        assert (
            llm.get_token_count(
                messages, tools=tools, add_security_risk_prediction=True
            )
            > 1000
        )
        with pytest.raises(LLMContextWindowExceedError, match="Serialized prompt"):
            llm.generate(messages, tools=tools, add_security_risk_prediction=True)
        assert llm.max_output_tokens == 49152
        assert llm.max_input_tokens == 131066
    finally:
        session.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_exact_provider_overflow_is_sdk_recoverable(monkeypatch, asynchronous):
    """Only replace the paid transport; retain SDK exception and condenser policy."""
    import asyncio

    from litellm.exceptions import BadRequestError
    from openhands.sdk import LLM
    from openhands.sdk.llm import Message, TextContent
    from openhands.sdk.llm.exceptions import LLMContextWindowExceedError

    from pyforger._sdk_runtime import PromptBudgetLLM

    error = BadRequestError(
        "Requested token count exceeds the model's maximum context length of 131072 tokens. You requested a total of 131906 tokens",
        model="test",
        llm_provider="openai",
    )

    def fail(*args, **kwargs):
        raise error

    async def afail(*args, **kwargs):
        raise error

    monkeypatch.setattr(LLM, "generate", fail)
    monkeypatch.setattr(LLM, "agenerate", afail)
    llm = PromptBudgetLLM(model="openai/gpt-4o", api_key="test")
    messages = [Message(role="user", content=[TextContent(text="task")])]
    with pytest.raises(LLMContextWindowExceedError):
        if asynchronous:
            asyncio.run(llm.agenerate(messages))
        else:
            llm.generate(messages)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_hard_reset_retries_smaller_summaries_preserving_prefix(
    monkeypatch, asynchronous
):
    import asyncio

    from completion import completion_server
    from openhands.sdk import LLM
    from openhands.sdk.context.view import View
    from openhands.sdk.event import MessageEvent, SystemPromptEvent
    from openhands.sdk.llm import Message, TextContent

    from pyforger._sdk_runtime import TaskPreservingCondenser

    events = [SystemPromptEvent(system_prompt=TextContent(text="system"), tools=[])]
    events += [
        MessageEvent(
            source="user",
            llm_message=Message(role="user", content=[TextContent(text=text)]),
        )
        for text in ["ORIGINAL_TASK", "large diagnostic " * 1000]
    ]
    calls = []
    original = TaskPreservingCondenser._build_summary_messages

    def reject_first(self, strings):
        calls.append(strings)
        if len(calls) == 1:
            raise ValueError("summary context overflow")
        return original(self, strings)

    monkeypatch.setattr(
        TaskPreservingCondenser, "_build_summary_messages", reject_first
    )
    with completion_server({"role": "assistant", "content": "summary"}) as (url, _):
        llm = LLM(model="openai/gpt-4o", base_url=url, api_key="test")
        condenser = TaskPreservingCondenser(llm=llm, max_tokens=80000, keep_first=2)
        result = (
            asyncio.run(condenser.ahard_context_reset(View.from_events(events), llm))
            if asynchronous
            else condenser.hard_context_reset(View.from_events(events), llm)
        )
        assert result.forgotten_event_ids == {events[2].id}
        assert result.summary_offset == 2
        assert len(calls) == 2
        assert len(calls[1][0]) < len(calls[0][0])
        assert condenser.max_tokens == 80000


def test_muse_tokenizer_counts_full_prompt_with_tools(tmp_path):
    from pathlib import Path

    from openhands.sdk.llm import Message, TextContent
    from openhands.sdk.llm.exceptions import LLMContextWindowExceedError

    from pyforger._sdk_runtime import PromptBudgetLLM

    cached = (
        Path.home()
        / ".cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots"
    )
    paths = sorted(cached.glob("*/tokenizer_config.json"))
    if not paths:
        pytest.skip("Muse tokenizer not cached")
    session = LocalBackend()._sdk_session(
        tmp_path, WorkspaceTask(instruction="test"), spec()
    )
    try:
        session.agent._initialize(session.state)
        tools = list(session.agent.tools_map.values())
        llm = PromptBudgetLLM(
            model="openai/test",
            api_key="test",
            custom_tokenizer=str(paths[0].parent),
            prompt_token_limit=80000,
            max_input_tokens=131066,
            max_output_tokens=49152,
        )
        messages = [
            Message(
                role="system", content=[TextContent(text="scientific task system")]
            ),
            Message(
                role="user",
                content=[TextContent(text="ORIGINAL_TASK " + "measurement " * 78000)],
            ),
        ]
        assert llm._chat_template_tokenizer is not None
        count = llm.get_token_count(
            messages, tools=tools, add_security_risk_prediction=True
        )
        without = llm.get_token_count(messages)
        assert count > without
        if count <= 80000:
            messages.append(
                Message(role="user", content=[TextContent(text="measurement " * 3000)])
            )
        with pytest.raises(LLMContextWindowExceedError, match="Serialized prompt"):
            llm.generate(messages, tools=tools, add_security_risk_prediction=True)
        assert llm.prompt_token_limit == 80000
        assert llm.max_output_tokens == 49152
        assert llm.max_input_tokens == 131066
    finally:
        session.close()


def test_provider_overflow_condenses_and_retries_real_sdk_loop(tmp_path, monkeypatch):
    """Inject only the unavailable provider rejection; run real SDK/tools/summary."""
    import json

    from completion import completion_server
    from litellm.exceptions import BadRequestError
    from openhands.sdk import LLM
    from openhands.sdk.event.condenser import Condensation

    from pyforger._sdk_runtime import PromptBudgetLLM

    responses = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_terminal",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps(
                            {"command": "true # diagnostic history"}
                        ),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "The diagnostic command completed."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_finish",
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": json.dumps({"message": "done"}),
                    },
                }
            ],
        },
    ]
    original = LLM.generate
    tool_calls = []

    def provider(self, *args, **kwargs):
        if kwargs.get("tools"):
            tool_calls.append(kwargs)
            if len(tool_calls) == 2:
                raise BadRequestError(
                    "Requested token count exceeds the model's maximum context length of 131072 tokens",
                    model="test",
                    llm_provider="openai",
                )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LLM, "generate", provider)
    with completion_server(responses) as (url, requests):
        from pyforger import local_backend

        llm = PromptBudgetLLM(
            model="openai/gpt-4o",
            base_url=url,
            api_key="test",
            prompt_token_limit=80000,
        )
        monkeypatch.setattr(local_backend, "agent_llm", lambda model: llm)
        session = LocalBackend()._sdk_session(
            tmp_path,
            WorkspaceTask(instruction="test"),
            AgentSpec(
                model=Model("openai/gpt-4o", api_base=url, api_key="test"),
                max_iterations=8,
                runtime=LearnerRuntime(condenser_max_tokens=80000),
            ),
        )
        try:
            session.send_message("ORIGINAL_TASK: run a diagnostic then finish")
            session.run()
            events = list(session.state.events)
            condensations = [e for e in events if isinstance(e, Condensation)]
            assert condensations
            assert all(e.summary_offset == 2 for e in condensations)
            assert all(events[1].id not in e.forgotten_event_ids for e in condensations)
            assert len(tool_calls) == 3
            assert "ORIGINAL_TASK" in json.dumps(requests[-1]["messages"])
            assert "The diagnostic command completed." in json.dumps(
                requests[-1]["messages"]
            )
        finally:
            session.close()


def test_non_context_provider_error_not_reclassified(monkeypatch):
    from litellm.exceptions import BadRequestError
    from openhands.sdk import LLM
    from openhands.sdk.llm import Message, TextContent

    from pyforger._sdk_runtime import PromptBudgetLLM

    error = BadRequestError("invalid tool schema", model="test", llm_provider="openai")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(LLM, "generate", fail)
    with pytest.raises(BadRequestError) as caught:
        PromptBudgetLLM(model="openai/gpt-4o", api_key="test").generate(
            [Message(role="user", content=[TextContent(text="task")])]
        )
    assert caught.value is error


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("message", "context_error"),
    [
        (
            (
                "Requested token count exceeds the model's maximum context length "
                "of 131066 tokens. You requested a total of 140000 tokens."
            ),
            True,
        ),
        ("Requested token count is invalid", False),
        ("invalid tool schema", False),
    ],
)
def test_installed_sdk_sglang_error_mapping(asynchronous, message, context_error):
    import asyncio

    from litellm.exceptions import BadRequestError
    from openhands.sdk import LLM
    from openhands.sdk.llm.exceptions import (
        LLMBadRequestError,
        LLMContextWindowExceedError,
    )

    expected = LLMContextWindowExceedError if context_error else LLMBadRequestError
    error = BadRequestError(message, model="test", llm_provider="openai")
    llm = LLM(model="openai/test", api_key="offline-test", num_retries=0)

    def no_fallback(_llm):
        pytest.fail("No fallback configured")

    with pytest.raises(expected) as caught:
        try:
            raise error
        except BadRequestError as original:
            if asynchronous:
                asyncio.run(llm._ahandle_error(original, no_fallback))
            else:
                llm._handle_error(original, no_fallback)
    assert type(caught.value) is expected
    assert caught.value.__cause__ is error
    assert str(caught.value) == str(error)
