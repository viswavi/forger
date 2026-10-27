"""Model boundary tests exercise native configuration without provider calls."""

from __future__ import annotations

import json
import os

import pytest

from pyforger.acquire.model import Model
from pyforger.harbor import harbor_agent_config


def test_model_settings_and_secret_reference(monkeypatch):
    """Preserve every SDK setting and retain only a credential reference."""
    pytest.importorskip("harbor")
    monkeypatch.setenv("FIXTURE_API_KEY", "fixture-private-value")
    model = Model(
        "openai/gpt-4o",
        api_key="fixture-private-value",
        api_base="https://fixture.invalid/v1",
        temperature=None,
        max_tokens=1234,
        num_retries=4,
        timeout=42,
        reasoning_effort="low",
        extra_body={"seed": 3},
        api_mode="responses",
        capability_overrides={"vision": False},
        disable_prompt_cache_key=True,
    )
    before = dict(os.environ)
    config = harbor_agent_config(model, api_key_env="FIXTURE_API_KEY")
    assert dict(os.environ) == before
    assert "fixture-private-value" not in repr(config)
    assert "fixture-private-value" not in config.model_dump_json()
    settings = json.loads(config.env["FORGER_LLM_CONFIG"])
    assert settings == dict(
        model="openai/gpt-4o",
        base_url="https://fixture.invalid/v1",
        temperature=None,
        max_output_tokens=1234,
        num_retries=5,
        timeout=42,
        reasoning_effort="low",
        litellm_extra_body={"seed": 3},
        api_mode="responses",
        capability_overrides={"vision": False},
    )
    assert config.env["LLM_API_KEY"] == "${FIXTURE_API_KEY}"


def test_boundary_rejects_lossy_or_conflicting_options(monkeypatch):
    """Unknown provider kwargs and competing model options fail closed."""
    monkeypatch.setenv("LITELLM_API_KEY", "fixture")
    with pytest.raises(ValueError, match="kwargs"):
        harbor_agent_config(Model("openai/gpt-4o", organization="x"))
    with pytest.raises(ValueError, match="model settings"):
        harbor_agent_config(Model("openai/gpt-4o"), agent_kwargs={"temperature": 1})


def test_native_factory_and_generated_runtime(tmp_path, monkeypatch):
    """Use the real factory and execute the runner through LLM construction."""
    import ast

    from harbor.agents.factory import AgentFactory

    from pyforger._harbor_model import ForgerOpenHandsSDK, runner_source
    from pyforger._sdk_llm import ConfiguredLLM

    monkeypatch.setenv("LITELLM_API_KEY", "fixture-runtime-private")
    model = Model(
        "openai/gpt-4o",
        temperature=None,
        max_tokens=1536,
        num_retries=3,
        timeout=51,
        reasoning_effort="low",
        extra_body={"seed": 91},
        api_mode="responses",
        capability_overrides={"vision": False},
        disable_prompt_cache_key=True,
    )
    config = harbor_agent_config(model, max_iterations=7)
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
    assert isinstance(agent, ForgerOpenHandsSDK)
    assert agent.extra_env["LLM_API_KEY"] == "fixture-runtime-private"
    assert agent.options.max_iterations == 7
    # Trim only after native LLM construction, before tools/conversation can run.
    tree = ast.parse(runner_source())
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.If)
        and not (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("openhands.tools")
        )
    ]
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    index = next(
        i
        for i, node in enumerate(main.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "llm" for t in node.targets)
    )
    main.body = main.body[: index + 1] + [
        ast.Return(value=ast.Name(id="llm", ctx=ast.Load()))
    ]
    ast.fix_missing_locations(tree)
    for name, value in agent.extra_env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--instruction",
            "fixture",
            "--logs-dir",
            str(tmp_path),
            "--trajectory-path",
            str(tmp_path / "trajectory.json"),
        ],
    )
    namespace = {"__name__": "forger_runner_test"}
    exec(compile(tree, "<native-forger-runner>", "exec"), namespace)
    llm = namespace["main"]()
    assert llm.__class__.__name__ == ConfiguredLLM.__name__
    expected = json.loads(config.env["FORGER_LLM_CONFIG"])
    assert llm.model_dump(mode="json", include=set(expected)) == expected
    assert llm.api_key.get_secret_value() == "fixture-runtime-private"


def test_no_literal_credential_or_embedded_secret(monkeypatch):
    """Refuse credentials that cannot be safely replayed from host references."""
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key_env"):
        harbor_agent_config(Model("openai/gpt-4o", api_key="unregistered-private"))
    monkeypatch.setenv("LITELLM_API_KEY", "fixture")
    with pytest.raises(ValueError, match="embedded credentials"):
        harbor_agent_config(
            Model("openai/gpt-4o", extra_body={"auth_token": "other-private"})
        )


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.org",
        "https://u:p@example.org/v1",
        "https://example.org/?token=secret",
    ],
)
def test_endpoint_rejects_credentials_and_non_http(url, monkeypatch):
    """Endpoints must not retain authentication material in public config."""
    monkeypatch.setenv("LITELLM_API_KEY", "fixture")
    with pytest.raises(ValueError, match="credential-free"):
        harbor_agent_config(Model("openai/gpt-4o", api_base=url))


def test_endpoint_and_network_policy(monkeypatch):
    """Allow the configured endpoint in a native restricted network plan."""
    from harbor.models.task.config import TaskConfig, VerifierEnvironmentMode
    from harbor.models.trial.config import EnvironmentConfig, VerifierConfig
    from harbor.trial.network_policy import resolve_trial_network_plan

    monkeypatch.setenv("LITELLM_API_KEY", "fixture")
    config = harbor_agent_config(
        Model("openai/gpt-4o", api_base="https://example.org/v1")
    )
    assert config.extra_allowed_hosts == ["example.org"]
    assert config.env["LLM_BASE_URL"] == "https://example.org/v1"
    task = TaskConfig.model_validate({"environment": {"network_mode": "no-network"}})
    policy = resolve_trial_network_plan(
        task,
        config,
        EnvironmentConfig(type="sail"),
        VerifierConfig(),
        None,
        verifier_mode=VerifierEnvironmentMode.SHARED,
    ).agent_phase
    assert policy.allowed_hosts == ["example.org"]


def test_bridge_control_environment_does_not_match_native_credentials(monkeypatch):
    """Native scrubbing must not treat a numeric configuration flag as a secret."""
    from harbor.utils.env import is_sensitive_env_key

    from pyforger.harbor import harbor_agent_config

    monkeypatch.setenv("LLM_API_KEY", "fixture-model-credential")
    model = Model(
        "openai/gpt-6-luna",
        api_key="fixture-model-credential",
        api_base="https://llm-proxy.app.all-hands.dev/v1",
        disable_prompt_cache_key=True,
    )
    config = harbor_agent_config(model, api_key_env="LLM_API_KEY")
    for name, value in config.env.items():
        if value in {"0", "1"}:
            assert not is_sensitive_env_key(name), name


def test_runtime_uses_shared_token_condenser_and_unlimited_iteration_adapter():
    from pyforger._harbor_model import runner_source

    source = runner_source()
    assert "TaskPreservingCondenser(llm=llm, max_tokens=int(limit)" in source
    assert "keep_first=2" in source
    assert 'os.environ.get("MAX_ITERATIONS") or str(sys.maxsize)' in source
