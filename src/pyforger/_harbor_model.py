"""Thin installed-agent bridge reusing Harbor's SDK runner and lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harbor.agents.installed import openhands_sdk
from harbor.agents.installed.openhands_sdk import OpenHandsSDK


def runner_source() -> str:
    """Extend the native runner only at its LLM construction boundary."""
    source = (
        Path(openhands_sdk.__file__).with_name("openhands_sdk_runner.py").read_text()
    )
    anchor = "    llm = build_llm()"
    if source.count(anchor) != 1:
        raise RuntimeError("Unsupported Harbor SDK runner: LLM construction changed")
    compatibility = Path(__file__).with_name("_sdk_llm.py").read_text()
    compatibility = compatibility.replace("from __future__ import annotations", "")
    replacement = """    settings = json.loads(os.environ["FORGER_LLM_CONFIG"])
    settings["api_key"] = os.environ.get("LLM_API_KEY")
    disable_cache_key = os.environ.get("FORGER_DISABLE_PROMPT_CACHE") == "1"
    llm_type = ConfiguredLLM if disable_cache_key else LLM
    llm = llm_type(**settings)"""
    runtime = Path(__file__).with_name("_sdk_runtime.py").read_text()
    runtime = runtime.replace("from __future__ import annotations", "")
    source = source.replace("import argparse", "import argparse\nimport sys")
    source = source.replace(
        "def main():", compatibility + "\n" + runtime + "\n\ndef main():"
    ).replace(anchor, replacement)
    condenser_anchor = "    agent = Agent(**agent_kwargs)"
    condenser = """    limit = os.environ.get("FORGER_CONDENSER_MAX_TOKENS")
    if limit:
        agent_kwargs["condenser"] = TaskPreservingCondenser(llm=llm, max_tokens=int(limit), max_size=sys.maxsize, keep_first=2)
    agent = Agent(**agent_kwargs)"""
    if source.count(condenser_anchor) != 1:
        raise RuntimeError("Unsupported Harbor SDK runner: agent construction changed")
    return source.replace(condenser_anchor, condenser)


class ForgerOpenHandsSDK(OpenHandsSDK):  # type: ignore[misc]
    """Keep Harbor behavior while supplying the complete common Model settings.

    Harbor owns authentication and agent lifecycle. This bridge preserves Forger
    model settings, prompt-cache behavior, and the shared condenser contract.
    """

    async def install(self, environment: Any) -> None:
        """Install the native SDK then upload the minimally extended runner."""
        source = runner_source()
        await super().install(environment)
        local = self.logs_dir / "run_agent.py"
        local.write_text(source)
        await environment.upload_file(
            source_path=local, target_path="/installed-agent/run_agent.py"
        )

    async def exec_as_agent(
        self,
        environment: Any,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        """Forward bridge settings explicitly without mutating the host env."""
        combined = dict(env or {})
        for name in (
            "FORGER_LLM_CONFIG",
            "FORGER_DISABLE_PROMPT_CACHE",
            "FORGER_CONDENSER_MAX_TOKENS",
        ):
            value = self._get_env(name)
            if value is not None:
                combined[name] = value
        return await super().exec_as_agent(
            environment, command, env=combined, cwd=cwd, timeout_sec=timeout_sec
        )
