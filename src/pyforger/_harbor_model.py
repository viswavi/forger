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
    anchor = "    llm = LLM(**llm_kwargs)"
    if source.count(anchor) != 1:
        raise RuntimeError("Unsupported Harbor SDK runner: LLM construction changed")
    compatibility = Path(__file__).with_name("_sdk_llm.py").read_text()
    compatibility = compatibility.replace("from __future__ import annotations", "")
    replacement = """    settings = json.loads(os.environ["FORGER_LLM_CONFIG"])
    settings["api_key"] = api_key
    disable_cache_key = os.environ.get("FORGER_DISABLE_PROMPT_CACHE") == "1"
    llm_type = ConfiguredLLM if disable_cache_key else LLM
    llm = llm_type(**settings)"""
    return source.replace("def main():", compatibility + "\n\ndef main():").replace(
        anchor, replacement
    )


class ForgerOpenHandsSDK(OpenHandsSDK):  # type: ignore[misc]
    """Keep Harbor behavior while supplying the complete common Model settings.

    Remove this bridge after https://github.com/harbor-framework/harbor/pull/3689
    is merged and the Harbor dependency is updated to use native ``llm_config``.
    Preserve the separate SDK prompt-cache-key workaround when migrating.
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
        for name in ("FORGER_LLM_CONFIG", "FORGER_DISABLE_PROMPT_CACHE"):
            value = self._get_env(name)
            if value is not None:
                combined[name] = value
        return await super().exec_as_agent(
            environment, command, env=combined, cwd=cwd, timeout_sec=timeout_sec
        )
