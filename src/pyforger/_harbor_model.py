"""Thin installed-agent bridge reusing Harbor's SDK runner and lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harbor.agents.installed import openhands_sdk
from harbor.agents.installed.openhands_sdk import OpenHandsSDK
from harbor.models.agent.context import ModelUsage
from harbor.models.trajectories.agent import Agent
from harbor.models.trajectories.step import Step
from harbor.models.trajectories.trajectory import Trajectory
from harbor.utils.trajectory_utils import compute_model_usage

from pyforger._streaming_json import object_items


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
    if settings.get("custom_tokenizer"):
        settings["custom_tokenizer"] = "/opt/tokenizer"
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
    source = source.replace(
        '    max_iter_raw = os.environ.get("MAX_ITERATIONS")',
        '    max_iter_raw = os.environ.get("MAX_ITERATIONS") or str(sys.maxsize)',
    )
    return "from __future__ import annotations\n" + source.replace(
        condenser_anchor, condenser
    )


class ForgerOpenHandsSDK(OpenHandsSDK):  # type: ignore[misc]
    """Keep Harbor behavior while supplying the complete common Model settings.

    Harbor owns authentication and agent lifecycle. This bridge preserves Forger
    model settings, prompt-cache behavior, and the shared condenser contract.
    """

    def populate_context_post_run(self, context: Any) -> None:
        """Read final metrics and per-model usage without loading the ATIF history."""
        path = self.logs_dir / self._TRAJECTORY_FILENAME
        if not path.exists():
            return
        try:
            agent = {}
            metrics = {}
            for key, value in object_items(path):
                if key == "agent":
                    agent = value
                elif key == "final_metrics":
                    metrics = value or {}
            usage = {}
            unpriced = set()
            for key, value in object_items(path, ("steps", "subagent_trajectories")):
                if key == "steps":
                    fragment = Trajectory.model_construct(
                        agent=Agent.model_validate(agent),
                        steps=[Step.model_validate(value)],
                        subagent_trajectories=None,
                    )
                elif key == "subagent_trajectories":
                    fragment = Trajectory.model_validate(value)
                else:
                    continue
                partial = compute_model_usage(fragment)
                for model, amount in partial.items():
                    total = usage.setdefault(model, ModelUsage())
                    total.n_input_tokens += amount.n_input_tokens
                    total.n_cache_tokens += amount.n_cache_tokens
                    total.n_output_tokens += amount.n_output_tokens
                    if amount.cost_usd is None:
                        unpriced.add(model)
                    else:
                        total.cost_usd = (total.cost_usd or 0.0) + amount.cost_usd
            for model in unpriced:
                usage[model].cost_usd = None
            context.cost_usd = metrics.get("total_cost_usd")
            context.n_input_tokens = metrics.get("total_prompt_tokens", 0)
            context.n_output_tokens = metrics.get("total_completion_tokens", 0)
            context.n_cache_tokens = metrics.get("total_cached_tokens", 0)
            # An explicit empty mapping prevents Harbor's whole-file fallback.
            context.model_usage = usage
        except (ValueError, OSError, TypeError) as error:
            context.model_usage = {}
            self.logger.error("Failed to parse trajectory metrics: %s", error)

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
