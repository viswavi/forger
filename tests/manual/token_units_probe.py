"""Run four bounded live requests and print only usage metadata."""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path


def main():
    """Verify fixed weighting, cumulative accounting, and reasoning forwarding."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["LITELLM_LOG"] = "ERROR"
    import litellm

    from pyforger.acquire.model import Model
    from pyforger.token_units import TokenUnitBudget, token_units

    litellm.suppress_debug_info = True
    if not all(os.environ.get(key) for key in ("LITELLM_API_KEY", "LITELLM_BASE_URL")):
        raise SystemExit(
            "Set LITELLM_API_KEY and LITELLM_BASE_URL in the environment or .env"
        )
    requests = [
        ("litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731", effort)
        for effort in ("none", "low", "high")
    ] + [("litellm_proxy/azure_ai/FW-GLM-5.3", "none")]
    with tempfile.TemporaryDirectory(prefix="forger-token-probe-") as directory:
        path = str(Path(directory) / "usage.json")
        os.environ["FORGER_TOKEN_USAGE_FILE"] = path
        os.environ["FORGER_TOKEN_UNIT_LIMIT"] = "100000"
        total = 0.0
        for model_id, effort in requests:
            model = Model(
                model_id,
                api_key=os.environ["LITELLM_API_KEY"],
                api_base=os.environ["LITELLM_BASE_URL"],
                temperature=1.0,
                max_tokens=256,
                num_retries=0,
                timeout=45,
                reasoning={"effort": effort},
            )
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Cost unavailable.*")
                    result = model.generate("What is 2 + 2? Reply briefly.")
            except Exception as exc:
                # Provider exceptions may embed credentials; do not print their bodies.
                print(
                    json.dumps(
                        {
                            "model": model_id,
                            "effort": effort,
                            "error_type": type(exc).__name__,
                        }
                    )
                )
                return 1
            usage = result.usage
            expected = token_units(
                model_id, usage["prompt_tokens"], usage["completion_tokens"]
            )
            assert usage["token_units"] == model.total_token_units == expected
            total += expected
            print(
                json.dumps(
                    {
                        "model": model_id,
                        "effort": effort,
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                        "token_units": expected,
                        "cost_usd": usage["cost_usd"],
                    }
                ),
                flush=True,
            )
        state = TokenUnitBudget(path, 100000).snapshot()
        assert state["calls"] == 4
        assert state["token_units"] == total
        print(
            json.dumps(
                {
                    "calls": state["calls"],
                    "total_token_units": total,
                    "cost_usd": state["cost_usd"],
                    "unpriced_calls": state["unpriced_calls"],
                    "passed": True,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
