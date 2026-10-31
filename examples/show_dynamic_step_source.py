"""Print cached dynamic-step programs and their @op cache-hit status.

Use the same model settings, goal, inputs, and FORGER_CACHE as the pipeline run.
``_cache_only=True`` raises on a miss and never launches an agent.
"""

import argparse
import json
import os

from dotenv import load_dotenv

from pyforger.acquire import Model
from pyforger.agent_backend import AgentSpec
from pyforger.alter import dynamic_step_source
from pyforger.op import clear_trace, trace


def main() -> None:
    """Show cached source for each requested input without running OpenHands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("inputs", nargs="+", help="JSON object for each input record")
    args = parser.parse_args()

    load_dotenv()
    model = Model(
        os.environ.get("MODEL_ID") or os.environ["LLM_MODEL"],
        api_key=os.environ.get("LITELLM_API_KEY"),
        api_base=os.environ.get("LITELLM_BASE_URL"),
        max_tokens=args.max_tokens,
    )
    clear_trace()
    for raw in args.inputs:
        item = json.loads(raw)
        if not isinstance(item, dict):
            parser.error("each input must be a JSON object")
        try:
            code = dynamic_step_source(
                item,
                args.goal,
                AgentSpec(model, max_iterations=args.max_iterations),
                _cache_only=True,
            )
        except FileNotFoundError:
            print(f"input: {item}\ncache_hit: False\n")
            continue
        record = trace()[-1]
        print(f"input: {item}\ncache_hit: {record.cache_hit}\ncode:\n{code}\n")


if __name__ == "__main__":
    main()
