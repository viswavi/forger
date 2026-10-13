"""Live, bounded judge-budget test. Prints only allowlisted diagnostic fields.

Uses the dedicated LITELLM_JUDGE_API_KEY as the disposable test credential;
its value is never printed or persisted. Does not change production key routing.
"""
import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coding.rl import rubric_rm as rm  # noqa: E402 - isolated worker/module bootstrap


async def main(model, effort, limit, usage_file):
    """Run the command-line utility described in the module documentation."""
    os.environ.update(
        {
            "JUDGE_MODEL": model,
            "JUDGE_REASONING_EFFORT": effort,
            "JUDGE_TOKEN_UNIT_LIMIT": str(limit),
            "JUDGE_TOKEN_USAGE_FILE": str(usage_file),
        }
    )
    rm.judge_configuration()
    ledger = rm.TokenUnitBudget(usage_file, limit)
    if not os.environ.get("LITELLM_JUDGE_API_KEY"):
        raise RuntimeError(
            "Set LITELLM_JUDGE_API_KEY; generation-key fallback is forbidden"
        )
    os.environ["LITELLM_JUDGE_BASE_URL"] = os.environ["LITELLM_BASE_URL"]
    os.environ["JUDGE_MAX_CONSECUTIVE_FAILURES"] = "5"
    logging.disable(logging.CRITICAL)
    report = {
        "successes": 0,
        "exhausted_calls": 0,
        "budget_failures": 0,
        "connection_failures": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "token_units": 0,
        "cost_usd": 0.0,
        "unpriced_calls": 0,
        "api_calls": 0,
        "local_quota_exhausted": False,
        "circuit_open": False,
        "blocked_without_api_call": False,
    }
    output = sys.stdout

    def emit(**values):
        print(json.dumps(values), file=output, flush=True)
        if values.get("event") in {"stopped", "call_limit", "time_limit"}:
            Path(__file__).with_name("judge_budget_result.json").write_text(
                json.dumps({"model": model, **values}, indent=2) + "\n"
            )

    # Suppress third-party SDK diagnostics and error bodies entirely.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(
        sink
    ), contextlib.redirect_stderr(sink):
        import litellm

        litellm.suppress_debug_info = True
        original = litellm.acompletion

        async def observed(**kwargs):
            report["api_calls"] += 1
            try:
                result = await original(**kwargs)
            except Exception as exc:
                report["exhausted_calls"] += 1
                # Inspect internally; expose neither the exception nor its body.
                description = str(exc).lower()
                is_budget = any(
                    word in description
                    for word in (
                        "budget",
                        "insufficient credit",
                        "quota exceeded",
                        "insufficient_quota",
                    )
                )
                is_connection = isinstance(exc, litellm.APIConnectionError)
                report["budget_failures"] += int(is_budget)
                report["connection_failures"] += int(is_connection)
                status = getattr(exc, "status_code", None)
                emit(
                    event="judge_failure",
                    budget_related=is_budget,
                    connection_error=is_connection,
                    status=status if isinstance(status, int) else None,
                    diagnostic_terms=[
                        term
                        for term in (
                            "temperature",
                            "reasoning",
                            "unsupported",
                            "not found",
                            "permission",
                            "authentication",
                        )
                        if term in description
                    ],
                    exhausted_calls=report["exhausted_calls"],
                )
                raise
            usage = result.usage
            report["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            report["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            return result

        litellm.acompletion = observed
        started = time.monotonic()
        for i in range(10):
            if time.monotonic() - started > 120:
                emit(event="time_limit", **report)
                break
            count = 1
            pairs = [(j + 173 + i, j + 249) for j in range(count)]
            prompt = [
                {
                    "role": "user",
                    "content": "Compute each product:\n"
                    + "\n".join(f"{a} * {b}" for a, b in pairs),
                }
            ]
            answer = "\n".join(f"{a} * {b} = {a*b}" for a, b in pairs)
            try:
                reward = await rm.prompt_score(
                    (
                        "Verify the arithmetic answers. Return 1 if all products "
                        "are correct, otherwise 0."
                    ),
                    prompt,
                    answer,
                    answer,
                )
            except rm.JudgeUnavailable:
                report["circuit_open"] = True
                snapshot = ledger.snapshot()
                report["token_units"] = snapshot["token_units"]
                report["cost_usd"] = snapshot["cost_usd"]
                report["unpriced_calls"] = snapshot["unpriced_calls"]
                report["local_quota_exhausted"] = snapshot["remaining_token_units"] <= 0
                before = report["api_calls"]
                try:
                    await rm.prompt_score("Correctness", prompt, answer, answer)
                except rm.JudgeUnavailable:
                    report["blocked_without_api_call"] = before == report["api_calls"]
                emit(event="stopped", **report)
                break
            report["successes"] += 1
            snapshot = ledger.snapshot()
            report["token_units"] = snapshot["token_units"]
            report["cost_usd"] = snapshot["cost_usd"]
            report["unpriced_calls"] = snapshot["unpriced_calls"]
            emit(event="scored", reward=reward, **report)
        else:
            emit(event="call_limit", **report)
    return (
        0
        if report["token_units"] > 0
        and report["local_quota_exhausted"]
        and report["circuit_open"]
        and report["blocked_without_api_call"]
        else 2
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=rm.JUDGE_MODEL)
    parser.add_argument(
        "--reasoning-effort", choices=("none", "low", "high"), default="low"
    )
    parser.add_argument("--token-unit-limit", type=float, default=250)
    parser.add_argument("--usage-file", type=Path)
    args = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix="judge-token-probe-") as temporary:
            code = asyncio.run(
                main(
                    args.model,
                    args.reasoning_effort,
                    args.token_unit_limit,
                    args.usage_file or Path(temporary) / "usage.json",
                )
            )
    except BaseException:
        # Avoid any accidental traceback containing SDK/provider context.
        print('{"event":"probe_failed_without_exception_details"}', flush=True)
        code = 3
    raise SystemExit(code)
