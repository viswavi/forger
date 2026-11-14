"""Rubric grading of clinician-chat answers: one judge call per criterion.

The grader prompt, the per-criterion true/false verdict, the score (points of the
met criteria over the sum of positive points, so negative criteria subtract) and the
length adjustment are the benchmark's own. The grader sees the conversation and
the final answer only, never the reasoning.

Two judges:
- "official": the host's grader, gpt-6-luna at high reasoning effort with the host
  key `LITELLM_OFFICIAL_JUDGE_KEY`. It scores the secret test set.
- "agent": DeepSeek-V4.1-Flash (the RL judge) at JUDGE_REASONING_EFFORT
  (default "none" for the clinical task, as in RL), through the judge
  token-unit ledger (`rl.token_budget`, budget="judge") and
  `LITELLM_JUDGE_API_KEY`. Validation runs in the agent's workspace use this one.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

# Length adjustment: s - 2.94e-5 * (answer characters - 2,000), i.e. 0.0147 per 500.
LENGTH_CENTER = 2000
LENGTH_PENALTY_PER_CHAR = 2.94e-5

OFFICIAL_MODEL = "openai/wine-gpt-6-luna"
OFFICIAL_EFFORT = "high"
ATTEMPTS = 6
# Attempts before a criterion the provider's content filter keeps blocking counts as
# ungraded. Azure blocks DeepSeek-V4-Flash's verdict on some chats about self-harm
# ("Response content blocked by label 'MultiSeverity_SelfHarmScore'"): 1 of the 194
# validation criteria, on every attempt (2026-11-11).
FILTER_ATTEMPTS = 3
JUDGE_MAX_TOKENS = 4096

# The benchmark's grader prompt, verbatim (typos included), with <<conversation>>
# and <<rubric_item>> slots.
GRADER_TEMPLATE = (Path(__file__).with_name("grader_prompt.txt")).read_text().strip()


class JudgeError(RuntimeError):
    """A criterion could not be graded after every attempt."""


def grader_prompt(messages: list[dict], answer: str, item: dict) -> str:
    r"""The grading prompt for one criterion: the chat plus the answer as its last turn.

    E.g. "user: ...\\n\\nassistant: <answer>" and the item "[5] Recommends ...".
    """
    convo = messages + [{"role": "assistant", "content": answer}]
    text = "\n\n".join(f"{m['role']}: {m['content']}" for m in convo)
    rubric_item = f"[{item['points']}] {item['criterion_text']}"
    return GRADER_TEMPLATE.replace("<<conversation>>", text).replace(
        "<<rubric_item>>", rubric_item
    )


def parse_verdict(text: str | None) -> bool | None:
    """The boolean "criteria_met" of a grader reply, or None when it has none."""
    cleaned = re.sub(r"^```json\s*|\s*```$", "", (text or "").strip())
    try:
        value = json.loads(cleaned).get("criteria_met")
    except (json.JSONDecodeError, AttributeError):
        return None
    return value if isinstance(value, bool) else None


def raw_score(items: list[dict], met: list[bool | None]) -> float:
    """Points of the met criteria (negative ones subtract) over the positive total.

    An ungraded criterion (None) is left out of both, unless it leaves no positive
    points, e.g. the only positive criterion of a row; then the full positive
    total stays the denominator.
    """
    total = sum(item["points"] for item in items if item["points"] > 0)
    if total <= 0:
        raise ValueError("a rubric needs at least one positive criterion")
    graded = [
        (item, hit) for item, hit in zip(items, met, strict=True) if hit is not None
    ]
    possible = sum(item["points"] for item, _ in graded if item["points"] > 0) or total
    achieved = sum(item["points"] for item, hit in graded if hit)
    return achieved / possible


def length_adjusted(score: float, answer: str) -> float:
    """The score minus 2.94e-5 per answer character above 2,000 (plus below it)."""
    return score - LENGTH_PENALTY_PER_CHAR * (len(answer) - LENGTH_CENTER)


def default_judge() -> str:
    """Return "official" where the host's official key is set, else "agent"."""
    return "official" if os.environ.get("LITELLM_OFFICIAL_JUDGE_KEY") else "agent"


def _official_base_url() -> str:
    for name in (
        "LITELLM_OFFICIAL_JUDGE_BASE_URL",
        "LITELLM_WORKER_BASE_URL",
        "LITELLM_BASE_URL",
    ):
        if os.environ.get(name):
            return os.environ[name]
    raise JudgeError("Set LITELLM_OFFICIAL_JUDGE_BASE_URL for the official judge")


async def _call(judge: str, prompt: str):
    """One grader call; returns (reply text, usage dict)."""
    messages = [{"role": "user", "content": prompt}]
    if judge == "official":
        import litellm

        result = await litellm.acompletion(
            model=OFFICIAL_MODEL,
            messages=messages,
            api_key=os.environ["LITELLM_OFFICIAL_JUDGE_KEY"],
            api_base=_official_base_url(),
            reasoning_effort=OFFICIAL_EFFORT,
            allowed_openai_params=["reasoning_effort"],
            timeout=600,
            num_retries=0,
        )
    elif judge == "agent":
        try:  # the agent's copy: rl/ sits next to this file
            from rl.token_budget import acompletion
        except ImportError:  # the host: post_data_bench/coding/rl
            from coding.rl.token_budget import acompletion

        # budget="judge" sets the judge model, key, effort and the ledger.
        result = await acompletion(
            budget="judge",
            messages=messages,
            max_tokens=JUDGE_MAX_TOKENS,
            timeout=600,
        )
    else:
        raise ValueError(f"unknown judge {judge!r}")
    usage = result.usage
    return result.choices[0].message.content, {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }


async def grade_criterion(judge: str, prompt: str, sem: asyncio.Semaphore) -> dict:
    """Grade one criterion, retrying failed calls and replies without a verdict.

    Returns {"criteria_met", "explanation", "attempts", "usage"}. The reference
    implementation retries a bad reply until it parses; this one stops after
    ATTEMPTS and raises, so a broken judge fails the run instead of looping. A
    criterion the provider's content filter blocks FILTER_ATTEMPTS times returns
    "criteria_met": None (ungraded), which raw_score leaves out.
    """
    last = None
    filtered = 0
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    for attempt in range(1, ATTEMPTS + 1):
        try:
            async with sem:
                text, usage = await _call(judge, prompt)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "TokenUnitQuotaExceeded":
                raise
            last = f"{type(exc).__name__}: {str(exc)[:200]}"
            if "content_filter" in str(exc):
                filtered += 1
                if filtered >= FILTER_ATTEMPTS:
                    return {
                        "criteria_met": None,
                        "explanation": "ungraded: the provider's content filter "
                        "blocked every grader reply",
                        "attempts": attempt,
                        "usage": usage_total,
                    }
            await asyncio.sleep(min(2**attempt, 30))
            continue
        for key in usage_total:
            usage_total[key] += usage[key] or 0
        verdict = parse_verdict(text)
        if verdict is not None:
            try:
                explanation = json.loads(
                    re.sub(r"^```json\s*|\s*```$", "", text.strip())
                ).get("explanation", "")
            except json.JSONDecodeError:
                explanation = ""
            return {
                "criteria_met": verdict,
                "explanation": explanation,
                "attempts": attempt,
                "usage": usage_total,
            }
        last = f"no verdict in reply: {(text or '')[:200]!r}"
    raise JudgeError(f"criterion not graded after {ATTEMPTS} attempts ({last})")


async def grade_answer(
    judge: str, messages: list[dict], answer: str, items: list[dict], sem
) -> dict:
    """Grade one answer on every criterion of its row.

    Returns {"score", "score_length_adjusted", "answer_chars", "criteria": [...]}.
    """
    grades = await asyncio.gather(
        *(
            grade_criterion(judge, grader_prompt(messages, answer, item), sem)
            for item in items
        )
    )
    met = [g["criteria_met"] for g in grades]
    score = raw_score(items, met)
    return {
        "score": score,
        "score_length_adjusted": length_adjusted(score, answer),
        "answer_chars": len(answer),
        "criteria": [
            {
                "points": item["points"],
                "criteria_met": g["criteria_met"],
                "explanation": g["explanation"],
                "attempts": g["attempts"],
                "usage": g["usage"],
            }
            for item, g in zip(items, grades)
        ],
    }
