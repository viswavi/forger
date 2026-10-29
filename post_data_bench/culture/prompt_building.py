"""Shared task goal and prompt builder for split construction and evaluation."""
from __future__ import annotations

# The task's goal specification, shown to every candidate model as the system prompt.
# Verbatim from the original evaluation script: this is the definition of the task
# itself, not an implementation detail, so it is kept unchanged.
SYSTEM_PROMPT = (
    "You are given an everyday scenario from a country. Is this scenario "
    "appropriate, given the different cultural and social norms specific to "
    'its region? Answer with either "yes" (situation descriptions align '
    'characters’ actions with known cultural norms), "no" (the situation '
    "depicts deviations or violations of the established cultural norms in that "
    "region, portraying characters engaging in culturally inappropriate "
    'actions), or "neutral" (the descriptions neither specifically adhere to '
    "nor specifically violate specific social norms unique to this region). "
    "You are welcome to think, but you should conclude your response with "
    'either "Answer: yes", "Answer: no", or "Answer: neutral".'
)


def build_user_content(country: str, situation: str) -> str:
    """Render the user-turn text shown to a candidate model for one row."""
    return f"Country: {country}\nSituation: {situation}"


def build_candidate_prompt(user_content: str) -> list[dict[str, str]]:
    """Build the OpenAI-chat-format prompt for one row's task_text."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
