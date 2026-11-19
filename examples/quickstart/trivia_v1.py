"""Version 1: ask one prompt 50 times, drop near-duplicates, keep what a judge likes.

    FORGER_LIMIT=3 python trivia_v1.py   # 3 rows, to catch mistakes cheaply
    python trivia_v1.py                  # the full run (about $0.16)

Read the step reports it prints; trivia_v2.py fixes what they show.
"""
import json

from dotenv import load_dotenv

from pyforger import acquire, filter, pipeline, step
from pyforger.acquire import Model


@step
def write_questions(model: Model, topic: str) -> list[dict]:
    """Fifty questions on one topic, with short answers and no near-duplicates."""
    rows = acquire.sample_from_LLM(
        f"Write one hard trivia question about {topic}, and its short answer.",
        model,
        n=50,
        output_schema={"question": str, "answer": str},
    )
    return filter.by_rouge_l(rows, field="question", threshold=0.7, min_yield=0.5)


@step
def keep_good(rows: list[dict], judge: Model) -> list[dict]:
    """Keep questions the judge rates 4+ of 5: a correct answer, one clear reading."""
    return filter.by_llm_score(
        rows,
        model=judge,
        input_field="question",
        output_field="answer",
        criteria="the answer is correct, and the question has one reasonable answer",
        score_threshold=4,
        min_yield=0.3,
    )


@pipeline
def main(model: Model, judge: Model) -> list[dict]:
    """Write questions, then judge them."""
    return keep_good(write_questions(model, "astronomy"), judge)


if __name__ == "__main__":
    load_dotenv()  # reads ANTHROPIC_API_KEY from .env
    rows = main(
        Model("anthropic/claude-haiku-5-5"), Model("anthropic/claude-sonnet-5-5")
    )
    with open("trivia_v1.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
