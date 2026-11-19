"""Version 2: fixes the two problems version 1's reports show.

1. Repetition. v1's report flagged "white dwarf" in 12 of 35 questions: one prompt
   asked 50 times keeps returning the same few facts. v2 first lists distinct
   subtopics, then writes one question per subtopic, and drops repeated answers.
2. A judge that rejects nothing. v1's judge gave all 35 questions 5 of 5. v2's
   checker answers each question without seeing the answer, and a second call checks
   whether the two answers agree. Questions where they disagree are dropped.

    FORGER_LIMIT=3 python trivia_v2.py
    python trivia_v2.py
"""
import json

from dotenv import load_dotenv

from pyforger import acquire, alter, filter, pipeline, step
from pyforger.acquire import Model

TOPIC = "astronomy"


@step
def subtopics(model: Model, topic: str) -> list[dict]:
    """About 40 distinct subtopics of the topic, without near-duplicates."""
    lists = acquire.sample_from_LLM(
        f"List 40 distinct, specific subtopics of {topic} that a trivia question "
        "could be about (e.g. a person, an object, a discovery, a phenomenon). "
        "Cover the whole field, not one corner of it.",
        model,
        n=1,
        output_schema={"subtopics": list},
    )
    rows = [{"subtopic": str(s)} for s in lists[0]["subtopics"]]
    return filter.by_rouge_l(rows, field="subtopic", threshold=0.7, min_yield=0.8)


@step
def write_questions(rows: list[dict], model: Model, topic: str) -> list[dict]:
    """One question with a short answer per subtopic; no two with the same answer."""
    rows = alter.generate_from_context(
        rows,
        "Write one hard " + topic + " trivia question about {subtopic}, and its "
        "short answer (a name, number, or term).",
        model,
        output_schema={"question": str, "answer": str},
        merge=True,
    )
    rows = filter.by_rouge_l(rows, field="answer", threshold=0.7, min_yield=0.7)
    return filter.by_rouge_l(rows, field="question", threshold=0.7, min_yield=0.8)


@step
def check_answers(rows: list[dict], checker: Model) -> list[dict]:
    """Keep questions whose answer a second model reaches on its own."""
    rows = alter.generate_from_context(
        rows,
        "{question}\n\nAnswer in a few words.",
        checker,
        output_schema={"answer": str},
        output_field="checker",
        merge=True,
    )
    rows = [
        {
            **r,
            "answers": (
                f"Reference: {r['answer']}\nIndependent: {r['checker']['answer']}"
            ),
        }
        for r in rows
    ]
    return filter.by_llm_score(
        rows,
        model=checker,
        input_field="question",
        output_field="answers",
        criteria="the independent answer names the same thing as the reference answer",
        rubric={
            1: "they name different things",
            2: "they mostly disagree",
            3: "partly the same, or one is much vaguer",
            4: "the same thing, with a small difference in detail",
            5: "exactly the same thing",
        },
        score_threshold=4,
        min_yield=0.5,
        max_yield=0.98,
    )


@pipeline
def main(writer: Model, checker: Model) -> list[dict]:
    """Subtopics, then one question each, then an independent check."""
    rows = subtopics(writer, TOPIC)
    rows = write_questions(rows, writer, TOPIC)
    return check_answers(rows, checker)


if __name__ == "__main__":
    load_dotenv()  # reads ANTHROPIC_API_KEY from .env
    rows = main(
        Model("anthropic/claude-haiku-5-5"), Model("anthropic/claude-sonnet-5-5")
    )
    with open("trivia_v2.jsonl", "w") as f:
        f.writelines(
            json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n"
            for r in rows
        )
    print(f"wrote {len(rows)} questions to trivia_v2.jsonl")
