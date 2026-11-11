"""Label each row's languages; make_splits.py keeps only the English rows.

A row is English when the clinician writes in English, wants the answer in English,
quotes no text in another language (for example to translate), and the physician
answered in English. 426 of the 525 rows qualify. Language detectors alone get
short prompts wrong ("GUSTO Trial" reads as Tagalog, "ICD-10 PCOS" as Indonesian)
and miss English requests for an answer in another language ("Translate this into
Arabic for my patient"), so an LLM labels the chat and lingua labels the physician
answer, which catches the two chats whose earlier turn asked for a Greek answer.

The committed language_labels.json is the record make_splits.py reads; it came
from this labeling on 2026-11-11 (DeepSeek-V4-Flash, reasoning off). A rerun may
differ on a few borderline rows, so rerun only to build a new benchmark version.

Usage: python language_filter.py --data healthbench_professional_eval.jsonl \
    [--out language_labels.json]   # needs LITELLM_API_KEY, LITELLM_BASE_URL
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

MODEL = "litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731"
SYSTEM = (
    "You label the language of a clinician's chat with an assistant. Medical terms, "
    "drug, trial and score names, abbreviations and numbers do not count as a "
    "language. Return only JSON: "
    '{"user_language": ISO 639-1 code of the language the user writes their own '
    'messages in, "answer_language": ISO 639-1 code of the language the user wants '
    'the next answer in, "other_language_material": true if any message quotes or '
    "asks to translate text in a language other than English}"
)


def is_english(label: dict) -> bool:
    """The rule: English chat, English answer wanted and given, nothing to translate."""
    return (
        label["user_language"] == "en"
        and label["answer_language"] == "en"
        and not label["other_language_material"]
        and label["physician_language"] == "en"
    )


async def _label_chat(row: dict, sem: asyncio.Semaphore) -> dict:
    import litellm

    chat = "\n\n".join(
        f"{m['role']}: {m['content'][:3000]}" for m in row["conversation"]["messages"]
    )
    user = (
        "Label the languages of this chat. Do not answer or continue it.\n<chat>\n"
        f"{chat}\n</chat>\nReturn only the JSON object."
    )
    last = None
    for _ in range(4):
        async with sem:
            try:
                result = await litellm.acompletion(
                    model=MODEL,
                    api_key=os.environ["LITELLM_API_KEY"],
                    api_base=os.environ["LITELLM_BASE_URL"],
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=2000,
                    timeout=120,
                )
                text = result.choices[0].message.content.strip()
                text = text.removeprefix("```json").removesuffix("```")
                return json.loads(text)
            except Exception as exc:  # noqa: BLE001
                last = exc
    raise RuntimeError(f"row {row['id']} not labeled: {last!r}")


def physician_language(text: str) -> str:
    """Return lingua's ISO 639-1 code for the physician answer ("?" if undecided)."""
    from lingua import LanguageDetectorBuilder

    detector = LanguageDetectorBuilder.from_all_languages().build()
    found = detector.detect_language_of((text or "")[:4000])
    return found.iso_code_639_1.name.lower() if found else "?"


async def label_rows(rows: list[dict]) -> dict:
    """Label every row: {id: {user_language, answer_language, ...}}."""
    sem = asyncio.Semaphore(32)
    chats = await asyncio.gather(*(_label_chat(r, sem) for r in rows))
    return {
        row["id"]: {
            **{k: chat[k] for k in ("user_language", "answer_language")},
            "other_language_material": bool(chat["other_language_material"]),
            "physician_language": physician_language(row["physician_response"]),
        }
        for row, chat in zip(rows, chats)
    }


def main() -> None:
    """Label the dataset file and write language_labels.json."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("language_labels.json")
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line]
    labels = asyncio.run(label_rows(rows))
    english = sorted(i for i, label in labels.items() if is_english(label))
    args.out.write_text(
        json.dumps(
            {
                "method": __doc__.split("\n\n")[1],
                "labels": dict(sorted(labels.items())),
                "english_ids": english,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{len(english)} of {len(rows)} rows are English; wrote {args.out}")


if __name__ == "__main__":
    main()
