"""Regenerate the readable validation artifact from a verified benchmark build."""
from pathlib import Path
import checkers
from split_contract import verify_manifest


def main():
    root = Path(__file__).resolve().parent
    verify_manifest(root)
    rows = checkers.read_jsonl(root / "validation_tasks.jsonl")
    if not rows:
        raise ValueError("Missing or empty validation split")
    for row in rows:
        if row.get('gold_response') is not None:
            row['gold_response'] = checkers.decode_answer(row['gold_response'])
    checkers.write_jsonl(root / "validation_tasks_de_obfuscated.jsonl", rows)


if __name__ == "__main__":
    main()
