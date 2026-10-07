"""Render worked examples from structured tests uniformly for both classes.

Only explicitly recognized example sections are replaced. Unknown formats fail the
build for inspection rather than silently retaining a contradictory statement.
"""
import json
import re


def normalize_wrong_output(true, wrong, functional):
    if functional:
        expected = json.loads(true)
        try:
            replacement = json.loads(wrong)
        except json.JSONDecodeError:
            if not isinstance(expected, str):
                raise ValueError("Functional perturbation is not valid JSON") from None
            replacement = wrong.strip()
        if isinstance(expected, str) and not isinstance(replacement, str):
            replacement = wrong.strip()
        if type(replacement) is not type(expected):
            # Older frozen integer perturbations sometimes contain 1.0.
            if type(expected) is int and type(replacement) is float and replacement.is_integer():
                replacement = int(replacement)
            else:
                raise ValueError("Functional perturbation changes the output type")
        if replacement == expected:
            raise ValueError("Perturbation does not change the output")
        return json.dumps(replacement, ensure_ascii=False, separators=(",", ":"))
    suffix = true[len(true.rstrip()):]
    if wrong.strip() == true.strip():
        raise ValueError("Perturbation does not change the output")
    return wrong.strip() + suffix


def harmonize_statement(row):
    text = row["question_content"].replace("\r\n", "\n").replace("\r", "")
    match = re.search(r"(?:Sample Input\s*1\s*:?|^[ \t]*Example\s*1\s*:)", text, re.M)
    if not match:
        if re.search(r"(?i)sample input|example\s*\d+", text):
            raise ValueError(f"Unrecognized example section for {row['question_id']}")
        return text  # No duplicated examples to harmonize (e.g. interactive statements).
    # Constraints/follow-up after the examples are part of the specification.
    tail = text[match.start():]
    ending = re.search(r"(?m)^[ \t]*(?:Constraints|Follow[ -]?up)\s*:?\s*$", tail)
    suffix = tail[ending.start():] if ending else ""
    tests = json.loads(row["public_test_cases"])
    examples = "\n\n".join(f"Example {i}:\nInput: {test['input'].rstrip()}\nOutput: {test['output'].rstrip()}" for i, test in enumerate(tests, 1))
    return text[:match.start()].rstrip() + "\n\n" + examples + "\n\n" + suffix
