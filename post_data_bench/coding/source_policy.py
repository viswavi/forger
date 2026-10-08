"""Exclude sources whose output contract allows multiple correct answers.

Apply before generating originals or perturbations. Known-source audits and
conservative wording patterns supplement semantic audits for each new release.
"""
import re

POLICY_VERSION = 1
# Public source IDs, with the reason taken from each problem's output contract.
NON_UNIQUE_OUTPUTS = {
    "abc311_c": "Any directed cycle may be printed.",
    "abc315_e": "Any valid reading order may be printed.",
    "abc326_d": "Any valid grid filling may be printed.",
    "abc333_e": "Any optimal potion-picking plan may be printed.",
    "abc343_a": "Any digit other than A+B may be printed.",
    "abc343_e": "Any valid cube positions may be printed.",
    "abc350_c": "Any valid sequence of swaps may be printed.",
    "abc362_c": "Any feasible sequence whose sum is zero may be printed.",
    "abc366_g": "Any satisfying sequence may be printed.",
    "abc373_g": "Any valid matching may be printed.",
    "abc396_e": "Any minimum-sum satisfying sequence may be printed.",
    "abc397_d": "Any satisfying pair may be printed.",
    "abc398_f": "Any shortest palindrome may be printed.",
    "arc181_c": "Any grid satisfying the row/column ordering may be printed.",
    "arc183_d": "Any valid sequence of operations may be printed.",
    "arc185_c": "Any satisfying triple may be printed.",
    "arc190_a": "Any minimum-cost operation plan may be printed.",
    "arc191_c": "Any satisfying pair may be printed.",
    "3194": "The returned indices may be in any order.",
    "3221": "The returned peak indices may be in any order.",
    "3234": "The returned good indices may be in any order.",
    "3453": "The returned strings may be in any order.",
}
# "Any" does not imply non-unique output when the required output is an optimum
# value, or mathematics ensures uniqueness. These were inspected explicitly.
UNIQUE_OUTPUT_EXCEPTIONS = {"1899_B", "3765", "abc327_b"}
_PATTERNS = (
    r"\b(?:print|output|return)\s+(?:any|one of)\b",
    r"\b(?:return|print|output|returned array)[^\n]{0,240}\bany order\b",
    (
        "\\bmultiple\\s+(?:valid\\s+|possible\\s+|correct\\s+)?(?:soluti"
        "ons?|answers?|outputs?)\\s+exist[^.]{0,100}\\bany\\b"
    ),
    r"\bany output that[^.]{0,180}\b(?:correct|accepted)\b",
    r"\bother outputs[^.]{0,100}\b(?:correct|accepted)\b",
)


def exclusion_reason(row):
    """Return the audited or detected reason to exclude a source, if any."""
    qid = str(row.get("source_question_id") or row["question_id"]).split("__perturbed")[
        0
    ]
    if qid in NON_UNIQUE_OUTPUTS:
        return NON_UNIQUE_OUTPUTS[qid]
    if qid in UNIQUE_OUTPUT_EXCEPTIONS:
        return None
    statement = row.get("question_content", "")
    if any(re.search(pattern, statement, re.I) for pattern in _PATTERNS):
        return "Output contract explicitly permits alternative answers or ordering."
    return None


def eligible_sources(rows):
    """Return sources whose output contracts satisfy the benchmark policy."""
    return [row for row in rows if exclusion_reason(row) is None]
