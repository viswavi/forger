"""Filter operations for Forger.

Filter operations remove data based on criteria.
If a filter removes ALL data, the pipeline terminates early.
"""
from pyforger.filter.agreement import (
    by_exact_agreement,
    by_group_agreement,
    by_rouge_l_agreement,
)
from pyforger.filter.diversity import by_exact_match, by_n_gram_overlap, by_rouge_l
from pyforger.filter.format import by_invalid_response, by_keyword_blocklist, by_length
from pyforger.filter.judge import by_llm_score
from pyforger.filter.model_difficulty import by_comparative_difficulty, by_pass_at_k
from pyforger.filter.validation import by_syntax

__all__ = [
    "by_comparative_difficulty",
    "by_exact_agreement",
    "by_exact_match",
    "by_group_agreement",
    "by_invalid_response",
    "by_keyword_blocklist",
    "by_length",
    "by_llm_score",
    "by_n_gram_overlap",
    "by_pass_at_k",
    "by_rouge_l",
    "by_rouge_l_agreement",
    "by_syntax",
]
