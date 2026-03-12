"""Filter operations for Forger.

Filter operations remove data based on criteria.
If a filter removes ALL data, the pipeline terminates early.
"""
from pyforger.filter.diversity import by_exact_match, by_n_gram_overlap, by_rouge_l

__all__ = ["by_exact_match", "by_n_gram_overlap", "by_rouge_l"]
