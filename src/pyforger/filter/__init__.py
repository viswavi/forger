"""Filter operations for Forger.

Filter operations remove data based on criteria.
If a filter removes ALL data, the pipeline terminates early.
"""
from pyforger.filter.agreement import by_exact_agreement, by_rouge_l_agreement

__all__ = ["by_exact_agreement", "by_rouge_l_agreement"]
