"""Filter operations for Forger.

Filter operations remove data based on criteria.
If a filter removes ALL data, the pipeline terminates early.
"""
from pyforger.filter.validation import by_syntax

__all__ = ["by_syntax"]
