"""Filter operations for Forger.

Filter operations remove data based on criteria.
If a filter removes ALL data, the pipeline can terminate early.
"""

from pyforger.core import FilterOperation

__all__ = ["FilterOperation"]
