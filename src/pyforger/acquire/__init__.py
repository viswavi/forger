"""Acquire operations for Forger.

Acquire operations generate or retrieve new data from external sources.
They are typically "source" nodes in the DAG with no DataSpec inputs.
"""

from pyforger.core import AcquireOperation

__all__ = ["AcquireOperation"]
