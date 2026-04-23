"""Alter operations for Forger.

Alter operations transform existing data without removing items.
"""
from pyforger.alter.concat import concat
from pyforger.alter.generate import generate_from_context

__all__ = ["concat", "generate_from_context"]
