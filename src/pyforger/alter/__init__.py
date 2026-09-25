"""Alter operations for Forger.

Alter operations transform existing data without removing items.
"""
from pyforger.alter import translate
from pyforger.alter.execute_code import execute_code
from pyforger.alter.generate import generate_from_context, self_taught_reasoner

__all__ = ["execute_code", "generate_from_context", "self_taught_reasoner", "translate"]
