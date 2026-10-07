"""Use the common benchmark manifest contract."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coding.split_contract import (  # noqa: E402 - standalone script path setup
    assert_disjoint,
    sha256,
    staged_build,
    verify_manifest,
    write_manifest,
)

__all__ = [
    "assert_disjoint",
    "sha256",
    "verify_manifest",
    "write_manifest",
    "staged_build",
]
