"""Use the common benchmark manifest contract."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coding.split_contract import assert_disjoint, sha256, verify_manifest, write_manifest, staged_build
