#!/usr/bin/env bash
# Initialize the pinned grader and apply the local timeout fix idempotently.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LCB="$SCRIPT_DIR/coding/LiveCodeBench"
git -C "$REPO_ROOT" submodule update --init -- "$LCB"
# Apply the timeout fix before the metadata race fix; detect both independently.
# The first patch's final context changes after the second, so check for the
# fully patched source first.
if git -C "$LCB" apply --reverse --check "$SCRIPT_DIR/patches/livecodebench-metadata-race.patch" 2>/dev/null; then
    echo "LiveCodeBench metadata/timeout fixes already applied"
else
    PATCH="$SCRIPT_DIR/patches/livecodebench-timeout.patch"
    if ! git -C "$LCB" apply --reverse --check "$PATCH" 2>/dev/null; then
        git -C "$LCB" apply --check "$PATCH"
        git -C "$LCB" apply "$PATCH"
    fi
    git -C "$LCB" apply "$SCRIPT_DIR/patches/livecodebench-metadata-race.patch"
fi
