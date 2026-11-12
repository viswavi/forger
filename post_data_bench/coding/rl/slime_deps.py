"""Give Slime's Python the LM judge's client library from the TRL env.

Prompt criteria call the judge through litellm inside Slime's reward workers, but
the combined image installs litellm only in the TRL env, so every RL run with a
prompt criterion stopped at its first reward ("No module named 'litellm'", job
168462). Run with the TRL env's Python: it walks litellm's requirements and copies
each distribution Slime's Python lacks (files and metadata) into DEST; ones Slime
already has are left alone. Both are CPython 3.12 in the same image, so compiled
extensions load in either.

    python slime_deps.py /usr/local/lib/python3.12/dist-packages /usr/bin/python3
"""
from __future__ import annotations

import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement

ROOT_DISTRIBUTIONS = ("litellm",)


def closure(names) -> list[str]:
    """Names of the distributions `names` need (no extras), themselves included."""
    seen, queue = {}, list(names)
    while queue:
        name = queue.pop()
        dist = metadata.distribution(name)
        key = dist.metadata["Name"].lower()
        if key in seen:
            continue
        seen[key] = dist.metadata["Name"]
        for text in dist.requires or ():
            req = Requirement(text)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                queue.append(req.name)
    return sorted(seen.values())


def present(python: str, names: list[str]) -> set[str]:
    """The subset of `names` installed for `python`."""
    script = (
        "import importlib.metadata as m, json, sys\n"
        "have = []\n"
        "for n in json.loads(sys.argv[1]):\n"
        "    try: m.version(n); have.append(n)\n"
        "    except m.PackageNotFoundError: pass\n"
        "print(json.dumps(have))"
    )
    out = subprocess.run(
        [python, "-c", script, json.dumps(names)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return set(json.loads(out))


def copy_distribution(name: str, dest: Path) -> int:
    """Copy one distribution's site-packages files into `dest`; returns the count."""
    dist = metadata.distribution(name)
    count = 0
    for file in dist.files or ():
        if str(file).startswith(".."):  # console scripts outside site-packages
            continue
        source = Path(dist.locate_file(file))
        if not source.is_file():
            continue
        target = dest / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
    return count


def main() -> None:
    """Copy what Slime's Python lacks and report each distribution copied."""
    dest, python = Path(sys.argv[1]), sys.argv[2]
    names = closure(ROOT_DISTRIBUTIONS)
    have = present(python, names)
    missing = [n for n in names if n not in have]
    for name in missing:
        files = copy_distribution(name, dest)
        print(f">>> {name} {metadata.version(name)}: {files} files")
    print(f">>> litellm closure: {len(names)} distributions, {len(missing)} copied")


if __name__ == "__main__":
    main()
