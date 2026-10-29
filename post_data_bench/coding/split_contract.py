"""Public row schema and checks binding a benchmark build to its artifacts."""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

PUBLIC_FIELDS = (
    "question_title",
    "question_content",
    "platform",
    "question_id",
    "contest_id",
    "contest_date",
    "starter_code",
    "difficulty",
    "public_test_cases",
    "metadata",
    "gold_response",
    "private_test_case_idx",
    "task_text",
)


def sha256(path):
    """Hash a file incrementally without loading its contents into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(row):
    """Return the identity shared by all variants of one source."""
    if "private_test_case_idx" in row:
        return str(row["private_test_case_idx"])
    if "problem_id" in row:  # FrontierCS: one row per problem
        return f"frontiercs:{row['problem_id']}"
    return row["situation"].strip()


def assert_disjoint(validation, test):
    """Reject empty splits or overlapping task and source identities."""
    if not validation or not test:
        raise ValueError("Both benchmark splits must be nonempty")
    if "private_test_case_idx" in validation[0]:
        for fields in (("question_id",),):
            left = {
                tuple(row[k] for k in fields)
                for row in validation
                if all(k in row for k in fields)
            }
            right = {
                tuple(row[k] for k in fields)
                for row in test
                if all(k in row for k in fields)
            }
            if left & right:
                raise ValueError("Validation and secret test identities overlap")
    if {source_id(row) for row in validation} & {source_id(row) for row in test}:
        raise ValueError("Validation and secret test sources overlap")


def write_manifest(root, paths, validation, test, *, seed, inputs):
    """Bind generated artifacts and source membership to a versioned manifest."""
    root = Path(root).resolve()
    assert_disjoint(validation, test)
    manifest = {
        "version": 2,
        "seed": seed,
        "inputs": inputs,
        "sources": {
            "validation": sorted({source_id(r) for r in validation}),
            "test": sorted({source_id(r) for r in test}),
        },
        "outputs": {str(Path(p).resolve().relative_to(root)): sha256(p) for p in paths},
    }
    path = root / ".hidden/split_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def verify_manifest(root, *, public=False):
    """Verify artifact hashes, contained paths and disjoint source membership."""
    root = Path(root)
    path = root / (
        "validation_manifest.json" if public else ".hidden/split_manifest.json"
    )
    manifest = json.loads(path.read_text())
    for name, expected in manifest["outputs"].items():
        target = (root / name).resolve(strict=True)
        if not target.is_relative_to(root.resolve()) or sha256(target) != expected:
            raise ValueError("Benchmark artifact differs from its split manifest")
    if not public:
        if set(manifest["sources"]["validation"]) & set(manifest["sources"]["test"]):
            raise ValueError("Manifest sources overlap")
    return manifest


def public_rows(rows):
    """Select the public schema without exposing perturbation provenance."""
    result = []
    for row in rows:
        if "__perturbed" in str(row["question_id"]):
            raise ValueError(
                "Validation ids expose perturbation provenance; rebuild the data"
            )
        result.append({name: row[name] for name in PUBLIC_FIELDS})
    return result


def _finish_build(root, journal):
    """Resume interrupted publication; a matching manifest is the commit marker."""
    import os
    import shutil

    root = Path(root).resolve()
    stage = root / journal["stage"]
    if stage.parent != root or not stage.name.startswith(".build-"):
        raise ValueError("Invalid build recovery directory")
    for name, expected in journal["manifest"]["outputs"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Build output escapes its directory")
        pending = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if pending.exists():
            if sha256(pending) != expected:
                raise ValueError("Staged build changed before publication")
            os.replace(pending, target)
        elif not target.is_file() or sha256(target) != expected:
            raise ValueError("Interrupted build cannot be recovered: missing artifact")
    manifest = root / ".hidden/split_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(journal["manifest"], indent=2) + "\n")
    os.replace(temporary, manifest)
    (root / ".build_pending.json").unlink()
    shutil.rmtree(stage)


@contextmanager
def staged_build(root):
    """Stage all artifacts, then publish under a writer lock with journaled recovery.

    Readers fail closed on manifest mismatches during publication.
    """
    import fcntl
    import os
    import shutil
    import tempfile

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        journal_path = root / ".build_pending.json"
        if journal_path.exists():
            _finish_build(root, json.loads(journal_path.read_text()))
        stage = Path(tempfile.mkdtemp(prefix=".build-", dir=root))
        try:
            yield stage
            manifest = verify_manifest(stage)
            journal = {"stage": stage.name, "manifest": manifest}
            temporary = journal_path.with_suffix(".tmp")
            with temporary.open("w") as stream:
                json.dump(journal, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, journal_path)
            _finish_build(root, journal)
        finally:
            if not journal_path.exists():
                shutil.rmtree(stage, ignore_errors=True)
