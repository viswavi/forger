"""Artifact primitives tested against real synthetic temporary files."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest
from pyforger.artifacts import ArtifactRef, ArtifactStore
from pyforger.op import clear_trace, op, trace


def tree(root):
    """Create a synthetic tree with nested, binary, and empty files."""
    root.mkdir()
    (root / "nested").mkdir()
    (root / "nested/data.bin").write_bytes(b"synthetic\x00data" * 100_000)
    (root / "a.txt").write_text("synthetic fixture")
    (root / "empty").write_bytes(b"")
    return root


def test_deterministic_immutable_manifest(tmp_path):
    """Equivalent contents yield immutable manifests independent of timestamps."""
    ref = ArtifactRef.capture(tree(tmp_path / "one"))
    other = ArtifactRef.capture(tree(tmp_path / "two"))
    os.utime(other.root / "a.txt", (100, 100))
    assert ref.digest == other.cache_key()
    assert [f.path for f in ref.files] == ["a.txt", "empty", "nested/data.bin"]
    assert ref.files[0].size == len(b"synthetic fixture")
    with pytest.raises(FrozenInstanceError):
        ref.digest = "changed"
    with pytest.raises(FrozenInstanceError):
        ref.files[0].size = 2


def test_empty_artifact(tmp_path):
    """Empty directories do not change an empty artifact manifest."""
    ref = ArtifactRef.capture(tmp_path)
    assert ref.files == ()
    (tmp_path / "directory").mkdir()
    ref.verify()


def test_json_roundtrip(tmp_path):
    """Restored references retain identity and detect subsequent missing files."""
    ref = ArtifactRef.capture(tree(tmp_path / "source"))
    restored = ArtifactRef.from_dict(json.loads(json.dumps(ref.to_dict())))
    assert restored == ref
    restored.verify()
    (ref.root / "a.txt").unlink()
    with pytest.raises(ValueError, match="missing"):
        restored.verify()


@pytest.mark.parametrize("change", ["missing", "tampered", "additional"])
def test_mutation(tmp_path, change):
    """All artifact access paths reject missing, changed, or additional files."""
    ref = ArtifactRef.capture(tree(tmp_path / "source"))
    if change == "missing":
        (ref.root / "a.txt").unlink()
    elif change == "tampered":
        (ref.root / "a.txt").write_text("SYNTHETIC FIXTURE")
    else:
        (ref.root / "extra").write_text("extra")
    for check in (ref.verify, ref.cache_key, lambda: ref.path("empty")):
        with pytest.raises(ValueError, match=change):
            check()


def test_missing_and_non_directory_root(tmp_path):
    """Capture and verification reject missing and regular-file roots."""
    ref = ArtifactRef.capture(tree(tmp_path / "source"))
    ref.root.rename(tmp_path / "moved")
    with pytest.raises(ValueError, match="Cannot read artifact"):
        ref.verify()
    with pytest.raises(ValueError, match="directory"):
        ArtifactRef.capture(tmp_path / "moved/a.txt")


@pytest.mark.parametrize(
    "relative",
    [
        "",
        ".",
        "..",
        "../outside",
        "nested/../a.txt",
        "/abs",
        "a//b",
        "a/./b",
        "C:/file",
        "a\\b",
        "a\x00b",
    ],
)
def test_unsafe_paths(tmp_path, relative):
    """Artifact lookup rejects traversal and malformed relative paths."""
    ref = ArtifactRef.capture(tree(tmp_path / "source"))
    with pytest.raises(ValueError, match="relative path"):
        ref.path(relative)


def test_path_requires_manifest_file(tmp_path):
    """Lookup exposes only regular files present in the verified manifest."""
    ref = ArtifactRef.capture(tree(tmp_path / "source"))
    assert ref.path("nested/data.bin").read_bytes().startswith(b"synthetic")
    for path in ("nested", "unknown"):
        with pytest.raises(ValueError, match="not in artifact"):
            ref.path(path)


@pytest.mark.parametrize("location", ["root", "file", "directory", "broken"])
def test_symlinks(tmp_path, location):
    """Capture rejects root and descendant symlinks, including broken links."""
    root = tree(tmp_path / "source")
    if location == "root":
        link = tmp_path / "link"
        link.symlink_to(root, target_is_directory=True)
        root = link
    else:
        target = {
            "file": root / "a.txt",
            "directory": root / "nested",
            "broken": root / "absent",
        }[location]
        (root / "link").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        ArtifactRef.capture(root)


def test_fifo_and_symlink_mutation(tmp_path):
    """Verification rejects new special files and replacement symlinks."""
    ref = ArtifactRef.capture(tree(tmp_path / "source"))
    os.mkfifo(ref.root / "fifo")
    with pytest.raises(ValueError, match="special file"):
        ref.verify()
    (ref.root / "fifo").unlink()
    (ref.root / "a.txt").unlink()
    (ref.root / "a.txt").symlink_to(ref.root / "empty")
    with pytest.raises(ValueError, match="symlink"):
        ref.cache_key()


def test_invalid_ledger(tmp_path):
    """Deserialization rejects malformed and inconsistent manifests."""
    ref = ArtifactRef.capture(tree(tmp_path / "source"))
    for mutate in (
        lambda d: d.update(digest="bad"),
        lambda d: d["files"][0].update(path="../escape"),
        lambda d: d["files"][0].update(size=-1),
        lambda d: d["files"][0].update(size=True),
        lambda d: d["files"].reverse(),
        lambda d: d["files"].append(d["files"][0]),
        lambda d: d["files"][0].update(sha256="0" * 64),
        lambda d: d.pop("root"),
    ):
        data = ref.to_dict()
        mutate(data)
        with pytest.raises(ValueError):
            ArtifactRef.from_dict(data)


def test_op_cache_invalidation(tmp_path, monkeypatch):
    """Cached operations verify artifacts and recompute for new content digests."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "cache"))

    @op
    def read(ref):
        return ref.path("a.txt").read_text()

    ref = ArtifactRef.capture(tree(tmp_path / "source"))
    clear_trace()
    assert read(ref) == read(ref) == "synthetic fixture"
    assert [record.cache_hit for record in trace()] == [False, True]
    (ref.root / "a.txt").write_text("changed fixture")
    with pytest.raises(ValueError, match="tampered"):
        read(ref)
    assert read(ArtifactRef.capture(ref.root)) == "changed fixture"
    assert not trace()[-1].cache_hit


def test_store_copy_dedup_corruption(tmp_path):
    """Stored copies are independent, deduplicated, and checked for corruption."""
    source = tree(tmp_path / "source")
    store = ArtifactStore(tmp_path / "store")
    ref = store.put(source)
    assert ref == store.get(ref.digest) == store.put(source)
    assert ref.root == store.root / ref.digest
    (source / "a.txt").write_text("source changed")
    ref.verify()
    (ref.root / "a.txt").write_text("store changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        store.get(ref.digest)
    with pytest.raises(ValueError):
        store.put(tree(tmp_path / "original"))
    with pytest.raises(ValueError):
        store.get("../escape")


def test_concurrent_store_publication(tmp_path):
    """Concurrent publication retains one verified content-addressed directory."""
    source = tree(tmp_path / "source")
    store = ArtifactStore(tmp_path / "store")
    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(store.put, [source] * 16))
    assert all(ref == refs[0] for ref in refs)
    assert list(store.root.iterdir()) == [refs[0].root]
    refs[0].verify()


def test_store_symlinks(tmp_path):
    """Publication rejects symlink destinations and symlink-bearing sources."""
    source = tree(tmp_path / "source")
    store = ArtifactStore(tmp_path / "store")
    ref = ArtifactRef.capture(source)
    (store.root / ref.digest).symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.put(source)
    (source / "link").symlink_to(source / "a.txt")
    with pytest.raises(ValueError, match="symlink"):
        store.put(source)


def test_trusted_ancestor_alias(tmp_path):
    """Ancestor aliases normalize once without allowing root symlinks."""
    physical = tmp_path / "physical"
    physical.mkdir()
    source = tree(physical / "source")
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    ref = ArtifactRef.capture(alias / "source")
    assert ref.root == source.resolve()
    assert ref.path("a.txt") == source.resolve() / "a.txt"
    ref.verify()
    ledger = ref.to_dict()
    ledger["root"] = str(alias / "source")
    assert ArtifactRef.from_dict(ledger) == ref
    store = ArtifactStore(alias / "store")
    assert store.root == physical.resolve() / "store"
    stored = store.put(alias / "source")
    assert store.get(stored.digest) == stored
    alias.unlink()
    alias.symlink_to(tmp_path, target_is_directory=True)
    ref.verify()
    assert store.get(stored.digest) == stored
    (physical / "root-link").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        ArtifactRef.capture(physical / "root-link")


@pytest.mark.parametrize("replacement", ["root", "ancestor"])
def test_store_rejects_replaced_physical_path(tmp_path, replacement):
    """Replacing a normalized store root or ancestor with a symlink fails."""
    physical = tmp_path / "physical"
    physical.mkdir()
    source = tree(tmp_path / "source")
    store = ArtifactStore(physical / "store")
    ref = store.put(source)
    target = store.root if replacement == "root" else physical
    moved = tmp_path / "moved"
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=True)
    for action in (lambda: store.get(ref.digest), lambda: store.put(source)):
        with pytest.raises(ValueError, match="symlink"):
            action()


def test_store_missing_digest(tmp_path):
    """An absent stored digest raises a descriptive validation error."""
    store = ArtifactStore(tmp_path / "store")
    with pytest.raises(ValueError, match="Cannot read stored artifact"):
        store.get("0" * 64)


def test_bounded_text_reads_regular_files_only(tmp_path):
    from pyforger.artifacts import read_text

    path = tmp_path / "text"
    path.write_text("example")
    assert read_text(path, 7) == "example"
    with pytest.raises(ValueError, match="limit"):
        read_text(path, 6)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        read_text(link)
    for limit in (0, -1, True):
        with pytest.raises(ValueError):
            read_text(path, limit)
