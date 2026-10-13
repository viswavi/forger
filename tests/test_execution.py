"""Validation tests are offline; real Docker tests require explicit opt-in."""
from __future__ import annotations

import io
import os
import subprocess
import tarfile
from dataclasses import FrozenInstanceError

import pytest
from pyforger.execution import DockerBackend, ExecutionSpec


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", ""),
        ("image", "--privileged"),
        ("image", "image name"),
        ("command", ()),
        ("command", ["python"]),
        ("command", ("",)),
        ("command", ("python", "\x00")),
        ("workdir", "/tmp"),
        ("timeout_s", 0),
        ("timeout_s", float("nan")),
        ("timeout_s", float("inf")),
        ("memory_mb", -1),
        ("memory_mb", 1.5),
        ("cpus", 0),
        ("pids_limit", True),
        ("pids_limit", 1.5),
        ("outputs", ["ok"]),
        ("outputs", ("same", "same")),
        ("outputs", ("a", "a/b")),
    ],
)
def test_spec_validation(field, value):
    kwargs = {"image": "python:3.12-slim", "command": ("python", "-V"), field: value}
    with pytest.raises(ValueError):
        ExecutionSpec(**kwargs)


@pytest.mark.parametrize(
    "path",
    ["", "/etc/passwd", "../escape", "a/../b", "./a", "a//b", "a/", "a\\b", "a\x00b"],
)
def test_unsafe_output_paths(path):
    with pytest.raises(ValueError):
        ExecutionSpec("python:3.12-slim", ("python",), outputs=(path,))


def test_spec_is_frozen():
    spec = ExecutionSpec("python:3.12-slim", ("python",), outputs=("a/b.txt",))
    with pytest.raises(FrozenInstanceError):
        spec.image = "other"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_log_bytes": 0},
        {"max_log_bytes": True},
        {"cleanup_timeout_s": 0},
        {"cleanup_timeout_s": float("nan")},
    ],
)
def test_backend_validation(kwargs):
    with pytest.raises(ValueError):
        DockerBackend(**kwargs)


def make_archive(path, entries):
    with tarfile.open(path, "w") as tar:
        for name, kind, data in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
            else:
                member.linkname = data.decode()
                tar.addfile(member)


@pytest.mark.parametrize(
    "entries",
    [
        [("../outside", tarfile.REGTYPE, b"bad")],
        [("/outside", tarfile.REGTYPE, b"bad")],
        [("link", tarfile.SYMTYPE, b"/etc"), ("link/passwd", tarfile.REGTYPE, b"bad")],
        [("hard", tarfile.LNKTYPE, b"/etc/passwd")],
        [("fifo", tarfile.FIFOTYPE, b"")],
        [("dup", tarfile.REGTYPE, b"a"), ("dup", tarfile.REGTYPE, b"b")],
        [("a", tarfile.REGTYPE, b"a"), ("a/b", tarfile.REGTYPE, b"b")],
    ],
)
def test_archive_rejected_before_writing(tmp_path, entries):
    archive = tmp_path / "archive.tar"
    destination = tmp_path / "output"
    destination.mkdir()
    make_archive(archive, entries)
    with pytest.raises(ValueError):
        DockerBackend._extract(archive, destination, (), 1024)
    assert list(destination.iterdir()) == []


def test_archive_selects_only_declared_files(tmp_path):
    archive = tmp_path / "archive.tar"
    destination = tmp_path / "output"
    destination.mkdir()
    make_archive(
        archive,
        [
            ("nested", tarfile.DIRTYPE, b""),
            ("nested/out", tarfile.REGTYPE, b"result"),
            ("unrequested", tarfile.REGTYPE, b"private"),
        ],
    )
    DockerBackend._extract(archive, destination, ("nested/out",), 1024)
    assert (destination / "nested/out").read_bytes() == b"result"
    assert not (destination / "unrequested").exists()


@pytest.mark.parametrize("outputs,limit", [(("missing",), 1024), (("out",), 1)])
def test_missing_or_oversized_output(tmp_path, outputs, limit):
    archive = tmp_path / "archive.tar"
    destination = tmp_path / "output"
    destination.mkdir()
    make_archive(archive, [("out", tarfile.REGTYPE, b"result")])
    with pytest.raises(ValueError):
        DockerBackend._extract(archive, destination, outputs, limit)
    assert not list(destination.iterdir())


@pytest.fixture
def docker_image():
    if os.environ.get("FORGER_DOCKER_TESTS") != "1":
        pytest.skip("real Docker execution requires FORGER_DOCKER_TESTS=1")
    image = os.environ.get("FORGER_DOCKER_IMAGE", "python:3.12-slim")
    try:
        checked = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip("Docker unavailable: " + str(exc))
    if checked.returncode:
        pytest.skip("trusted image not installed; ask parent before pulling: " + image)
    pytest.importorskip("pyforger.artifacts")
    return image


def execute(image, tmp_path, script, **kwargs):
    return DockerBackend(max_log_bytes=1024).execute(
        ExecutionSpec(image, ("python", "-c", script), timeout_s=20, **kwargs),
        tmp_path / "output",
    )


def test_docker_round_trip(docker_image, tmp_path):
    from pyforger.artifacts import ArtifactRef

    root = tmp_path / "input"
    root.mkdir()
    (root / "seed.txt").write_text("seed")
    result = execute(
        docker_image,
        tmp_path,
        "from pathlib import Path; print('hello'); "
        "Path('nested').mkdir(); "
        "Path('nested/out.txt').write_text(Path('seed.txt').read_text()+'!')",
        inputs=ArtifactRef.capture(root),
        outputs=("nested/out.txt",),
    )
    assert result.status == "completed", result.stderr
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    result.artifacts.verify()
    assert result.artifacts.path("nested/out.txt").read_text() == "seed!"
    assert not (tmp_path / "output/seed.txt").exists()
    assert (root / "seed.txt").read_text() == "seed"


def test_docker_failure_and_bounded_logs(docker_image, tmp_path):
    result = execute(
        docker_image,
        tmp_path,
        "import sys; print('x'*100000); print('y'*100000,file=sys.stderr); sys.exit(7)",
    )
    assert result.status == "failed", result.stderr
    assert result.exit_code == 7
    assert len(result.stdout) <= 1024
    assert len(result.stderr) <= 1024
    assert result.artifacts is None


def test_docker_timeout_and_cleanup(docker_image, tmp_path):
    before = subprocess.check_output(
        ["docker", "ps", "-aq", "--filter", "name=forger-"]
    )
    result = DockerBackend().execute(
        ExecutionSpec(
            docker_image, ("python", "-c", "import time; time.sleep(60)"), timeout_s=3
        ),
        tmp_path / "output",
    )
    assert result.status == "timed_out", result.stderr
    assert result.exit_code is None
    assert result.duration_s < 20
    after = subprocess.check_output(["docker", "ps", "-aq", "--filter", "name=forger-"])
    assert before == after


@pytest.mark.parametrize(
    "script",
    [
        "from pathlib import Path; Path('out').symlink_to('/etc/passwd')",
        "from pathlib import Path; Path('dir').symlink_to('/etc'); print('done')",
        "from pathlib import Path; Path('out').mkdir()",
        "print('missing')",
    ],
)
def test_docker_rejects_unsafe_or_missing_outputs(docker_image, tmp_path, script):
    result = execute(docker_image, tmp_path, script, outputs=("out",))
    assert result.status == "infrastructure_error"
    assert result.artifacts is None
    assert not (tmp_path / "output").exists()


def test_docker_isolation(docker_image, tmp_path):
    script = """
import os, socket
from pathlib import Path
assert os.getuid() == 65534
assert not Path('/var/run/docker.sock').exists()
assert 'GITHUB_TOKEN' not in os.environ
try:
    Path('/forger-write-test').write_text('bad')
except OSError:
    pass
else:
    raise AssertionError('writable root')
s = socket.socket()
s.settimeout(1)
assert s.connect_ex(('192.0.2.1', 80)) != 0
Path('out').write_text('isolated')
"""
    result = execute(docker_image, tmp_path, script, outputs=("out",))
    assert result.status == "completed", result.stderr


def test_docker_fresh_workdir(docker_image, tmp_path):
    first = execute(
        docker_image, tmp_path, "from pathlib import Path; Path('marker').touch()"
    )
    assert first.status == "completed", first.stderr
    second = execute(
        docker_image,
        tmp_path,
        "from pathlib import Path; assert not Path('marker').exists()",
    )
    assert second.status == "completed", second.stderr


def test_stage_matches_manifest(tmp_path):
    from pyforger.artifacts import ArtifactRef

    root = tmp_path / "source"
    (root / "nested").mkdir(parents=True)
    (root / "nested/input").write_bytes(b"contents")
    (root / "empty").mkdir()
    ref = ArtifactRef.capture(root)
    stage = tmp_path / "stage"
    stage.mkdir()
    DockerBackend._stage_inputs(ref, stage)
    assert ArtifactRef.capture(stage).digest == ref.digest
    assert not (stage / "empty").exists()


def test_stage_rejects_changed_inputs(tmp_path):
    from pyforger.artifacts import ArtifactRef

    root = tmp_path / "source"
    root.mkdir()
    (root / "input").write_bytes(b"original")
    ref = ArtifactRef.capture(root)
    (root / "input").write_bytes(b"changed")
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(ValueError):
        DockerBackend._stage_inputs(ref, stage)
    assert not list(stage.iterdir())


def test_output_parent_alias_and_leaf_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    assert DockerBackend._output_path(alias / "new") == real / "new"
    leaf = real / "leaf"
    leaf.symlink_to(tmp_path / "absent", target_is_directory=True)
    with pytest.raises(ValueError):
        DockerBackend._output_path(alias / "leaf")


def test_docker_failure_missing_outputs_stays_failed(docker_image, tmp_path):
    result = execute(
        docker_image, tmp_path, "import sys; sys.exit(7)", outputs=("missing",)
    )
    assert result.status == "failed", result.stderr
    assert result.exit_code == 7
    assert result.artifacts is None
    assert not (tmp_path / "output").exists()


def test_input_archive_normalizes_permissions(tmp_path):
    stage = tmp_path / "stage"
    (stage / "nested").mkdir(parents=True)
    (stage / "nested/file").write_text("input")
    archive = tmp_path / "input.tar"
    DockerBackend._input_archive(stage, archive)
    with tarfile.open(archive) as tar:
        assert [(m.name, m.mode, m.uid, m.gid) for m in tar] == [
            ("nested", 0o777, 0, 0),
            ("nested/file", 0o666, 0, 0),
        ]


def test_docker_nested_inputs_are_writable(docker_image, tmp_path):
    from pyforger.artifacts import ArtifactRef

    root = tmp_path / "input"
    (root / "nested").mkdir(parents=True)
    (root / "nested/seed").write_text("input")
    result = execute(
        docker_image,
        tmp_path,
        "from pathlib import Path; p=Path('nested/seed'); "
        "p.write_text(p.read_text()+' modified'); "
        "Path('nested/new').write_text('new')",
        inputs=ArtifactRef.capture(root),
        outputs=("nested/seed", "nested/new"),
    )
    assert result.status == "completed", result.stderr
    assert result.artifacts.path("nested/seed").read_text() == "input modified"
    assert result.artifacts.path("nested/new").read_text() == "new"
    assert (root / "nested/seed").read_text() == "input"


def test_docker_rejects_linked_output_ancestor(docker_image, tmp_path):
    result = execute(
        docker_image,
        tmp_path,
        "from pathlib import Path; Path('nested').symlink_to('/etc')",
        outputs=("nested/passwd",),
    )
    assert result.status == "infrastructure_error", result.stderr
    assert result.artifacts is None
    assert not (tmp_path / "output").exists()


def test_docker_rejects_output_hardlinks(docker_image, tmp_path):
    result = execute(
        docker_image,
        tmp_path,
        "from pathlib import Path; import os; "
        "Path('original').write_text('data'); os.link('original','out')",
        outputs=("out",),
    )
    assert result.status == "infrastructure_error", result.stderr
    assert result.artifacts is None
    assert not (tmp_path / "output").exists()
