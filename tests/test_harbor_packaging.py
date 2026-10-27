"""Native separate verifier packaging parsed by the installed Harbor SDK."""

from harbor.models.task.config import TaskConfig, VerifierEnvironmentMode
from harbor.models.task.paths import TaskPaths
from harbor.models.task.verifier_mode import resolve_verifier_environment_definition

from pyforger.agent_backend import AgentSpec, WorkspaceTask
from pyforger.harbor_backend import _package


def test_separate_root_verifier_never_builds_private_material_into_agent(tmp_path):
    task = WorkspaceTask(
        instruction="submit",
        files={"public.txt": "observations"},
        outputs=("output/result.json",),
        verifier="echo 1 > $WORKSPACE/reward.txt",
        verifier_files={"labels.json": "private-target"},
    )
    artifact = _package(
        task, AgentSpec(kind="nop"), tmp_path / "task", ["/work/output/result.json"]
    )
    config = TaskConfig.model_validate_toml(artifact.path("task.toml").read_text())
    assert config.verifier.environment_mode == VerifierEnvironmentMode.SEPARATE
    assert config.verifier.user == "root"
    assert config.agent.user == "65534"
    definition = resolve_verifier_environment_definition(
        config, TaskPaths(artifact.root)
    )
    assert definition.directory == artifact.root / "tests"
    assert definition.bundled_tests
    assert not any(
        "private-target" in p.read_text()
        for p in (artifact.root / "environment").rglob("*")
        if p.is_file()
    )
    dockerfile = artifact.path("tests/Dockerfile").read_text()
    assert "USER root" in dockerfile and "chmod 700 /tests /logs/verifier" in dockerfile
    assert "reward.txt" not in config.artifacts


def test_host_trusted_helper_is_only_in_separate_verifier_image(tmp_path):
    task = WorkspaceTask(
        instruction="submit",
        files={"public.txt": "x"},
        outputs=("output.json",),
        verifier="echo 0 > $WORKSPACE/reward.txt",
    )
    artifact = _package(
        task,
        AgentSpec(kind="nop"),
        tmp_path / "task",
        ["/work/output.json"],
        trusted_verifier_files={"submission_harness.py": b"trusted-host-bytes"},
    )
    assert (
        artifact.path("tests/trusted/submission_harness.py").read_bytes()
        == b"trusted-host-bytes"
    )
    assert "python3 -I /tests/launch.py" in artifact.path("tests/test.sh").read_text()
    assert (
        "cp -r /tests/trusted/. /opt/trusted/"
        in artifact.path("tests/Dockerfile").read_text()
    )
    assert not (artifact.root / "environment/trusted").exists()
    assert "COPY public /work" in artifact.path("tests/Dockerfile").read_text()


def test_verifier_root_bootstrap_executes_without_submitted_imports(tmp_path):
    import ast

    task = WorkspaceTask(
        instruction="submit",
        files={},
        outputs=("result.json",),
        verifier="python3 -I /tests/verifier/grade.py",
    )
    artifact = _package(
        task, AgentSpec(kind="nop"), tmp_path / "task", ["/work/result.json"]
    )
    tree = ast.parse(artifact.path("tests/launch.py").read_text())
    compile(tree, "trusted-launch", "exec")
    assert "PYTHONSAFEPATH" in artifact.path("tests/launch.py").read_text()
    command = artifact.path("tests/verifier-command.sh").read_text()
    assert "chmod 600 /work/reward.txt" in command
    assert "if [ ! -s /logs/verifier/reward.txt ]" in command


def test_public_custom_tokenizer_is_built_without_credentials(tmp_path):
    from pyforger.acquire.model import Model

    model = Model("openai/gpt-4o", custom_tokenizer="meta-models/Muse-Glimmer-30B")
    task = WorkspaceTask(instruction="submit", files={}, outputs=("result.json",))
    artifact = _package(
        task, AgentSpec(model=model), tmp_path / "task", ["/work/result.json"]
    )
    dockerfile = artifact.path("environment/Dockerfile").read_text()
    assert "snapshot_download" in dockerfile
    assert "token=False" in dockerfile
    assert "/opt/tokenizer" in dockerfile
    assert "model.safetensors" not in dockerfile
    assert "snapshot_download" not in artifact.path("tests/Dockerfile").read_text()


def test_empty_public_inputs_have_no_docker_copy_from_missing_directory(tmp_path):
    task = WorkspaceTask(
        instruction="submit",
        files={},
        outputs=("result.json",),
        verifier="echo 0 > $WORKSPACE/reward.txt",
    )
    artifact = _package(
        task, AgentSpec(kind="nop"), tmp_path / "task", ["/work/result.json"]
    )
    assert (
        "COPY --chown=65534:65534 files /work"
        not in artifact.path("environment/Dockerfile").read_text()
    )
    assert "COPY public /work" not in artifact.path("tests/Dockerfile").read_text()
