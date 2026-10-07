"""Small host-side helpers for the container boundary."""
import os
import tempfile
from pathlib import Path


def atomic_text(path, text):
    """Publish text atomically without following the destination symlink."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".host-write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(
            temporary, path
        )  # replaces a planted symlink instead of following it
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def model_cache_bind(host_home, sandbox_home, model):
    """Bind only the selected model cache into the agent filesystem."""
    source = Path(host_home) / "hub" / ("models--" + model.replace("/", "--"))
    if not source.is_dir():
        raise ValueError("Required model is absent from the host HF cache")
    source = source.resolve(strict=True)
    for path in source.rglob("*"):
        if path.is_symlink() and not path.resolve(strict=True).is_relative_to(source):
            raise ValueError(
                "Model cache contains a symlink outside its model directory"
            )
    target = Path(sandbox_home) / "hub" / source.name
    target.mkdir(parents=True, exist_ok=True)
    return f"{source}:{target}:ro"
