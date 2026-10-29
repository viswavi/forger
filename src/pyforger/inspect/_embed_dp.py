"""Data-parallel sentence-transformers embedding: one worker process per GPU.

The texts are sorted by length and dealt round-robin, so each GPU gets an even
mix of long and short texts; the embeddings come back in input order.

Each worker is a fresh `python -m pyforger.inspect._embed_dp_worker` process with
`CUDA_VISIBLE_DEVICES` set to its GPU, not sentence-transformers'
`start_multi_process_pool`: that pool uses multiprocessing "spawn", which
re-imports the caller's main script and so reruns a pipeline script that has no
`if __name__ == "__main__":` guard.
"""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
import traceback
from multiprocessing.connection import wait
from typing import Any


def serve(conn: Any) -> None:
    """Embed one shard: receive `(model_id, texts, encode_kwargs)`, send the array.

    `CUDA_VISIBLE_DEVICES` was set to this worker's GPU by the parent.
    """
    try:
        model_id, texts, encode_kwargs = conn.recv()
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model_id)
        conn.send(("ok", encoder.encode(texts, **encode_kwargs)))
    except BaseException:  # report any failure to the parent, then exit
        conn.send(("error", traceback.format_exc()))


class EmbedProcess:
    """One worker process embedding one shard of texts on one GPU."""

    def __init__(self, gpu: str) -> None:
        """Start the worker; `send` gives it texts, `result` waits for vectors."""
        self.gpu = gpu
        self.connection, child = multiprocessing.Pipe()
        # The worker imports this pyforger, e.g. a checkout on sys.path that is
        # not the installed package.
        src = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=path)
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pyforger.inspect._embed_dp_worker",
                str(child.fileno()),
            ],
            pass_fds=[child.fileno()],
            env=env,
        )
        child.close()

    def _exited(self) -> RuntimeError:
        return RuntimeError(
            f"embedding worker on GPU {self.gpu} exited "
            f"(exit code {self._process.poll()})"
        )

    def send(self, model_id: str, texts: list[str], encode_kwargs: dict) -> None:
        """Give the worker its texts; blocks until the worker has read them."""
        try:
            self.connection.send((model_id, texts, encode_kwargs))
        except OSError:  # e.g. BrokenPipeError: the worker died while starting
            raise self._exited() from None

    def result(self) -> Any:
        """The shard's embeddings; a worker error or exit raises RuntimeError."""
        try:
            status, payload = self.connection.recv()
        except EOFError:
            raise self._exited() from None
        if status == "error":
            raise RuntimeError(f"embedding worker on GPU {self.gpu} failed:\n{payload}")
        return payload

    def close(self) -> None:
        """Stop the worker, if it is still running, and free its GPU."""
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self.connection.close()


def encode(
    texts: list[str],
    model_id: str,
    gpus: list[str],
    encode_kwargs: dict[str, Any],
) -> Any:
    """Embed `texts` with `model_id`, one worker per GPU in `gpus`, in input order.

    Returns an array of shape ``(len(texts), dim)``, as
    ``SentenceTransformer.encode(texts, **encode_kwargs)`` with
    ``convert_to_numpy=True`` would. Every worker is stopped before this
    returns or raises.
    """
    import numpy as np

    by_length = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    shards = [by_length[k :: len(gpus)] for k in range(len(gpus))]
    workers: list[tuple[list[int], EmbedProcess]] = []
    try:
        # Start every worker before sending any texts, so they import and load
        # the model at the same time.
        for gpu, indices in zip(gpus, shards):
            if indices:
                workers.append((indices, EmbedProcess(gpu)))
        for indices, worker in workers:
            worker.send(model_id, [texts[i] for i in indices], encode_kwargs)
        embeddings = None
        # Take results as they come, so a failing worker raises (and the finally
        # stops the others) without waiting for the slower ones.
        pending = {worker.connection: (indices, worker) for indices, worker in workers}
        while pending:
            for connection in wait(list(pending)):
                indices, worker = pending.pop(connection)  # type: ignore[arg-type]
                vectors = np.asarray(worker.result())
                if embeddings is None:
                    shape = (len(texts), *vectors.shape[1:])
                    embeddings = np.empty(shape, vectors.dtype)
                embeddings[indices] = vectors
        return embeddings
    finally:
        for _, worker in workers:
            worker.close()
