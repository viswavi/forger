"""Data-parallel sentence-transformers embedding: one worker process per GPU.

The texts are sorted by length and dealt round-robin, so each GPU gets an even
mix of long and short texts; the embeddings come back in input order.

Each worker is a fresh `python -m pyforger.inspect._embed_dp_worker` process with
`CUDA_VISIBLE_DEVICES` set to its GPU, not sentence-transformers'
`start_multi_process_pool`: that pool uses multiprocessing "spawn", which
re-imports the caller's main script and so reruns a pipeline script that has no
`if __name__ == "__main__":` guard.

Every encoder, in a worker or in the caller's process, embeds in token-budgeted
batches (:func:`encode_in_token_batches`) rather than a fixed number of texts,
so a batch of long texts does not run the GPU out of memory.
"""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
import traceback
from multiprocessing.connection import wait
from typing import Any

# Padded tokens per batch (texts in the batch x tokens in its longest text);
# FORGER_EMBED_BATCH_TOKENS overrides it. A fixed 64 texts per batch ran a 48 GB
# L40S out of memory on SFT targets of a few thousand tokens. With this budget,
# Qwen3-Embedding-0.6B embedded 5112 uncut targets (up to 16K tokens) on one
# L40S with at most 4.5 GB reserved, and 2500 ~500-token texts in half the time
# of 64 per batch (less padding). On the 300 longest, budgets of 8192 / 16384 /
# 32768 took 48 / 79 / 100 s (4.7 / 5.4 / 9.6 GB); short texts were not timed
# at 8192.
DEFAULT_BATCH_TOKENS = 16384


def batch_tokens() -> int:
    """The token budget per batch: `FORGER_EMBED_BATCH_TOKENS`, else the default."""
    value = os.environ.get("FORGER_EMBED_BATCH_TOKENS")
    if value is None or not value.strip():
        return DEFAULT_BATCH_TOKENS
    budget = int(value)
    if budget < 1:
        raise ValueError(f"FORGER_EMBED_BATCH_TOKENS must be positive, not {value!r}")
    return budget


def token_lengths(encoder: Any, texts: list[str]) -> list[int]:
    """Tokens per text as the encoder sees them (cut at its `max_seq_length`).

    Falls back to character counts, which are at least the token counts for
    usual tokenizers, when the encoder has no text tokenizer.
    """
    tokenizer = getattr(encoder, "tokenizer", None)
    max_length = getattr(encoder, "max_seq_length", None)
    try:
        ids = tokenizer(  # type: ignore[misc]
            texts, truncation=max_length is not None, max_length=max_length
        )["input_ids"]
        return [len(i) for i in ids]
    except Exception:  # no tokenizer, or not a text tokenizer
        return [len(t) for t in texts]


def token_batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Group text indices, longest first, into batches of at most `budget` tokens.

    A batch costs its size times its longest text, as padded; a text longer
    than the budget is a batch of its own. e.g. lengths [5, 50, 20, 30] with
    budget 60 -> [[1], [3, 2], [0]].
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches: list[list[int]] = []
    for i in order:
        # Longest first, so a batch's first text is its longest.
        if batches and (len(batches[-1]) + 1) * lengths[batches[-1][0]] <= budget:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


def encode_in_token_batches(
    encoder: Any, texts: list[str], *, batch_tokens: int, **encode_kwargs: Any
) -> Any:
    """`encoder.encode(texts, **encode_kwargs)` in token-budgeted batches, in order.

    Returns an array of shape ``(len(texts), dim)``; `encode_kwargs` should ask
    for numpy (``convert_to_numpy=True``).
    """
    import numpy as np

    if not texts:
        return np.asarray(encoder.encode(texts, **encode_kwargs))
    embeddings = None
    for batch in token_batches(token_lengths(encoder, texts), batch_tokens):
        vectors = np.asarray(
            encoder.encode(
                [texts[i] for i in batch],
                batch_size=len(batch),
                show_progress_bar=False,
                **encode_kwargs,
            )
        )
        if embeddings is None:
            embeddings = np.empty((len(texts), *vectors.shape[1:]), vectors.dtype)
        embeddings[batch] = vectors
    return embeddings


def serve(conn: Any) -> None:
    """Embed one shard: receive `(model_id, texts, encode_kwargs)`, send the array.

    `encode_kwargs` go to :func:`encode_in_token_batches`. `CUDA_VISIBLE_DEVICES`
    was set to this worker's GPU by the parent.
    """
    try:
        model_id, texts, encode_kwargs = conn.recv()
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model_id)
        conn.send(("ok", encode_in_token_batches(encoder, texts, **encode_kwargs)))
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
    :func:`encode_in_token_batches` with `encode_kwargs` would in one process.
    Every worker is stopped before this returns or raises.
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
