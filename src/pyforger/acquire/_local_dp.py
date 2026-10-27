"""Data-parallel vLLM for `LocalModel`: one engine per GPU group, each in a process.

`vllm.LLM` runs one engine in the calling process. A model that fits on one GPU
gets the most throughput from one full copy per GPU, each sampling its share of
the prompts, so `DataParallelEngine` starts a worker process per GPU group
(with `CUDA_VISIBLE_DEVICES` set to that group before vllm is imported) and
offers the same `chat(...)` call as `vllm.LLM`. Prompts are dealt round-robin
so long and short prompts spread evenly, and outputs come back in input order.
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any


def visible_gpus() -> list[str]:
    """GPU ids this process may use: `CUDA_VISIBLE_DEVICES`, else `nvidia-smi -L`.

    e.g. ``["0", "1", "2", "3"]``; empty when no GPU is visible or detectable.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [d.strip() for d in env.split(",") if d.strip() and d.strip() != "-1"]
    try:
        listing = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        str(i) for i, line in enumerate(listing.splitlines()) if line.startswith("GPU ")
    ]


def device_groups(
    gpus: list[str], tensor_parallel_size: int, data_parallel_size: int | str
) -> list[list[str]]:
    """Split `gpus` into one group per engine.

    ``"auto"`` gives as many engines as whole groups of `tensor_parallel_size`
    fit (8 GPUs, tensor_parallel_size=2 -> 4 engines); an int asks for exactly
    that many. Fewer than two groups means a single in-process engine (``[]``).
    """
    if data_parallel_size == "auto":
        count = len(gpus) // tensor_parallel_size
    elif isinstance(data_parallel_size, int) and data_parallel_size >= 1:
        count = data_parallel_size
        if count * tensor_parallel_size > len(gpus) and count > 1:
            raise ValueError(
                f"data_parallel_size={count} x "
                f"tensor_parallel_size={tensor_parallel_size} needs "
                f"{count * tensor_parallel_size} GPUs; {len(gpus)} are visible ({gpus})"
            )
    else:
        raise ValueError(
            "data_parallel_size must be 'auto' or a positive int, "
            f"not {data_parallel_size!r}"
        )
    if count < 2:
        return []
    return [
        gpus[i * tensor_parallel_size : (i + 1) * tensor_parallel_size]
        for i in range(count)
    ]


def _plain(output: Any) -> SimpleNamespace:
    """The fields LocalModel reads from a vLLM RequestOutput, as plain data."""
    return SimpleNamespace(
        prompt_token_ids=list(output.prompt_token_ids or []),
        outputs=[
            SimpleNamespace(
                text=c.text,
                token_ids=list(c.token_ids or []),
                finish_reason=getattr(c, "finish_reason", None),
            )
            for c in output.outputs
        ],
    )


def _worker(conn: Any, devices: list[str], engine_kwargs: dict[str, Any]) -> None:
    """Hold one `vllm.LLM` on `devices`; answer `chat` requests until told to stop."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    try:
        import vllm

        llm = vllm.LLM(**engine_kwargs)
    except BaseException:  # report any startup failure to the parent, then exit
        conn.send(("error", traceback.format_exc()))
        return
    conn.send(("ready", None))
    while True:
        request = conn.recv()
        if request is None:
            return
        messages, sampling_params, kwargs = request
        try:
            outputs = llm.chat(messages, sampling_params, **kwargs)
            conn.send(("ok", [_plain(o) for o in outputs]))
        except BaseException:
            conn.send(("error", traceback.format_exc()))


class EngineProcess:
    """One worker process holding a vLLM engine on a group of GPUs."""

    def __init__(self, devices: list[str], engine_kwargs: dict[str, Any]) -> None:
        """Start the worker; `wait_ready` blocks until its engine has loaded."""
        context = multiprocessing.get_context("spawn")  # CUDA cannot be forked
        self.devices = devices
        self._conn, child = context.Pipe()
        self._process = context.Process(
            target=_worker, args=(child, devices, engine_kwargs), daemon=True
        )
        self._process.start()
        child.close()

    def _receive(self) -> Any:
        """The worker's next reply; a worker error or exit raises RuntimeError."""
        try:
            status, payload = self._conn.recv()
        except EOFError:
            raise RuntimeError(
                f"LocalModel engine on GPU(s) {','.join(self.devices)} exited "
                f"(exit code {self._process.exitcode})"
            ) from None
        if status == "error":
            raise RuntimeError(
                f"LocalModel engine on GPU(s) {','.join(self.devices)} failed:\n"
                f"{payload}"
            )
        return payload

    def wait_ready(self) -> None:
        """Block until the engine has loaded its model."""
        self._receive()

    def chat(self, messages: list, sampling_params: Any, **kwargs: Any) -> list:
        """Run `vllm.LLM.chat` in the worker and return its outputs."""
        self._conn.send((messages, sampling_params, kwargs))
        return self._receive()

    def close(self) -> None:
        """Stop the worker and free its GPUs."""
        if self._process.is_alive():
            try:
                self._conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            self._process.join(timeout=30)
            if self._process.is_alive():
                self._process.kill()
        self._conn.close()


class DataParallelEngine:
    """Several engines behind `vllm.LLM`'s `chat` call.

    `engines` are objects with `chat(messages, sampling_params, **kwargs)`;
    `DataParallelEngine.start` builds them as worker processes.
    """

    def __init__(self, engines: list[Any]) -> None:
        """Wrap already-started engines."""
        self.engines = engines
        self._pool = ThreadPoolExecutor(max_workers=len(engines))

    @classmethod
    def start(
        cls, groups: list[list[str]], engine_kwargs: dict[str, Any]
    ) -> DataParallelEngine:
        """Start one engine per GPU group, loading them in parallel."""
        engines = [EngineProcess(group, engine_kwargs) for group in groups]
        try:
            for engine in engines:
                engine.wait_ready()
        except BaseException:
            for engine in engines:
                engine.close()
            raise
        dp = cls(engines)
        atexit.register(dp.close)
        return dp

    def chat(self, messages: list, sampling_params: Any, **kwargs: Any) -> list:
        """Deal `messages` round-robin to the engines; outputs return in input order."""
        shards = [
            list(range(i, len(messages), len(self.engines)))
            for i in range(len(self.engines))
        ]
        jobs = []
        for i, (engine, indices) in enumerate(zip(self.engines, shards)):
            if not indices:
                continue
            # One progress bar is enough; the others would interleave with it.
            shard_kwargs = kwargs if i == 0 else {**kwargs, "use_tqdm": False}
            shard = [messages[j] for j in indices]
            jobs.append(
                (
                    indices,
                    self._pool.submit(
                        engine.chat, shard, sampling_params, **shard_kwargs
                    ),
                )
            )
        outputs: list[Any] = [None] * len(messages)
        for indices, future in jobs:
            for j, output in zip(indices, future.result()):
                outputs[j] = output
        return outputs

    def close(self) -> None:
        """Stop every engine and free the GPUs."""
        for engine in self.engines:
            close = getattr(engine, "close", None)
            if close is not None:
                close()
        self._pool.shutdown(wait=False)
