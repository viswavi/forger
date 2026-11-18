"""Data-parallel vLLM for `LocalModel`: one engine per GPU group, each in a process.

`vllm.LLM` runs one engine in the calling process. A model that fits on one GPU
gets the most throughput from one full copy per GPU, each sampling its share of
the prompts, so `DataParallelEngine` starts a worker process per GPU group
(with `CUDA_VISIBLE_DEVICES` set to that group before vllm is imported) and
offers the same `chat(...)` call as `vllm.LLM`, plus `target_logprobs(...)`.
Prompts are dealt round-robin so long and short prompts spread evenly, and
outputs come back in input order.
"""

from __future__ import annotations

import atexit
import gc
import getpass
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

_logger = logging.getLogger("pyforger.acquire.model")


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


# Where Triton and torch.compile (Inductor) write compiled kernels, and the
# subdirectory each gets in an engine's own cache directory. By default every
# process on the machine shares /tmp/torchinductor_<user> (and ~/.triton):
# eight engines compiling at once on an apptainer /tmp failed with
# "IsADirectoryError: .../tmp.pid_40242_.../triton_.cubin".
_COMPILE_CACHES = {"TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor"}


def _torch_default_cache_dir() -> str:
    """The Inductor cache torch picks when TORCHINDUCTOR_CACHE_DIR is unset.

    torch also writes it back to ``os.environ`` the first time it compiles,
    so a variable at this value was not set by the user.
    """
    try:
        user = getpass.getuser()
    except Exception:  # no user name, e.g. a uid without a passwd entry
        return ""
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", user)
    return os.path.join(tempfile.gettempdir(), "torchinductor_" + sanitized)


def own_compile_caches(environ: Any) -> tuple[dict[str, str], str | None]:
    """Fresh Triton and Inductor cache directories for one engine process.

    Returns the variables to set and the directory to delete once the engine
    stops, e.g. ``({"TRITON_CACHE_DIR": "/dev/shm/pyforger-engine-x1/triton",
    "TORCHINDUCTOR_CACHE_DIR": "/dev/shm/pyforger-engine-x1/inductor"},
    "/dev/shm/pyforger-engine-x1")``. The directory is under /dev/shm when it
    is writable (memory, not a container's overlay /tmp), else the temp dir.
    A variable already set in `environ` is kept, so ``({}, None)`` when the
    user set both; TORCHINDUCTOR_CACHE_DIR at torch's own default counts as
    unset. The cost: each engine compiles its kernels afresh (start-up only).
    """
    default = _torch_default_cache_dir()
    missing = [
        name
        for name in _COMPILE_CACHES
        if not environ.get(name)
        or (name == "TORCHINDUCTOR_CACHE_DIR" and environ[name] == default)
    ]
    if not missing:
        return {}, None
    shm = "/dev/shm"
    roots = [shm] if os.path.isdir(shm) and os.access(shm, os.W_OK) else []
    for root in [*roots, None]:  # None: the temp dir
        try:
            scratch = tempfile.mkdtemp(prefix="pyforger-engine-", dir=root)
            break
        except OSError:
            if root is None:
                raise
    caches = {name: os.path.join(scratch, _COMPILE_CACHES[name]) for name in missing}
    return caches, scratch


def prepare_in_process_engine() -> None:
    """Set up this process to hold a `vllm.LLM`, as each worker process is set up.

    The engine core runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0)
    rather than in a child vLLM forks: under apptainer --fakeroot that
    child's CUDA init fails ("CUDA driver initialization failed"). Compiled
    kernels go to this process's own cache directories (`own_compile_caches`),
    deleted at exit. Variables already set are kept, e.g.
    VLLM_ENABLE_V1_MULTIPROCESSING=1 keeps vLLM's engine-core process.
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    caches, scratch = own_compile_caches(os.environ)
    os.environ.update(caches)
    if scratch is not None:
        atexit.register(shutil.rmtree, scratch, True)


# In-process `vllm.LLM` engines started and not yet shut down. vLLM's
# distributed state is per process, so it is torn down only when this is 0.
_live_engines = 0


def start_in_process_engine(llm_class: Any, engine_kwargs: dict[str, Any]) -> Any:
    """``llm_class(**engine_kwargs)``, e.g. a `vllm.LLM`, in this process."""
    global _live_engines
    prepare_in_process_engine()
    llm = llm_class(**engine_kwargs)
    _live_engines += 1
    return llm


def shutdown_in_process_engine(llm: Any) -> None:
    """Stop a `vllm.LLM`'s engine core, which frees its executor and workers.

    With VLLM_ENABLE_V1_MULTIPROCESSING=0 (see `prepare_in_process_engine`)
    the core runs in this process; with it on, this stops the core's process
    (and the workers it started for tensor_parallel_size > 1). Errors are
    logged, not raised: a failed close must not fail a run that has its samples.
    """
    global _live_engines
    _live_engines = max(0, _live_engines - 1)
    core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception as e:  # noqa: BLE001
            _logger.warning("vLLM engine shutdown failed: %r", e)


def release_in_process_engine() -> None:
    """Free GPU memory once the last reference to a `vllm.LLM` is dropped.

    Runs vLLM's `cleanup_dist_env_and_memory` (what vLLM's own tests run
    between engines in one process): it destroys the model-parallel groups
    and torch.distributed, collects garbage, and empties the CUDA cache.
    While another in-process engine is running, which still needs those
    groups, or where vLLM lacks the function, only `gc.collect()` and
    `torch.cuda.empty_cache()` run. Best effort: memory an object still
    referenced elsewhere holds, and the CUDA context itself, stay allocated.
    """
    if _live_engines == 0:
        _cleanup_vllm()
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cleanup_vllm() -> None:
    """vLLM's `cleanup_dist_env_and_memory`, when this vLLM has it."""
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    except Exception:  # noqa: BLE001  # moved or missing in this vLLM
        cleanup_dist_env_and_memory = None
    if cleanup_dist_env_and_memory is not None:
        try:
            cleanup_dist_env_and_memory()
        except Exception as e:  # noqa: BLE001
            _logger.warning("vLLM cleanup_dist_env_and_memory failed: %r", e)


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


def target_logprobs(
    llm: Any,
    pairs: list[tuple[list[dict], str]],
    *,
    chat_template_kwargs: dict[str, Any],
    use_tqdm: bool = False,
) -> list[dict[str, Any]]:
    """Log-probability of each target token given its conversation, under `llm`.

    Each conversation is rendered with the chat template and the generation
    prompt, as `chat` renders it for sampling, and its target follows as the
    assistant's response. vLLM's prompt logprobs give each token's
    log-probability given everything before it. A target with `</think>` is
    tokenized in two parts, through `</think>` and after it, so the reasoning
    tokens can be counted apart.

    Returns one dict per pair: `logprobs` (natural log, one per scored target
    token), `reasoning_tokens` (target tokens through `</think>`, 0 without
    one), and `truncated` (the target did not fit in the context and its end
    was not scored).
    """
    import vllm

    tokenizer = llm.get_tokenizer()
    max_len = llm.llm_engine.model_config.max_model_len
    requests, spans = [], []
    for messages, target in pairs:
        prefix = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        head, close, tail = target.partition("</think>")
        parts = [head + close, tail] if close else [target]
        part_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in parts]
        ids = prompt_ids + [t for part in part_ids for t in part]
        end = min(len(ids), max_len - 1)  # room for the one generated token
        spans.append((len(prompt_ids), end, len(ids), len(part_ids[0]) if close else 0))
        requests.append({"prompt_token_ids": ids[: max(end, 1)]})
    params = vllm.SamplingParams(
        max_tokens=1, temperature=0.0, prompt_logprobs=0, detokenize=False
    )
    outputs = llm.generate(requests, params, use_tqdm=use_tqdm)
    scores = []
    for request, output, (start, end, full, reasoning) in zip(requests, outputs, spans):
        ids = request["prompt_token_ids"]
        logprobs = [
            output.prompt_logprobs[j][ids[j]].logprob for j in range(start, end)
        ]
        scores.append(
            {
                "logprobs": logprobs,
                "reasoning_tokens": min(reasoning, len(logprobs)),
                "truncated": end < full,
            }
        )
    return scores


def serve(conn: Any) -> None:
    """Hold one `vllm.LLM`; answer requests until told to stop.

    A request is `(method, args, kwargs)`: `"chat"` runs `vllm.LLM.chat` and
    `"target_logprobs"` runs :func:`target_logprobs` on the engine.

    The first message is the engine kwargs; `CUDA_VISIBLE_DEVICES` was set by
    the parent before this process started.
    """
    try:
        engine_kwargs = conn.recv()
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
        method, args, kwargs = request
        if kwargs.pop("progress", False):
            kwargs["use_tqdm"], flush = _progress_sender(conn)
        else:
            flush = lambda: None  # noqa: E731
        try:
            if method == "chat":
                result: Any = [_plain(o) for o in llm.chat(*args, **kwargs)]
            else:
                result = target_logprobs(llm, *args, **kwargs)
            flush()
            conn.send(("ok", result))
        except BaseException:
            conn.send(("error", traceback.format_exc()))


def _progress_sender(conn: Any) -> tuple[Any, Any]:
    """A vLLM ``use_tqdm`` that sends counts to the parent at most twice a second.

    Returns it and a flush for the counts not yet sent.
    """
    from pyforger._progress import vllm_tqdm

    pending, last = [0], [time.monotonic()]

    def flush() -> None:
        if pending[0]:
            conn.send(("progress", pending[0]))
            pending[0] = 0
        last[0] = time.monotonic()

    def sink(n: int) -> None:
        pending[0] += n
        if time.monotonic() - last[0] >= 0.5:
            flush()

    return vllm_tqdm(sink), flush


class EngineProcess:
    """One worker process holding a vLLM engine on a group of GPUs.

    The worker is a fresh `python -m pyforger.acquire._local_dp_worker` process, not a
    multiprocessing child: "spawn" re-imports the caller's main script, which
    reruns any script without an `if __name__ == "__main__":` guard. Nor is it
    forked, so it never inherits a CUDA context from this process. It gets its
    own Triton and Inductor cache directories (`own_compile_caches`), so
    engines compiling at once do not share one /tmp.
    """

    def __init__(self, devices: list[str], engine_kwargs: dict[str, Any]) -> None:
        """Start the worker; `wait_ready` blocks until its engine has loaded."""
        self.devices = devices
        self._conn, child = multiprocessing.Pipe()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices))
        # The worker is already a process of its own, so its engine core runs in
        # it rather than in a forked child: under apptainer --fakeroot a forked
        # child's CUDA init fails ("CUDA driver initialization failed").
        env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        caches, self._scratch = own_compile_caches(os.environ)
        env.update(caches)
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "pyforger.acquire._local_dp_worker",
                    str(child.fileno()),
                ],
                pass_fds=[child.fileno()],
                env=env,
            )
        except BaseException:
            self._remove_scratch()
            raise
        finally:
            child.close()
        self._conn.send(engine_kwargs)

    def _receive(self, on_progress: Any = None) -> Any:
        """The worker's reply, after passing its progress counts to `on_progress`.

        A worker error or exit raises RuntimeError.
        """
        while True:
            try:
                status, payload = self._conn.recv()
            except EOFError:
                raise RuntimeError(
                    f"LocalModel engine on GPU(s) {','.join(self.devices)} exited "
                    f"(exit code {self._process.poll()})"
                ) from None
            if status != "progress":
                break
            if on_progress is not None:
                on_progress(payload)
        if status == "error":
            raise RuntimeError(
                f"LocalModel engine on GPU(s) {','.join(self.devices)} failed:\n"
                f"{payload}"
            )
        return payload

    def wait_ready(self) -> None:
        """Block until the engine has loaded its model."""
        self._receive()

    def chat(
        self,
        messages: list,
        sampling_params: Any,
        on_progress: Any = None,
        **kwargs: Any,
    ) -> list:
        """Run `vllm.LLM.chat` in the worker; `on_progress(n)` gets its counts."""
        kwargs["progress"] = on_progress is not None
        self._conn.send(("chat", (messages, sampling_params), kwargs))
        return self._receive(on_progress)

    def target_logprobs(
        self, pairs: list, on_progress: Any = None, **kwargs: Any
    ) -> list:
        """Run :func:`target_logprobs` in the worker; `on_progress(n)` gets counts."""
        kwargs["progress"] = on_progress is not None
        self._conn.send(("target_logprobs", (pairs,), kwargs))
        return self._receive(on_progress)

    def close(self) -> None:
        """Stop the worker and free its GPUs."""
        if self._process.poll() is None:
            try:
                self._conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._conn.close()
        self._remove_scratch()

    def _remove_scratch(self) -> None:
        """Delete the worker's compile caches (see `own_compile_caches`)."""
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None


class DataParallelEngine:
    """Several engines behind `vllm.LLM`'s `chat` call and `target_logprobs`.

    `engines` are objects with `chat(messages, sampling_params, **kwargs)` (and
    `target_logprobs(pairs, **kwargs)` for scoring);
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

    def chat(
        self, messages: list, sampling_params: Any, progress: Any = None, **kwargs: Any
    ) -> list:
        """Deal `messages` round-robin to the engines; outputs return in input order.

        `progress` (a `pyforger._progress.Progress`) counts every engine's work.
        """
        return self._deal(
            "chat", messages, sampling_params, progress=progress, **kwargs
        )

    def target_logprobs(self, pairs: list, progress: Any = None, **kwargs: Any) -> list:
        """Deal `(conversation, target)` pairs round-robin; scores return in order."""
        return self._deal("target_logprobs", pairs, progress=progress, **kwargs)

    def _deal(
        self, method: str, items: list, *args: Any, progress: Any = None, **kwargs: Any
    ) -> list:
        """Call `method` on each engine with its share of `items`, in input order."""
        shards = [
            list(range(i, len(items), len(self.engines)))
            for i in range(len(self.engines))
        ]
        jobs = []
        for engine, indices in zip(self.engines, shards):
            if not indices:
                continue
            # Every engine reports to `progress`; vLLM's own bar would show one.
            shard_kwargs = {**kwargs, "use_tqdm": False}
            if progress is not None:
                shard_kwargs["on_progress"] = progress.update
            shard = [items[j] for j in indices]
            jobs.append(
                (
                    indices,
                    self._pool.submit(
                        getattr(engine, method), shard, *args, **shard_kwargs
                    ),
                )
            )
        outputs: list[Any] = [None] * len(items)
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
