"""Real native Trial lifecycle with only remote phases replaced by async gates."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import pytest

from pyforger.artifacts import ArtifactRef
from pyforger.harbor import HarborRequest, HarborTaskRunner


@pytest.fixture
def native_task(tmp_path):
    root = tmp_path / "task"
    (root / "environment").mkdir(parents=True)
    (root / "environment/Dockerfile").write_text("FROM python:3.12-slim\n")
    (root / "instruction.md").write_text("No learner execution in this fixture.")
    (root / "tests").mkdir()
    (root / "tests/test.sh").write_text("exit 0\n")
    (root / "task.toml").write_text(
        'version="1.0"\n[agent]\ntimeout_sec=60\n[verifier]\ntimeout_sec=60\n'
    )
    return ArtifactRef.capture(root)


@pytest.mark.parametrize("phase", ["setup", "verifier"])
def test_absolute_deadline_cancels_real_trial_and_settles_cleanup(
    native_task, tmp_path, monkeypatch, phase
):
    from harbor.trial.single_step import SingleStepTrial
    from harbor.trial.trial import Trial

    async def warmup():
        await Trial.create(
            HarborTaskRunner(
                agent={"name": "nop"}, environment={"type": "docker"}
            )._configuration(
                HarborRequest(trial_id="warmup", task=native_task),
                native_task.root,
                tmp_path / "warmup",
            )
        )

    asyncio.run(warmup())
    events = []

    async def prepare(self):
        events.append("setup")
        if phase == "setup":
            await asyncio.sleep(10)

    async def run(self):
        events.append("verifier")
        await asyncio.sleep(10)

    async def recover(self):
        events.append("recover")

    async def finalize(self):
        await asyncio.sleep(0.01)
        events.append("cleanup")

    # Native Trial.run(), cancellation dispatch, exception/result handling are real.
    # Only phases requiring an external sandbox are substituted to avoid provisioning.
    monkeypatch.setattr(SingleStepTrial, "_prepare", prepare)
    monkeypatch.setattr(SingleStepTrial, "_run", run)
    monkeypatch.setattr(SingleStepTrial, "_recover_outputs", recover)
    monkeypatch.setattr(SingleStepTrial, "_finalize", finalize)
    runner = HarborTaskRunner(
        agent={"name": "nop"}, environment={"type": "docker"}, timeout_s=0.5
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        runner.run(HarborRequest(trial_id=phase, task=native_task), tmp_path / "native")
    assert time.monotonic() - started < 2
    assert events[-1] == "cleanup"
    if phase == "setup":
        assert "verifier" not in events


def test_retry_backoff_spends_same_absolute_budget(native_task, tmp_path, monkeypatch):
    from harbor.models.trial.result import ExceptionInfo
    from harbor.trial.single_step import SingleStepTrial

    starts = []

    async def prepare(self):
        starts.append(time.monotonic())

    async def run(self):
        self.result.exception_info = ExceptionInfo(
            exception_type="ApiError",
            exception_message="HTTP 503",
            exception_traceback="fixture",
            occurred_at=datetime.now(timezone.utc),
        )

    async def finalize(self):
        self.paths.result_path.write_text(self.result.model_dump_json())

    async def recover(self):
        pass

    monkeypatch.setattr(SingleStepTrial, "_recover_outputs", recover)
    monkeypatch.setattr(SingleStepTrial, "_prepare", prepare)
    monkeypatch.setattr(SingleStepTrial, "_run", run)
    monkeypatch.setattr(SingleStepTrial, "_finalize", finalize)
    runner = HarborTaskRunner(
        agent={"name": "nop"}, environment={"type": "docker"}, timeout_s=0.5
    )
    with pytest.raises(TimeoutError):
        runner.run(
            HarborRequest(trial_id="retry", task=native_task), tmp_path / "native"
        )
    assert len(starts) == 1


def test_async_caller_cancellation_waits_for_shielded_cleanup():
    from pyforger._harbor_deadline import run_owned

    async def scenario():
        cleaned = asyncio.Event()
        started = asyncio.Event()

        async def cleanup():
            await asyncio.sleep(0.03)
            cleaned.set()

        async def operation():
            started.set()
            try:
                await asyncio.sleep(10)
            finally:
                await asyncio.shield(cleanup())

        task = asyncio.create_task(run_owned(operation, time.monotonic() + 10))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(scenario())


def test_concurrent_deadlines_preserve_unrelated_tasks_and_factory():
    from pyforger._harbor_deadline import run_owned

    async def scenario():
        loop = asyncio.get_running_loop()

        def factory(loop, coro, **kwargs):
            return asyncio.Task(coro, loop=loop, **kwargs)

        loop.set_task_factory(factory)
        finished = []

        async def operation(name, duration):
            await asyncio.sleep(duration)
            finished.append(name)
            return name

        async def short():
            with pytest.raises(TimeoutError):
                await run_owned(lambda: operation("short", 10), time.monotonic() + 0.02)

        unrelated = asyncio.create_task(operation("unrelated", 0.06))
        long = asyncio.create_task(
            run_owned(lambda: operation("long", 0.08), time.monotonic() + 1)
        )
        await short()
        assert await long == "long"
        await unrelated
        assert finished == ["unrelated", "long"]
        assert loop.get_task_factory() is factory
        loop.set_task_factory(None)

    asyncio.run(scenario())


def test_cleanup_grace_is_bounded_and_not_scientific_time(monkeypatch):
    import pyforger._harbor_deadline as module

    monkeypatch.setattr(module, "CLEANUP_GRACE_S", 0.03)

    async def scenario():
        canceled = asyncio.Event()

        async def operation():
            try:
                await asyncio.sleep(10)
            finally:
                try:
                    await asyncio.sleep(10)
                finally:
                    canceled.set()

        started = time.monotonic()
        with pytest.raises(RuntimeError, match="cleanup exceeded grace"):
            await module.run_owned(operation, time.monotonic() + 0.02)
        assert time.monotonic() - started < 0.3
        assert canceled.is_set()
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(scenario())


def test_blocked_native_create_is_owned_and_canceled(
    native_task, tmp_path, monkeypatch
):
    from harbor.trial.trial import Trial

    canceled = []

    async def create(cls, config):
        try:
            await asyncio.sleep(10)
        finally:
            canceled.append(True)

    monkeypatch.setattr(Trial, "create", classmethod(create))

    async def scenario():
        runner = HarborTaskRunner(
            agent={"name": "nop"}, environment={"type": "docker"}, timeout_s=0.05
        )
        with pytest.raises(TimeoutError):
            await runner.arun(
                HarborRequest(trial_id="create", task=native_task), tmp_path / "native"
            )
        assert canceled == [True]
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(scenario())


def test_phase_caps_decrease_after_setup(native_task, tmp_path, monkeypatch):
    from harbor.trial.single_step import SingleStepTrial
    from harbor.trial.hooks import TrialEvent

    caps = []

    async def prepare(self):
        await asyncio.sleep(0.04)

    async def run(self):
        await self._emit(TrialEvent.VERIFICATION_START)
        caps.append(self._verifier_timeout_sec)
        self.result.verifier_result = None

    async def finalize(self):
        self.paths.result_path.write_text(self.result.model_dump_json())

    monkeypatch.setattr(SingleStepTrial, "_prepare", prepare)
    monkeypatch.setattr(SingleStepTrial, "_run", run)
    monkeypatch.setattr(SingleStepTrial, "_finalize", finalize)
    runner = HarborTaskRunner(
        agent={"name": "nop"}, environment={"type": "docker"}, timeout_s=1
    )
    result = runner.run(
        HarborRequest(trial_id="caps", task=native_task), tmp_path / "native"
    )
    assert result.status == "completed"
    assert 0 < caps[0] < 0.97


def test_expired_backend_budget_never_provisions(native_task, tmp_path):
    from pyforger.harbor_backend import HarborBackend
    from pyforger.agent_backend import AgentSpec, WorkspaceTask

    backend = HarborBackend()
    with pytest.raises(TimeoutError):
        backend.run_agent(
            WorkspaceTask("no time", timeout_s=1e-9), AgentSpec(kind="nop")
        )


def test_native_provider_build_timeout_matches_remaining_deadline(
    native_task, tmp_path, monkeypatch
):
    from harbor.trial.single_step import SingleStepTrial
    from harbor.trial.hooks import TrialEvent

    caps = []

    async def prepare(self):
        await self._emit(TrialEvent.ENVIRONMENT_START)
        caps.append(
            (
                self._environment_build_timeout_sec,
                self.agent_environment.task_env_config.build_timeout_sec,
            )
        )

    async def run(self):
        pass

    async def finalize(self):
        self.paths.result_path.write_text(self.result.model_dump_json())

    monkeypatch.setattr(SingleStepTrial, "_prepare", prepare)
    monkeypatch.setattr(SingleStepTrial, "_run", run)
    monkeypatch.setattr(SingleStepTrial, "_finalize", finalize)
    result = HarborTaskRunner(
        agent={"name": "nop"}, environment={"type": "sail"}, timeout_s=2
    ).run(HarborRequest(trial_id="provider-cap", task=native_task), tmp_path / "native")
    assert result.status == "completed"
    assert 0 < caps[0][1] <= caps[0][0] <= 2


def test_native_stop_failure_is_not_successful_cleanup():
    from types import SimpleNamespace
    from harbor.trial.trial import Trial
    from pyforger._harbor_deadline import run_owned

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        async def stop(delete):
            started.set()
            await release.wait()
            raise RuntimeError("remote stop failed")

        trial = SimpleNamespace(
            _is_agent_environment_stopped=False,
            agent_environment=SimpleNamespace(stop=stop),
            config=SimpleNamespace(
                environment=SimpleNamespace(delete=True), trial_name="fixture"
            ),
            logger=SimpleNamespace(debug=lambda *a: None),
            config_name="fixture",
            trial_name="fixture",
        )

        async def operation():
            stop_task = asyncio.create_task(Trial._stop_agent_environment(trial))
            await started.wait()
            stop_task.cancel()
            await stop_task
            release.set()
            return "native finished"

        with pytest.raises(RuntimeError, match="cleanup") as error:
            await run_owned(operation, time.monotonic() + 1)
        assert error.value.diagnostics["cleanup"] == "failed"
        assert "remote stop failed" in str(error.value.diagnostics)

    asyncio.run(scenario())


def test_normal_native_stop_settlement_survives_repeated_caller_cancel():
    from types import SimpleNamespace
    from harbor.trial.trial import Trial
    from pyforger._harbor_deadline import run_owned

    async def scenario():
        started = asyncio.Event()
        root_done = asyncio.Event()
        finished = asyncio.Event()

        async def stop(delete):
            started.set()
            await asyncio.sleep(0.08)
            finished.set()

        trial = SimpleNamespace(
            _is_agent_environment_stopped=False,
            agent_environment=SimpleNamespace(stop=stop),
            config=SimpleNamespace(
                environment=SimpleNamespace(delete=True), trial_name="fixture"
            ),
            logger=SimpleNamespace(debug=lambda *a: None),
            config_name="fixture",
            trial_name="fixture",
        )

        async def operation():
            stop_task = asyncio.create_task(Trial._stop_agent_environment(trial))
            await started.wait()
            stop_task.cancel()
            await stop_task
            root_done.set()

        owner = asyncio.create_task(run_owned(operation, time.monotonic() + 1))
        await root_done.wait()
        await asyncio.sleep(0.01)
        owner.cancel()
        await asyncio.sleep(0.01)
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert finished.is_set()
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(scenario())


def test_grace_failure_retains_pending_source_stack_before_cancel(monkeypatch):
    import pyforger._harbor_deadline as module

    monkeypatch.setattr(module, "CLEANUP_GRACE_S", 0.02)

    async def scenario():
        async def late_stop():
            await asyncio.sleep(10)

        async def operation():
            asyncio.create_task(late_stop())

        with pytest.raises(module.HarborCleanupError) as caught:
            await module.run_owned(operation, time.monotonic() + 1)
        pending = caught.value.diagnostics["pending_tasks"]
        assert any("late_stop" in row["coroutine"] and row["stack"] for row in pending)
        assert caught.value.diagnostics["cleanup"] == "failed"

    asyncio.run(scenario())


def test_real_sail_reclaimer_settles_inside_owned_grace():
    from sail.sailbox import _reclaimed_op
    from pyforger._harbor_deadline import run_owned

    async def scenario():
        created = asyncio.Event()
        reclaimed = []

        async def create():
            await asyncio.sleep(0.05)
            created.set()
            return "real-operation-result"

        async def undo(value):
            reclaimed.append(value)

        async def operation():
            return await _reclaimed_op(asyncio.create_task(create()), undo)

        with pytest.raises(TimeoutError):
            await run_owned(operation, time.monotonic() + 0.01)
        assert created.is_set()
        assert reclaimed == ["real-operation-result"]
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(scenario())


def test_handled_auxiliary_failure_is_not_mislabeled_cleanup_failure():
    from pyforger._harbor_deadline import run_owned

    async def scenario():
        async def auxiliary():
            raise ValueError("native phase handled this error")

        async def operation():
            child = asyncio.create_task(auxiliary())
            try:
                await child
            except ValueError:
                pass
            return "handled"

        assert await run_owned(operation, time.monotonic() + 1) == "handled"

    asyncio.run(scenario())
