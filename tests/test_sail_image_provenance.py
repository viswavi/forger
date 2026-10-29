"""Real image/protobuf paths; only paid provider transport is replaced."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("sail")
from sail import Sailbox
from sail.errors import SailboxCreationError
from sail.image import ImageDefinition
from sail.pb.image.v1 import image_pb2 as pb
from sail.pb.imagebuilder.v1 import imagebuilder_pb2 as builder

from pyforger import _sail_image_provenance as provenance
from pyforger.sail_provision import create_sailbox
from pyforger.session_budget import SessionBudget

ID = "sha256:" + "a" * 64
SECRET = "credential-not-for-evidence"


def image():
    return ImageDefinition(
        pb.ImageSpec(
            oci=pb.OciImage(ref="registry/secret@" + ID), env={"API_KEY": SECRET}
        ),
        _image_id=ID,
    )


def failure():
    return SailboxCreationError(f"resolve image: image {ID} has not been built")


def test_real_built_image_and_create_serialization_preserve_identity():
    events = []
    observed = provenance.ObservedImage(image(), events)
    built = observed._ensure_built(timeout=3)
    request = Sailbox._create_request("app", "box", [], [], built, None, None, None)
    assert request["image"]["env"]["API_KEY"] == SECRET
    assert [e["stage"] for e in events] == [
        "ensure_started",
        "ensure_returned",
        "spec_serialized",
    ]
    assert len({e["canonical_spec_sha256"] for e in events}) == 1
    assert events[-1]["image_id"] == ID
    assert SECRET not in json.dumps(events)
    assert "registry" not in json.dumps(events)


def test_real_async_built_image_serialization():
    events = []

    async def run():
        result = await provenance.ObservedImage(image(), events)._ensure_built_aio(2)
        assert result.to_proto() == image().to_proto()

    asyncio.run(run())
    assert [e["stage"] for e in events] == [
        "ensure_started",
        "ensure_returned",
        "spec_serialized",
    ]


def test_failed_create_preserves_original_and_no_retry(monkeypatch):
    error = failure()
    calls = []
    monkeypatch.delenv("SAIL_API_KEY", raising=False)

    def create(**kwargs):
        calls.append(kwargs)
        built = kwargs["image"]._ensure_built(timeout=2)
        built.to_proto()
        raise error

    with pytest.raises(SailboxCreationError) as caught:
        create_sailbox(create, budget=SessionBudget(1, 10), image=image())
    assert caught.value is error
    assert len(calls) == 1
    assert error.sail_image_provenance["status_probe"] == {"skipped": True}
    assert SECRET not in str(error.__notes__)


def test_probe_status_transport_allowlist_and_deadline(monkeypatch):
    import grpc

    calls = []

    class Channel:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def unary_unary(self, rpc, **kwargs):
            assert rpc == provenance._RPC

            def call(request, **options):
                assert (
                    builder.GetImageBuildStatusRequest.FromString(
                        kwargs["request_serializer"](request)
                    ).image_id
                    == ID
                )
                calls.append(options)
                return builder.GetImageBuildStatusResponse(
                    image_id=ID,
                    status=pb.IMAGE_BUILD_STATUS_READY,
                    filesystem_built=True,
                    error_message=SECRET,
                    resolved_oci_ref="private-credential@" + ID,
                )

            return call

    monkeypatch.setenv("SAIL_API_KEY", SECRET)
    monkeypatch.setattr(grpc, "secure_channel", lambda endpoint, credentials: Channel())
    evidence = {}
    provenance.probe(failure(), evidence, time.monotonic() + 1)
    assert 0 < calls[0]["timeout"] <= 1
    assert calls[0]["metadata"] == (("authorization", "Bearer " + SECRET),)
    assert evidence["status_probe"]["status"] == "IMAGE_BUILD_STATUS_READY"
    assert SECRET not in json.dumps(evidence)
    assert "private-credential" not in json.dumps(evidence)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("image has not been built " + ID),
        SailboxCreationError("other " + ID),
    ],
)
def test_nonmatching_error_does_not_probe(error, monkeypatch):
    monkeypatch.setenv("SAIL_API_KEY", SECRET)
    assert provenance._probe_options(error, time.monotonic() + 20) is None


def test_expired_deadline_skips_and_probe_failure_is_redacted(monkeypatch):
    import grpc

    monkeypatch.setenv("SAIL_API_KEY", SECRET)
    assert provenance._probe_options(failure(), time.monotonic() - 1) is None
    assert provenance._probe_options(failure(), None) is None
    assert provenance._probe_options(failure(), time.monotonic() + 100)[1] <= 12

    def unavailable(*args):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(grpc, "secure_channel", unavailable)
    evidence = {}
    provenance.probe(failure(), evidence, time.monotonic() + 3)
    assert evidence == {"status_probe": {"error_type": "RuntimeError"}}


def test_real_build_return_pin_diff_visible_at_create(monkeypatch):
    from sail._core_client import CoreClient

    calls = []

    class Transport:
        def is_builtin_base_spec(self, spec):
            return False

        def build_spec_to_ready(self, spec, **options):
            calls.append(spec)
            return ID, "registry/python@" + ID, None, None

    monkeypatch.setattr(CoreClient, "get", lambda: Transport())
    original = ImageDefinition(pb.ImageSpec(oci=pb.OciImage(ref="registry/python:tag")))
    events = []
    built = provenance.ObservedImage(original, events)._ensure_built(timeout=3)
    request = Sailbox._create_request("app", "box", [], [], built, None, None, None)
    assert request["image"]["oci"]["ref"].endswith(ID)
    assert events[0]["canonical_spec_sha256"] != events[-1]["canonical_spec_sha256"]
    assert events[-2]["canonical_spec_sha256"] == events[-1]["canonical_spec_sha256"]
    assert len(calls) == 1


def test_async_status_transport_failure_preserves_native_exception(
    monkeypatch, tmp_path
):
    import grpc
    from harbor.environments.sail import SailEnvironment

    from pyforger._harbor_deadline import IMAGE_DEADLINE
    from pyforger._harbor_sail import ObservedSailEnvironment

    error = failure()

    async def remote(self, image):
        built = await image._ensure_built_aio(2)
        built.to_proto()
        raise error

    monkeypatch.setattr(SailEnvironment, "_create_sailbox", remote)
    monkeypatch.setenv("SAIL_API_KEY", SECRET)

    def unavailable(*args):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(grpc.aio, "secure_channel", unavailable)
    env = object.__new__(ObservedSailEnvironment)
    env.trial_paths = SimpleNamespace(trial_dir=tmp_path)

    async def run():
        token = IMAGE_DEADLINE.set(time.monotonic() + 2)
        try:
            with pytest.raises(SailboxCreationError) as caught:
                await env._create_sailbox(image())
            assert caught.value is error
        finally:
            IMAGE_DEADLINE.reset(token)

    asyncio.run(run())
    evidence = json.loads(next(tmp_path.glob("image-provenance-*.json")).read_text())
    assert evidence["status_probe"] == {"error_type": "RuntimeError"}
    assert len(evidence["events"]) == 3
    assert SECRET not in json.dumps(evidence)


def test_real_grpc_status_only_roundtrip(monkeypatch):
    """Local gRPC server is the provider boundary; no external network is used."""
    from concurrent.futures import ThreadPoolExecutor

    import grpc

    requests = []

    def status(request, context):
        requests.append((request.image_id, dict(context.invocation_metadata())))
        return builder.GetImageBuildStatusResponse(
            image_id=ID, status=pb.IMAGE_BUILD_STATUS_READY, filesystem_built=True
        )

    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "imagebuilder.v1.ImageBuilderService",
                {
                    "GetImageBuildStatus": grpc.unary_unary_rpc_method_handler(
                        status,
                        request_deserializer=builder.GetImageBuildStatusRequest.FromString,
                        response_serializer=builder.GetImageBuildStatusResponse.SerializeToString,
                    )
                },
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    monkeypatch.setenv("SAIL_API_KEY", SECRET)
    monkeypatch.setattr(
        grpc,
        "secure_channel",
        lambda endpoint, credentials: grpc.insecure_channel(f"127.0.0.1:{port}"),
    )
    try:
        evidence = {}
        provenance.probe(failure(), evidence, time.monotonic() + 2)
        assert evidence["status_probe"]["filesystem_built"] is True
        assert requests == [
            (
                ID,
                {
                    "user-agent": requests[0][1]["user-agent"],
                    "authorization": "Bearer " + SECRET,
                },
            )
        ]
    finally:
        server.stop(0).wait()


def test_async_status_rpc_uses_bounded_timeout_and_preserves_cancel(monkeypatch):
    import grpc

    calls = []

    class Channel:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def unary_unary(self, rpc, **serializers):
            async def call(request, **options):
                calls.append(options)
                return builder.GetImageBuildStatusResponse(
                    image_id=ID, status=pb.IMAGE_BUILD_STATUS_READY
                )

            return call

    monkeypatch.setenv("SAIL_API_KEY", SECRET)
    monkeypatch.setattr(grpc.aio, "secure_channel", lambda *args: Channel())
    evidence = {}
    asyncio.run(provenance.probe_async(failure(), evidence, time.monotonic() + 1))
    assert 0 < calls[0]["timeout"] <= 1
    assert evidence["status_probe"]["image_id"] == ID


def test_status_probe_missing_optional_grpc_preserves_error(monkeypatch):
    import sys

    monkeypatch.setenv("SAIL_API_KEY", SECRET)
    monkeypatch.setitem(sys.modules, "grpc", None)
    evidence = {}
    provenance.probe(failure(), evidence, time.monotonic() + 1)
    assert evidence == {"status_probe": {"error_type": "ModuleNotFoundError"}}


def test_real_async_build_pin_transition(monkeypatch):
    from sail._core_client import CoreClient

    class Transport:
        def is_builtin_base_spec(self, spec):
            return False

        async def build_spec_to_ready_aio(self, spec, **options):
            return ID, "registry/python@" + ID, None, None

    monkeypatch.setattr(CoreClient, "get", lambda: Transport())
    events = []
    original = ImageDefinition(pb.ImageSpec(oci=pb.OciImage(ref="registry/python:tag")))

    async def run():
        result = await provenance.ObservedImage(original, events)._ensure_built_aio(3)
        result.to_proto()

    asyncio.run(run())
    assert events[0]["canonical_spec_sha256"] != events[-1]["canonical_spec_sha256"]
    assert events[-2]["stage"] == "ensure_returned"
    assert events[-2]["canonical_spec_sha256"] == events[-1]["canonical_spec_sha256"]


def test_native_factory_retains_observer_for_separate_verifier(tmp_path):
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import EnvironmentConfig
    from harbor.utils.import_path import import_symbol

    from pyforger._harbor_sail import ObservedSailEnvironment

    config = EnvironmentConfig(
        type=EnvironmentType.SAIL,
        import_path="pyforger._harbor_sail:ObservedSailEnvironment",
    )
    verifier_config = config.model_copy(update={"extra_docker_compose": []})
    assert import_symbol(verifier_config.import_path) is ObservedSailEnvironment


def test_async_probe_cancellation_propagates_without_waiter(monkeypatch):
    import grpc

    closed = []

    class Channel:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)

        def unary_unary(self, *args, **kwargs):
            async def wait(*args, **kwargs):
                await asyncio.sleep(10)

            return wait

    monkeypatch.setenv("SAIL_API_KEY", SECRET)
    monkeypatch.setattr(grpc.aio, "secure_channel", lambda *args: Channel())

    async def run():
        task = asyncio.create_task(
            provenance.probe_async(failure(), {}, time.monotonic() + 10)
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert closed == [True]


def test_local_status_rpc_deadline_is_enforced(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import grpc

    def blocked(request, context):
        while context.is_active():
            time.sleep(0.005)
        return builder.GetImageBuildStatusResponse()

    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "imagebuilder.v1.ImageBuilderService",
                {
                    "GetImageBuildStatus": grpc.unary_unary_rpc_method_handler(
                        blocked,
                        request_deserializer=builder.GetImageBuildStatusRequest.FromString,
                        response_serializer=builder.GetImageBuildStatusResponse.SerializeToString,
                    )
                },
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    monkeypatch.setenv("SAIL_API_KEY", SECRET)
    monkeypatch.setattr(
        grpc, "secure_channel", lambda *args: grpc.insecure_channel(f"127.0.0.1:{port}")
    )
    try:
        evidence = {}
        start = time.monotonic()
        provenance.probe(failure(), evidence, start + 0.1)
        assert time.monotonic() - start < 1
        assert evidence["status_probe"]["error_type"] == "_InactiveRpcError"
    finally:
        server.stop(0).wait()


@pytest.mark.parametrize(
    "text",
    [
        f"prefix resolve image: image {ID} has not been built",
        f"resolve image: image {ID} has not been built extra",
        f"resolve image: image {ID} has not been built {ID}",
        f"resolve image {ID}: image has not been built",
    ],
)
def test_only_exact_production_image_not_built_is_probed(text):
    assert provenance.failure_image(SailboxCreationError(text)) is None


def test_diagnostic_snapshot_and_package_metadata_cannot_abort_create(monkeypatch):
    monkeypatch.setattr(
        provenance.importlib.metadata,
        "version",
        lambda *a: (_ for _ in ()).throw(RuntimeError(SECRET)),
    )
    monkeypatch.setattr(
        provenance, "_snapshot", lambda *a: (_ for _ in ()).throw(ValueError(SECRET))
    )
    box = object()

    def create(**kwargs):
        built = kwargs["image"]._ensure_built(timeout=2)
        assert built.to_proto() == image().to_proto()
        return box

    assert create_sailbox(create, budget=SessionBudget(1, 3), image=image()) is box


def test_diagnostic_snapshot_failure_preserves_create_exception(monkeypatch):
    monkeypatch.delenv("SAIL_API_KEY", raising=False)
    monkeypatch.setattr(
        provenance, "_snapshot", lambda *a: (_ for _ in ()).throw(ValueError(SECRET))
    )
    error = failure()

    def create(**kwargs):
        kwargs["image"]._ensure_built(timeout=2).to_proto()
        raise error

    with pytest.raises(SailboxCreationError) as caught:
        create_sailbox(create, budget=SessionBudget(1, 3), image=image())
    assert caught.value is error
    assert SECRET not in str(error.__notes__)


@pytest.mark.parametrize("broken", ["journal", "ObservedImage"])
def test_observer_setup_failure_uses_unchanged_native_image(monkeypatch, broken):
    original = image()

    def fail(*args):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(provenance, broken, fail)
    calls = []

    def create(**kwargs):
        calls.append(kwargs["image"])
        return "native-success"

    assert (
        create_sailbox(create, budget=SessionBudget(1, 3), image=original)
        == "native-success"
    )
    assert calls == [original]
    error = failure()

    def failed_create(**kwargs):
        raise error

    with pytest.raises(SailboxCreationError) as caught:
        create_sailbox(failed_create, budget=SessionBudget(1, 3), image=original)
    assert caught.value is error


def test_harbor_snapshot_failure_keeps_original_native_error(monkeypatch, tmp_path):
    from harbor.environments.sail import SailEnvironment

    from pyforger._harbor_sail import ObservedSailEnvironment

    monkeypatch.delenv("SAIL_API_KEY", raising=False)
    monkeypatch.setattr(
        provenance, "_snapshot", lambda *a: (_ for _ in ()).throw(ValueError(SECRET))
    )
    error = failure()

    async def remote(self, image):
        (await image._ensure_built_aio(1)).to_proto()
        raise error

    monkeypatch.setattr(SailEnvironment, "_create_sailbox", remote)
    env = object.__new__(ObservedSailEnvironment)
    env.trial_paths = SimpleNamespace(trial_dir=tmp_path)

    async def run():
        with pytest.raises(SailboxCreationError) as caught:
            await env._create_sailbox(image())
        assert caught.value is error

    asyncio.run(run())
    assert SECRET not in next(tmp_path.glob("image-provenance-*.json")).read_text()
