"""Allowlisted, per-create Sail image evidence; never builds or retries images."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import time
from typing import Any

from google.protobuf.json_format import MessageToDict
from sail.image import ImageDefinition

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ENDPOINT = "sailbox-imagebuilder-dispatcher.sailresearch.com:443"
_RPC = "/imagebuilder.v1.ImageBuilderService/GetImageBuildStatus"


def _digest(value: str) -> str | None:
    match = _DIGEST.fullmatch(value)
    return match.group() if match else None


def _snapshot(image: ImageDefinition) -> dict[str, Any]:
    spec = ImageDefinition.to_proto(image)
    canonical = json.dumps(MessageToDict(spec), sort_keys=True, separators=(",", ":"))
    pins = []
    if spec.HasField("oci"):
        pins.extend(_DIGEST.findall(spec.oci.ref))
    if spec.HasField("dockerfile"):
        pins.extend(_DIGEST.findall(str(spec.dockerfile.pinned_from)))
    return {
        "image_id": _digest(image._image_id or ""),
        "canonical_spec_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "resolved_digests": sorted(set(pins))[:32],
    }


class ObservedImage(ImageDefinition):  # type: ignore[misc]
    def __init__(self, image: ImageDefinition, events: list[dict[str, Any]]):
        super().__init__(
            image._spec, image._image_id, image._source_ref, image._context_dir
        )
        object.__setattr__(self, "events", events)

    def record(self, stage: str) -> None:
        try:
            if len(self.events) >= 16:
                return
            self.events.append(
                {
                    "stage": stage,
                    "wall_time": time.time(),
                    "monotonic": time.monotonic(),
                    **_snapshot(self),
                }
            )
        except Exception:  # noqa: BLE001 — observation cannot change provisioning
            return

    def to_proto(self) -> Any:
        self.record("spec_serialized")
        return super().to_proto()

    def _ensure_built(self, *, timeout: int = 3600) -> ImageDefinition:
        self.record("ensure_started")
        result = ObservedImage(super()._ensure_built(timeout=timeout), self.events)
        result.record("ensure_returned")
        return result

    async def _ensure_built_aio(self, timeout: float) -> ImageDefinition:
        self.record("ensure_started")
        result = ObservedImage(await super()._ensure_built_aio(timeout), self.events)
        result.record("ensure_returned")
        return result


def journal() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("sail")
    except Exception:  # noqa: BLE001 — packaging metadata is not a create prerequisite
        version = "unavailable"
    return {"sail_version": version, "events": []}


def observe(image: Any) -> tuple[Any, dict[str, Any] | None]:
    try:
        if not isinstance(image, ImageDefinition):
            return image, None
        evidence = journal()
        return ObservedImage(image, evidence["events"]), evidence
    except Exception:  # noqa: BLE001 — keep the original image if observation fails
        return image, None


def failure_image(error: Exception) -> str | None:
    from sail.errors import SailboxCreationError

    if not isinstance(error, SailboxCreationError):
        return None
    match = re.fullmatch(
        r"resolve image: image (sha256:[0-9a-f]{64}) has not been built", str(error)
    )
    return match.group(1) if match else None


def _probe_options(
    error: Exception, deadline: float | None
) -> tuple[str, float] | None:
    identity = failure_image(error)
    left = min(12.0, deadline - time.monotonic()) if deadline is not None else 0
    if not identity or left <= 0 or not os.environ.get("SAIL_API_KEY"):
        return None
    return identity, left


def _status(response: Any) -> dict[str, Any]:
    from sail.pb.image.v1 import image_pb2 as pb

    return {
        "image_id": _digest(response.image_id),
        "status": pb.ImageBuildStatus.Name(response.status),
        "filesystem_built": response.filesystem_built,
        "resolved_digests": sorted(
            set(
                _DIGEST.findall(
                    response.resolved_oci_ref + str(response.dockerfile_pins)
                )
            )
        )[:32],
    }


def probe(error: Exception, evidence: dict[str, Any], deadline: float | None) -> None:
    options = _probe_options(error, deadline)
    if options is None:
        evidence["status_probe"] = {"skipped": True}
        return
    try:
        import grpc
        from sail.pb.imagebuilder.v1 import imagebuilder_pb2 as pb

        identity, timeout = options
        started = time.time()
        with grpc.secure_channel(_ENDPOINT, grpc.ssl_channel_credentials()) as channel:
            call = channel.unary_unary(
                _RPC,
                request_serializer=pb.GetImageBuildStatusRequest.SerializeToString,
                response_deserializer=pb.GetImageBuildStatusResponse.FromString,
            )
            response = call(
                pb.GetImageBuildStatusRequest(image_id=identity),
                timeout=timeout,
                metadata=(("authorization", "Bearer " + os.environ["SAIL_API_KEY"]),),
            )
        evidence["status_probe"] = {
            "requested_image": identity,
            "started": started,
            "ended": time.time(),
            **_status(response),
        }
    except Exception as diagnostic_error:  # noqa: BLE001 — preserve the create failure
        evidence["status_probe"] = {"error_type": type(diagnostic_error).__name__}


async def probe_async(
    error: Exception, evidence: dict[str, Any], deadline: float | None
) -> None:
    options = _probe_options(error, deadline)
    if options is None:
        evidence["status_probe"] = {"skipped": True}
        return
    try:
        import grpc
        from sail.pb.imagebuilder.v1 import imagebuilder_pb2 as pb

        identity, timeout = options
        started = time.time()
        async with grpc.aio.secure_channel(
            _ENDPOINT, grpc.ssl_channel_credentials()
        ) as channel:
            call = channel.unary_unary(
                _RPC,
                request_serializer=pb.GetImageBuildStatusRequest.SerializeToString,
                response_deserializer=pb.GetImageBuildStatusResponse.FromString,
            )
            response = await call(
                pb.GetImageBuildStatusRequest(image_id=identity),
                timeout=timeout,
                metadata=(("authorization", "Bearer " + os.environ["SAIL_API_KEY"]),),
            )
        evidence["status_probe"] = {
            "requested_image": identity,
            "started": started,
            "ended": time.time(),
            **_status(response),
        }
    except Exception as diagnostic_error:  # noqa: BLE001 — preserve the create failure
        evidence["status_probe"] = {"error_type": type(diagnostic_error).__name__}
