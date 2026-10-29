"""Native Sail environment with host-only image failure evidence."""

import json
import uuid
from typing import Any

from harbor.environments.sail import SailEnvironment

from pyforger._harbor_deadline import IMAGE_DEADLINE
from pyforger._sail_image_provenance import (
    failure_image,
    observe,
    probe_async,
)


class ObservedSailEnvironment(SailEnvironment):  # type: ignore[misc]
    async def _create_sailbox(self, image: Any) -> None:
        observed, evidence = observe(image)
        try:
            await super()._create_sailbox(observed)
        except Exception as error:
            if evidence is not None and failure_image(error):
                await probe_async(error, evidence, IMAGE_DEADLINE.get())
                try:
                    path = self.trial_paths.trial_dir / (
                        "image-provenance-" + uuid.uuid4().hex + ".json"
                    )
                    path.write_text(json.dumps(evidence))
                except OSError:
                    error.add_note("Sail image provenance could not be retained")
            raise
