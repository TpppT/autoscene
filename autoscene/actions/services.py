from __future__ import annotations

from autoscene.actions.advanced import (
    DebugArtifactWriter,
    LocateActions,
    ObjectActions,
    RetryPolicy,
    TextActions,
)
from autoscene.actions.base import BaseActions
from autoscene.actions.vision_runtime import ActionVisionRuntime


class ActionServices:
    """Compose action-layer collaborators without acting as a forwarding facade.

    Text/object locate evaluation stays in vision-side pipeline helpers; this
    object only wires the runtime-facing collaborators together.
    """

    def __init__(self, capture, detector, ocr, detectors=None) -> None:
        base_actions = BaseActions(capture=capture)
        vision_runtime = ActionVisionRuntime(
            detector=detector,
            ocr=ocr,
            detectors=detectors,
        )
        retry_policy = RetryPolicy()
        debug_artifact_writer = DebugArtifactWriter(base_actions)

        self.base_actions = base_actions
        self.screenshot_actions = base_actions
        self.vision_runtime = vision_runtime
        self.retry_policy = retry_policy
        self.debug_artifact_writer = debug_artifact_writer
        self.text_actions = TextActions(
            base_actions=base_actions,
            vision_runtime=vision_runtime,
            retry_policy=retry_policy,
            debug_artifact_writer=debug_artifact_writer,
        )
        self.object_actions = ObjectActions(
            base_actions=base_actions,
            vision_runtime=vision_runtime,
            retry_policy=retry_policy,
            debug_artifact_writer=debug_artifact_writer,
        )
        self.locate_actions = LocateActions(
            text_actions=self.text_actions,
            object_actions=self.object_actions,
        )
