from __future__ import annotations

from typing import Any

from autoscene.actions.advanced.debug_artifacts import DebugArtifactWriter
from autoscene.actions.advanced.protocols import (
    BaseActionRuntimeProtocol,
    VisionRuntimeProtocol,
)
from autoscene.actions.advanced.retry import RetryPolicy
from autoscene.actions.advanced.shared import capture_active_frame
from autoscene.capture.window_capture import CaptureResult
from autoscene.core.models import BoundingBox, ObjectLocateSpec
from autoscene.vision.pipeline import (
    filter_object_locate_nodes,
    run_object_locate_pipeline,
)


class ObjectActions:
    def __init__(
        self,
        *,
        base_actions: BaseActionRuntimeProtocol,
        vision_runtime: VisionRuntimeProtocol,
        retry_policy: RetryPolicy,
        debug_artifact_writer: DebugArtifactWriter,
    ) -> None:
        self.base_actions = base_actions
        self.vision_runtime = vision_runtime
        self.retry_policy = retry_policy
        self.debug_artifact_writer = debug_artifact_writer

    def click_object(
        self,
        locate: ObjectLocateSpec,
        debug_path: str | None = None,
    ) -> None:
        capture_result, target, region = self._locate_required_object(locate)
        self.debug_artifact_writer.save_object_debug(
            image=capture_result.image,
            region=region,
            target_box=target,
            label=locate.label,
            debug_path=debug_path,
            capture_result=capture_result,
        )
        x, y = self.base_actions.capture_to_screen(
            *target.center,
            capture_result=capture_result,
        )
        self.base_actions.click(x, y)

    def drag_object_to_position(
        self,
        locate: ObjectLocateSpec,
        target_x: int,
        target_y: int,
        duration_ms: int = 500,
        debug_path: str | None = None,
    ) -> None:
        capture_result, source, region = self._locate_required_object(locate)
        self.debug_artifact_writer.save_object_debug(
            image=capture_result.image,
            region=region,
            target_box=source,
            label=locate.label,
            debug_path=debug_path,
            capture_result=capture_result,
        )
        start_x, start_y = self.base_actions.capture_to_screen(
            *source.center,
            capture_result=capture_result,
        )
        end_x, end_y = self.base_actions.capture_to_screen(
            int(target_x),
            int(target_y),
            capture_result=capture_result,
        )
        self.base_actions.logger.info(
            "selected object %s bbox=(%s, %s, %s, %s) center=(%s, %s) score=%.3f region=%s",
            locate.label,
            source.x1,
            source.y1,
            source.x2,
            source.y2,
            source.center[0],
            source.center[1],
            float(source.score),
            region,
        )
        self.base_actions.logger.info(
            "drag coordinate mapping capture=(%s, %s)->(%s, %s) screen=(%s, %s)->(%s, %s)",
            source.center[0],
            source.center[1],
            int(target_x),
            int(target_y),
            start_x,
            start_y,
            end_x,
            end_y,
        )
        self.base_actions.drag(start_x, start_y, end_x, end_y, duration_ms=duration_ms)

    def drag_object_to_object(
        self,
        source: ObjectLocateSpec,
        target: ObjectLocateSpec,
        duration_ms: int = 500,
    ) -> None:
        source_locate = self._require_object_locate(source, field_name="source")
        target_locate = self._require_object_locate(target, field_name="target")
        capture_result = capture_active_frame(self.base_actions)
        image = capture_result.image
        sources, targets = self._locate_drag_object_boxes(
            image=image,
            source=source_locate,
            target=target_locate,
        )
        if not sources:
            raise AssertionError(f"Source object not found: {source_locate.label}")
        if not targets:
            raise AssertionError(f"Target object not found: {target_locate.label}")
        source_box = self.pick_box(sources, pick=source_locate.pick)
        target_box = self.pick_box(targets, pick=target_locate.pick)
        sx, sy = self.base_actions.capture_to_screen(
            *source_box.center,
            capture_result=capture_result,
        )
        tx, ty = self.base_actions.capture_to_screen(
            *target_box.center,
            capture_result=capture_result,
        )
        self.base_actions.drag(sx, sy, tx, ty, duration_ms=duration_ms)

    def verify_object_exists(self, locate: ObjectLocateSpec) -> bool:
        _, boxes, _ = self.locate_object_boxes(locate=locate)
        return bool(boxes)

    def detect_object_boxes(
        self,
        image: object,
        locate: ObjectLocateSpec,
    ) -> tuple[list[BoundingBox], BoundingBox | None]:
        detector = self.vision_runtime.resolve_detector(locate.detector)
        result = run_object_locate_pipeline(
            image,
            detector=detector,
            locate=locate,
        )
        return result.boxes, result.region

    def filter_detected_object_boxes(
        self,
        image: object,
        locate: ObjectLocateSpec,
        *,
        boxes: list[BoundingBox],
        nodes: list[Any] | None = None,
    ) -> tuple[list[BoundingBox], BoundingBox | None]:
        result = filter_object_locate_nodes(
            image,
            boxes=boxes,
            nodes=nodes,
            locate=locate,
        )
        return result.boxes, result.region

    def locate_object_boxes(
        self,
        locate: ObjectLocateSpec,
        attempts: int = 3,
        retry_interval_seconds: float = 0.3,
    ) -> tuple[CaptureResult, list[BoundingBox], BoundingBox | None]:
        capture_result, boxes, region = self.retry_policy.run_with_retry(
            lambda: self._locate_object_once(locate),
            attempts=attempts,
            retry_interval_seconds=retry_interval_seconds,
            should_retry=lambda result: not bool(result[1]),
        )
        return capture_result, boxes, region

    def _locate_object_once(
        self,
        locate: ObjectLocateSpec,
    ) -> tuple[CaptureResult, list[BoundingBox], BoundingBox | None]:
        capture_result = capture_active_frame(self.base_actions)
        boxes, region = self.detect_object_boxes(
            image=capture_result.image,
            locate=locate,
        )
        return capture_result, boxes, region

    def _locate_required_object(
        self,
        locate: ObjectLocateSpec,
    ) -> tuple[CaptureResult, BoundingBox, BoundingBox | None]:
        capture_result, boxes, region = self.locate_object_boxes(locate=locate)
        if not boxes:
            raise AssertionError(f"Object not found by detector: {locate.label}")
        return capture_result, self.pick_box(boxes, pick=locate.pick), region

    @staticmethod
    def clip_search_region(region: BoundingBox | None, image: object) -> BoundingBox | None:
        if region is None:
            return None
        image_size = getattr(image, "size", None)
        if image_size is None:
            raise AssertionError("Locate region requires an image capture with size.")
        width, height = image_size
        x1 = max(0, min(int(region.x1), width))
        y1 = max(0, min(int(region.y1), height))
        x2 = max(0, min(int(region.x2), width))
        y2 = max(0, min(int(region.y2), height))
        if x2 <= x1 or y2 <= y1:
            raise AssertionError("Invalid locate.region for detector click.")
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def box_center_in_region(box: BoundingBox, region: BoundingBox) -> bool:
        center_x, center_y = box.center
        return region.x1 <= center_x <= region.x2 and region.y1 <= center_y <= region.y2

    @staticmethod
    def pick_box(boxes: list[BoundingBox], pick: str) -> BoundingBox:
        mode = pick.lower()
        if mode == "topmost":
            return min(boxes, key=lambda box: (box.y1, -box.score, box.x1))
        if mode == "bottommost":
            return max(boxes, key=lambda box: (box.y2, box.score, -box.x1))
        if mode == "leftmost":
            return min(boxes, key=lambda box: (box.x1, -box.score, box.y1))
        if mode == "rightmost":
            return max(boxes, key=lambda box: (box.x2, box.score, -box.y1))
        return max(boxes, key=lambda box: box.score)

    def _locate_drag_object_boxes(
        self,
        *,
        image: object,
        source: ObjectLocateSpec,
        target: ObjectLocateSpec,
    ) -> tuple[list[BoundingBox], list[BoundingBox]]:
        shared_detector = self._shared_detector_name(source, target)
        if shared_detector is None:
            source_boxes, _ = self.detect_object_boxes(image=image, locate=source)
            target_boxes, _ = self.detect_object_boxes(image=image, locate=target)
            return source_boxes, target_boxes

        detector = self.vision_runtime.resolve_detector(
            None if shared_detector == "" else shared_detector
        )
        labels = [source.label]
        if target.label != source.label:
            labels.append(target.label)
        boxes = list(detector.detect(image, labels=labels))
        pipeline_result = getattr(detector, "last_pipeline_result", None)
        detector_nodes = None
        if pipeline_result is not None:
            candidate_nodes = list(getattr(pipeline_result, "nodes", []) or [])
            if len(candidate_nodes) == len(boxes):
                detector_nodes = candidate_nodes

        source_boxes, _ = self.filter_detected_object_boxes(
            image=image,
            locate=source,
            boxes=boxes,
            nodes=detector_nodes,
        )
        target_boxes, _ = self.filter_detected_object_boxes(
            image=image,
            locate=target,
            boxes=boxes,
            nodes=detector_nodes,
        )
        return source_boxes, target_boxes

    @staticmethod
    def _shared_detector_name(
        source: ObjectLocateSpec,
        target: ObjectLocateSpec,
    ) -> str | None:
        if source.detector == target.detector:
            return "" if source.detector is None else source.detector
        return None

    @staticmethod
    def _require_object_locate(
        locate: ObjectLocateSpec,
        *,
        field_name: str,
    ) -> ObjectLocateSpec:
        if not isinstance(locate, ObjectLocateSpec):
            raise TypeError(
                f"{field_name} requires ObjectLocateSpec, got {type(locate).__name__}."
            )
        return locate
