from __future__ import annotations

import time

from autoscene.actions.advanced.debug_artifacts import DebugArtifactWriter
from autoscene.actions.advanced.protocols import (
    BaseActionRuntimeProtocol,
    VisionRuntimeProtocol,
)
from autoscene.actions.advanced.retry import RetryPolicy
from autoscene.actions.advanced.shared import capture_active_frame
from autoscene.capture.window_capture import CaptureResult
from autoscene.core.models import BoundingBox, OCRText, TextLocateSpec
from autoscene.vision.interfaces import OCREngine
from autoscene.vision.pipeline import run_text_locate_pipeline


class _TextLocateOCREngine(OCREngine):
    def __init__(self, text_actions: "TextActions", *, ocr_options: dict | None) -> None:
        self._text_actions = text_actions
        self._ocr_options = ocr_options

    def read(self, image: object) -> list[OCRText]:
        return self._text_actions.read_ocr(image, ocr_options=self._ocr_options)


class TextActions:
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

    def click_text(
        self,
        locate: TextLocateSpec,
        debug_path: str | None = None,
        debug_crop_path: str | None = None,
    ) -> None:
        resolved_locate = self._require_text_locate(locate)
        capture_result, matched = self._locate_text_with_retry(
            resolved_locate,
            attempts=3,
            retry_interval_seconds=0.3,
        )
        matched = self._require_matched_text(matched, locate=resolved_locate)
        self.debug_artifact_writer.save_text_match_debug(
            capture_result.image,
            matched,
            target_text=resolved_locate.text,
            debug_path=debug_path,
            debug_crop_path=debug_crop_path,
        )
        x, y = self.base_actions.capture_to_screen(
            *matched.bbox.center,
            capture_result=capture_result,
        )
        self.base_actions.click(x, y)

    def click_relative_to_text(
        self,
        locate: TextLocateSpec,
        offset_x: int = 0,
        offset_y: int = 0,
        anchor: str = "center",
    ) -> None:
        resolved_locate = self._require_text_locate(locate)
        capture_result, matched = self.locate_text_match(resolved_locate)
        matched = self._require_matched_text(matched, locate=resolved_locate)
        base_x, base_y = self.bbox_anchor(matched.bbox, anchor)
        x, y = self.base_actions.capture_to_screen(
            base_x + int(offset_x),
            base_y + int(offset_y),
            capture_result=capture_result,
        )
        self.base_actions.click(x, y)

    def wait_for_text(
        self,
        locate: TextLocateSpec,
        timeout: float = 10.0,
        interval: float = 0.5,
    ) -> bool:
        resolved_locate = self._require_text_locate(locate)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.verify_text_exists(resolved_locate):
                return True
            time.sleep(interval)
        return False

    def verify_text_exists(self, locate: TextLocateSpec) -> bool:
        resolved_locate = self._require_text_locate(locate)
        _, matched = self.locate_text_match(resolved_locate)
        return matched is not None

    def read_ocr(self, image: object, ocr_options: dict | None = None) -> list[OCRText]:
        return self.vision_runtime.read_ocr(image, ocr_options=ocr_options)

    def locate_text_match(
        self,
        locate: TextLocateSpec,
    ) -> tuple[CaptureResult, OCRText | None]:
        resolved_locate = self._require_text_locate(locate)
        return self._locate_text_once(resolved_locate)

    def _locate_text_once(
        self,
        locate: TextLocateSpec,
    ) -> tuple[CaptureResult, OCRText | None]:
        capture_result = capture_active_frame(self.base_actions)
        result = run_text_locate_pipeline(
            capture_result.image,
            ocr_engine=_TextLocateOCREngine(self, ocr_options=locate.ocr),
            locate=locate,
        )
        return capture_result, result.match

    def _locate_text_with_retry(
        self,
        locate: TextLocateSpec,
        *,
        attempts: int,
        retry_interval_seconds: float,
    ) -> tuple[CaptureResult, OCRText | None]:
        return self.retry_policy.run_with_retry(
            lambda: self._locate_text_once(locate),
            attempts=attempts,
            retry_interval_seconds=retry_interval_seconds,
            should_retry=lambda result: result[1] is None,
        )

    @staticmethod
    def _require_text_locate(locate: TextLocateSpec) -> TextLocateSpec:
        if not isinstance(locate, TextLocateSpec):
            raise TypeError(
                f"text actions require TextLocateSpec, got {type(locate).__name__}."
            )
        return locate

    @staticmethod
    def _require_matched_text(
        matched: OCRText | None,
        *,
        locate: TextLocateSpec,
    ) -> OCRText:
        if matched is None:
            raise AssertionError(f"Text not found by OCR: {locate.text}")
        return matched

    @staticmethod
    def bbox_anchor(bbox: BoundingBox, anchor: str) -> tuple[int, int]:
        anchor_name = anchor.lower()
        if anchor_name == "top_left":
            return (bbox.x1, bbox.y1)
        if anchor_name == "top_right":
            return (bbox.x2, bbox.y1)
        if anchor_name == "bottom_left":
            return (bbox.x1, bbox.y2)
        if anchor_name == "bottom_right":
            return (bbox.x2, bbox.y2)
        if anchor_name == "left_center":
            return (bbox.x1, bbox.center[1])
        if anchor_name == "right_center":
            return (bbox.x2, bbox.center[1])
        return bbox.center
