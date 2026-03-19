from __future__ import annotations

from pathlib import Path

from autoscene.actions.advanced.protocols import BaseActionRuntimeProtocol
from autoscene.capture.window_capture import CaptureResult
from autoscene.core.models import BoundingBox, OCRText


class DebugArtifactWriter:
    def __init__(self, base_actions: BaseActionRuntimeProtocol) -> None:
        self.base_actions = base_actions

    def save_text_match_debug(
        self,
        image: object,
        matched: OCRText,
        target_text: str,
        debug_path: str | None,
        debug_crop_path: str | None,
    ) -> None:
        if not debug_path and not debug_crop_path:
            return
        try:
            from PIL import ImageDraw
        except ImportError:
            return

        if debug_path:
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            bbox = matched.bbox
            draw.rectangle((bbox.x1, bbox.y1, bbox.x2, bbox.y2), outline="red", width=4)
            text_y = max(0, bbox.y1 - 24)
            draw.text((bbox.x1, text_y), f"OCR: {target_text}", fill="red")
            debug_file = Path(debug_path)
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(debug_file)
            self.base_actions.logger.info("saved click_text debug image: %s", debug_file)

        if debug_crop_path:
            bbox = matched.bbox
            crop = image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
            debug_crop_file = Path(debug_crop_path)
            debug_crop_file.parent.mkdir(parents=True, exist_ok=True)
            crop.save(debug_crop_file)
            self.base_actions.logger.info("saved click_text debug crop: %s", debug_crop_file)

    def save_object_debug(
        self,
        image: object,
        region: BoundingBox | None,
        target_box: BoundingBox,
        label: str,
        debug_path: str | None,
        capture_result: CaptureResult | None = None,
    ) -> None:
        if not debug_path:
            return
        try:
            from PIL import ImageDraw
        except ImportError:
            return

        annotated = image.copy()
        scale_x = 1.0
        scale_y = 1.0
        candidate = None
        if capture_result is not None:
            candidate = capture_result.artifact_image
        if candidate is None:
            get_last_capture_result = getattr(
                self.base_actions.capture_engine,
                "get_last_capture_result",
                None,
            )
            if callable(get_last_capture_result):
                last_result = get_last_capture_result()
                if last_result is not None:
                    candidate = getattr(last_result, "artifact_image", None)
        if candidate is not None and self.images_share_uniform_scale(image, candidate):
            annotated = candidate.copy()
            image_size = getattr(image, "size", None)
            candidate_size = getattr(candidate, "size", None)
            if image_size and candidate_size:
                scale_x = float(candidate_size[0]) / max(float(image_size[0]), 1.0)
                scale_y = float(candidate_size[1]) / max(float(image_size[1]), 1.0)

        draw = ImageDraw.Draw(annotated)
        if region is not None:
            scaled_region = self.scale_bbox(region, scale_x=scale_x, scale_y=scale_y)
            draw.rectangle(
                (scaled_region.x1, scaled_region.y1, scaled_region.x2, scaled_region.y2),
                outline="yellow",
                width=3,
            )
        scaled_target_box = self.scale_bbox(target_box, scale_x=scale_x, scale_y=scale_y)
        draw.rectangle(
            (scaled_target_box.x1, scaled_target_box.y1, scaled_target_box.x2, scaled_target_box.y2),
            outline="red",
            width=4,
        )
        scaled_region = None
        if region is not None:
            scaled_region = self.scale_bbox(region, scale_x=scale_x, scale_y=scale_y)
        anchor_x = scaled_target_box.x1 if scaled_region is None else scaled_region.x1
        anchor_y = scaled_target_box.y1 if scaled_region is None else scaled_region.y1
        draw.text((anchor_x, max(0, anchor_y - 22)), f"Object: {label}", fill="red")
        debug_file = Path(debug_path)
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(debug_file)
        self.base_actions.logger.info("saved object debug image: %s", debug_file)

    @staticmethod
    def scale_bbox(box: BoundingBox, scale_x: float, scale_y: float) -> BoundingBox:
        return BoundingBox(
            x1=int(round(float(box.x1) * scale_x)),
            y1=int(round(float(box.y1) * scale_y)),
            x2=int(round(float(box.x2) * scale_x)),
            y2=int(round(float(box.y2) * scale_y)),
            score=box.score,
            label=box.label,
        )

    @staticmethod
    def images_share_uniform_scale(
        image: object,
        candidate: object,
        tolerance: float = 0.05,
    ) -> bool:
        image_size = getattr(image, "size", None)
        candidate_size = getattr(candidate, "size", None)
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            return False
        if not isinstance(candidate_size, tuple) or len(candidate_size) != 2:
            return False
        scale_x = float(candidate_size[0]) / max(float(image_size[0]), 1.0)
        scale_y = float(candidate_size[1]) / max(float(image_size[1]), 1.0)
        return scale_x > 0.0 and scale_y > 0.0 and abs(scale_x - scale_y) <= tolerance
