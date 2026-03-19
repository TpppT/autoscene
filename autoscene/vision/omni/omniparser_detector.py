from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from autoscene.core.exceptions import DependencyMissingError
from autoscene.core.models import BoundingBox
from autoscene.vision.algorithms.opencv.template_matcher import (
    TemplateMatcher,
    coerce_pil_image,
)
from autoscene.vision.interfaces import Detector

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None


class OmniParserDetector(Detector):
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.05,
        iou_threshold: float = 0.1,
        image_size: int | Sequence[int] | None = None,
        templates_dir: str | Path | None = None,
        template_paths: Mapping[str, str | Path | Sequence[str | Path]] | None = None,
        template_match_threshold: float = 0.65,
        match_size: int | Sequence[int] = 64,
        default_label: str = "icon",
        unknown_label: str = "unknown_icon",
    ) -> None:
        if YOLO is None:
            raise DependencyMissingError(
                "ultralytics is not installed. Install the optional YOLO support: "
                "pip install -r requirements-yolo.txt"
            )
        self._model = YOLO(model_path)
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._image_size = self._normalize_image_size(image_size)
        self._default_label = default_label
        self._unknown_label = unknown_label
        self._template_match_threshold = template_match_threshold
        self._matcher: TemplateMatcher | None = None
        if templates_dir is not None or template_paths:
            self._matcher = TemplateMatcher(
                templates_dir=templates_dir,
                template_paths=template_paths,
                match_size=match_size,
            )

    def detect(
        self, image: Any, labels: Optional[Sequence[str]] = None
    ) -> list[BoundingBox]:
        pil_image = coerce_pil_image(image)
        predict_kwargs: dict[str, Any] = {
            "conf": self._confidence,
            "iou": self._iou_threshold,
            "verbose": False,
        }
        if self._image_size is not None:
            predict_kwargs["imgsz"] = self._image_size

        result = self._model.predict(pil_image, **predict_kwargs)[0]
        allowed = set(labels) if labels else None
        output: list[BoundingBox] = []

        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            clamped = self._clamp_box(xyxy, pil_image.size)
            if clamped is None:
                continue
            x1, y1, x2, y2 = clamped
            detector_score = float(box.conf[0].item())
            label, score = self._classify_crop(
                pil_image.crop((x1, y1, x2, y2)),
                detector_score=detector_score,
                labels=allowed,
            )
            if allowed and label not in allowed:
                continue
            output.append(
                BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                    label=label,
                )
            )
        return output

    def _classify_crop(
        self,
        crop: Any,
        detector_score: float,
        labels: Optional[Sequence[str]] = None,
    ) -> tuple[str, float]:
        if self._matcher is None:
            return self._default_label, detector_score

        allowed = set(labels) if labels else None
        if allowed:
            matcher_labels = getattr(self._matcher, "labels", set())
            if matcher_labels and allowed.isdisjoint(matcher_labels):
                return self._unknown_label, 0.0

            filtered_match = self._matcher.match(crop, labels=allowed)
            if filtered_match is None:
                return self._unknown_label, 0.0
            if filtered_match.score < self._template_match_threshold:
                return self._unknown_label, detector_score * filtered_match.score

            if matcher_labels and not matcher_labels.issubset(allowed):
                match = self._matcher.match(crop)
                if match is None:
                    return self._unknown_label, 0.0
                if match.label not in allowed:
                    return self._unknown_label, detector_score * match.score
            else:
                match = filtered_match
        else:
            match = self._matcher.match(crop)
        if match is None:
            return self._unknown_label, 0.0
        if match.score < self._template_match_threshold:
            return self._unknown_label, detector_score * match.score
        return match.label, detector_score * match.score

    @staticmethod
    def _normalize_image_size(
        image_size: int | Sequence[int] | None,
    ) -> int | tuple[int, int] | None:
        if image_size is None:
            return None
        if isinstance(image_size, int):
            return image_size
        values = tuple(int(value) for value in image_size)
        if len(values) != 2:
            raise ValueError("image_size must be an int or a sequence of two ints.")
        return values

    @staticmethod
    def _clamp_box(
        xyxy: Sequence[float], image_size: tuple[int, int]
    ) -> tuple[int, int, int, int] | None:
        width, height = image_size
        x1 = max(0, min(width, int(xyxy[0])))
        y1 = max(0, min(height, int(xyxy[1])))
        x2 = max(0, min(width, int(xyxy[2])))
        y2 = max(0, min(height, int(xyxy[3])))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)
