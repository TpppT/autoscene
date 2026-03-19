from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from autoscene.core.exceptions import DependencyMissingError
from autoscene.core.models import BoundingBox
from autoscene.vision.algorithms.opencv.template_matcher import coerce_pil_image
from autoscene.vision.interfaces import Detector

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None


class YoloDetector(Detector):
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        inference_region: BoundingBox | Mapping[str, Any] | None = None,
    ) -> None:
        if YOLO is None:
            raise DependencyMissingError(
                "ultralytics is not installed. Install the optional YOLO support: "
                "pip install -r requirements-yolo.txt"
            )
        self._model = YOLO(model_path)
        self._confidence = confidence
        self._inference_region = self._coerce_region(inference_region)

    def detect(
        self, image: Any, labels: Optional[Sequence[str]] = None
    ) -> list[BoundingBox]:
        inference_image = image
        inference_size: tuple[int, int] | None = None
        offset_x = 0
        offset_y = 0
        if self._inference_region is not None:
            pil_image = coerce_pil_image(image)
            region = self._clip_region(self._inference_region, pil_image.size)
            inference_image = pil_image.crop((region.x1, region.y1, region.x2, region.y2))
            inference_size = inference_image.size
            offset_x = region.x1
            offset_y = region.y1

        result = self._model.predict(inference_image, conf=self._confidence, verbose=False)[0]
        names = result.names
        output: list[BoundingBox] = []
        allowed = set(labels) if labels else None

        for box in result.boxes:
            cls_idx = int(box.cls[0].item())
            label = str(names.get(cls_idx, cls_idx))
            if allowed and label not in allowed:
                continue
            xyxy = box.xyxy[0].tolist()
            if inference_size is not None:
                clamped = self._clip_box(xyxy, inference_size)
                if clamped is None:
                    continue
                x1, y1, x2, y2 = clamped
                xyxy = [
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                ]
            output.append(
                BoundingBox(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    score=float(box.conf[0].item()),
                    label=label,
                )
            )
        return output

    @staticmethod
    def _coerce_region(
        region: BoundingBox | Mapping[str, Any] | None,
    ) -> BoundingBox | None:
        if region is None:
            return None
        if isinstance(region, BoundingBox):
            return region
        if not isinstance(region, Mapping):
            raise TypeError("inference_region must be a BoundingBox or mapping.")
        try:
            return BoundingBox(
                x1=int(region["x1"]),
                y1=int(region["y1"]),
                x2=int(region["x2"]),
                y2=int(region["y2"]),
            )
        except KeyError as exc:
            raise ValueError(
                "inference_region requires x1, y1, x2, and y2 fields."
            ) from exc

    @staticmethod
    def _clip_region(region: BoundingBox, image_size: tuple[int, int]) -> BoundingBox:
        width, height = image_size
        x1 = max(0, min(int(region.x1), width))
        y1 = max(0, min(int(region.y1), height))
        x2 = max(0, min(int(region.x2), width))
        y2 = max(0, min(int(region.y2), height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("inference_region must overlap the detector input image.")
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def _clip_box(
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
