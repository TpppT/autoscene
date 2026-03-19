from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from autoscene.core.models import BoundingBox
from autoscene.imaging.opencv.base import OpenCVAdapterBase
from autoscene.vision.interfaces import Detector


class OpenCVColorDetector(OpenCVAdapterBase, Detector):
    def __init__(
        self,
        lower: Sequence[int],
        upper: Sequence[int],
        *,
        label: str = "color_match",
        color_space: str = "HSV",
        min_area: int = 25,
        max_results: int = 10,
    ) -> None:
        self.lower = tuple(int(value) for value in lower)
        self.upper = tuple(int(value) for value in upper)
        self.label = str(label)
        self.color_space = str(color_space)
        self.min_area = int(min_area)
        self.max_results = int(max_results)

    def detect(
        self,
        image: Any,
        labels: Sequence[str] | None = None,
    ) -> list[BoundingBox]:
        raise NotImplementedError(
            "OpenCVColorDetector is a skeleton. Implement color conversion, "
            "cv2.inRange masking, contour extraction, and box filtering."
        )
