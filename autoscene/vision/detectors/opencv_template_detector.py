from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from autoscene.core.models import BoundingBox
from autoscene.imaging.opencv.base import OpenCVAdapterBase
from autoscene.vision.interfaces import Detector


class OpenCVTemplateDetector(OpenCVAdapterBase, Detector):
    def __init__(
        self,
        templates_dir: str | Path | None = None,
        template_paths: Mapping[str, str | Path | Sequence[str | Path]] | None = None,
        threshold: float = 0.85,
        match_method: str = "TM_CCOEFF_NORMED",
        grayscale: bool = True,
        max_results: int = 5,
        nms_iou: float = 0.3,
    ) -> None:
        self.templates_dir = None if templates_dir is None else Path(templates_dir)
        self.template_paths = dict(template_paths or {})
        self.threshold = float(threshold)
        self.match_method = str(match_method)
        self.grayscale = bool(grayscale)
        self.max_results = int(max_results)
        self.nms_iou = float(nms_iou)

    def detect(
        self,
        image: Any,
        labels: Sequence[str] | None = None,
    ) -> list[BoundingBox]:
        raise NotImplementedError(
            "OpenCVTemplateDetector is a skeleton. Implement template loading, "
            "cv2.matchTemplate scoring, and NMS before using it in runtime steps."
        )
