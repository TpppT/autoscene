from __future__ import annotations

from typing import Any

from autoscene.core.models import BoundingBox
from autoscene.vision.interfaces import ComparatorAdapter
from autoscene.vision.models import CompareResult
from autoscene.imaging.opencv.base import OpenCVAdapterBase


class OpenCVImageSimilarityComparator(OpenCVAdapterBase, ComparatorAdapter):
    def __init__(
        self,
        *,
        method: str = "normalized_correlation",
        threshold: float = 0.95,
        grayscale: bool = True,
    ) -> None:
        self.method = str(method)
        self.threshold = float(threshold)
        self.grayscale = bool(grayscale)

    @property
    def backend(self) -> str:
        return "opencv"

    def compare(
        self,
        image: Any,
        expected: Any,
        region: BoundingBox | None = None,
    ) -> CompareResult:
        raise NotImplementedError(
            "OpenCVImageSimilarityComparator is a skeleton. Implement image "
            "alignment/preprocessing and a similarity metric before runtime use."
        )
