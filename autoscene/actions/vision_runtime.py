from __future__ import annotations

from collections.abc import Mapping

from autoscene.core.exceptions import ActionExecutionError
from autoscene.core.models import OCRText
from autoscene.vision.interfaces import Detector, OCREngine


class ActionVisionRuntime:
    """Own vision backends used by action-layer locate helpers."""

    def __init__(
        self,
        *,
        detector: Detector,
        ocr: OCREngine,
        detectors: Mapping[str, Detector] | None = None,
    ) -> None:
        self.detector = detector
        self.detectors = dict(detectors or {})
        self.ocr = ocr

    def resolve_detector(self, detector_name: str | None = None) -> Detector:
        if detector_name is None or str(detector_name).strip() == "":
            return self.detector
        normalized_name = str(detector_name).strip()
        if normalized_name == "default":
            return self.detector
        detector = self.detectors.get(normalized_name)
        if detector is None:
            available = ["default", *sorted(self.detectors)]
            raise ActionExecutionError(
                f"Unknown detector alias: {normalized_name}. Available: {', '.join(available)}"
            )
        return detector

    def read_ocr(
        self,
        image: object,
        *,
        ocr_options: dict | None = None,
    ) -> list[OCRText]:
        if ocr_options and hasattr(self.ocr, "read_with_overrides"):
            return self.ocr.read_with_overrides(image, overrides=ocr_options)
        return self.ocr.read(image)
