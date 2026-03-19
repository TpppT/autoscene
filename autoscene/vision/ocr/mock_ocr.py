from __future__ import annotations

from typing import Any

from autoscene.core.models import BoundingBox, OCRText
from autoscene.vision.interfaces import OCREngine


class MockOCREngine(OCREngine):
    def __init__(self, fixtures: list[dict] | None = None) -> None:
        self._fixtures = fixtures or []

    def read(self, image: Any) -> list[OCRText]:
        return [
            OCRText(
                text=item["text"],
                bbox=BoundingBox(
                    x1=item["x1"],
                    y1=item["y1"],
                    x2=item["x2"],
                    y2=item["y2"],
                    score=item.get("score", 1.0),
                ),
                score=item.get("score", 1.0),
            )
            for item in self._fixtures
        ]
