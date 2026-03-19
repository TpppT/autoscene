from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from autoscene.core.models import BoundingBox
from autoscene.vision.interfaces import Detector


class MockDetector(Detector):
    def __init__(self, fixtures: Optional[list[dict]] = None) -> None:
        self._fixtures = fixtures or []

    def detect(
        self, image: Any, labels: Optional[Sequence[str]] = None
    ) -> list[BoundingBox]:
        output = [
            BoundingBox(
                x1=item["x1"],
                y1=item["y1"],
                x2=item["x2"],
                y2=item["y2"],
                score=item.get("score", 1.0),
                label=item.get("label", ""),
            )
            for item in self._fixtures
        ]
        if labels:
            allowed = set(labels)
            output = [box for box in output if box.label in allowed]
        return output
