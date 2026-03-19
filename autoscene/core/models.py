from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 1.0
    label: str = ""

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_payload(self) -> dict[str, Any]:
        return {
            "x1": int(self.x1),
            "y1": int(self.y1),
            "x2": int(self.x2),
            "y2": int(self.y2),
        }


@dataclass
class OCRText:
    text: str
    bbox: BoundingBox
    score: float = 1.0


@dataclass(frozen=True)
class TextLocateSpec:
    text: str
    exact: bool = False
    region: BoundingBox | None = None
    ocr: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": str(self.text),
            "exact": bool(self.exact),
        }
        if self.region is not None:
            payload["region"] = self.region.to_payload()
        if self.ocr:
            payload["ocr"] = dict(self.ocr)
        return payload


@dataclass(frozen=True)
class ObjectLocateSpec:
    label: str
    min_score: float = 0.3
    pick: str = "highest_score"
    detector: str | None = None
    region: BoundingBox | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": str(self.label),
            "min_score": float(self.min_score),
            "pick": str(self.pick),
        }
        if self.detector is not None:
            payload["detector"] = str(self.detector)
        if self.region is not None:
            payload["region"] = self.region.to_payload()
        return payload


@dataclass
class TestCase:
    name: str
    emulator: dict = field(default_factory=dict)
    detector: dict = field(default_factory=dict)
    detectors: dict = field(default_factory=dict)
    readers: dict = field(default_factory=dict)
    log_sources: dict = field(default_factory=dict)
    ocr: dict = field(default_factory=dict)
    capture: dict = field(default_factory=dict)
    setup: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    verification_setup: list[dict] = field(default_factory=list)
    verification: list[dict] = field(default_factory=list)
    teardown: list[dict] = field(default_factory=list)
