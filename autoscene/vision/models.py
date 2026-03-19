from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autoscene.core.models import BoundingBox


@dataclass
class VisionNodeTraceEntry:
    stage_name: str
    stage_kind: str
    label: str = ""
    score: float = 0.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionNode:
    region: BoundingBox | None = None
    label: str = ""
    score: float = 1.0
    text: str = ""
    value: float | int | str | None = None
    unit: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: list[VisionNodeTraceEntry] = field(default_factory=list)

    @property
    def bbox(self) -> BoundingBox | None:
        return self.region

    def to_bounding_box(self) -> BoundingBox:
        if self.region is None:
            raise ValueError("VisionNode does not have a region to convert into BoundingBox.")
        return BoundingBox(
            x1=int(self.region.x1),
            y1=int(self.region.y1),
            x2=int(self.region.x2),
            y2=int(self.region.y2),
            score=float(self.score),
            label=str(self.label),
        )


@dataclass
class MatchResult:
    score: float
    region: BoundingBox
    label: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompareResult:
    passed: bool
    score: float
    threshold: float | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadResult:
    value: float | int | str
    score: float
    label: str = ""
    unit: str = ""
    source: str = ""
    region: BoundingBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionOperatorOutput:
    nodes: list[VisionNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
