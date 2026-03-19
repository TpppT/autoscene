from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from autoscene.core.models import BoundingBox, OCRText
from autoscene.vision.models import (
    CompareResult,
    MatchResult,
    ReadResult,
    VisionNode,
    VisionOperatorOutput,
)


class Detector(ABC):
    @abstractmethod
    def detect(
        self,
        image: Any,
        labels: Sequence[str] | None = None,
    ) -> list[BoundingBox]:
        raise NotImplementedError


class OCREngine(ABC):
    @abstractmethod
    def read(self, image: Any) -> list[OCRText]:
        raise NotImplementedError


class VisionAdapter(ABC):
    @property
    @abstractmethod
    def backend(self) -> str:
        raise NotImplementedError


class MatcherAdapter(VisionAdapter, ABC):
    @abstractmethod
    def match(
        self,
        image: Any,
        query: Any = None,
        region: BoundingBox | None = None,
    ) -> MatchResult | None:
        raise NotImplementedError


class ComparatorAdapter(VisionAdapter, ABC):
    @abstractmethod
    def compare(
        self,
        image: Any,
        expected: Any,
        region: BoundingBox | None = None,
    ) -> CompareResult:
        raise NotImplementedError


class ReaderAdapter(VisionAdapter, ABC):
    @abstractmethod
    def read(
        self,
        image: Any,
        query: Any = None,
        region: BoundingBox | None = None,
    ) -> ReadResult:
        raise NotImplementedError


class VisionOperator(VisionAdapter, ABC):
    @abstractmethod
    def run(
        self,
        image: Any,
        nodes: Sequence[VisionNode],
        *,
        context: Any,
        params: dict[str, Any] | None = None,
    ) -> VisionOperatorOutput | Sequence[VisionNode]:
        raise NotImplementedError
