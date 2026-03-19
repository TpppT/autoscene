from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from autoscene.core.models import BoundingBox
from autoscene.vision.interfaces import Detector, MatcherAdapter
from autoscene.vision.pipeline import (
    DetectorRefinementStage,
    DetectorRegionStage,
    MatcherClassificationStage,
    VisionPipeline,
    VisionPipelineResult,
)


class CascadeDetector(Detector):
    """Compose multiple vision stages inside one detector entry.

    Common use case:
    - stage 1: YOLO finds a panel / card / slot region
    - stage 2: matcher or nested detector recognizes the concrete content in that region
    """

    def __init__(
        self,
        region_detector: Detector | Mapping[str, Any],
        *,
        detail_detector: Detector | Mapping[str, Any] | None = None,
        detail_matcher: MatcherAdapter | Mapping[str, Any] | None = None,
        region_labels: Sequence[str] | None = None,
        detail_labels: Sequence[str] | None = None,
        match_threshold: float = 0.0,
        unknown_label: str | None = None,
        max_regions: int | None = None,
        detector_factory: Callable[[dict[str, Any]], Detector] | None = None,
        matcher_factory: Callable[[dict[str, Any]], MatcherAdapter] | None = None,
        registry_bundle=None,
    ) -> None:
        if detail_detector is None and detail_matcher is None:
            raise ValueError(
                "CascadeDetector requires either 'detail_detector' or 'detail_matcher'."
            )
        if detail_detector is not None and detail_matcher is not None:
            raise ValueError(
                "CascadeDetector accepts only one of 'detail_detector' or 'detail_matcher'."
            )

        detector_factory = detector_factory or getattr(registry_bundle, "create_detector", None)
        matcher_factory = matcher_factory or getattr(registry_bundle, "create_matcher_adapter", None)
        if detector_factory is None:
            raise ValueError("CascadeDetector requires a detector_factory.")
        if detail_matcher is not None and matcher_factory is None and not isinstance(
            detail_matcher,
            MatcherAdapter,
        ):
            raise ValueError("CascadeDetector requires a matcher_factory.")

        self._region_detector = self._coerce_detector(region_detector, detector_factory)
        self._detail_detector = (
            None
            if detail_detector is None
            else self._coerce_detector(detail_detector, detector_factory)
        )
        self._detail_matcher = (
            None
            if detail_matcher is None
            else self._coerce_matcher(detail_matcher, matcher_factory)
        )
        self._region_labels = None if region_labels is None else tuple(str(v) for v in region_labels)
        self._detail_labels = None if detail_labels is None else tuple(str(v) for v in detail_labels)
        self._match_threshold = max(float(match_threshold), 0.0)
        self._unknown_label = None if unknown_label is None else str(unknown_label)
        self._max_regions = None if max_regions is None else max(int(max_regions), 1)
        self._pipeline = self._build_pipeline()

    @property
    def last_pipeline_result(self) -> VisionPipelineResult | None:
        return self._pipeline.last_result

    def detect(
        self,
        image: Any,
        labels: Sequence[str] | None = None,
    ) -> list[BoundingBox]:
        return self.run_pipeline(image, labels=labels).boxes

    def run_pipeline(
        self,
        image: Any,
        labels: Sequence[str] | None = None,
    ) -> VisionPipelineResult:
        return self._pipeline.run(image, labels=labels)

    def _build_pipeline(self) -> VisionPipeline:
        stages = [
            DetectorRegionStage(
                self._region_detector,
                labels=self._region_labels,
                max_regions=self._max_regions,
            )
        ]
        if self._detail_detector is not None:
            stages.append(
                DetectorRefinementStage(
                    self._detail_detector,
                    labels=self._detail_labels,
                )
            )
        else:
            assert self._detail_matcher is not None
            stages.append(
                MatcherClassificationStage(
                    self._detail_matcher,
                    labels=self._detail_labels,
                    match_threshold=self._match_threshold,
                    unknown_label=self._unknown_label,
                )
            )
        return VisionPipeline(stages)

    @staticmethod
    def _coerce_detector(
        detector: Detector | Mapping[str, Any],
        detector_factory: Callable[[dict[str, Any]], Detector],
    ) -> Detector:
        if isinstance(detector, Detector):
            return detector
        if not isinstance(detector, Mapping):
            raise TypeError("CascadeDetector detector stage must be a Detector or mapping.")
        return detector_factory(dict(detector))

    @staticmethod
    def _coerce_matcher(
        matcher: MatcherAdapter | Mapping[str, Any],
        matcher_factory: Callable[[dict[str, Any]], MatcherAdapter] | None,
    ) -> MatcherAdapter:
        if isinstance(matcher, MatcherAdapter):
            return matcher
        if matcher_factory is None:
            raise ValueError("CascadeDetector requires a matcher_factory.")
        if not isinstance(matcher, Mapping):
            raise TypeError("CascadeDetector matcher stage must be a MatcherAdapter or mapping.")
        return matcher_factory(dict(matcher))
