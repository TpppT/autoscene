from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from autoscene.core.models import BoundingBox
from autoscene.vision.algorithms.opencv.template_matcher import coerce_pil_image
from autoscene.vision.interfaces import (
    ComparatorAdapter,
    Detector,
    MatcherAdapter,
    OCREngine,
    ReaderAdapter,
    VisionOperator,
)
from autoscene.vision.models import VisionNode

from .core import (
    VisionPipelineContext,
    VisionPipelineResult,
    VisionPipelineStage,
    VisionStageRegistry,
)
from .registry import _coerce_stage
from .utils import _normalize_labels


class VisionPipeline:
    def __init__(self, stages: Sequence[VisionPipelineStage]) -> None:
        self._stages = list(stages)
        if not self._stages:
            raise ValueError("VisionPipeline requires at least one stage.")
        self._last_result: VisionPipelineResult | None = None

    @property
    def last_result(self) -> VisionPipelineResult | None:
        return self._last_result

    def run(
        self,
        image: Any,
        *,
        labels: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VisionPipelineResult:
        pil_image = coerce_pil_image(image)
        context = VisionPipelineContext(
            allowed_labels=_normalize_labels(labels),
            metadata=dict(metadata or {}),
        )
        nodes: list[VisionNode] = []
        for stage in self._stages:
            nodes = stage.execute(pil_image, nodes, context)
        result = VisionPipelineResult(nodes=nodes, trace=list(context.trace))
        self._last_result = result
        return result

    def detect(
        self,
        image: Any,
        labels: Sequence[str] | None = None,
    ) -> list[BoundingBox]:
        return self.run(image, labels=labels).boxes


class VisionPipelineDetector(Detector):
    def __init__(
        self,
        stages: Sequence[VisionPipelineStage | Mapping[str, Any]],
        *,
        detector_factory: Callable[[dict[str, Any]], Detector] | None = None,
        ocr_engine_factory: Callable[[dict[str, Any]], OCREngine] | None = None,
        matcher_factory: Callable[[dict[str, Any]], MatcherAdapter] | None = None,
        comparator_factory: Callable[[dict[str, Any]], ComparatorAdapter] | None = None,
        reader_factory: Callable[[dict[str, Any]], ReaderAdapter] | None = None,
        operator_factory: Callable[[dict[str, Any]], VisionOperator] | None = None,
        stage_registry: VisionStageRegistry | None = None,
        registry_bundle: Any = None,
    ) -> None:
        resolved_factories = _resolve_pipeline_builders(
            detector_factory=detector_factory,
            ocr_engine_factory=ocr_engine_factory,
            matcher_factory=matcher_factory,
            comparator_factory=comparator_factory,
            reader_factory=reader_factory,
            operator_factory=operator_factory,
            stage_registry=stage_registry,
            registry_bundle=registry_bundle,
        )
        self._pipeline = build_vision_pipeline(
            stages,
            **resolved_factories,
        )

    @property
    def last_pipeline_result(self) -> VisionPipelineResult | None:
        return self._pipeline.last_result

    def run_pipeline(
        self,
        image: Any,
        *,
        labels: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VisionPipelineResult:
        return self._pipeline.run(image, labels=labels, metadata=metadata)

    def detect(
        self,
        image: Any,
        labels: Sequence[str] | None = None,
    ) -> list[BoundingBox]:
        return self._pipeline.detect(image, labels=labels)


def build_vision_pipeline(
    stages: Sequence[VisionPipelineStage | Mapping[str, Any]],
    *,
    detector_factory: Callable[[dict[str, Any]], Detector] | None = None,
    ocr_engine_factory: Callable[[dict[str, Any]], OCREngine] | None = None,
    matcher_factory: Callable[[dict[str, Any]], MatcherAdapter] | None = None,
    comparator_factory: Callable[[dict[str, Any]], ComparatorAdapter] | None = None,
    reader_factory: Callable[[dict[str, Any]], ReaderAdapter] | None = None,
    operator_factory: Callable[[dict[str, Any]], VisionOperator] | None = None,
    stage_registry: VisionStageRegistry | None = None,
) -> VisionPipeline:
    return VisionPipeline(
        _coerce_pipeline_stages(
            stages,
            detector_factory=detector_factory,
            ocr_engine_factory=ocr_engine_factory,
            matcher_factory=matcher_factory,
            comparator_factory=comparator_factory,
            reader_factory=reader_factory,
            operator_factory=operator_factory,
            stage_registry=stage_registry,
        )
    )


def _resolve_pipeline_builders(
    *,
    detector_factory: Callable[[dict[str, Any]], Detector] | None,
    ocr_engine_factory: Callable[[dict[str, Any]], OCREngine] | None,
    matcher_factory: Callable[[dict[str, Any]], MatcherAdapter] | None,
    comparator_factory: Callable[[dict[str, Any]], ComparatorAdapter] | None,
    reader_factory: Callable[[dict[str, Any]], ReaderAdapter] | None,
    operator_factory: Callable[[dict[str, Any]], VisionOperator] | None,
    stage_registry: VisionStageRegistry | None,
    registry_bundle: Any,
) -> dict[str, Any]:
    return {
        "detector_factory": _resolve_registry_bundle_attr(
            detector_factory,
            registry_bundle=registry_bundle,
            attr_name="create_detector",
        ),
        "ocr_engine_factory": _resolve_registry_bundle_attr(
            ocr_engine_factory,
            registry_bundle=registry_bundle,
            attr_name="create_ocr_engine",
        ),
        "matcher_factory": _resolve_registry_bundle_attr(
            matcher_factory,
            registry_bundle=registry_bundle,
            attr_name="create_matcher_adapter",
        ),
        "comparator_factory": _resolve_registry_bundle_attr(
            comparator_factory,
            registry_bundle=registry_bundle,
            attr_name="create_comparator_adapter",
        ),
        "reader_factory": _resolve_registry_bundle_attr(
            reader_factory,
            registry_bundle=registry_bundle,
            attr_name="create_reader_adapter",
        ),
        "operator_factory": _resolve_registry_bundle_attr(
            operator_factory,
            registry_bundle=registry_bundle,
            attr_name="create_operator",
        ),
        "stage_registry": _resolve_registry_bundle_attr(
            stage_registry,
            registry_bundle=registry_bundle,
            attr_name="pipeline_stages",
        ),
    }


def _resolve_registry_bundle_attr(
    value: Any,
    *,
    registry_bundle: Any,
    attr_name: str,
) -> Any:
    if value is not None:
        return value
    return getattr(registry_bundle, attr_name, None)


def _coerce_pipeline_stages(
    stages: Sequence[VisionPipelineStage | Mapping[str, Any]],
    *,
    detector_factory: Callable[[dict[str, Any]], Detector] | None,
    ocr_engine_factory: Callable[[dict[str, Any]], OCREngine] | None,
    matcher_factory: Callable[[dict[str, Any]], MatcherAdapter] | None,
    comparator_factory: Callable[[dict[str, Any]], ComparatorAdapter] | None,
    reader_factory: Callable[[dict[str, Any]], ReaderAdapter] | None,
    operator_factory: Callable[[dict[str, Any]], VisionOperator] | None,
    stage_registry: VisionStageRegistry | None,
) -> list[VisionPipelineStage]:
    return [
        _coerce_stage(
            stage,
            detector_factory=detector_factory,
            ocr_engine_factory=ocr_engine_factory,
            matcher_factory=matcher_factory,
            comparator_factory=comparator_factory,
            reader_factory=reader_factory,
            operator_factory=operator_factory,
            stage_registry=stage_registry,
        )
        for stage in stages
    ]
