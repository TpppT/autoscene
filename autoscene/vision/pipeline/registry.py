from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from autoscene.core.models import BoundingBox
from autoscene.vision.interfaces import (
    ComparatorAdapter,
    Detector,
    MatcherAdapter,
    OCREngine,
    ReaderAdapter,
    VisionOperator,
)

from .core import (
    VisionPipelineStage,
    VisionStageBuildContext,
    VisionStageBuilder,
    VisionStageRegistry,
)
from .stages import (
    ComparatorFilterStage,
    DetectorRefinementStage,
    DetectorRegionStage,
    MatcherClassificationStage,
    NodeFilterStage,
    OCRClassificationStage,
    OperatorStage,
    ReaderClassificationStage,
    TextLocateStage,
)


def _build_stage_context(
    *,
    detector_factory: Callable[[dict[str, Any]], Detector] | None,
    ocr_engine_factory: Callable[[dict[str, Any]], OCREngine] | None,
    matcher_factory: Callable[[dict[str, Any]], MatcherAdapter] | None,
    comparator_factory: Callable[[dict[str, Any]], ComparatorAdapter] | None,
    reader_factory: Callable[[dict[str, Any]], ReaderAdapter] | None,
    operator_factory: Callable[[dict[str, Any]], VisionOperator] | None,
    stage_registry: VisionStageRegistry,
) -> VisionStageBuildContext:
    return VisionStageBuildContext(
        detector_factory=detector_factory,
        ocr_engine_factory=ocr_engine_factory,
        matcher_factory=matcher_factory,
        comparator_factory=comparator_factory,
        reader_factory=reader_factory,
        operator_factory=operator_factory,
        stage_registry=stage_registry,
    )


def _coerce_stage(
    stage: VisionPipelineStage | Mapping[str, Any],
    *,
    detector_factory: Callable[[dict[str, Any]], Detector] | None,
    ocr_engine_factory: Callable[[dict[str, Any]], OCREngine] | None,
    matcher_factory: Callable[[dict[str, Any]], MatcherAdapter] | None,
    comparator_factory: Callable[[dict[str, Any]], ComparatorAdapter] | None,
    reader_factory: Callable[[dict[str, Any]], ReaderAdapter] | None,
    operator_factory: Callable[[dict[str, Any]], VisionOperator] | None,
    stage_registry: VisionStageRegistry | None,
) -> VisionPipelineStage:
    if isinstance(stage, VisionPipelineStage):
        return stage
    if not isinstance(stage, Mapping):
        raise TypeError("Vision pipeline stage must be a stage object or mapping.")
    resolved_registry = resolve_vision_stage_registry(stage_registry)
    build_context = _build_stage_context(
        detector_factory=detector_factory,
        ocr_engine_factory=ocr_engine_factory,
        matcher_factory=matcher_factory,
        comparator_factory=comparator_factory,
        reader_factory=reader_factory,
        operator_factory=operator_factory,
        stage_registry=resolved_registry,
    )
    return resolved_registry.create(dict(stage), build_context=build_context)


def register_vision_stage(
    name: str,
    builder: VisionStageBuilder,
    *,
    namespace: str | None = None,
    override: bool = True,
) -> None:
    _DEFAULT_STAGE_REGISTRY.register(
        name,
        builder,
        namespace=namespace,
        override=override,
    )


def build_vision_stage_registry() -> VisionStageRegistry:
    registry = VisionStageRegistry()
    _register_builtin_stage_builders(registry)
    return registry


def resolve_vision_stage_registry(
    stage_registry: VisionStageRegistry | None = None,
) -> VisionStageRegistry:
    return _DEFAULT_STAGE_REGISTRY if stage_registry is None else stage_registry


def _register_builtin_stage_builders(registry: VisionStageRegistry) -> None:
    _register_stage_aliases(
        registry,
        _build_detector_region_stage,
        "detector_region",
        "region_detector",
    )
    _register_stage_aliases(
        registry,
        _build_detector_refinement_stage,
        "detector_refinement",
        "detail_detector",
    )
    _register_stage_aliases(
        registry,
        _build_matcher_classification_stage,
        "matcher_classification",
        "detail_matcher",
    )
    _register_stage_aliases(
        registry,
        _build_ocr_classification_stage,
        "ocr_classification",
        "detail_ocr",
        "ocr",
    )
    _register_stage_aliases(
        registry,
        _build_text_locate_stage,
        "text_locate",
        "locate_text",
    )
    _register_stage_aliases(
        registry,
        _build_reader_classification_stage,
        "reader_classification",
        "detail_reader",
        "reader",
    )
    _register_stage_aliases(
        registry,
        _build_node_filter_stage,
        "node_filter",
        "filter",
    )
    _register_stage_aliases(
        registry,
        _build_comparator_filter_stage,
        "comparator_filter",
        "detail_comparator",
        "comparator",
    )
    _register_stage_aliases(
        registry,
        _build_operator_stage,
        "operator",
    )


def _register_stage_aliases(
    registry: VisionStageRegistry,
    builder: VisionStageBuilder,
    *names: str,
) -> None:
    for name in names:
        registry.register(name, builder)


def _build_detector_region_stage(
    payload: dict[str, Any],
    build_context: VisionStageBuildContext,
) -> VisionPipelineStage:
    name = payload.pop("name", None)
    detector = _coerce_detector_stage(payload.pop("detector", None), build_context.detector_factory)
    return DetectorRegionStage(
        detector,
        labels=payload.pop("labels", None),
        max_regions=payload.pop("max_regions", None),
        name=name or "region_detector",
    )


def _build_detector_refinement_stage(
    payload: dict[str, Any],
    build_context: VisionStageBuildContext,
) -> VisionPipelineStage:
    name = payload.pop("name", None)
    detector = _coerce_detector_stage(payload.pop("detector", None), build_context.detector_factory)
    return DetectorRefinementStage(
        detector,
        labels=payload.pop("labels", None),
        name=name or "detail_detector",
    )


def _build_matcher_classification_stage(
    payload: dict[str, Any],
    build_context: VisionStageBuildContext,
) -> VisionPipelineStage:
    name = payload.pop("name", None)
    matcher = _coerce_matcher_stage(payload.pop("matcher", None), build_context.matcher_factory)
    return MatcherClassificationStage(
        matcher,
        labels=payload.pop("labels", None),
        match_threshold=payload.pop("match_threshold", 0.0),
        unknown_label=payload.pop("unknown_label", None),
        name=name or "detail_matcher",
    )


def _build_ocr_classification_stage(
    payload: dict[str, Any],
    build_context: VisionStageBuildContext,
) -> VisionPipelineStage:
    name = payload.pop("name", None)
    ocr_engine = _coerce_ocr_stage(payload.pop("ocr", None), build_context.ocr_engine_factory)
    return OCRClassificationStage(
        ocr_engine,
        labels=payload.pop("labels", None),
        match_mode=payload.pop("match_mode", "contains"),
        unknown_label=payload.pop("unknown_label", None),
        output_label=payload.pop("output_label", None),
        min_score=payload.pop("min_score", 0.0),
        name=name or "detail_ocr",
    )


def _build_text_locate_stage(
    payload: dict[str, Any],
    build_context: VisionStageBuildContext,
) -> VisionPipelineStage:
    name = payload.pop("name", None)
    ocr_engine = _coerce_ocr_stage(payload.pop("ocr", None), build_context.ocr_engine_factory)
    return TextLocateStage(
        ocr_engine,
        query=str(payload.pop("query")),
        exact=bool(payload.pop("exact", False)),
        name=name or "locate_text",
    )


def _build_reader_classification_stage(
    payload: dict[str, Any],
    build_context: VisionStageBuildContext,
) -> VisionPipelineStage:
    name = payload.pop("name", None)
    reader = _coerce_reader_stage(payload.pop("reader", None), build_context.reader_factory)
    return ReaderClassificationStage(
        reader,
        query=payload.pop("query", None),
        labels=payload.pop("labels", None),
        label_source=payload.pop("label_source", "label"),
        output_label=payload.pop("output_label", None),
        unknown_label=payload.pop("unknown_label", None),
        min_score=payload.pop("min_score", 0.0),
        name=name or "detail_reader",
    )


def _build_node_filter_stage(
    payload: dict[str, Any],
    build_context: VisionStageBuildContext,
) -> VisionPipelineStage:
    del build_context
    name = payload.pop("name", None)
    return NodeFilterStage(
        labels=payload.pop("labels", None),
        min_score=payload.pop("min_score", 0.0),
        region=_coerce_region(payload.pop("region", None), field_name="region"),
        name=name or "node_filter",
    )


def _build_comparator_filter_stage(
    payload: dict[str, Any],
    build_context: VisionStageBuildContext,
) -> VisionPipelineStage:
    name = payload.pop("name", None)
    comparator = _coerce_comparator_stage(
        payload.pop("comparator", None),
        build_context.comparator_factory,
    )
    return ComparatorFilterStage(
        comparator,
        expected=payload.pop("expected", None),
        pass_label=payload.pop("pass_label", None),
        fail_label=payload.pop("fail_label", None),
        keep_failed=payload.pop("keep_failed", False),
        name=name or "detail_comparator",
    )


def _build_operator_stage(
    payload: dict[str, Any],
    build_context: VisionStageBuildContext,
) -> VisionPipelineStage:
    name = payload.pop("name", None)
    operator = _coerce_operator_stage(
        payload.pop("operator", None),
        build_context.operator_factory,
    )
    explicit_params = payload.pop("params", None)
    if explicit_params is not None and not isinstance(explicit_params, Mapping):
        raise TypeError("Vision pipeline operator 'params' must be a mapping when provided.")
    params = dict(explicit_params or {})
    params.update(payload)
    return OperatorStage(
        operator,
        params=params,
        name=name or "operator",
    )


def _coerce_detector_stage(
    detector: Detector | Mapping[str, Any] | None,
    detector_factory: Callable[[dict[str, Any]], Detector] | None,
) -> Detector:
    return _coerce_stage_component(
        detector,
        detector_factory,
        component_type=Detector,
        field_name="detector",
        component_name="detector",
        factory_name="detector_factory",
    )


def _coerce_matcher_stage(
    matcher: MatcherAdapter | Mapping[str, Any] | None,
    matcher_factory: Callable[[dict[str, Any]], MatcherAdapter] | None,
) -> MatcherAdapter:
    return _coerce_stage_component(
        matcher,
        matcher_factory,
        component_type=MatcherAdapter,
        field_name="matcher",
        component_name="matcher",
        factory_name="matcher_factory",
    )


def _coerce_ocr_stage(
    ocr_engine: OCREngine | Mapping[str, Any] | None,
    ocr_engine_factory: Callable[[dict[str, Any]], OCREngine] | None,
) -> OCREngine:
    return _coerce_stage_component(
        ocr_engine,
        ocr_engine_factory,
        component_type=OCREngine,
        field_name="ocr",
        component_name="OCR",
        factory_name="ocr_engine_factory",
    )


def _coerce_reader_stage(
    reader: ReaderAdapter | Mapping[str, Any] | None,
    reader_factory: Callable[[dict[str, Any]], ReaderAdapter] | None,
) -> ReaderAdapter:
    return _coerce_stage_component(
        reader,
        reader_factory,
        component_type=ReaderAdapter,
        field_name="reader",
        component_name="reader",
        factory_name="reader_factory",
    )


def _coerce_comparator_stage(
    comparator: ComparatorAdapter | Mapping[str, Any] | None,
    comparator_factory: Callable[[dict[str, Any]], ComparatorAdapter] | None,
) -> ComparatorAdapter:
    return _coerce_stage_component(
        comparator,
        comparator_factory,
        component_type=ComparatorAdapter,
        field_name="comparator",
        component_name="comparator",
        factory_name="comparator_factory",
    )


def _coerce_operator_stage(
    operator: VisionOperator | Mapping[str, Any] | None,
    operator_factory: Callable[[dict[str, Any]], VisionOperator] | None,
) -> VisionOperator:
    return _coerce_stage_component(
        operator,
        operator_factory,
        component_type=VisionOperator,
        field_name="operator",
        component_name="operator",
        factory_name="operator_factory",
    )


def _coerce_stage_component(
    component: Any,
    factory: Callable[[dict[str, Any]], Any] | None,
    *,
    component_type: type,
    field_name: str,
    component_name: str,
    factory_name: str,
) -> Any:
    if component is None:
        raise ValueError(f"Vision pipeline {component_name} stage requires '{field_name}'.")
    if isinstance(component, component_type):
        return component
    if factory is None:
        raise ValueError(
            f"Vision pipeline {component_name} stage requires a {factory_name}."
        )
    if not isinstance(component, Mapping):
        raise TypeError(
            f"Vision pipeline {component_name} config must be a {component_type.__name__} or mapping."
        )
    return factory(dict(component))


def _coerce_region(
    value: Any,
    *,
    field_name: str,
) -> BoundingBox | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"Vision pipeline stage field '{field_name}' must be a mapping.")
    try:
        return BoundingBox(
            x1=int(value["x1"]),
            y1=int(value["y1"]),
            x2=int(value["x2"]),
            y2=int(value["y2"]),
        )
    except KeyError as exc:
        raise ValueError(
            f"Vision pipeline stage field '{field_name}' requires x1, y1, x2, y2."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Vision pipeline stage field '{field_name}' coordinates must be integers."
        ) from exc


_DEFAULT_STAGE_REGISTRY = build_vision_stage_registry()
