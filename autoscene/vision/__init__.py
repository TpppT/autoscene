from autoscene.vision.interfaces import (
    ComparatorAdapter,
    Detector,
    MatcherAdapter,
    OCREngine,
    ReaderAdapter,
    VisionOperator,
)
from autoscene.vision.models import (
    CompareResult,
    MatchResult,
    ReadResult,
    VisionNode,
    VisionNodeTraceEntry,
    VisionOperatorOutput,
)
from autoscene.vision.pipeline import (
    ComparatorFilterStage,
    DetectorRefinementStage,
    DetectorRegionStage,
    filter_object_locate_nodes,
    MatcherClassificationStage,
    NodeFilterStage,
    OperatorStage,
    ObjectLocatePipelineResult,
    OCRClassificationStage,
    ReaderClassificationStage,
    ScopedVisionStageRegistry,
    TextLocatePipelineResult,
    TextLocateStage,
    VisionPipelineDetector,
    VisionPipeline,
    VisionStageBuildContext,
    VisionStageRegistry,
    build_vision_pipeline,
    build_vision_stage_registry,
    register_vision_stage,
    VisionPipelineContext,
    VisionPipelineResult,
    VisionPipelineStage,
    VisionPipelineTraceEntry,
    run_object_locate_pipeline,
    run_text_locate_pipeline,
)


def create_detector(config, **kwargs):
    from autoscene.vision.registry import create_detector as _create_detector

    return _create_detector(config, **kwargs)


def create_ocr_engine(config, **kwargs):
    from autoscene.vision.registry import create_ocr_engine as _create_ocr_engine

    return _create_ocr_engine(config, **kwargs)


def create_matcher_adapter(config, **kwargs):
    from autoscene.vision.registry import (
        create_matcher_adapter as _create_matcher_adapter,
    )

    return _create_matcher_adapter(config, **kwargs)


def create_comparator_adapter(config, **kwargs):
    from autoscene.vision.registry import (
        create_comparator_adapter as _create_comparator_adapter,
    )

    return _create_comparator_adapter(config, **kwargs)


def create_reader_adapter(config, **kwargs):
    from autoscene.vision.registry import create_reader_adapter as _create_reader_adapter

    return _create_reader_adapter(config, **kwargs)


def create_operator(config, **kwargs):
    from autoscene.vision.registry import create_operator as _create_operator

    return _create_operator(config, **kwargs)


def register_detector(name, factory, **kwargs):
    from autoscene.vision.registry import register_detector as _register_detector

    _register_detector(name, factory, **kwargs)


def register_ocr_engine(name, factory, **kwargs):
    from autoscene.vision.registry import register_ocr_engine as _register_ocr_engine

    _register_ocr_engine(name, factory, **kwargs)


def register_matcher_adapter(name, factory, **kwargs):
    from autoscene.vision.registry import (
        register_matcher_adapter as _register_matcher_adapter,
    )

    _register_matcher_adapter(name, factory, **kwargs)


def register_comparator_adapter(name, factory, **kwargs):
    from autoscene.vision.registry import (
        register_comparator_adapter as _register_comparator_adapter,
    )

    _register_comparator_adapter(name, factory, **kwargs)


def register_reader_adapter(name, factory, **kwargs):
    from autoscene.vision.registry import (
        register_reader_adapter as _register_reader_adapter,
    )

    _register_reader_adapter(name, factory, **kwargs)


def register_operator(name, factory, **kwargs):
    from autoscene.vision.registry import register_operator as _register_operator

    _register_operator(name, factory, **kwargs)


def build_vision_registry_bundle(plugins=None):
    from autoscene.vision.registry import (
        build_vision_registry_bundle as _build_vision_registry_bundle,
    )

    return _build_vision_registry_bundle(plugins=plugins)


def install_vision_plugins(bundle, plugins):
    from autoscene.vision.registry import (
        install_vision_plugins as _install_vision_plugins,
    )

    return _install_vision_plugins(bundle, plugins)


__all__ = [
    "CompareResult",
    "ComparatorAdapter",
    "Detector",
    "MatchResult",
    "MatcherAdapter",
    "OCREngine",
    "ReadResult",
    "ReaderAdapter",
    "VisionNode",
    "VisionOperator",
    "VisionOperatorOutput",
    "VisionNodeTraceEntry",
    "create_detector",
    "create_ocr_engine",
    "create_matcher_adapter",
    "create_comparator_adapter",
    "create_reader_adapter",
    "create_operator",
    "register_detector",
    "register_ocr_engine",
    "register_matcher_adapter",
    "register_comparator_adapter",
    "register_reader_adapter",
    "register_operator",
    "build_vision_registry_bundle",
    "install_vision_plugins",
    "DetectorRefinementStage",
    "DetectorRegionStage",
    "ComparatorFilterStage",
    "MatcherClassificationStage",
    "NodeFilterStage",
    "OperatorStage",
    "ObjectLocatePipelineResult",
    "OCRClassificationStage",
    "ReaderClassificationStage",
    "ScopedVisionStageRegistry",
    "TextLocatePipelineResult",
    "TextLocateStage",
    "VisionPipeline",
    "VisionPipelineDetector",
    "VisionPipelineContext",
    "VisionPipelineResult",
    "VisionPipelineStage",
    "VisionStageBuildContext",
    "VisionStageRegistry",
    "VisionPipelineTraceEntry",
    "build_vision_pipeline",
    "build_vision_stage_registry",
    "filter_object_locate_nodes",
    "run_object_locate_pipeline",
    "run_text_locate_pipeline",
    "register_vision_stage",
]
