from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from autoscene.vision.detectors.cascade_detector import CascadeDetector
from autoscene.vision.detectors.mock_detector import MockDetector
from autoscene.vision.detectors.opencv_color_detector import OpenCVColorDetector
from autoscene.vision.detectors.opencv_template_detector import (
    OpenCVTemplateDetector,
)
from autoscene.vision.detectors.yolo_detector import YoloDetector
from autoscene.vision.interfaces import (
    ComparatorAdapter,
    Detector,
    MatcherAdapter,
    OCREngine,
    ReaderAdapter,
    VisionOperator,
)
from autoscene.vision.ocr.mock_ocr import MockOCREngine
from autoscene.vision.ocr.tesseract_ocr import TesseractOCREngine
from autoscene.vision.omni.omniparser_detector import OmniParserDetector
from autoscene.vision.pipeline import (
    ScopedVisionStageRegistry,
    VisionPipelineDetector,
    build_vision_stage_registry,
)
from autoscene.vision.opencv.comparators.image_similarity import (
    OpenCVImageSimilarityComparator,
)
from autoscene.vision.opencv.matchers.feature_matcher import (
    OpenCVFeatureMatcher,
)
from autoscene.vision.opencv.matchers.template_matcher_adapter import (
    TemplateMatcherAdapter,
)
from autoscene.vision.opencv.readers.qt_cluster_static_reader import (
    OpenCVQtClusterStaticReader,
)

DetectorFactory = Callable[..., Detector]
OCRFactory = Callable[..., OCREngine]
MatcherAdapterFactory = Callable[..., MatcherAdapter]
ComparatorAdapterFactory = Callable[..., ComparatorAdapter]
ReaderAdapterFactory = Callable[..., ReaderAdapter]
VisionOperatorFactory = Callable[..., VisionOperator]


class DetectorPlugin(Protocol):
    namespace: str | None
    override: bool

    def register_detectors(self, registry: "ScopedDetectorRegistry") -> None: ...


class OCRPlugin(Protocol):
    namespace: str | None
    override: bool

    def register_ocr_engines(self, registry: "ScopedOCRRegistry") -> None: ...


class MatcherPlugin(Protocol):
    namespace: str | None
    override: bool

    def register_matcher_adapters(self, registry: "ScopedMatcherRegistry") -> None: ...


class ComparatorPlugin(Protocol):
    namespace: str | None
    override: bool

    def register_comparator_adapters(
        self,
        registry: "ScopedComparatorRegistry",
    ) -> None: ...


class ReaderPlugin(Protocol):
    namespace: str | None
    override: bool

    def register_reader_adapters(self, registry: "ScopedReaderRegistry") -> None: ...


class OperatorPlugin(Protocol):
    namespace: str | None
    override: bool

    def register_operators(self, registry: "ScopedVisionOperatorRegistry") -> None: ...


class StagePlugin(Protocol):
    namespace: str | None
    override: bool

    def register_pipeline_stages(self, registry: "ScopedVisionStageRegistry") -> None: ...


class VisionPlugin(
    DetectorPlugin,
    OCRPlugin,
    MatcherPlugin,
    ComparatorPlugin,
    ReaderPlugin,
    OperatorPlugin,
    StagePlugin,
    Protocol,
):
    pass


@dataclass(frozen=True)
class _FactoryRegistration:
    name: str
    factory: Callable[..., Any]


@dataclass(frozen=True)
class _PluginRegistryBinding:
    plugin_method_name: str
    bundle_attr_name: str
    scoped_registry_type: type


class _BaseFactoryRegistry:
    registry_name = "factory"

    def __init__(self, bundle: "VisionRegistryBundle") -> None:
        self._bundle = bundle
        self._registrations: dict[str, _FactoryRegistration] = {}

    def clone_for_bundle(self, bundle: "VisionRegistryBundle"):
        cloned = type(self)(bundle)
        cloned._registrations = dict(self._registrations)
        return cloned

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        *,
        namespace: str | None = None,
        override: bool = True,
    ) -> None:
        qualified_name = _qualify_registry_name(name, namespace=namespace)
        if not override and qualified_name in self._registrations:
            raise ValueError(
                f"{self.registry_name.title()} already registered: {qualified_name}"
            )
        self._registrations[qualified_name] = _FactoryRegistration(
            name=qualified_name,
            factory=factory,
        )

    def resolve(self, name: str) -> Callable[..., Any] | None:
        registration = self._registrations.get(str(name).strip().lower())
        if registration is None:
            return None
        return registration.factory

    def create(self, config: dict[str, Any]) -> Any:
        payload = dict(config or {})
        type_name = str(payload.pop("type", "")).strip().lower()
        if not type_name:
            raise ValueError(
                f"{self.registry_name} config requires a non-empty 'type' field."
            )
        factory = self.resolve(type_name)
        if factory is None:
            available = ", ".join(sorted(self._registrations))
            raise ValueError(
                f"Unknown {self.registry_name} '{type_name}'. Available: {available}"
            )
        return _invoke_factory(factory, payload, registry_bundle=self._bundle)


class DetectorRegistry(_BaseFactoryRegistry):
    registry_name = "detector"


class OCREngineRegistry(_BaseFactoryRegistry):
    registry_name = "OCR engine"


class MatcherAdapterRegistry(_BaseFactoryRegistry):
    registry_name = "matcher adapter"


class ComparatorAdapterRegistry(_BaseFactoryRegistry):
    registry_name = "comparator adapter"


class ReaderAdapterRegistry(_BaseFactoryRegistry):
    registry_name = "reader adapter"


class VisionOperatorRegistry(_BaseFactoryRegistry):
    registry_name = "vision operator"


class _ScopedFactoryRegistry:
    def __init__(
        self,
        registry: _BaseFactoryRegistry,
        *,
        namespace: str | None = None,
        override: bool = False,
    ) -> None:
        self._registry = registry
        self._namespace = _normalize_namespace(namespace)
        self._override = bool(override)

    def register(self, name: str, factory: Callable[..., Any], **kwargs: Any) -> None:
        kwargs.setdefault("namespace", self._namespace)
        kwargs.setdefault("override", self._override)
        self._registry.register(name, factory, **kwargs)


class ScopedDetectorRegistry(_ScopedFactoryRegistry):
    pass


class ScopedOCRRegistry(_ScopedFactoryRegistry):
    pass


class ScopedMatcherRegistry(_ScopedFactoryRegistry):
    pass


class ScopedComparatorRegistry(_ScopedFactoryRegistry):
    pass


class ScopedReaderRegistry(_ScopedFactoryRegistry):
    pass


class ScopedVisionOperatorRegistry(_ScopedFactoryRegistry):
    pass


class VisionRegistryBundle:
    def __init__(self) -> None:
        self.detectors = DetectorRegistry(self)
        self.ocr_engines = OCREngineRegistry(self)
        self.matcher_adapters = MatcherAdapterRegistry(self)
        self.comparator_adapters = ComparatorAdapterRegistry(self)
        self.reader_adapters = ReaderAdapterRegistry(self)
        self.operators = VisionOperatorRegistry(self)
        self.pipeline_stages = build_vision_stage_registry()

    def clone(self) -> "VisionRegistryBundle":
        cloned = VisionRegistryBundle()
        cloned.detectors = self.detectors.clone_for_bundle(cloned)
        cloned.ocr_engines = self.ocr_engines.clone_for_bundle(cloned)
        cloned.matcher_adapters = self.matcher_adapters.clone_for_bundle(cloned)
        cloned.comparator_adapters = self.comparator_adapters.clone_for_bundle(cloned)
        cloned.reader_adapters = self.reader_adapters.clone_for_bundle(cloned)
        cloned.operators = self.operators.clone_for_bundle(cloned)
        cloned.pipeline_stages = self.pipeline_stages.clone()
        return cloned

    def create_detector(self, config: dict[str, Any]) -> Detector:
        return self.detectors.create(config)

    def create_ocr_engine(self, config: dict[str, Any]) -> OCREngine:
        return self.ocr_engines.create(config)

    def create_matcher_adapter(self, config: dict[str, Any]) -> MatcherAdapter:
        return self.matcher_adapters.create(config)

    def create_comparator_adapter(self, config: dict[str, Any]) -> ComparatorAdapter:
        return self.comparator_adapters.create(config)

    def create_reader_adapter(self, config: dict[str, Any]) -> ReaderAdapter:
        return self.reader_adapters.create(config)

    def create_operator(self, config: dict[str, Any]) -> VisionOperator:
        return self.operators.create(config)


_PLUGIN_REGISTRY_BINDINGS = (
    _PluginRegistryBinding("register_detectors", "detectors", ScopedDetectorRegistry),
    _PluginRegistryBinding("register_ocr_engines", "ocr_engines", ScopedOCRRegistry),
    _PluginRegistryBinding("register_matcher_adapters", "matcher_adapters", ScopedMatcherRegistry),
    _PluginRegistryBinding(
        "register_comparator_adapters",
        "comparator_adapters",
        ScopedComparatorRegistry,
    ),
    _PluginRegistryBinding("register_reader_adapters", "reader_adapters", ScopedReaderRegistry),
    _PluginRegistryBinding("register_operators", "operators", ScopedVisionOperatorRegistry),
    _PluginRegistryBinding(
        "register_pipeline_stages",
        "pipeline_stages",
        ScopedVisionStageRegistry,
    ),
)


def build_vision_registry_bundle(
    plugins: tuple[object, ...] | list[object] | None = None,
) -> VisionRegistryBundle:
    bundle = VisionRegistryBundle()
    _register_builtin_factories(bundle)
    install_vision_plugins(bundle, plugins)
    return bundle


def install_vision_plugins(
    bundle: VisionRegistryBundle,
    plugins: tuple[object, ...] | list[object] | None,
) -> VisionRegistryBundle:
    for plugin in tuple(plugins or ()):
        namespace = _normalize_namespace(getattr(plugin, "namespace", None))
        override = bool(getattr(plugin, "override", False))
        for binding in _PLUGIN_REGISTRY_BINDINGS:
            _install_plugin_registry_binding(
                bundle=bundle,
                plugin=plugin,
                binding=binding,
                namespace=namespace,
                override=override,
            )
    return bundle


def resolve_vision_registry_bundle(
    *,
    plugins: tuple[object, ...] | list[object] | None = None,
    registry_bundle: VisionRegistryBundle | None = None,
) -> VisionRegistryBundle:
    if registry_bundle is None:
        if not plugins:
            return _DEFAULT_BUNDLE
        registry_bundle = _DEFAULT_BUNDLE.clone()
    elif plugins:
        registry_bundle = registry_bundle.clone()

    if plugins:
        install_vision_plugins(registry_bundle, plugins)
    return registry_bundle


def register_detector(
    name: str,
    factory: DetectorFactory,
    *,
    namespace: str | None = None,
    override: bool = True,
) -> None:
    _DEFAULT_BUNDLE.detectors.register(
        name,
        factory,
        namespace=namespace,
        override=override,
    )


def register_ocr_engine(
    name: str,
    factory: OCRFactory,
    *,
    namespace: str | None = None,
    override: bool = True,
) -> None:
    _DEFAULT_BUNDLE.ocr_engines.register(
        name,
        factory,
        namespace=namespace,
        override=override,
    )


def register_matcher_adapter(
    name: str,
    factory: MatcherAdapterFactory,
    *,
    namespace: str | None = None,
    override: bool = True,
) -> None:
    _DEFAULT_BUNDLE.matcher_adapters.register(
        name,
        factory,
        namespace=namespace,
        override=override,
    )


def register_comparator_adapter(
    name: str,
    factory: ComparatorAdapterFactory,
    *,
    namespace: str | None = None,
    override: bool = True,
) -> None:
    _DEFAULT_BUNDLE.comparator_adapters.register(
        name,
        factory,
        namespace=namespace,
        override=override,
    )


def register_reader_adapter(
    name: str,
    factory: ReaderAdapterFactory,
    *,
    namespace: str | None = None,
    override: bool = True,
) -> None:
    _DEFAULT_BUNDLE.reader_adapters.register(
        name,
        factory,
        namespace=namespace,
        override=override,
    )


def register_operator(
    name: str,
    factory: VisionOperatorFactory,
    *,
    namespace: str | None = None,
    override: bool = True,
) -> None:
    _DEFAULT_BUNDLE.operators.register(
        name,
        factory,
        namespace=namespace,
        override=override,
    )


def create_detector(
    config: dict[str, Any],
    *,
    plugins: tuple[object, ...] | list[object] | None = None,
    registry_bundle: VisionRegistryBundle | None = None,
) -> Detector:
    return _create_bundle_component(
        config,
        creator_name="create_detector",
        plugins=plugins,
        registry_bundle=registry_bundle,
    )


def create_ocr_engine(
    config: dict[str, Any],
    *,
    plugins: tuple[object, ...] | list[object] | None = None,
    registry_bundle: VisionRegistryBundle | None = None,
) -> OCREngine:
    return _create_bundle_component(
        config,
        creator_name="create_ocr_engine",
        plugins=plugins,
        registry_bundle=registry_bundle,
    )


def create_matcher_adapter(
    config: dict[str, Any],
    *,
    plugins: tuple[object, ...] | list[object] | None = None,
    registry_bundle: VisionRegistryBundle | None = None,
) -> MatcherAdapter:
    return _create_bundle_component(
        config,
        creator_name="create_matcher_adapter",
        plugins=plugins,
        registry_bundle=registry_bundle,
    )


def create_comparator_adapter(
    config: dict[str, Any],
    *,
    plugins: tuple[object, ...] | list[object] | None = None,
    registry_bundle: VisionRegistryBundle | None = None,
) -> ComparatorAdapter:
    return _create_bundle_component(
        config,
        creator_name="create_comparator_adapter",
        plugins=plugins,
        registry_bundle=registry_bundle,
    )


def create_reader_adapter(
    config: dict[str, Any],
    *,
    plugins: tuple[object, ...] | list[object] | None = None,
    registry_bundle: VisionRegistryBundle | None = None,
) -> ReaderAdapter:
    return _create_bundle_component(
        config,
        creator_name="create_reader_adapter",
        plugins=plugins,
        registry_bundle=registry_bundle,
    )


def create_operator(
    config: dict[str, Any],
    *,
    plugins: tuple[object, ...] | list[object] | None = None,
    registry_bundle: VisionRegistryBundle | None = None,
) -> VisionOperator:
    return _create_bundle_component(
        config,
        creator_name="create_operator",
        plugins=plugins,
        registry_bundle=registry_bundle,
    )


def _register_builtin_factories(bundle: VisionRegistryBundle) -> None:
    _register_bundle_factories(
        bundle.detectors,
        (
            ("cascade", CascadeDetector),
            ("mock", MockDetector),
            ("omniparser", OmniParserDetector),
            ("opencv_color", OpenCVColorDetector),
            ("opencv_template", OpenCVTemplateDetector),
            ("pipeline", VisionPipelineDetector),
            ("yolo", YoloDetector),
        ),
    )
    _register_bundle_factories(
        bundle.ocr_engines,
        (
            ("mock", MockOCREngine),
            ("tesseract", TesseractOCREngine),
        ),
    )
    _register_bundle_factories(
        bundle.matcher_adapters,
        (
            ("opencv_feature", OpenCVFeatureMatcher),
            ("template_matcher", TemplateMatcherAdapter),
        ),
    )
    _register_bundle_factories(
        bundle.comparator_adapters,
        (("opencv_image_similarity", OpenCVImageSimilarityComparator),),
    )
    _register_bundle_factories(
        bundle.reader_adapters,
        (("opencv_qt_cluster_static", OpenCVQtClusterStaticReader),),
    )


def _register_bundle_factories(
    registry: _BaseFactoryRegistry,
    registrations: tuple[tuple[str, Callable[..., Any]], ...],
) -> None:
    for name, factory in registrations:
        registry.register(name, factory)


def _install_plugin_registry_binding(
    *,
    bundle: VisionRegistryBundle,
    plugin: object,
    binding: _PluginRegistryBinding,
    namespace: str | None,
    override: bool,
) -> None:
    register_method = getattr(plugin, binding.plugin_method_name, None)
    if not callable(register_method):
        return
    registry = getattr(bundle, binding.bundle_attr_name)
    register_method(
        binding.scoped_registry_type(
            registry,
            namespace=namespace,
            override=override,
        )
    )


def _create_bundle_component(
    config: dict[str, Any],
    *,
    creator_name: str,
    plugins: tuple[object, ...] | list[object] | None,
    registry_bundle: VisionRegistryBundle | None,
):
    bundle = resolve_vision_registry_bundle(
        plugins=plugins,
        registry_bundle=registry_bundle,
    )
    return getattr(bundle, creator_name)(config)


def _normalize_namespace(namespace: str | None) -> str | None:
    if namespace is None:
        return None
    normalized = str(namespace).strip().lower().strip(".")
    return normalized or None


def _qualify_registry_name(name: str, *, namespace: str | None = None) -> str:
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("Vision registry name must not be empty.")
    normalized_namespace = _normalize_namespace(namespace)
    if normalized_namespace is None:
        return normalized_name
    return f"{normalized_namespace}.{normalized_name}"


def _invoke_factory(
    factory: Callable[..., Any],
    config: dict[str, Any],
    *,
    registry_bundle: VisionRegistryBundle,
) -> Any:
    kwargs = dict(config)
    extras = {
        "registry_bundle": registry_bundle,
        "detector_factory": registry_bundle.create_detector,
        "ocr_engine_factory": registry_bundle.create_ocr_engine,
        "matcher_factory": registry_bundle.create_matcher_adapter,
        "comparator_factory": registry_bundle.create_comparator_adapter,
        "reader_factory": registry_bundle.create_reader_adapter,
        "operator_factory": registry_bundle.create_operator,
    }
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return factory(**{**kwargs, **extras})

    filtered_kwargs = dict(kwargs)
    for name, value in extras.items():
        if name in signature.parameters:
            filtered_kwargs[name] = value
    return factory(**filtered_kwargs)


_DEFAULT_BUNDLE = build_vision_registry_bundle()
