from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
from autoscene.vision.models import VisionNode, VisionNodeTraceEntry

VisionStageBuilder = Callable[[dict[str, Any], "VisionStageBuildContext"], "VisionPipelineStage"]


@dataclass(frozen=True)
class VisionStageBuildContext:
    detector_factory: Callable[[dict[str, Any]], Detector] | None = None
    ocr_engine_factory: Callable[[dict[str, Any]], OCREngine] | None = None
    matcher_factory: Callable[[dict[str, Any]], MatcherAdapter] | None = None
    comparator_factory: Callable[[dict[str, Any]], ComparatorAdapter] | None = None
    reader_factory: Callable[[dict[str, Any]], ReaderAdapter] | None = None
    operator_factory: Callable[[dict[str, Any]], VisionOperator] | None = None
    stage_registry: "VisionStageRegistry | None" = None


@dataclass(frozen=True)
class _StageRegistration:
    name: str
    builder: VisionStageBuilder


class VisionStageRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, _StageRegistration] = {}

    def clone(self) -> "VisionStageRegistry":
        cloned = VisionStageRegistry()
        cloned._registrations = dict(self._registrations)
        return cloned

    def register(
        self,
        name: str,
        builder: VisionStageBuilder,
        *,
        namespace: str | None = None,
        override: bool = True,
    ) -> None:
        qualified_name = _qualify_stage_name(name, namespace=namespace)
        if not override and qualified_name in self._registrations:
            raise ValueError(f"Vision stage already registered: {qualified_name}")
        self._registrations[qualified_name] = _StageRegistration(
            name=qualified_name,
            builder=builder,
        )

    def resolve(self, name: str) -> VisionStageBuilder | None:
        registration = self._registrations.get(str(name).strip().lower())
        if registration is None:
            return None
        return registration.builder

    def create(
        self,
        config: dict[str, Any],
        *,
        build_context: VisionStageBuildContext,
    ) -> "VisionPipelineStage":
        payload = dict(config or {})
        stage_type = str(payload.pop("type", "")).strip().lower()
        if not stage_type:
            raise ValueError("Vision pipeline stage config requires a non-empty 'type'.")
        builder = self.resolve(stage_type)
        if builder is None:
            available = ", ".join(sorted(self._registrations))
            raise ValueError(
                f"Unknown vision pipeline stage '{stage_type}'. Available: {available}"
            )
        return builder(payload, build_context)


class ScopedVisionStageRegistry:
    def __init__(
        self,
        registry: VisionStageRegistry,
        *,
        namespace: str | None = None,
        override: bool = False,
    ) -> None:
        self._registry = registry
        self._namespace = _normalize_stage_namespace(namespace)
        self._override = bool(override)

    def register(self, name: str, builder: VisionStageBuilder, **kwargs: Any) -> None:
        kwargs.setdefault("namespace", self._namespace)
        kwargs.setdefault("override", self._override)
        self._registry.register(name, builder, **kwargs)


@dataclass
class VisionPipelineTraceEntry:
    stage_name: str
    stage_kind: str
    input_count: int
    output_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionPipelineContext:
    allowed_labels: tuple[str, ...] | None = None
    trace: list[VisionPipelineTraceEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _allowed_label_set: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._allowed_label_set = frozenset(self.allowed_labels or ())

    def record_stage(
        self,
        *,
        stage_name: str,
        stage_kind: str,
        input_count: int,
        output_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.trace.append(
            VisionPipelineTraceEntry(
                stage_name=stage_name,
                stage_kind=stage_kind,
                input_count=int(input_count),
                output_count=int(output_count),
                metadata=dict(metadata or {}),
            )
        )

    def allows_label(self, label: str) -> bool:
        if not self._allowed_label_set:
            return True
        return label in self._allowed_label_set


@dataclass
class VisionPipelineResult:
    nodes: list[VisionNode]
    trace: list[VisionPipelineTraceEntry] = field(default_factory=list)

    @property
    def boxes(self) -> list[BoundingBox]:
        return [node.to_bounding_box() for node in self.nodes if node.region is not None]


class VisionPipelineStage(ABC):
    stage_kind = "stage"

    def __init__(self, *, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    def execute(
        self,
        image: Any,
        nodes: list[VisionNode] | tuple[VisionNode, ...],
        context: VisionPipelineContext,
    ) -> list[VisionNode]:
        inputs = list(nodes)
        outputs = list(self.run(image=image, nodes=inputs, context=context))
        for node in outputs:
            _append_node_trace(
                node,
                stage_name=self.name,
                stage_kind=self.stage_kind,
            )
        context.record_stage(
            stage_name=self.name,
            stage_kind=self.stage_kind,
            input_count=len(inputs),
            output_count=len(outputs),
            metadata=self.describe(context),
        )
        return outputs

    @abstractmethod
    def run(
        self,
        *,
        image: Any,
        nodes: list[VisionNode] | tuple[VisionNode, ...],
        context: VisionPipelineContext,
    ) -> list[VisionNode]:
        raise NotImplementedError

    def describe(self, context: VisionPipelineContext) -> dict[str, Any]:
        return {}


def _append_node_trace(
    node: VisionNode,
    *,
    stage_name: str,
    stage_kind: str,
) -> None:
    node.trace.append(
        VisionNodeTraceEntry(
            stage_name=stage_name,
            stage_kind=stage_kind,
            label=str(node.label),
            score=float(node.score),
            source=str(node.source),
            metadata=dict(node.metadata),
        )
    )


def _normalize_stage_namespace(namespace: str | None) -> str | None:
    if namespace is None:
        return None
    normalized = str(namespace).strip().lower().strip(".")
    return normalized or None


def _qualify_stage_name(name: str, *, namespace: str | None = None) -> str:
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("Vision stage name must not be empty.")
    normalized_namespace = _normalize_stage_namespace(namespace)
    if normalized_namespace is None:
        return normalized_name
    return f"{normalized_namespace}.{normalized_name}"
