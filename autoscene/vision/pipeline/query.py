from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from autoscene.core.models import BoundingBox, OCRText, ObjectLocateSpec, TextLocateSpec
from autoscene.vision.interfaces import Detector, OCREngine
from autoscene.vision.models import VisionNode, VisionNodeTraceEntry

from .core import VisionPipelineContext, VisionPipelineResult
from .stages import DetectorRegionStage, NodeFilterStage, TextLocateStage
from .utils import _backend_name, box_center_in_region, find_ocr_text_match, normalize_ocr_text


@dataclass(frozen=True)
class ObjectLocatePipelineResult:
    pipeline_result: VisionPipelineResult
    region: BoundingBox | None = None

    @property
    def nodes(self):
        return self.pipeline_result.nodes

    @property
    def boxes(self) -> list[BoundingBox]:
        return self.pipeline_result.boxes


@dataclass(frozen=True)
class TextLocatePipelineResult:
    pipeline_result: VisionPipelineResult
    region: BoundingBox | None = None

    @property
    def nodes(self):
        return self.pipeline_result.nodes

    @property
    def boxes(self) -> list[BoundingBox]:
        return self.pipeline_result.boxes

    @property
    def match(self) -> OCRText | None:
        if not self.pipeline_result.nodes:
            return None
        node = self.pipeline_result.nodes[0]
        if node.region is None:
            return None
        return OCRText(
            text=node.text,
            bbox=node.to_bounding_box(),
            score=float(node.score),
        )


def run_object_locate_pipeline(
    image: Any,
    *,
    detector: Detector,
    locate: ObjectLocateSpec,
    metadata: Mapping[str, Any] | None = None,
) -> ObjectLocatePipelineResult:
    detector_stage = DetectorRegionStage(
        detector,
        labels=[locate.label],
        name="locate_detector",
    )
    filter_stage = NodeFilterStage(
        min_score=locate.min_score,
        region=locate.region,
        name="locate_filter",
    )
    payload = dict(metadata or {})
    payload["locate"] = locate.to_payload()
    context = VisionPipelineContext(
        allowed_labels=(str(locate.label),),
        metadata=payload,
    )
    nodes = detector_stage.execute(image, [], context)
    nodes = filter_stage.execute(image, nodes, context)
    result = VisionPipelineResult(nodes=list(nodes), trace=list(context.trace))
    return ObjectLocatePipelineResult(
        pipeline_result=result,
        region=filter_stage.last_region,
    )


def filter_object_locate_nodes(
    image: Any,
    *,
    locate: ObjectLocateSpec,
    boxes: Sequence[BoundingBox] | None = None,
    nodes: Sequence[VisionNode] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ObjectLocatePipelineResult:
    input_nodes = _coerce_object_nodes(boxes=boxes, nodes=nodes)
    filter_stage = NodeFilterStage(
        labels=[locate.label],
        min_score=locate.min_score,
        region=locate.region,
        name="locate_filter",
    )
    payload = dict(metadata or {})
    payload["locate"] = locate.to_payload()
    context = VisionPipelineContext(
        allowed_labels=(str(locate.label),),
        metadata=payload,
    )
    if not _supports_region_clips(image):
        return _run_object_filter_without_image_size(
            input_nodes,
            locate=locate,
            context=context,
        )
    filtered_nodes = filter_stage.execute(image, input_nodes, context)
    result = VisionPipelineResult(nodes=list(filtered_nodes), trace=list(context.trace))
    return ObjectLocatePipelineResult(
        pipeline_result=result,
        region=filter_stage.last_region,
    )


def run_text_locate_pipeline(
    image: Any,
    *,
    ocr_engine: OCREngine,
    locate: TextLocateSpec,
    metadata: Mapping[str, Any] | None = None,
) -> TextLocatePipelineResult:
    locate_stage = TextLocateStage(
        ocr_engine,
        query=locate.text,
        exact=locate.exact,
        name="locate_text",
    )
    payload = dict(metadata or {})
    payload["locate"] = locate.to_payload()
    context = VisionPipelineContext(
        allowed_labels=(str(locate.text),),
        metadata=payload,
    )
    if not _supports_region_crops(image):
        return _run_text_locate_without_crops(
            image,
            ocr_engine=ocr_engine,
            locate=locate,
            context=context,
        )
    seed_nodes: list[VisionNode] = []
    if locate.region is not None:
        seed_nodes.append(
            VisionNode(
                region=locate.region,
                score=1.0,
                source="locate_region",
                metadata={"synthetic_region": True},
            )
        )
    nodes = locate_stage.execute(image, seed_nodes, context)
    result = VisionPipelineResult(nodes=list(nodes), trace=list(context.trace))
    return TextLocatePipelineResult(
        pipeline_result=result,
        region=locate.region,
    )


def _supports_region_crops(image: Any) -> bool:
    return hasattr(image, "size") and hasattr(image, "crop")


def _supports_region_clips(image: Any) -> bool:
    return hasattr(image, "size")


def _coerce_object_nodes(
    *,
    boxes: Sequence[BoundingBox] | None,
    nodes: Sequence[VisionNode] | None,
) -> list[VisionNode]:
    if nodes is not None:
        output: list[VisionNode] = []
        for node in nodes:
            output.append(
                VisionNode(
                    region=node.region,
                    label=str(node.label),
                    score=float(node.score),
                    text=str(node.text),
                    value=node.value,
                    unit=str(node.unit),
                    source=str(node.source),
                    metadata=dict(node.metadata),
                    trace=list(node.trace),
                )
            )
        return output
    output: list[VisionNode] = []
    for box in boxes or ():
        output.append(
            VisionNode(
                region=box,
                label=str(box.label),
                score=float(box.score),
                source="locate_detector",
            )
        )
    return output


def _run_object_filter_without_image_size(
    nodes: Sequence[VisionNode],
    *,
    locate: ObjectLocateSpec,
    context: VisionPipelineContext,
) -> ObjectLocatePipelineResult:
    filtered_nodes: list[VisionNode] = []
    for node in nodes:
        if node.region is None:
            continue
        if str(node.label) != str(locate.label):
            continue
        if float(node.score) < float(locate.min_score):
            continue
        if locate.region is not None and not box_center_in_region(node.to_bounding_box(), locate.region):
            continue
        node.trace.append(
            VisionNodeTraceEntry(
                stage_name="locate_filter",
                stage_kind="node_filter",
                label=str(node.label),
                score=float(node.score),
                source=str(node.source),
                metadata=dict(node.metadata),
            )
        )
        filtered_nodes.append(node)
    context.record_stage(
        stage_name="locate_filter",
        stage_kind="node_filter",
        input_count=len(nodes),
        output_count=len(filtered_nodes),
        metadata={
            "labels": [str(locate.label)],
            "min_score": float(locate.min_score),
            "region": None if locate.region is None else locate.region.to_payload(),
        },
    )
    return ObjectLocatePipelineResult(
        pipeline_result=VisionPipelineResult(
            nodes=list(filtered_nodes),
            trace=list(context.trace),
        ),
        region=locate.region,
    )


def _run_text_locate_without_crops(
    image: Any,
    *,
    ocr_engine: OCREngine,
    locate: TextLocateSpec,
    context: VisionPipelineContext,
) -> TextLocatePipelineResult:
    texts = list(ocr_engine.read(image))
    normalized_query = normalize_ocr_text(locate.text)
    matched = find_ocr_text_match(
        texts,
        locate.text,
        exact=locate.exact,
        normalized_text=normalized_query,
    )
    if matched is not None and locate.region is not None:
        if not box_center_in_region(matched.bbox, locate.region):
            matched = None
    nodes: list[VisionNode] = []
    if matched is not None:
        node = VisionNode(
            region=matched.bbox,
            label=locate.text,
            score=float(matched.score),
            text=matched.text,
            source=_backend_name(ocr_engine),
            metadata={
                "ocr_hits": len(texts),
                "query": locate.text,
                "exact": locate.exact,
            },
        )
        node.trace.append(
            VisionNodeTraceEntry(
                stage_name="locate_text",
                stage_kind="text_locate",
                label=str(node.label),
                score=float(node.score),
                source=str(node.source),
                metadata=dict(node.metadata),
            )
        )
        nodes.append(node)
    context.record_stage(
        stage_name="locate_text",
        stage_kind="text_locate",
        input_count=0 if locate.region is None else 1,
        output_count=len(nodes),
        metadata={
            "query": locate.text,
            "exact": locate.exact,
        },
    )
    return TextLocatePipelineResult(
        pipeline_result=VisionPipelineResult(nodes=nodes, trace=list(context.trace)),
        region=locate.region,
    )
