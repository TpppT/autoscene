from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from autoscene.core.models import BoundingBox, OCRText
from autoscene.vision.interfaces import (
    ComparatorAdapter,
    Detector,
    MatcherAdapter,
    OCREngine,
    ReaderAdapter,
    VisionOperator,
)
from autoscene.vision.models import VisionNode, VisionOperatorOutput

from .core import VisionPipelineContext, VisionPipelineStage
from .utils import (
    _backend_name,
    _candidate_nodes,
    _derive_node,
    _is_full_image_region,
    _iter_prepared_region_nodes,
    _new_node,
    _normalize_label_source,
    _normalize_labels,
    _normalize_match_mode,
    _normalize_text,
    _reader_label_candidates,
    box_center_in_region,
    clip_region,
    find_ocr_text_match,
    normalize_ocr_text,
    translate_box,
)


class DetectorRegionStage(VisionPipelineStage):
    stage_kind = "detector_region"

    def __init__(
        self,
        detector: Detector,
        *,
        labels: Sequence[str] | None = None,
        max_regions: int | None = None,
        name: str = "region_detector",
    ) -> None:
        super().__init__(name=name)
        self._detector = detector
        self._labels = _normalize_labels(labels)
        self._max_regions = None if max_regions is None else max(int(max_regions), 1)

    def run(
        self,
        *,
        image: Any,
        nodes: Sequence[VisionNode],
        context: VisionPipelineContext,
    ) -> list[VisionNode]:
        del nodes, context
        results = list(self._detector.detect(image, labels=self._labels))
        detector_nodes = self._extract_detector_nodes(results)
        if detector_nodes is not None:
            if self._max_regions is not None:
                detector_nodes = detector_nodes[: self._max_regions]
            return detector_nodes
        if self._max_regions is not None:
            results = results[: self._max_regions]
        detector_backend = _backend_name(self._detector)
        return [
            _new_node(
                region=box,
                label=box.label,
                score=float(box.score),
                source=detector_backend,
            )
            for box in results
        ]

    def _extract_detector_nodes(
        self,
        boxes: Sequence[BoundingBox],
    ) -> list[VisionNode] | None:
        last_result = getattr(self._detector, "last_pipeline_result", None)
        nodes = getattr(last_result, "nodes", None)
        if not isinstance(nodes, (list, tuple)):
            return None
        region_nodes = [
            node
            for node in nodes
            if isinstance(node, VisionNode) and node.region is not None
        ]
        if len(region_nodes) != len(boxes):
            return None
        for node, box in zip(region_nodes, boxes):
            node_box = node.to_bounding_box()
            if (
                int(node_box.x1) != int(box.x1)
                or int(node_box.y1) != int(box.y1)
                or int(node_box.x2) != int(box.x2)
                or int(node_box.y2) != int(box.y2)
                or str(node.label) != str(box.label)
            ):
                return None
        return [_derive_node(node) for node in region_nodes]

    def describe(self, context: VisionPipelineContext) -> dict[str, Any]:
        del context
        return {
            "labels": list(self._labels) if self._labels is not None else [],
            "max_regions": self._max_regions,
        }


class DetectorRefinementStage(VisionPipelineStage):
    stage_kind = "detector_refinement"

    def __init__(
        self,
        detector: Detector,
        *,
        labels: Sequence[str] | None = None,
        name: str = "detail_detector",
    ) -> None:
        super().__init__(name=name)
        self._detector = detector
        self._labels = _normalize_labels(labels)

    def run(
        self,
        *,
        image: Any,
        nodes: Sequence[VisionNode],
        context: VisionPipelineContext,
    ) -> list[VisionNode]:
        nested_labels = self._labels or context.allowed_labels
        allows_label = context.allows_label
        detector_backend = _backend_name(self._detector)
        output: list[VisionNode] = []
        for prepared_node in _iter_prepared_region_nodes(
            image,
            nodes,
            include_crop=True,
        ):
            region_node = prepared_node.node
            clamped_region = prepared_node.region
            nested_boxes = self._detector.detect(
                prepared_node.require_crop(),
                labels=nested_labels,
            )
            for box in nested_boxes:
                translated = translate_box(
                    box,
                    offset_x=clamped_region.x1,
                    offset_y=clamped_region.y1,
                )
                if not allows_label(str(translated.label)):
                    continue
                output.append(
                    _derive_node(
                        region_node,
                        region=translated,
                        label=translated.label,
                        score=float(region_node.score) * float(translated.score),
                        source=detector_backend,
                        metadata={"parent_label": region_node.label},
                    )
                )
        return output

    def describe(self, context: VisionPipelineContext) -> dict[str, Any]:
        return {
            "labels": list(self._labels or context.allowed_labels or ()),
        }


class MatcherClassificationStage(VisionPipelineStage):
    stage_kind = "matcher_classification"

    def __init__(
        self,
        matcher: MatcherAdapter,
        *,
        labels: Sequence[str] | None = None,
        match_threshold: float = 0.0,
        unknown_label: str | None = None,
        name: str = "detail_matcher",
    ) -> None:
        super().__init__(name=name)
        self._matcher = matcher
        self._labels = _normalize_labels(labels)
        self._match_threshold = max(float(match_threshold), 0.0)
        self._unknown_label = None if unknown_label is None else str(unknown_label)

    def run(
        self,
        *,
        image: Any,
        nodes: Sequence[VisionNode],
        context: VisionPipelineContext,
    ) -> list[VisionNode]:
        query = self._labels or context.allowed_labels
        allows_label = context.allows_label
        matcher_backend = _backend_name(self._matcher)
        unknown_label = self._unknown_label
        unknown_label_allowed = unknown_label is not None and allows_label(unknown_label)
        output: list[VisionNode] = []
        for prepared_node in _iter_prepared_region_nodes(
            image,
            nodes,
            use_candidate_nodes=True,
            include_crop=True,
        ):
            region_node = prepared_node.node
            clamped_region = prepared_node.region
            match = self._matcher.match(prepared_node.require_crop(), query=query)
            if match is None or float(match.score) < self._match_threshold:
                if not unknown_label_allowed:
                    continue
                score = 0.0 if match is None else float(region_node.score) * float(match.score)
                output.append(
                    _derive_node(
                        region_node,
                        region=clamped_region,
                        label=unknown_label,
                        score=score,
                        source=matcher_backend,
                        metadata={"matched": False},
                    )
                )
                continue
            matched_label = str(match.label)
            if not allows_label(matched_label):
                continue
            target_region = translate_box(
                match.region,
                offset_x=clamped_region.x1,
                offset_y=clamped_region.y1,
            )
            output.append(
                _derive_node(
                    region_node,
                    region=target_region,
                    label=matched_label,
                    score=float(region_node.score) * float(match.score),
                    source=str(match.source or matcher_backend),
                    metadata=dict(getattr(match, "metadata", {}) or {}),
                )
            )
        return output

    def describe(self, context: VisionPipelineContext) -> dict[str, Any]:
        return {
            "labels": list(self._labels or context.allowed_labels or ()),
            "match_threshold": self._match_threshold,
            "unknown_label": self._unknown_label,
        }


class OCRClassificationStage(VisionPipelineStage):
    stage_kind = "ocr_classification"

    def __init__(
        self,
        ocr_engine: OCREngine,
        *,
        labels: Sequence[str] | None = None,
        match_mode: str = "contains",
        unknown_label: str | None = None,
        output_label: str | None = None,
        min_score: float = 0.0,
        name: str = "detail_ocr",
    ) -> None:
        super().__init__(name=name)
        self._ocr_engine = ocr_engine
        self._labels = _normalize_labels(labels)
        self._match_mode = _normalize_match_mode(match_mode)
        self._unknown_label = None if unknown_label is None else str(unknown_label)
        self._output_label = None if output_label is None else str(output_label)
        self._min_score = max(float(min_score), 0.0)

    def run(
        self,
        *,
        image: Any,
        nodes: Sequence[VisionNode],
        context: VisionPipelineContext,
    ) -> list[VisionNode]:
        allowed = context.allowed_labels
        queries = self._labels or allowed
        allows_label = context.allows_label
        ocr_backend = _backend_name(self._ocr_engine)
        normalized_queries = _normalize_query_pairs(queries)
        output: list[VisionNode] = []
        for prepared_node in _iter_prepared_region_nodes(
            image,
            nodes,
            use_candidate_nodes=True,
            include_crop=True,
        ):
            region_node = prepared_node.node
            clamped_region = prepared_node.region
            texts = list(self._ocr_engine.read(prepared_node.require_crop()))
            label, score, text_item = self._resolve_ocr_label(
                texts=texts,
                normalized_queries=normalized_queries,
            )
            if score < self._min_score:
                label = None
            if label is None:
                if self._unknown_label is None:
                    continue
                label = self._unknown_label
                score = 0.0
                target_region = clamped_region
                matched_text = ""
            else:
                matched_text = "" if text_item is None else text_item.text
                target_region = (
                    clamped_region
                    if text_item is None
                    else translate_box(
                        text_item.bbox,
                        offset_x=clamped_region.x1,
                        offset_y=clamped_region.y1,
                    )
                )
            if not allows_label(label):
                continue
            output.append(
                _derive_node(
                    region_node,
                    region=target_region,
                    label=label,
                    score=float(region_node.score) * float(score),
                    text=matched_text,
                    source=ocr_backend,
                    metadata={"ocr_hits": len(texts)},
                )
            )
        return output

    def _resolve_ocr_label(
        self,
        *,
        texts: Sequence[OCRText],
        normalized_queries: tuple[tuple[str, str], ...] | None,
    ) -> tuple[str | None, float, OCRText | None]:
        if not texts:
            return (None, 0.0, None)

        if self._output_label is not None:
            best_text = max(texts, key=lambda item: float(item.score))
            return (self._output_label, float(best_text.score), best_text)

        if normalized_queries:
            normalized_texts = [
                (item, _normalize_text(item.text), float(item.score))
                for item in texts
            ]
            best_label: str | None = None
            best_score = 0.0
            best_text: OCRText | None = None
            for query, query_text in normalized_queries:
                for item, haystack, item_score in normalized_texts:
                    if not haystack:
                        continue
                    matched = (
                        haystack == query_text
                        if self._match_mode == "equals"
                        else query_text in haystack
                    )
                    if matched and item_score >= best_score:
                        best_label = query
                        best_score = item_score
                        best_text = item
            return (best_label, best_score, best_text)

        best_text = max(texts, key=lambda item: float(item.score))
        label = best_text.text.strip()
        if label == "":
            return (None, 0.0, None)
        return (label, float(best_text.score), best_text)

    def describe(self, context: VisionPipelineContext) -> dict[str, Any]:
        return {
            "labels": list(self._labels or context.allowed_labels or ()),
            "match_mode": self._match_mode,
            "unknown_label": self._unknown_label,
            "output_label": self._output_label,
            "min_score": self._min_score,
        }


class TextLocateStage(VisionPipelineStage):
    stage_kind = "text_locate"

    def __init__(
        self,
        ocr_engine: OCREngine,
        *,
        query: str,
        exact: bool = False,
        name: str = "locate_text",
    ) -> None:
        super().__init__(name=name)
        self._ocr_engine = ocr_engine
        self._query = str(query)
        self._normalized_query = normalize_ocr_text(query)
        self._exact = bool(exact)

    def run(
        self,
        *,
        image: Any,
        nodes: Sequence[VisionNode],
        context: VisionPipelineContext,
    ) -> list[VisionNode]:
        del context
        ocr_backend = _backend_name(self._ocr_engine)
        output: list[VisionNode] = []
        for prepared_node in _iter_prepared_region_nodes(
            image,
            nodes,
            use_candidate_nodes=True,
            include_crop=True,
        ):
            region_node = prepared_node.node
            clamped_region = prepared_node.region
            texts = list(self._ocr_engine.read(prepared_node.require_crop()))
            matched = find_ocr_text_match(
                texts,
                self._query,
                exact=self._exact,
                normalized_text=self._normalized_query,
            )
            if matched is None:
                continue
            target_region = translate_box(
                matched.bbox,
                offset_x=clamped_region.x1,
                offset_y=clamped_region.y1,
            )
            output.append(
                _derive_node(
                    region_node,
                    region=target_region,
                    label=self._query,
                    score=float(region_node.score) * float(matched.score),
                    text=matched.text,
                    source=ocr_backend,
                    metadata={
                        "ocr_hits": len(texts),
                        "query": self._query,
                        "exact": self._exact,
                    },
                )
            )
        return output

    def describe(self, context: VisionPipelineContext) -> dict[str, Any]:
        del context
        return {
            "query": self._query,
            "exact": self._exact,
        }


class ReaderClassificationStage(VisionPipelineStage):
    stage_kind = "reader_classification"

    def __init__(
        self,
        reader: ReaderAdapter,
        *,
        query: Any = None,
        labels: Sequence[str] | None = None,
        label_source: str = "label",
        output_label: str | None = None,
        unknown_label: str | None = None,
        min_score: float = 0.0,
        name: str = "detail_reader",
    ) -> None:
        super().__init__(name=name)
        self._reader = reader
        self._query = query
        self._labels = _normalize_labels(labels)
        self._label_source = _normalize_label_source(label_source)
        self._output_label = None if output_label is None else str(output_label)
        self._unknown_label = None if unknown_label is None else str(unknown_label)
        self._min_score = max(float(min_score), 0.0)

    def run(
        self,
        *,
        image: Any,
        nodes: Sequence[VisionNode],
        context: VisionPipelineContext,
    ) -> list[VisionNode]:
        allowed = context.allowed_labels
        queries = self._labels or allowed
        allows_label = context.allows_label
        reader_backend = _backend_name(self._reader)
        normalized_queries = _normalize_query_pairs(queries)
        output: list[VisionNode] = []
        for prepared_node in _iter_prepared_region_nodes(
            image,
            nodes,
            use_candidate_nodes=True,
        ):
            region_node = prepared_node.node
            clamped_region = prepared_node.region
            reader_region = (
                None if _is_full_image_region(clamped_region, image.size) else clamped_region
            )
            result = self._reader.read(image, query=self._query, region=reader_region)
            label = self._resolve_reader_label(
                result=result,
                normalized_queries=normalized_queries,
            )
            score = float(result.score)
            if score < self._min_score:
                label = None
            if label is None:
                if self._unknown_label is None:
                    continue
                label = self._unknown_label
                score = 0.0
            if not allows_label(label):
                continue
            target_region = result.region or clamped_region
            output.append(
                _derive_node(
                    region_node,
                    region=target_region,
                    label=label,
                    score=float(region_node.score) * score,
                    value=result.value,
                    unit=result.unit,
                    source=str(result.source or reader_backend),
                    metadata=dict(getattr(result, "metadata", {}) or {}),
                )
            )
        return output

    def _resolve_reader_label(
        self,
        *,
        result: Any,
        normalized_queries: tuple[tuple[str, str], ...] | None,
    ) -> str | None:
        if self._output_label is not None:
            return self._output_label

        candidates = _reader_label_candidates(result, label_source=self._label_source)
        if normalized_queries:
            normalized_candidates = {_normalize_text(candidate) for candidate in candidates}
            for query, query_text in normalized_queries:
                if query_text in normalized_candidates:
                    return query
            return None
        return candidates[0] if candidates else None

    def describe(self, context: VisionPipelineContext) -> dict[str, Any]:
        return {
            "labels": list(self._labels or context.allowed_labels or ()),
            "query": self._query,
            "label_source": self._label_source,
            "output_label": self._output_label,
            "unknown_label": self._unknown_label,
            "min_score": self._min_score,
        }


class NodeFilterStage(VisionPipelineStage):
    stage_kind = "node_filter"

    def __init__(
        self,
        *,
        labels: Sequence[str] | None = None,
        min_score: float = 0.0,
        region: BoundingBox | None = None,
        name: str = "node_filter",
    ) -> None:
        super().__init__(name=name)
        self._labels = _normalize_labels(labels)
        self._min_score = max(float(min_score), 0.0)
        self._region = region
        self._last_region: BoundingBox | None = None

    @property
    def last_region(self) -> BoundingBox | None:
        return self._last_region

    def run(
        self,
        *,
        image: Any,
        nodes: Sequence[VisionNode],
        context: VisionPipelineContext,
    ) -> list[VisionNode]:
        del context
        self._last_region = None if self._region is None else clip_region(self._region, image.size)
        output: list[VisionNode] = []
        for node in nodes:
            if node.region is None:
                continue
            if self._labels is not None and str(node.label) not in self._labels:
                continue
            if float(node.score) < self._min_score:
                continue
            if self._last_region is not None and not box_center_in_region(
                node.to_bounding_box(),
                self._last_region,
            ):
                continue
            output.append(node)
        return output

    def describe(self, context: VisionPipelineContext) -> dict[str, Any]:
        del context
        return {
            "labels": list(self._labels) if self._labels is not None else [],
            "min_score": self._min_score,
            "region": None if self._last_region is None else {
                "x1": int(self._last_region.x1),
                "y1": int(self._last_region.y1),
                "x2": int(self._last_region.x2),
                "y2": int(self._last_region.y2),
            },
        }


class ComparatorFilterStage(VisionPipelineStage):
    stage_kind = "comparator_filter"

    def __init__(
        self,
        comparator: ComparatorAdapter,
        *,
        expected: Any,
        pass_label: str | None = None,
        fail_label: str | None = None,
        keep_failed: bool = False,
        name: str = "detail_comparator",
    ) -> None:
        super().__init__(name=name)
        self._comparator = comparator
        self._expected = expected
        self._pass_label = None if pass_label is None else str(pass_label)
        self._fail_label = None if fail_label is None else str(fail_label)
        self._keep_failed = bool(keep_failed)

    def run(
        self,
        *,
        image: Any,
        nodes: Sequence[VisionNode],
        context: VisionPipelineContext,
    ) -> list[VisionNode]:
        allows_label = context.allows_label
        output: list[VisionNode] = []
        for prepared_node in _iter_prepared_region_nodes(
            image,
            nodes,
            use_candidate_nodes=True,
        ):
            region_node = prepared_node.node
            clamped_region = prepared_node.region
            compare_region = (
                None if _is_full_image_region(clamped_region, image.size) else clamped_region
            )
            result = self._comparator.compare(
                image,
                expected=self._expected,
                region=compare_region,
            )
            if bool(result.passed):
                label = self._pass_label or clamped_region.label or "matched"
            else:
                if not self._keep_failed and self._fail_label is None:
                    continue
                label = self._fail_label or clamped_region.label or "mismatch"
            if not allows_label(label):
                continue
            output.append(
                _derive_node(
                    region_node,
                    region=clamped_region,
                    label=label,
                    score=float(region_node.score) * float(result.score),
                    source=str(result.source or _backend_name(self._comparator)),
                    metadata={
                        **dict(getattr(result, "metadata", {}) or {}),
                        "passed": bool(result.passed),
                        "threshold": result.threshold,
                    },
                )
            )
        return output

    def describe(self, context: VisionPipelineContext) -> dict[str, Any]:
        del context
        return {
            "expected": self._expected,
            "pass_label": self._pass_label,
            "fail_label": self._fail_label,
            "keep_failed": self._keep_failed,
        }


class OperatorStage(VisionPipelineStage):
    stage_kind = "operator"

    def __init__(
        self,
        operator: VisionOperator,
        *,
        params: Mapping[str, Any] | None = None,
        name: str = "operator",
    ) -> None:
        super().__init__(name=name)
        self._operator = operator
        self._params = dict(params or {})
        self._last_metadata: dict[str, Any] = {}

    def run(
        self,
        *,
        image: Any,
        nodes: Sequence[VisionNode],
        context: VisionPipelineContext,
    ) -> list[VisionNode]:
        self._last_metadata = {}
        candidate_nodes = _candidate_nodes(image, nodes)
        result = self._operator.run(
            image,
            candidate_nodes,
            context=context,
            params=dict(self._params),
        )
        if isinstance(result, VisionOperatorOutput):
            self._last_metadata = dict(result.metadata)
            return list(result.nodes)
        return list(result)

    def describe(self, context: VisionPipelineContext) -> dict[str, Any]:
        del context
        metadata = {
            "backend": _backend_name(self._operator),
            "params": dict(self._params),
        }
        metadata.update(self._last_metadata)
        return metadata


def _normalize_query_pairs(
    queries: Sequence[str] | None,
) -> tuple[tuple[str, str], ...] | None:
    if not queries:
        return None
    return tuple((str(query), _normalize_text(query)) for query in queries)
