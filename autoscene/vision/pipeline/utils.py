from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from autoscene.core.models import BoundingBox, OCRText
from autoscene.vision.models import VisionNode, VisionNodeTraceEntry


@dataclass(frozen=True)
class _PreparedRegionNode:
    node: VisionNode
    region: BoundingBox
    crop: Any | None = None

    def require_crop(self) -> Any:
        if self.crop is None:
            raise ValueError("Prepared region node does not include a crop.")
        return self.crop


@dataclass
class _OCRLineGroup:
    entries: list[OCRText]
    center_y_sum: float
    max_height: int

    @property
    def center_y(self) -> float:
        return self.center_y_sum / max(len(self.entries), 1)

    def accepts(self, center_y: int, entry_height: int) -> bool:
        tolerance = max(self.max_height, entry_height) * 0.6
        return abs(float(center_y) - self.center_y) <= tolerance

    def append(self, entry: OCRText, center_y: int, entry_height: int) -> None:
        self.entries.append(entry)
        self.center_y_sum += float(center_y)
        self.max_height = max(self.max_height, entry_height)


def clip_region(
    box: BoundingBox,
    image_size: tuple[int, int],
) -> BoundingBox | None:
    width, height = image_size
    x1 = max(0, min(width, int(box.x1)))
    y1 = max(0, min(height, int(box.y1)))
    x2 = max(0, min(width, int(box.x2)))
    y2 = max(0, min(height, int(box.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        score=box.score,
        label=box.label,
    )


def translate_box(box: BoundingBox, *, offset_x: int, offset_y: int) -> BoundingBox:
    return BoundingBox(
        x1=int(box.x1) + int(offset_x),
        y1=int(box.y1) + int(offset_y),
        x2=int(box.x2) + int(offset_x),
        y2=int(box.y2) + int(offset_y),
        score=box.score,
        label=box.label,
    )


def box_center_in_region(box: BoundingBox, region: BoundingBox) -> bool:
    center_x, center_y = box.center
    return region.x1 <= center_x <= region.x2 and region.y1 <= center_y <= region.y2


def normalize_ocr_text(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def build_normalized_ocr_lookup(entries: Sequence[OCRText]) -> dict[int, str]:
    return {id(entry): normalize_ocr_text(entry.text) for entry in entries}


def resolve_normalized_ocr_entry_text(
    entry: OCRText,
    *,
    normalized_lookup: dict[int, str] | None = None,
) -> str:
    if normalized_lookup is None:
        return normalize_ocr_text(entry.text)
    normalized = normalized_lookup.get(id(entry))
    if normalized is not None:
        return normalized
    return normalize_ocr_text(entry.text)


def group_ocr_lines(entries: Sequence[OCRText]) -> list[list[OCRText]]:
    if not entries:
        return []
    sorted_entries = sorted(entries, key=lambda item: (item.bbox.y1, item.bbox.x1))
    line_groups: list[_OCRLineGroup] = []
    for entry in sorted_entries:
        placed = False
        center_y = entry.bbox.center[1]
        entry_height = max(entry.bbox.y2 - entry.bbox.y1, 1)
        for line_group in line_groups:
            if line_group.accepts(center_y, entry_height):
                line_group.append(entry, center_y, entry_height)
                placed = True
                break
        if not placed:
            line_groups.append(
                _OCRLineGroup(
                    entries=[entry],
                    center_y_sum=float(center_y),
                    max_height=entry_height,
                )
            )
    for line_group in line_groups:
        line_group.entries.sort(key=lambda item: item.bbox.x1)
    return [line_group.entries for line_group in line_groups]


def merge_ocr_entries(entries: Sequence[OCRText]) -> OCRText:
    selected = list(entries)
    x1 = min(entry.bbox.x1 for entry in selected)
    y1 = min(entry.bbox.y1 for entry in selected)
    x2 = max(entry.bbox.x2 for entry in selected)
    y2 = max(entry.bbox.y2 for entry in selected)
    score = sum(entry.score for entry in selected) / len(selected)
    text = " ".join(entry.text.strip() for entry in selected if entry.text.strip())
    return OCRText(
        text=text,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score),
        score=score,
    )


def match_phrase_in_line(
    line: Sequence[OCRText],
    target: str,
    *,
    normalized_lookup: dict[int, str] | None = None,
) -> OCRText | None:
    for start in range(len(line)):
        combined_parts: list[str] = []
        selected: list[OCRText] = []
        for end in range(start, len(line)):
            entry = line[end]
            normalized = resolve_normalized_ocr_entry_text(
                entry,
                normalized_lookup=normalized_lookup,
            )
            if not normalized:
                continue
            combined_parts.append(normalized)
            selected.append(entry)
            combined = " ".join(combined_parts)
            if combined == target:
                return merge_ocr_entries(selected)
            if not target.startswith(combined):
                break
    return None


def find_ocr_text_match(
    entries: Sequence[OCRText],
    text: str,
    *,
    exact: bool = False,
    normalized_text: str | None = None,
) -> OCRText | None:
    target = normalize_ocr_text(text) if normalized_text is None else str(normalized_text)
    if not target:
        return None
    normalized_lookup = build_normalized_ocr_lookup(entries)

    for entry in entries:
        normalized = normalized_lookup.get(id(entry), "")
        if exact and normalized == target:
            return entry
        if not exact and target in normalized:
            return entry

    for line in group_ocr_lines(entries):
        matched = match_phrase_in_line(
            line,
            target,
            normalized_lookup=normalized_lookup,
        )
        if matched is not None:
            return matched
    return None


def _normalize_labels(values: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        text = values.strip()
        return None if text == "" else (text,)
    normalized = tuple(str(value) for value in values if str(value).strip())
    return normalized or None


def _new_node(
    *,
    region: BoundingBox | None,
    label: str = "",
    score: float = 1.0,
    text: str = "",
    value: float | int | str | None = None,
    unit: str = "",
    source: str = "",
    metadata: dict[str, Any] | None = None,
    trace: list[VisionNodeTraceEntry] | None = None,
) -> VisionNode:
    return VisionNode(
        region=region,
        label=str(label),
        score=float(score),
        text=str(text),
        value=value,
        unit=str(unit),
        source=str(source),
        metadata=dict(metadata or {}),
        trace=list(trace or ()),
    )


def _derive_node(
    parent: VisionNode,
    *,
    region: BoundingBox | None = None,
    label: str | None = None,
    score: float | None = None,
    text: str | None = None,
    value: float | int | str | None = None,
    unit: str | None = None,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> VisionNode:
    merged_metadata = dict(parent.metadata)
    merged_metadata.update(dict(metadata or {}))
    return VisionNode(
        region=region if region is not None else parent.region,
        label=parent.label if label is None else str(label),
        score=parent.score if score is None else float(score),
        text=parent.text if text is None else str(text),
        value=parent.value if value is None else value,
        unit=parent.unit if unit is None else str(unit),
        source=parent.source if source is None else str(source),
        metadata=merged_metadata,
        trace=list(parent.trace),
    )


def _candidate_nodes(
    image: Any,
    nodes: Sequence[VisionNode],
) -> list[VisionNode]:
    if nodes:
        return list(nodes)
    width, height = image.size
    return [
        _new_node(
            region=BoundingBox(
                x1=0,
                y1=0,
                x2=int(width),
                y2=int(height),
                score=1.0,
                label="",
            ),
            metadata={"synthetic_region": True},
        )
    ]


def _iter_prepared_region_nodes(
    image: Any,
    nodes: Sequence[VisionNode],
    *,
    use_candidate_nodes: bool = False,
    include_crop: bool = False,
) -> Iterator[_PreparedRegionNode]:
    region_nodes = _candidate_nodes(image, nodes) if use_candidate_nodes else nodes
    for region_node in region_nodes:
        region_box = region_node.to_bounding_box()
        clamped_region = clip_region(region_box, image.size)
        if clamped_region is None:
            continue
        crop = None
        if include_crop:
            crop = image.crop(
                (clamped_region.x1, clamped_region.y1, clamped_region.x2, clamped_region.y2)
            )
        yield _PreparedRegionNode(
            node=region_node,
            region=clamped_region,
            crop=crop,
        )


def _is_full_image_region(box: BoundingBox, image_size: tuple[int, int]) -> bool:
    width, height = image_size
    return (
        int(box.x1) <= 0
        and int(box.y1) <= 0
        and int(box.x2) >= int(width)
        and int(box.y2) >= int(height)
    )


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _normalize_match_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"contains", ""}:
        return "contains"
    if normalized in {"equals", "equal", "exact"}:
        return "equals"
    raise ValueError(f"Unsupported OCR pipeline match mode: {mode}")


def _normalize_label_source(label_source: str) -> str:
    normalized = str(label_source).strip().lower()
    if normalized in {"label", ""}:
        return "label"
    if normalized == "value":
        return "value"
    raise ValueError(f"Unsupported reader pipeline label_source: {label_source}")


def _reader_label_candidates(result: Any, *, label_source: str) -> list[str]:
    values: list[str] = []
    if label_source == "label":
        if getattr(result, "label", ""):
            values.append(str(result.label))
        if getattr(result, "value", None) not in (None, ""):
            values.append(str(result.value))
    else:
        if getattr(result, "value", None) not in (None, ""):
            values.append(str(result.value))
        if getattr(result, "label", ""):
            values.append(str(result.label))
    return [value for value in values if str(value).strip()]


def _backend_name(component: Any) -> str:
    backend = getattr(component, "backend", None)
    if isinstance(backend, str) and backend.strip():
        return backend
    return component.__class__.__name__
