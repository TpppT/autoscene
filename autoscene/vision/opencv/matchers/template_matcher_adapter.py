from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autoscene.core.models import BoundingBox
from autoscene.vision.algorithms.opencv.template_matcher import (
    TemplateMatcher,
    coerce_pil_image,
)
from autoscene.vision.interfaces import MatcherAdapter
from autoscene.vision.models import MatchResult


class TemplateMatcherAdapter(MatcherAdapter):
    def __init__(
        self,
        templates_dir: str | Path | None = None,
        template_paths=None,
        *,
        match_size: int | Sequence[int] = 64,
        min_score: float = 0.0,
    ) -> None:
        self._matcher = TemplateMatcher(
            templates_dir=templates_dir,
            template_paths=template_paths,
            match_size=match_size,
        )
        self._min_score = max(float(min_score), 0.0)

    @property
    def backend(self) -> str:
        return "opencv_template"

    def match(
        self,
        image: Any,
        query: Any = None,
        region: BoundingBox | None = None,
    ) -> MatchResult | None:
        pil_image = coerce_pil_image(image)
        target_region = self._normalize_region(region, pil_image.size)
        if target_region is not None:
            pil_image = pil_image.crop(
                (target_region.x1, target_region.y1, target_region.x2, target_region.y2)
            )

        labels = self._normalize_query_labels(query)
        matched = self._matcher.match(pil_image, labels=labels)
        if matched is None or float(matched.score) < self._min_score:
            return None

        if target_region is None:
            target_region = BoundingBox(
                x1=0,
                y1=0,
                x2=int(pil_image.size[0]),
                y2=int(pil_image.size[1]),
                score=float(matched.score),
                label=str(matched.label),
            )
        return MatchResult(
            score=float(matched.score),
            region=target_region,
            label=str(matched.label),
            source=str(matched.source),
        )

    @staticmethod
    def _normalize_query_labels(query: Any) -> list[str] | None:
        if query is None:
            return None
        if isinstance(query, str):
            text = query.strip()
            return None if text == "" else [text]
        if isinstance(query, Sequence):
            values = [str(value) for value in query if str(value).strip() != ""]
            return values or None
        text = str(query).strip()
        return None if text == "" else [text]

    @staticmethod
    def _normalize_region(
        region: BoundingBox | None,
        image_size: tuple[int, int],
    ) -> BoundingBox | None:
        if region is None:
            return None
        width, height = image_size
        x1 = max(0, min(int(region.x1), width))
        y1 = max(0, min(int(region.y1), height))
        x2 = max(0, min(int(region.x2), width))
        y2 = max(0, min(int(region.y2), height))
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=region.score,
            label=region.label,
        )
