from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from autoscene.core.models import BoundingBox
from autoscene.vision.interfaces import MatcherAdapter
from autoscene.vision.models import MatchResult
from autoscene.imaging.opencv.base import OpenCVAdapterBase


class OpenCVFeatureMatcher(OpenCVAdapterBase, MatcherAdapter):
    def __init__(
        self,
        reference_paths: Mapping[str, str | Path] | None = None,
        *,
        algorithm: str = "ORB",
        ratio_test: float = 0.75,
        min_matches: int = 8,
    ) -> None:
        self.reference_paths = {
            str(label): Path(path) for label, path in dict(reference_paths or {}).items()
        }
        self.algorithm = str(algorithm)
        self.ratio_test = float(ratio_test)
        self.min_matches = int(min_matches)

    @property
    def backend(self) -> str:
        return "opencv"

    def match(
        self,
        image: Any,
        query: Any = None,
        region: BoundingBox | None = None,
    ) -> MatchResult | None:
        raise NotImplementedError(
            "OpenCVFeatureMatcher is a skeleton. Implement feature extraction, "
            "descriptor matching, homography estimation, and result scoring."
        )
