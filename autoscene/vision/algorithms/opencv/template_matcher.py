from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoscene.core.exceptions import DependencyMissingError

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageOps = None

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class TemplateMatch:
    label: str
    score: float
    source: str


@dataclass(frozen=True)
class _PreparedTemplate:
    label: str
    source: str
    pixels: Any
    mask: Any
    foreground: Any


def coerce_pil_image(image: Any) -> Any:
    if Image is None:
        raise DependencyMissingError("Pillow is not installed. Run: pip install Pillow")
    if isinstance(image, Image.Image):
        return image.copy()
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.copy()
    if np is not None and isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise TypeError("Expected a PIL image, numpy array, or image path.")


class TemplateMatcher:
    def __init__(
        self,
        templates_dir: str | Path | None = None,
        template_paths: Mapping[str, str | Path | Sequence[str | Path]] | None = None,
        match_size: int | Sequence[int] = 64,
    ) -> None:
        if np is None:
            raise DependencyMissingError("numpy is not installed. Run: pip install numpy")
        if Image is None or ImageOps is None:
            raise DependencyMissingError("Pillow is not installed. Run: pip install Pillow")

        self._match_size = self._normalize_size(match_size)
        self._templates: dict[str, list[_PreparedTemplate]] = {}

        if templates_dir is not None:
            self._load_from_directory(Path(templates_dir))
        if template_paths:
            self._load_from_mapping(template_paths)
        if not self._templates:
            raise ValueError("TemplateMatcher requires at least one template image.")

    @property
    def labels(self) -> set[str]:
        return set(self._templates)

    def match(
        self, image: Any, labels: Sequence[str] | None = None
    ) -> TemplateMatch | None:
        candidate_pixels, candidate_foreground = self._prepare_candidate(
            coerce_pil_image(image)
        )
        allowed = set(labels) if labels else None
        best: TemplateMatch | None = None

        for label, variants in self._templates.items():
            if allowed and label not in allowed:
                continue
            for variant in variants:
                score = self._score(
                    candidate_pixels,
                    candidate_foreground,
                    variant.pixels,
                    variant.mask,
                    variant.foreground,
                )
                if best is None or score > best.score:
                    best = TemplateMatch(label=label, score=score, source=variant.source)
        return best

    def _load_from_directory(self, templates_dir: Path) -> None:
        if not templates_dir.exists():
            raise FileNotFoundError(f"Template directory does not exist: {templates_dir}")
        for path in sorted(templates_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(templates_dir)
            label = relative.parts[0] if len(relative.parts) > 1 else path.stem
            self._add_template(label, path)

    def _load_from_mapping(
        self, template_paths: Mapping[str, str | Path | Sequence[str | Path]]
    ) -> None:
        for label, raw_paths in template_paths.items():
            if isinstance(raw_paths, (str, Path)):
                paths = [raw_paths]
            else:
                paths = list(raw_paths)
            for path in paths:
                self._add_template(str(label), Path(path))

    def _add_template(self, label: str, path: Path) -> None:
        with Image.open(path) as opened:
            pixels, mask, foreground = self._prepare_template(opened.copy())
        self._templates.setdefault(label, []).append(
            _PreparedTemplate(
                label=label,
                source=str(path),
                pixels=pixels,
                mask=mask,
                foreground=foreground,
            )
        )

    def _prepare_template(self, image: Any) -> tuple[Any, Any, Any]:
        rendered = ImageOps.pad(
            image.convert("RGBA"),
            self._match_size,
            method=Image.Resampling.BICUBIC,
            color=(255, 255, 255, 0),
        )
        pixels = np.asarray(rendered.convert("L"), dtype=np.float32) / 255.0
        mask = np.asarray(rendered.getchannel("A"), dtype=np.float32) / 255.0
        if float(mask.sum()) <= 0.0:
            mask = np.ones_like(pixels, dtype=np.float32)
        foreground = (mask > 0.05).astype(np.float32)
        if float(foreground.sum()) <= 0.0:
            foreground = (pixels < 0.95).astype(np.float32)
        if float(foreground.sum()) <= 0.0:
            foreground = np.ones_like(pixels, dtype=np.float32)
        return pixels, mask, foreground

    def _prepare_candidate(self, image: Any) -> tuple[Any, Any]:
        rendered = ImageOps.pad(
            image.convert("RGBA"),
            self._match_size,
            method=Image.Resampling.BICUBIC,
            color=(255, 255, 255, 255),
        )
        pixels = np.asarray(rendered.convert("L"), dtype=np.float32) / 255.0
        foreground = (pixels < 0.95).astype(np.float32)
        if float(foreground.sum()) <= 0.0:
            foreground = np.ones_like(pixels, dtype=np.float32)
        return pixels, foreground

    @staticmethod
    def _normalize_size(match_size: int | Sequence[int]) -> tuple[int, int]:
        if isinstance(match_size, int):
            return (match_size, match_size)
        values = tuple(int(value) for value in match_size)
        if len(values) != 2:
            raise ValueError("match_size must be an int or a sequence of two ints.")
        return values

    @staticmethod
    def _score(
        candidate_pixels: Any,
        candidate_foreground: Any,
        template_pixels: Any,
        mask: Any,
        template_foreground: Any,
    ) -> float:
        mask_sum = float(mask.sum())
        if mask_sum <= 0.0:
            mask = np.ones_like(template_pixels, dtype=np.float32)
            mask_sum = float(mask.sum())

        candidate_mean = float((candidate_pixels * mask).sum() / mask_sum)
        template_mean = float((template_pixels * mask).sum() / mask_sum)
        candidate_centered = (candidate_pixels - candidate_mean) * mask
        template_centered = (template_pixels - template_mean) * mask
        denominator = float(
            np.sqrt((candidate_centered**2).sum() * (template_centered**2).sum())
        )
        if denominator <= 1e-6:
            mse = float(
                (((candidate_pixels - template_pixels) ** 2) * mask).sum() / mask_sum
            )
            ncc_score = max(0.0, min(1.0, 1.0 - mse))
        else:
            ncc = float((candidate_centered * template_centered).sum() / denominator)
            ncc_score = max(0.0, min(1.0, (ncc + 1.0) / 2.0))

        intersection = float((candidate_foreground * template_foreground).sum())
        union = float(np.maximum(candidate_foreground, template_foreground).sum())
        iou_score = 1.0 if union <= 1e-6 else intersection / union
        return max(0.0, min(1.0, 0.3 * ncc_score + 0.7 * iou_score))
