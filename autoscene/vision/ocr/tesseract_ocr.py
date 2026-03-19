from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from autoscene.core.exceptions import DependencyMissingError
from autoscene.core.models import BoundingBox, OCRText
from autoscene.imaging.opencv.base import OpenCVAdapterBase
from autoscene.vision.interfaces import OCREngine

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None


class TesseractOCREngine(OpenCVAdapterBase, OCREngine):
    def __init__(
        self,
        lang: str = "eng",
        min_confidence: float = 40.0,
        tesseract_cmd: str | None = None,
        preprocess: bool | Mapping[str, Any] | None = None,
        tesseract_config: str | None = None,
    ) -> None:
        if pytesseract is None:
            raise DependencyMissingError(
                "pytesseract is not installed. Run: pip install pytesseract"
            )
        self._lang = lang
        self._min_confidence = min_confidence
        self._tesseract_cmd = self._find_tesseract_cmd(tesseract_cmd)
        self._preprocess = self._normalize_preprocess(preprocess)
        self._tesseract_config = self._normalize_tesseract_config(tesseract_config)

    def read(self, image: Any) -> list[OCRText]:
        return self._read_with_settings(
            image=image,
            lang=self._lang,
            min_confidence=self._min_confidence,
            preprocess=self._preprocess,
            tesseract_config=self._tesseract_config,
        )

    def read_with_overrides(
        self,
        image: Any,
        overrides: Mapping[str, Any] | None = None,
    ) -> list[OCRText]:
        normalized = self._normalize_overrides(overrides)
        return self._read_with_settings(
            image=image,
            lang=str(normalized.get("lang", self._lang)),
            min_confidence=float(normalized.get("min_confidence", self._min_confidence)),
            preprocess=self._resolve_preprocess_override(normalized),
            tesseract_config=(
                self._tesseract_config
                if "tesseract_config" not in normalized
                else self._normalize_tesseract_config(normalized.get("tesseract_config"))
            ),
        )

    def _read_with_settings(
        self,
        image: Any,
        *,
        lang: str,
        min_confidence: float,
        preprocess: dict[str, Any] | None,
        tesseract_config: str | None,
    ) -> list[OCRText]:
        self._configure_tesseract_cmd()
        image_for_ocr, coordinate_scale = self._prepare_image_for_ocr(
            image=image,
            preprocess=preprocess,
        )
        kwargs = {
            "lang": lang,
            "output_type": pytesseract.Output.DICT,
        }
        if tesseract_config:
            kwargs["config"] = tesseract_config
        try:
            data = pytesseract.image_to_data(image_for_ocr, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on local installation
            if self._is_tesseract_not_found(exc):
                raise DependencyMissingError(
                    "Tesseract OCR executable was not found. Install Tesseract or set "
                    "'ocr.tesseract_cmd' in the YAML case."
                ) from exc
            raise
        output: list[OCRText] = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if not text:
                continue
            conf = float(data["conf"][i])
            if conf < min_confidence:
                continue
            x = int(round(int(data["left"][i]) / coordinate_scale))
            y = int(round(int(data["top"][i]) / coordinate_scale))
            w = int(round(int(data["width"][i]) / coordinate_scale))
            h = int(round(int(data["height"][i]) / coordinate_scale))
            bbox = BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h, score=conf / 100.0)
            output.append(OCRText(text=text, bbox=bbox, score=conf / 100.0))
        return output

    def _configure_tesseract_cmd(self) -> None:
        if self._tesseract_cmd and hasattr(pytesseract, "pytesseract"):
            pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd

    def _prepare_image_for_ocr(
        self,
        image: Any,
        preprocess: dict[str, Any] | None = None,
    ) -> tuple[Any, float]:
        if preprocess is None:
            return (image, 1.0)
        processed = self._preprocess_image(image, preprocess)
        return (processed, float(preprocess["scale"]))

    def _preprocess_image(self, image: Any, config: dict[str, Any]) -> Any:
        cv2 = self.require_cv2()
        Image = self.require_pillow()
        frame = self.to_ndarray(image)

        scale = float(config["scale"])
        if scale != 1.0:
            interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=interpolation)

        if bool(config["grayscale"]):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clahe_clip_limit = float(config["clahe_clip_limit"])
        if clahe_clip_limit > 0.0:
            tile = int(config["clahe_tile_grid_size"])
            tile = max(tile, 1)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(tile, tile))
            if len(frame.shape) == 2:
                frame = clahe.apply(frame)
            else:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                enhanced_l = clahe.apply(l_channel)
                frame = cv2.cvtColor(
                    cv2.merge((enhanced_l, a_channel, b_channel)),
                    cv2.COLOR_LAB2BGR,
                )

        median_blur = int(config["median_blur"])
        if median_blur > 1:
            frame = cv2.medianBlur(frame, self._normalize_odd_kernel(median_blur, minimum=3))

        if bool(config["sharpen"]):
            frame = self._apply_unsharp_mask(frame)

        frame = self._apply_threshold(frame, config)

        if bool(config["invert"]):
            frame = cv2.bitwise_not(frame)

        if len(frame.shape) == 2:
            return Image.fromarray(frame)
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _apply_threshold(self, frame: Any, config: dict[str, Any]) -> Any:
        cv2 = self.require_cv2()
        mode = str(config["threshold"]).strip().lower()
        if mode in {"", "none"}:
            return frame
        if len(frame.shape) != 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if mode in {"adaptive", "adaptive_gaussian", "gaussian"}:
            return cv2.adaptiveThreshold(
                frame,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self._normalize_odd_kernel(int(config["adaptive_block_size"]), minimum=3),
                float(config["adaptive_c"]),
            )
        if mode in {"adaptive_mean", "mean"}:
            return cv2.adaptiveThreshold(
                frame,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                self._normalize_odd_kernel(int(config["adaptive_block_size"]), minimum=3),
                float(config["adaptive_c"]),
            )
        if mode == "otsu":
            _, thresholded = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresholded
        raise ValueError(f"Unsupported OCR preprocess threshold mode: {mode}")

    @staticmethod
    def _apply_unsharp_mask(frame: Any) -> Any:
        cv2 = __import__("cv2")
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.2)
        return cv2.addWeighted(frame, 1.6, blurred, -0.6, 0)

    @staticmethod
    def _normalize_preprocess(
        preprocess: bool | Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if preprocess in (None, False):
            return None
        if preprocess is True:
            raw: dict[str, Any] = {}
        elif isinstance(preprocess, Mapping):
            raw = dict(preprocess)
        else:
            raise ValueError("Field 'ocr.preprocess' must be a boolean or a mapping/object.")

        enabled = bool(raw.pop("enabled", True))
        if not enabled:
            return None

        config = {
            "scale": max(float(raw.pop("scale", 2.0)), 1.0),
            "grayscale": bool(raw.pop("grayscale", True)),
            "clahe_clip_limit": float(raw.pop("clahe_clip_limit", 3.0)),
            "clahe_tile_grid_size": int(raw.pop("clahe_tile_grid_size", 8)),
            "median_blur": int(raw.pop("median_blur", 0)),
            "sharpen": bool(raw.pop("sharpen", True)),
            "threshold": str(raw.pop("threshold", "adaptive_gaussian")),
            "adaptive_block_size": int(raw.pop("adaptive_block_size", 31)),
            "adaptive_c": float(raw.pop("adaptive_c", 11.0)),
            "invert": bool(raw.pop("invert", False)),
        }
        if raw:
            unknown = ", ".join(sorted(str(key) for key in raw))
            raise ValueError(f"Unknown OCR preprocess options: {unknown}")
        return config

    @staticmethod
    def _normalize_tesseract_config(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _resolve_preprocess_override(
        self,
        overrides: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if "preprocess" not in overrides:
            return self._preprocess

        preprocess_override = overrides.get("preprocess")
        if isinstance(preprocess_override, Mapping):
            merged = dict(self._preprocess or {})
            merged.update(dict(preprocess_override))
            return self._normalize_preprocess(merged)
        return self._normalize_preprocess(preprocess_override)

    @staticmethod
    def _normalize_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
        if overrides is None:
            return {}
        if not isinstance(overrides, Mapping):
            raise ValueError("OCR overrides must be a mapping/object.")
        normalized = dict(overrides)
        allowed = {"lang", "min_confidence", "preprocess", "tesseract_config"}
        unknown = sorted(str(key) for key in normalized if key not in allowed)
        if unknown:
            raise ValueError(f"Unknown OCR override options: {', '.join(unknown)}")
        return normalized

    @staticmethod
    def _normalize_odd_kernel(value: int, minimum: int = 1) -> int:
        normalized = max(int(value), int(minimum))
        if normalized % 2 == 0:
            normalized += 1
        return normalized

    @staticmethod
    def _find_tesseract_cmd(configured: str | None) -> str | None:
        if configured:
            resolved = TesseractOCREngine._resolve_executable(configured)
            if resolved is None:
                raise DependencyMissingError(
                    f"Tesseract executable not found: {configured}"
                )
            return resolved

        resolved = TesseractOCREngine._resolve_executable("tesseract")
        if resolved:
            return resolved

        candidates = [
            Path(os.environ[key]) / "Tesseract-OCR/tesseract.exe"
            for key in ("ProgramFiles", "ProgramFiles(x86)")
            if os.environ.get(key)
        ]
        if os.environ.get("LocalAppData"):
            candidates.append(
                Path(os.environ["LocalAppData"]) / "Programs/Tesseract-OCR/tesseract.exe"
            )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def _resolve_executable(value: str) -> str | None:
        resolved = shutil.which(value)
        if resolved:
            return resolved
        candidate = Path(value)
        if candidate.exists():
            return str(candidate)
        return None

    @staticmethod
    def _is_tesseract_not_found(exc: Exception) -> bool:
        message = str(exc).lower()
        if "tesseract is not installed" in message:
            return True
        if isinstance(exc, FileNotFoundError):
            return True
        not_found_type = getattr(pytesseract, "TesseractNotFoundError", None)
        return bool(not_found_type and isinstance(exc, not_found_type))
