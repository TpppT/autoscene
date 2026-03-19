from __future__ import annotations

from pathlib import Path
from typing import Any

from autoscene.capture.window_capture import (
    CaptureCoordinateSpace,
    CaptureRegion,
    CaptureResult,
)
from autoscene.core.exceptions import DependencyMissingError


class StaticImageCapture:
    def __init__(
        self,
        image_path: str | Path,
        *,
        default_region: CaptureRegion | dict[str, int] | None = None,
        coordinate_region: CaptureRegion | dict[str, int] | None = None,
        keep_full_frame_artifact: bool = True,
    ) -> None:
        self._image_path = Path(image_path)
        self._default_region = self._coerce_region(default_region)
        self._coordinate_region = self._coerce_region(coordinate_region)
        self._keep_full_frame_artifact = bool(keep_full_frame_artifact)
        self._last_capture_result: CaptureResult | None = None
        self._bound_window_handle: int | None = None

    def capture(
        self,
        window_title: str | None = None,
        region: CaptureRegion | dict[str, int] | None = None,
    ):
        del window_title
        return self.capture_result(region=region).image

    def capture_result(
        self,
        window_title: str | None = None,
        region: CaptureRegion | dict[str, int] | None = None,
    ) -> CaptureResult:
        del window_title
        frame = self._load_image()
        crop_region = self.resolve_capture_region(region=region)
        image = self._crop_frame(frame, crop_region)
        artifact_image = frame if self._keep_full_frame_artifact else image
        mapped_region = self._coordinate_region or crop_region
        coordinate_space = self._build_coordinate_space(image, mapped_region)
        result = CaptureResult(
            image=image,
            artifact_image=artifact_image,
            coordinate_space=coordinate_space,
            source=f"static_image:{self._image_path.as_posix()}",
            score=100.0,
            capture_region=mapped_region,
        )
        self._last_capture_result = result
        return result

    def resolve_capture_region(
        self,
        window_title: str | None = None,
        region: CaptureRegion | dict[str, int] | None = None,
    ) -> CaptureRegion | None:
        del window_title
        explicit_region = self._coerce_region(region)
        if explicit_region is not None:
            return explicit_region
        return self._default_region

    def bind_window_handle(self, window_handle: int | None) -> None:
        self._bound_window_handle = None if window_handle is None else int(window_handle)

    def get_bound_window_handle(self) -> int | None:
        return self._bound_window_handle

    def get_last_capture_result(self) -> CaptureResult | None:
        return self._last_capture_result

    def close(self) -> None:
        return None

    def _load_image(self):
        if not self._image_path.exists():
            raise FileNotFoundError(f"Static capture image does not exist: {self._image_path}")
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DependencyMissingError(
                "Pillow is not installed. Run: pip install Pillow"
            ) from exc

        with Image.open(self._image_path) as image:
            return image.copy()

    @staticmethod
    def _coerce_region(
        region: CaptureRegion | dict[str, int] | None,
    ) -> CaptureRegion | None:
        if region is None:
            return None
        if isinstance(region, CaptureRegion):
            return region
        return CaptureRegion(
            left=int(region["left"]),
            top=int(region["top"]),
            width=int(region["width"]),
            height=int(region["height"]),
        )

    @staticmethod
    def _crop_frame(frame: object, region: CaptureRegion | None) -> object:
        if region is None:
            return frame
        crop = getattr(frame, "crop", None)
        if not callable(crop):
            return frame
        left = int(region.left)
        top = int(region.top)
        right = int(region.left + region.width)
        bottom = int(region.top + region.height)
        return crop((left, top, right, bottom))

    @staticmethod
    def _build_coordinate_space(
        image: object,
        region: CaptureRegion | None,
    ) -> CaptureCoordinateSpace | None:
        if region is None:
            return None
        image_size = getattr(image, "size", None)
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            return None
        return CaptureCoordinateSpace(
            region=region,
            image_width=int(image_size[0]),
            image_height=int(image_size[1]),
        )


def create_static_image_capture(capture_config: dict[str, Any]) -> StaticImageCapture:
    image_path = capture_config.get("path") or capture_config.get("image_path")
    if image_path in (None, ""):
        raise ValueError("static_image capture requires 'path' or 'image_path'.")
    return StaticImageCapture(
        image_path=image_path,
        default_region=capture_config.get("region"),
        coordinate_region=(
            capture_config.get("coordinate_region")
            or capture_config.get("screen_region")
        ),
        keep_full_frame_artifact=bool(capture_config.get("keep_full_frame_artifact", True)),
    )


__all__ = [
    "StaticImageCapture",
    "create_static_image_capture",
]
