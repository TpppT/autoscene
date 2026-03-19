from __future__ import annotations

from typing import Any, Protocol

from autoscene.capture.window_capture import (
    CaptureCoordinateSpace,
    CaptureRegion,
    CaptureResult,
)
from autoscene.core.exceptions import DependencyMissingError

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


class VideoStreamFrameProvider(Protocol):
    @property
    def source_name(self) -> str: ...

    def read_frame(self) -> object: ...

    def close(self) -> None: ...


class OpenCVVideoStreamProvider:
    def __init__(
        self,
        source: str | int,
        *,
        api_preference: int | None = None,
    ) -> None:
        if cv2 is None:
            raise DependencyMissingError(
                "opencv-python is not installed. Run: pip install opencv-python"
            )
        try:
            from PIL import Image  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DependencyMissingError(
                "Pillow is not installed. Run: pip install Pillow"
            ) from exc

        self._source = source
        self._capture = (
            cv2.VideoCapture(source)
            if api_preference is None
            else cv2.VideoCapture(source, int(api_preference))
        )
        if not bool(self._capture.isOpened()):
            raise RuntimeError(f"Unable to open video stream source: {source!r}")

    @property
    def source_name(self) -> str:
        return str(self._source)

    def read_frame(self) -> object:
        from PIL import Image

        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Unable to read frame from video stream: {self.source_name}")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def close(self) -> None:
        release = getattr(self._capture, "release", None)
        if callable(release):
            release()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


class VideoStreamCapture:
    def __init__(
        self,
        frame_provider: VideoStreamFrameProvider,
        *,
        default_region: CaptureRegion | dict[str, int] | None = None,
        coordinate_region: CaptureRegion | dict[str, int] | None = None,
        keep_full_frame_artifact: bool = True,
    ) -> None:
        self._frame_provider = frame_provider
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
        frame = self._frame_provider.read_frame()
        crop_region = self.resolve_capture_region(region=region)
        image = self._crop_frame(frame, crop_region)
        artifact_image = frame if self._keep_full_frame_artifact else image
        mapped_region = self._coordinate_region or crop_region
        coordinate_space = self._build_coordinate_space(image, mapped_region)
        result = CaptureResult(
            image=image,
            artifact_image=artifact_image,
            coordinate_space=coordinate_space,
            source=f"video_stream:{self._frame_provider.source_name}",
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
        close = getattr(self._frame_provider, "close", None)
        if callable(close):
            close()

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


def create_video_stream_capture(capture_config: dict[str, Any]) -> VideoStreamCapture:
    backend = str(capture_config.get("backend", "opencv")).strip().lower()
    if backend != "opencv":
        raise ValueError(f"Unsupported video stream backend: {backend!r}")

    source: str | int | None
    if "device_index" in capture_config:
        source = int(capture_config["device_index"])
    else:
        source = capture_config.get("source")
    if source in (None, ""):
        raise ValueError("video_stream capture requires 'source' or 'device_index'.")

    provider = OpenCVVideoStreamProvider(
        source=source,
        api_preference=(
            None
            if capture_config.get("api_preference") is None
            else int(capture_config["api_preference"])
        ),
    )
    return VideoStreamCapture(
        frame_provider=provider,
        default_region=capture_config.get("region"),
        coordinate_region=(
            capture_config.get("coordinate_region")
            or capture_config.get("screen_region")
        ),
        keep_full_frame_artifact=bool(capture_config.get("keep_full_frame_artifact", True)),
    )
