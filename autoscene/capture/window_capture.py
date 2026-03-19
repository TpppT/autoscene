from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import Optional

from autoscene.core.exceptions import DependencyMissingError

try:
    import mss
except ImportError:  # pragma: no cover - optional dependency
    mss = None

try:
    import pygetwindow
except ImportError:  # pragma: no cover - optional dependency
    pygetwindow = None


_CAPTURE_MIN_USABLE_SCORE = 12.0
_WINDOW_IMAGE_GRAB_MAX_ATTEMPTS = 6
_CAPTURE_RETRY_DELAY_SECONDS = 0.2


@dataclass
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class CaptureCoordinateSpace:
    region: CaptureRegion
    image_width: int
    image_height: int

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        width = max(float(self.image_width), 1.0)
        height = max(float(self.image_height), 1.0)
        scale_x = float(self.region.width) / width
        scale_y = float(self.region.height) / height
        return (
            int(round(float(self.region.left) + float(x) * scale_x)),
            int(round(float(self.region.top) + float(y) * scale_y)),
        )


@dataclass(frozen=True)
class CaptureResult:
    image: object
    artifact_image: object | None = None
    coordinate_space: CaptureCoordinateSpace | None = None
    source: str = ""
    score: float = 0.0
    capture_region: CaptureRegion | None = None

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        if self.coordinate_space is not None:
            return self.coordinate_space.to_screen(int(x), int(y))
        if self.capture_region is not None:
            return (
                int(self.capture_region.left + int(x)),
                int(self.capture_region.top + int(y)),
            )
        return (int(x), int(y))


@dataclass(frozen=True)
class _CaptureAttempt:
    source: str
    image: object
    score: float
    native_image: object | None = None
    capture_region: CaptureRegion | None = None


class WindowLocator:
    def __init__(
        self,
        default_window_title: str | None = None,
        default_region: Optional[dict] = None,
    ) -> None:
        self._default_window_title = default_window_title
        self._default_region = default_region
        self._bound_window_handle: int | None = None
        self._bound_capture_region: CaptureRegion | None = None

    def resolve_region(
        self,
        window_title: str | None,
        region: CaptureRegion | dict | None,
    ) -> CaptureRegion | None:
        explicit_region = self._coerce_region(region)
        if explicit_region is not None:
            return explicit_region

        if self._default_region:
            return self._coerce_region(self._default_region)

        if self._bound_window_handle is not None:
            resolved = self._resolve_bound_window_region()
            if resolved is not None:
                return resolved

        title = window_title or self._default_window_title
        if title:
            return self._find_window_region(title)
        return None

    def resolve_capture_window_handle(
        self,
        window_title: str | None,
        region: CaptureRegion | dict | None,
    ) -> int | None:
        if region or self._default_region:
            return None
        if self._bound_window_handle is not None:
            return int(self._bound_window_handle)
        title = window_title or self._default_window_title
        if not title:
            return None
        try:
            window = self._find_window(title)
        except RuntimeError:
            return None
        return self._get_window_handle(window)

    def bind_window_handle(self, window_handle: int | None) -> None:
        self._bound_window_handle = None if window_handle is None else int(window_handle)
        self._bound_capture_region = None

    def get_bound_window_handle(self) -> int | None:
        return self._bound_window_handle

    @staticmethod
    def _coerce_region(region: CaptureRegion | dict | None) -> CaptureRegion | None:
        if region is None:
            return None
        if isinstance(region, dict):
            return CaptureRegion(**region)
        return region

    def _resolve_bound_window_region(self) -> CaptureRegion | None:
        if self._bound_window_handle is None:
            return None
        try:
            resolved = self._find_window_region_by_handle(self._bound_window_handle)
        except RuntimeError:
            self._clear_bound_window_binding()
            return None
        self._bound_capture_region = resolved
        return resolved

    def _clear_bound_window_binding(self) -> None:
        self._bound_window_handle = None
        self._bound_capture_region = None

    @staticmethod
    def _find_window(title: str):
        if pygetwindow is None:
            raise DependencyMissingError(
                "pygetwindow is not installed. Run: pip install pygetwindow"
            )
        windows = pygetwindow.getWindowsWithTitle(title)
        if not windows:
            raise RuntimeError(f"No window found for title: {title}")
        return windows[0]

    def _find_window_region(self, title: str) -> CaptureRegion:
        win = self._find_window(title)
        return self._window_region_from_window(win)

    def _find_window_region_by_handle(self, window_handle: int) -> CaptureRegion:
        outer_region = self._outer_region_from_handle(int(window_handle))
        client_region = self._client_region_from_handle(int(window_handle))
        preferred_region = self._prefer_window_region(
            client_region=client_region,
            outer_region=outer_region,
        )
        if preferred_region is not None:
            return preferred_region
        if pygetwindow is None:
            raise DependencyMissingError(
                "pygetwindow is not installed. Run: pip install pygetwindow"
            )
        if not self._can_query_pygetwindow():
            raise RuntimeError(
                f"Cannot query window handles through pygetwindow for handle: {window_handle}"
            )
        for window in pygetwindow.getAllWindows():
            if self._get_window_handle(window) == int(window_handle):
                return self._window_region_from_window(window)
        raise RuntimeError(f"No window found for handle: {window_handle}")

    def _window_region_from_window(self, window: object) -> CaptureRegion:
        outer_region = CaptureRegion(
            left=int(getattr(window, "left")),
            top=int(getattr(window, "top")),
            width=int(getattr(window, "width")),
            height=int(getattr(window, "height")),
        )
        handle = self._get_window_handle(window)
        if handle is not None:
            client_region = self._client_region_from_handle(handle)
            preferred_region = self._prefer_window_region(
                client_region=client_region,
                outer_region=outer_region,
            )
            if preferred_region is not None:
                return preferred_region
        return outer_region

    @staticmethod
    def _outer_region_from_handle(window_handle: int) -> CaptureRegion | None:
        if not hasattr(ctypes, "windll"):  # pragma: no cover - non-Windows fallback
            return None
        user32 = getattr(ctypes.windll, "user32", None)
        if user32 is None:
            return None

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        rect = RECT()
        if not user32.GetWindowRect(int(window_handle), ctypes.byref(rect)):
            return None
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            return None
        return CaptureRegion(
            left=int(rect.left),
            top=int(rect.top),
            width=width,
            height=height,
        )

    @staticmethod
    def _prefer_window_region(
        client_region: CaptureRegion | None,
        outer_region: CaptureRegion | None,
    ) -> CaptureRegion | None:
        if client_region is None:
            return outer_region
        if outer_region is None:
            return client_region
        width_ratio = float(client_region.width) / max(float(outer_region.width), 1.0)
        height_ratio = float(client_region.height) / max(float(outer_region.height), 1.0)
        if width_ratio < 0.9 or height_ratio < 0.85:
            return outer_region
        return client_region

    @staticmethod
    def _client_region_from_handle(window_handle: int) -> CaptureRegion | None:
        if not hasattr(ctypes, "windll"):  # pragma: no cover - non-Windows fallback
            return None
        user32 = getattr(ctypes.windll, "user32", None)
        if user32 is None:
            return None

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        class POINT(ctypes.Structure):
            _fields_ = [
                ("x", ctypes.c_long),
                ("y", ctypes.c_long),
            ]

        rect = RECT()
        origin = POINT()
        if not user32.GetClientRect(int(window_handle), ctypes.byref(rect)):
            return None
        if not user32.ClientToScreen(int(window_handle), ctypes.byref(origin)):
            return None

        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            return None
        return CaptureRegion(left=int(origin.x), top=int(origin.y), width=width, height=height)

    @staticmethod
    def _get_window_handle(window: object) -> int | None:
        handle = getattr(window, "_hWnd", None)
        if handle is None:
            return None
        return int(handle)

    @staticmethod
    def _can_query_pygetwindow() -> bool:
        if not hasattr(ctypes, "windll"):  # pragma: no cover - non-Windows fallback
            return False
        return getattr(ctypes.windll, "user32", None) is not None


class CaptureScorer:
    def build_capture_space(
        self,
        attempt: _CaptureAttempt,
    ) -> CaptureCoordinateSpace | None:
        if attempt.capture_region is None:
            return None
        image_size = getattr(attempt.image, "size", None)
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            return None
        return CaptureCoordinateSpace(
            region=attempt.capture_region,
            image_width=int(image_size[0]),
            image_height=int(image_size[1]),
        )

    def build_capture_result(self, attempt: _CaptureAttempt) -> CaptureResult:
        return CaptureResult(
            image=attempt.image,
            artifact_image=attempt.native_image or attempt.image,
            coordinate_space=self.build_capture_space(attempt),
            source=attempt.source,
            score=float(attempt.score),
            capture_region=attempt.capture_region,
        )

    def is_usable_capture_attempt(self, attempt: _CaptureAttempt | None) -> bool:
        return attempt is not None and float(attempt.score) >= _CAPTURE_MIN_USABLE_SCORE

    def prefer_capture_attempt(
        self,
        current: _CaptureAttempt | None,
        candidate: _CaptureAttempt | None,
    ) -> _CaptureAttempt | None:
        if candidate is None:
            return current
        if current is None or float(candidate.score) >= float(current.score):
            return candidate
        return current

    def score_capture_attempt(
        self,
        image: object,
        expected_region: CaptureRegion | None = None,
    ) -> float:
        if expected_region is not None and not self._capture_matches_region(
            image,
            expected_region,
        ):
            return 0.0
        return self._image_capture_score(image)

    @staticmethod
    def _image_capture_score(image: object) -> float:
        try:
            from PIL import ImageStat
        except ImportError:
            return 0.0
        try:
            stat = ImageStat.Stat(image.convert("RGB"))
        except (AttributeError, TypeError, ValueError):
            return 0.0
        return float(sum(stat.mean))

    @staticmethod
    def _capture_matches_region(image: object, expected_region: CaptureRegion) -> bool:
        image_size = getattr(image, "size", None)
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            return False
        return image_size == (int(expected_region.width), int(expected_region.height))


class ImageGrabWindowCaptureBackend:
    def __init__(
        self,
        window_locator: WindowLocator | None = None,
        scorer: CaptureScorer | None = None,
    ) -> None:
        self._window_locator = window_locator or WindowLocator()
        self._scorer = scorer or CaptureScorer()

    def capture(
        self,
        image_module: object,
        window_handle: int,
        expected_region: CaptureRegion | None = None,
    ) -> _CaptureAttempt | None:
        if not hasattr(ctypes, "windll"):  # pragma: no cover - non-Windows fallback
            return None
        try:
            from PIL import ImageGrab
        except ImportError:
            return None
        best_image = None
        best_native_image = None
        best_capture_region = None
        best_score = float("-inf")
        for attempt in range(_WINDOW_IMAGE_GRAB_MAX_ATTEMPTS):
            try:
                raw_image = ImageGrab.grab(window=int(window_handle))
            except (NotImplementedError, OSError, TypeError, ValueError):
                return None
            capture_region = self._infer_image_grab_region(
                window_handle=int(window_handle),
                image=raw_image,
                expected_region=expected_region,
            )
            image = self._normalize_image_grab_capture(
                image_module,
                raw_image,
                expected_region=expected_region,
            )
            score = self._scorer.score_capture_attempt(
                image,
                expected_region=expected_region,
            )
            if score > best_score:
                best_image = image
                best_native_image = raw_image
                best_capture_region = capture_region
                best_score = score
            if score >= _CAPTURE_MIN_USABLE_SCORE:
                break
            if attempt < _WINDOW_IMAGE_GRAB_MAX_ATTEMPTS - 1:
                time.sleep(_CAPTURE_RETRY_DELAY_SECONDS)
        if best_image is None:
            return None
        return _CaptureAttempt(
            source="imagegrab_window",
            image=best_image,
            score=best_score,
            native_image=best_native_image,
            capture_region=best_capture_region,
        )

    def _normalize_image_grab_capture(
        self,
        image_module: object,
        image: object,
        expected_region: CaptureRegion | None = None,
    ):
        if expected_region is None:
            return image
        image_size = getattr(image, "size", None)
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            return image
        expected_size = (int(expected_region.width), int(expected_region.height))
        if image_size == expected_size:
            return image
        if not self._is_uniform_scale(image_size, expected_size):
            return image
        resize = getattr(image, "resize", None)
        if not callable(resize):
            return image
        resampling = getattr(image_module, "Resampling", image_module)
        resample_filter = getattr(resampling, "LANCZOS", None)
        if resample_filter is None:
            return resize(expected_size)
        return resize(expected_size, resample=resample_filter)

    def _infer_image_grab_region(
        self,
        window_handle: int,
        image: object,
        expected_region: CaptureRegion | None = None,
    ) -> CaptureRegion | None:
        image_size = getattr(image, "size", None)
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            return expected_region
        candidates: list[CaptureRegion] = []
        outer_region = self._window_locator._outer_region_from_handle(int(window_handle))
        client_region = self._window_locator._client_region_from_handle(int(window_handle))
        for candidate in (outer_region, client_region, expected_region):
            if candidate is None:
                continue
            if candidate not in candidates:
                candidates.append(candidate)
        if not candidates:
            return expected_region
        return self._pick_region_for_image_size(
            image_size=(int(image_size[0]), int(image_size[1])),
            candidates=candidates,
        )

    def _pick_region_for_image_size(
        self,
        image_size: tuple[int, int],
        candidates: list[CaptureRegion],
    ) -> CaptureRegion:
        best_candidate = candidates[0]
        best_score = float("inf")
        for candidate in candidates:
            candidate_size = (int(candidate.width), int(candidate.height))
            if candidate_size == image_size:
                return candidate
            if not self._is_uniform_scale(image_size, candidate_size):
                continue
            scale_x = float(image_size[0]) / max(float(candidate.width), 1.0)
            scale_y = float(image_size[1]) / max(float(candidate.height), 1.0)
            score = abs(scale_x - 1.0) + abs(scale_y - 1.0)
            if score < best_score:
                best_candidate = candidate
                best_score = score
        return best_candidate

    @staticmethod
    def _is_uniform_scale(
        image_size: tuple[int, int],
        expected_size: tuple[int, int],
        tolerance: float = 0.05,
    ) -> bool:
        expected_width = max(float(expected_size[0]), 1.0)
        expected_height = max(float(expected_size[1]), 1.0)
        scale_x = float(image_size[0]) / expected_width
        scale_y = float(image_size[1]) / expected_height
        return scale_x > 0.0 and scale_y > 0.0 and abs(scale_x - scale_y) <= tolerance


class MSSCaptureBackend:
    def __init__(self, scorer: CaptureScorer | None = None) -> None:
        self._scorer = scorer or CaptureScorer()

    def capture(
        self,
        image_module: object,
        target_region: CaptureRegion | None,
    ) -> _CaptureAttempt:
        with mss.mss() as sct:
            if target_region is None:
                monitor = sct.monitors[1]
            else:
                monitor = {
                    "left": target_region.left,
                    "top": target_region.top,
                    "width": target_region.width,
                    "height": target_region.height,
                }
            raw = sct.grab(monitor)
            image = image_module.frombytes("RGB", raw.size, raw.rgb)
        capture_region = target_region
        if capture_region is None:
            capture_region = CaptureRegion(
                left=int(monitor["left"]),
                top=int(monitor["top"]),
                width=int(monitor["width"]),
                height=int(monitor["height"]),
            )
        return _CaptureAttempt(
            source="mss",
            image=image,
            score=self._scorer.score_capture_attempt(image, expected_region=target_region),
            native_image=image,
            capture_region=capture_region,
        )


class WindowCapture:
    def __init__(
        self,
        default_window_title: str | None = None,
        default_region: Optional[dict] = None,
        window_locator: WindowLocator | None = None,
        imagegrab_backend: ImageGrabWindowCaptureBackend | None = None,
        mss_backend: MSSCaptureBackend | None = None,
        scorer: CaptureScorer | None = None,
    ) -> None:
        self._window_locator = window_locator or WindowLocator(
            default_window_title=default_window_title,
            default_region=default_region,
        )
        self._scorer = scorer or CaptureScorer()
        self._imagegrab_backend = imagegrab_backend or ImageGrabWindowCaptureBackend(
            window_locator=self._window_locator,
            scorer=self._scorer,
        )
        self._mss_backend = mss_backend or MSSCaptureBackend(scorer=self._scorer)
        setattr(self._imagegrab_backend, "_window_locator", self._window_locator)
        setattr(self._imagegrab_backend, "_scorer", self._scorer)
        setattr(self._mss_backend, "_scorer", self._scorer)
        self._last_capture_result: CaptureResult | None = None

    def capture(
        self,
        window_title: str | None = None,
        region: CaptureRegion | dict | None = None,
    ):
        return self.capture_result(window_title=window_title, region=region).image

    def capture_result(
        self,
        window_title: str | None = None,
        region: CaptureRegion | dict | None = None,
    ) -> CaptureResult:
        image_module = self._require_image_module()

        self._last_capture_result = None
        fallback_attempt: _CaptureAttempt | None = None

        window_handle = self._window_locator.resolve_capture_window_handle(
            window_title=window_title,
            region=region,
        )
        expected_region: CaptureRegion | None = None
        if window_handle is not None:
            expected_region = self._resolve_expected_region(
                window_title=window_title,
                region=region,
            )
            attempt = self._imagegrab_backend.capture(
                image_module,
                window_handle,
                expected_region=expected_region,
            )
            fallback_attempt = self._scorer.prefer_capture_attempt(fallback_attempt, attempt)
            if self._scorer.is_usable_capture_attempt(attempt):
                return self._remember_capture_attempt(attempt)
            if expected_region is not None and mss is not None:
                attempt = self._mss_backend.capture(image_module, expected_region)
                fallback_attempt = self._scorer.prefer_capture_attempt(fallback_attempt, attempt)
                if self._scorer.is_usable_capture_attempt(attempt):
                    return self._remember_capture_attempt(attempt)

        if mss is None:
            if fallback_attempt is not None:
                return self._remember_capture_attempt(fallback_attempt)
            raise DependencyMissingError("mss is not installed. Run: pip install mss")

        target_region = expected_region or self._window_locator.resolve_region(
            window_title=window_title,
            region=region,
        )
        attempt = self._mss_backend.capture(image_module, target_region)
        fallback_attempt = self._scorer.prefer_capture_attempt(fallback_attempt, attempt)
        if self._scorer.is_usable_capture_attempt(attempt):
            return self._remember_capture_attempt(attempt)
        if fallback_attempt is not None:
            return self._remember_capture_attempt(fallback_attempt)
        raise RuntimeError("Unable to capture a usable image from any capture source.")

    def resolve_capture_region(
        self,
        window_title: str | None = None,
        region: CaptureRegion | dict | None = None,
    ) -> CaptureRegion | None:
        return self._window_locator.resolve_region(window_title=window_title, region=region)

    def _resolve_region(
        self, window_title: str | None, region: CaptureRegion | dict | None
    ) -> CaptureRegion | None:
        return self._window_locator.resolve_region(window_title=window_title, region=region)

    def bind_window_handle(self, window_handle: int | None) -> None:
        self._window_locator.bind_window_handle(window_handle)

    def get_bound_window_handle(self) -> int | None:
        return self._window_locator.get_bound_window_handle()

    def get_last_capture_result(self) -> CaptureResult | None:
        return self._last_capture_result

    def _resolve_capture_window_handle(
        self,
        window_title: str | None,
        region: CaptureRegion | dict | None,
    ) -> int | None:
        return self._window_locator.resolve_capture_window_handle(
            window_title=window_title,
            region=region,
        )

    def _remember_capture_attempt(self, attempt: _CaptureAttempt) -> CaptureResult:
        result = self._scorer.build_capture_result(attempt)
        self._last_capture_result = result
        return result

    @staticmethod
    def _require_image_module():
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DependencyMissingError(
                "Pillow is not installed. Run: pip install Pillow"
            ) from exc
        return Image

    def _resolve_expected_region(
        self,
        *,
        window_title: str | None,
        region: CaptureRegion | dict | None,
    ) -> CaptureRegion | None:
        try:
            return self._window_locator.resolve_region(
                window_title=window_title,
                region=region,
            )
        except (RuntimeError, DependencyMissingError):
            return None
