from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoscene.core.models import BoundingBox
from autoscene.vision.interfaces import ReaderAdapter
from autoscene.vision.models import ReadResult
from autoscene.imaging.opencv.base import OpenCVAdapterBase


@dataclass(frozen=True)
class _GaugeValueSpec:
    name: str
    center_ratio: tuple[float, float]
    angle_start_deg: float
    angle_end_deg: float
    sweep_direction: str
    value_start: float
    value_end: float
    round_digits: int | None = 0


@dataclass(frozen=True)
class _OverlaySpec:
    name: str
    center_ratio: tuple[float, float]
    radius_ratio: float
    angle_grid_deg: tuple[float, ...]
    angle_grid_unwrapped: tuple[float, ...] | None
    inner_ratio: float
    outer_ratio: float
    radial_samples: int
    neighbor_penalty: float
    min_score: float
    min_radius: int
    radial_weight_start: float | None = None


def _linspace(start: float, end: float, count: int) -> tuple[float, ...]:
    if count <= 1:
        return (float(start),)
    step = (float(end) - float(start)) / float(count - 1)
    return tuple(float(start) + step * index for index in range(count))


def _wrapped_linspace(start: float, mid: float, count1: int, end: float, count2: int) -> tuple[float, ...]:
    return tuple(value % 360.0 for value in (*_linspace(start, mid, count1), *_linspace(360.0, end, count2)))


_VALUE_SPECS: dict[str, _GaugeValueSpec] = {
    "speed": _GaugeValueSpec(
        name="speed",
        center_ratio=(190.0 / 800.0, 214.0 / 480.0),
        angle_start_deg=90.0,
        angle_end_deg=287.74,
        sweep_direction="cw",
        value_start=0.0,
        value_end=200.0,
        round_digits=0,
    ),
    "rpm": _GaugeValueSpec(
        name="rpm",
        center_ratio=(400.0 / 800.0, 232.0 / 480.0),
        angle_start_deg=135.0,
        angle_end_deg=15.0,
        sweep_direction="cw",
        value_start=0.0,
        value_end=7000.0,
        round_digits=0,
    ),
    "temp": _GaugeValueSpec(
        name="temp",
        center_ratio=(610.0 / 800.0, 214.0 / 480.0),
        angle_start_deg=90.0,
        angle_end_deg=270.0,
        sweep_direction="ccw",
        value_start=0.0,
        value_end=180.0,
        round_digits=0,
    ),
}

_OVERLAY_SPECS: dict[str, _OverlaySpec] = {
    "speed": _OverlaySpec(
        name="speed",
        center_ratio=_VALUE_SPECS["speed"].center_ratio,
        radius_ratio=145.0 / 480.0,
        angle_grid_deg=_linspace(88.0, 292.0, 385),
        angle_grid_unwrapped=None,
        inner_ratio=0.22,
        outer_ratio=0.82,
        radial_samples=70,
        neighbor_penalty=0.65,
        min_score=6.0,
        min_radius=32,
    ),
    "rpm": _OverlaySpec(
        name="rpm",
        center_ratio=_VALUE_SPECS["rpm"].center_ratio,
        radius_ratio=150.0 / 480.0,
        angle_grid_deg=_wrapped_linspace(135.0, 359.5, 450, 375.0, 31),
        angle_grid_unwrapped=tuple((*_linspace(135.0, 359.5, 450), *_linspace(360.0, 375.0, 31))),
        inner_ratio=0.20,
        outer_ratio=0.90,
        radial_samples=100,
        neighbor_penalty=0.65,
        min_score=8.0,
        min_radius=48,
        radial_weight_start=0.35,
    ),
    "temp": _OverlaySpec(
        name="temp",
        center_ratio=_VALUE_SPECS["temp"].center_ratio,
        radius_ratio=145.0 / 480.0,
        angle_grid_deg=_wrapped_linspace(270.0, 359.5, 180, 450.0, 181),
        angle_grid_unwrapped=tuple((*_linspace(270.0, 359.5, 180), *_linspace(360.0, 450.0, 181))),
        inner_ratio=0.18,
        outer_ratio=0.88,
        radial_samples=72,
        neighbor_penalty=0.65,
        min_score=5.0,
        min_radius=32,
    ),
}


class OpenCVQtClusterStaticReader(OpenCVAdapterBase, ReaderAdapter):
    """
    Single-frame Qt cluster gauge reader inspired by GaugeCVRecognizer.
    """

    def __init__(
        self,
        *,
        default_query: str | None = "speed",
        score_normalizer: float = 80.0,
    ) -> None:
        self.default_query = None if default_query is None else str(default_query)
        self.score_normalizer = max(float(score_normalizer), 1e-6)

    @property
    def backend(self) -> str:
        return "opencv"

    def read(
        self,
        image: Any,
        query: Any = None,
        region: BoundingBox | None = None,
    ) -> ReadResult:
        gauge_name = self._resolve_query(query)
        frame, offset = self.clip_region(image, region)
        return self._read_single(frame, offset, gauge_name)

    def read_all(
        self,
        image: Any,
        region: BoundingBox | None = None,
    ) -> dict[str, ReadResult]:
        frame, offset = self.clip_region(image, region)
        return {
            name: self._read_single(frame, offset, name)
            for name in _VALUE_SPECS
        }

    def _read_single(
        self,
        frame: Any,
        offset: tuple[int, int],
        gauge_name: str,
    ) -> ReadResult:
        spec = _VALUE_SPECS[gauge_name]
        overlay = self._detect_overlay(frame, _OVERLAY_SPECS[gauge_name])
        if overlay is None:
            value = spec.value_start
            raw_score = 0.0
            angle_deg = spec.angle_start_deg
            center_xy = self._center_from_frame(frame, spec.center_ratio)
            gauge_radius = max(8, int(round(min(frame.shape[:2]) * _OVERLAY_SPECS[gauge_name].radius_ratio)))
        else:
            angle_deg = float(overlay["angle_deg"])
            raw_score = float(overlay["score"])
            center_xy = tuple(int(value) for value in overlay["center_xy"])
            gauge_radius = int(overlay["gauge_radius"])
            value = self._angle_to_value(angle_deg, spec)
        if spec.round_digits is not None:
            rounded = round(value, spec.round_digits)
            value = int(rounded) if spec.round_digits == 0 else rounded
        return ReadResult(
            value=value,
            score=max(0.0, min(1.0, raw_score / self.score_normalizer)),
            label=gauge_name,
            source="opencv_qt_cluster_static",
            region=self._region_from_center(center_xy, gauge_radius, frame.shape, offset),
            metadata={
                "angle_deg": angle_deg,
                "center_xy": [int(center_xy[0] + offset[0]), int(center_xy[1] + offset[1])],
                "gauge_radius": gauge_radius,
                "raw_score": raw_score,
                "calibration": {
                    "angle_start_deg": spec.angle_start_deg,
                    "angle_end_deg": spec.angle_end_deg,
                    "sweep_direction": spec.sweep_direction,
                    "value_start": spec.value_start,
                    "value_end": spec.value_end,
                },
            },
        )

    def _detect_overlay(self, frame: Any, spec: _OverlaySpec) -> dict[str, Any] | None:
        np = self.require_numpy()
        pointer_map = self._build_red_pointer_map(frame)
        height, width = frame.shape[:2]
        center_xy = self._center_from_frame(frame, spec.center_ratio)
        gauge_radius = max(spec.min_radius, int(round(min(height, width) * spec.radius_ratio)))
        xs, ys = self._sampling_grid(
            width=width,
            height=height,
            center_xy=center_xy,
            gauge_radius=gauge_radius,
            spec=spec,
        )
        radial_values = pointer_map[ys, xs].astype(np.float32)
        radial_weights = None
        if spec.radial_weight_start is not None:
            radial_weights = np.linspace(
                float(spec.radial_weight_start),
                1.0,
                radial_values.shape[1],
                dtype=np.float32,
            )
        score = self._neighbor_suppressed_score(
            radial_values=radial_values,
            neighbor_penalty=spec.neighbor_penalty,
            radial_weights=radial_weights,
        )
        best_index = int(np.argmax(score))
        best_score = float(score[best_index])
        if best_score < spec.min_score:
            return None
        angle_deg = self._refine_peak_angle(
            spec.angle_grid_unwrapped or spec.angle_grid_deg,
            score,
            best_index,
        )
        return {
            "center_xy": list(center_xy),
            "angle_deg": angle_deg,
            "score": best_score,
            "gauge_radius": gauge_radius,
        }

    def _build_red_pointer_map(self, frame: Any) -> Any:
        cv2 = self.require_cv2()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_red = cv2.inRange(hsv, (0, 120, 100), (14, 255, 255))
        high_red = cv2.inRange(hsv, (166, 120, 100), (179, 255, 255))
        return cv2.GaussianBlur(cv2.bitwise_or(low_red, high_red), (5, 5), 0)

    def _sampling_grid(
        self,
        *,
        width: int,
        height: int,
        center_xy: tuple[int, int],
        gauge_radius: int,
        spec: _OverlaySpec,
    ) -> tuple[Any, Any]:
        np = self.require_numpy()
        radii = np.linspace(
            gauge_radius * spec.inner_ratio,
            gauge_radius * spec.outer_ratio,
            spec.radial_samples,
            dtype=np.float32,
        )
        angles = np.deg2rad(np.asarray(spec.angle_grid_deg, dtype=np.float32)).reshape(-1, 1)
        xs = np.clip(
            np.round(center_xy[0] + np.cos(angles) * radii.reshape(1, -1)).astype(np.int32),
            0,
            width - 1,
        )
        ys = np.clip(
            np.round(center_xy[1] + np.sin(angles) * radii.reshape(1, -1)).astype(np.int32),
            0,
            height - 1,
        )
        return xs, ys

    def _neighbor_suppressed_score(
        self,
        *,
        radial_values: Any,
        neighbor_penalty: float,
        radial_weights: Any = None,
    ) -> Any:
        np = self.require_numpy()
        if radial_weights is None:
            raw_score = radial_values.mean(axis=1)
        else:
            raw_score = np.average(radial_values, axis=1, weights=radial_weights)
        padded = np.pad(raw_score, (3, 3), mode="edge")
        neighbor_score = (
            padded[:-6]
            + padded[1:-5]
            + padded[2:-4]
            + padded[4:-2]
            + padded[5:-1]
            + padded[6:]
        ) / 6.0
        return raw_score - neighbor_score * float(neighbor_penalty)

    def _refine_peak_angle(
        self,
        angle_grid: tuple[float, ...],
        score: Any,
        best_index: int,
        radius: int = 2,
    ) -> float:
        np = self.require_numpy()
        start = max(0, int(best_index) - int(radius))
        end = min(len(angle_grid), int(best_index) + int(radius) + 1)
        local_scores = np.asarray(score[start:end], dtype=np.float32)
        if local_scores.size == 0:
            return float(angle_grid[int(best_index)] % 360.0)
        local_scores = np.maximum(local_scores - float(local_scores.min()) + 1e-3, 1e-3)
        local_angles = np.asarray(angle_grid[start:end], dtype=np.float32)
        return float(np.average(local_angles, weights=local_scores) % 360.0)

    def _resolve_query(self, query: Any) -> str:
        if query is not None:
            name = str(query)
            if name not in _VALUE_SPECS:
                available = ", ".join(sorted(_VALUE_SPECS))
                raise ValueError(f"Unknown gauge query '{name}'. Available: {available}")
            return name
        if self.default_query is not None:
            if self.default_query not in _VALUE_SPECS:
                available = ", ".join(sorted(_VALUE_SPECS))
                raise ValueError(
                    f"Unknown default_query '{self.default_query}'. Available: {available}"
                )
            return self.default_query
        if len(_VALUE_SPECS) == 1:
            return next(iter(_VALUE_SPECS))
        available = ", ".join(sorted(_VALUE_SPECS))
        raise ValueError(
            f"Reader query is required when multiple gauges are configured. Available: {available}"
        )

    @staticmethod
    def _center_from_frame(frame: Any, center_ratio: tuple[float, float]) -> tuple[int, int]:
        height, width = frame.shape[:2]
        return (
            int(round(center_ratio[0] * width)),
            int(round(center_ratio[1] * height)),
        )

    @staticmethod
    def _directed_angle_delta(start_deg: float, end_deg: float, direction: str) -> float:
        start = start_deg % 360.0
        end = end_deg % 360.0
        if direction == "cw":
            return (end - start) % 360.0
        return (start - end) % 360.0

    @classmethod
    def _angle_to_value(cls, angle_deg: float, spec: _GaugeValueSpec) -> float:
        sweep = cls._directed_angle_delta(
            spec.angle_start_deg,
            spec.angle_end_deg,
            spec.sweep_direction,
        )
        position = cls._directed_angle_delta(
            spec.angle_start_deg,
            angle_deg,
            spec.sweep_direction,
        )
        progress = 0.0 if sweep == 0.0 else max(0.0, min(1.0, position / sweep))
        return spec.value_start + (spec.value_end - spec.value_start) * progress

    @staticmethod
    def _region_from_center(
        center_xy: tuple[int, int],
        gauge_radius: int,
        frame_shape: tuple[int, ...],
        offset: tuple[int, int],
    ) -> BoundingBox:
        height, width = frame_shape[:2]
        x1 = max(0, center_xy[0] - gauge_radius) + offset[0]
        y1 = max(0, center_xy[1] - gauge_radius) + offset[1]
        x2 = min(width, center_xy[0] + gauge_radius) + offset[0]
        y2 = min(height, center_xy[1] + gauge_radius) + offset[1]
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
