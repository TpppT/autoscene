from __future__ import annotations

import logging
from typing import Any, Protocol

from autoscene.capture.window_capture import CaptureResult
from autoscene.core.models import OCRText


class BaseActionRuntimeProtocol(Protocol):
    logger: logging.Logger
    capture_engine: Any

    def click(self, x: int, y: int) -> None: ...

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300,
    ) -> None: ...

    def capture_frame(self) -> CaptureResult: ...

    def capture_to_screen(
        self,
        x: int,
        y: int,
        capture_result: CaptureResult | object | None = None,
    ) -> tuple[int, int]: ...

    def activate_bound_window(self, settle_seconds: float = 0.2) -> bool: ...


class VisionRuntimeProtocol(Protocol):
    def read_ocr(
        self,
        image: object,
        *,
        ocr_options: dict | None = None,
    ) -> list[OCRText]: ...

    def resolve_detector(self, detector_name: str | None = None) -> Any: ...
