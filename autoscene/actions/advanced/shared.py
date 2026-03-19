from __future__ import annotations

from autoscene.actions.advanced.protocols import BaseActionRuntimeProtocol
from autoscene.capture.window_capture import CaptureResult


def capture_active_frame(base_actions: BaseActionRuntimeProtocol) -> CaptureResult:
    base_actions.activate_bound_window()
    return base_actions.capture_frame()
