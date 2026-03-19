from __future__ import annotations

import ctypes
import logging
import time
from pathlib import Path

from autoscene.actions.browser import BrowserActionsMixin
from autoscene.capture.window_capture import CaptureResult, WindowCapture
from autoscene.core.exceptions import ActionExecutionError, DependencyMissingError

_DRAG_HOLD_SECONDS = 0.15
_DRAG_STEP_SECONDS = 0.05
_WINDOW_API_LOGGER = logging.getLogger(__name__)


class BaseActions(BrowserActionsMixin):
    def __init__(
        self,
        capture: WindowCapture,
    ) -> None:
        self.capture_engine = capture
        self.logger = logging.getLogger(self.__class__.__name__)

    def click(self, x: int, y: int) -> None:
        self.logger.info("click (%s, %s)", x, y)
        pyautogui = self._require_pyautogui()
        pyautogui.click(x=int(x), y=int(y))

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300,
    ) -> None:
        self.logger.info(
            "drag (%s, %s) -> (%s, %s), duration=%sms",
            start_x,
            start_y,
            end_x,
            end_y,
            duration_ms,
        )
        pyautogui = self._require_pyautogui()
        pyautogui.moveTo(int(start_x), int(start_y))
        pyautogui.mouseDown()
        time.sleep(_DRAG_HOLD_SECONDS)

        start_x = int(start_x)
        start_y = int(start_y)
        end_x = int(end_x)
        end_y = int(end_y)
        duration_seconds = max(float(duration_ms) / 1000.0, 0.0)
        steps = max(int(round(duration_seconds / _DRAG_STEP_SECONDS)), 1)
        per_step_seconds = duration_seconds / float(steps)

        for step_index in range(1, steps + 1):
            progress = float(step_index) / float(steps)
            current_x = int(round(float(start_x) + float(end_x - start_x) * progress))
            current_y = int(round(float(start_y) + float(end_y - start_y) * progress))
            pyautogui.moveTo(current_x, current_y)
            if step_index < steps and per_step_seconds > 0.0:
                time.sleep(per_step_seconds)

        pyautogui.mouseUp()

    def input_text(self, text: str) -> None:
        self.logger.info("input_text %s", text)
        pyautogui = self._require_pyautogui()
        pyautogui.write(text)

    def press_key(
        self,
        key: str,
        presses: int = 1,
        interval_seconds: float = 0.0,
    ) -> None:
        normalized_key = str(key)
        normalized_presses = max(int(presses), 1)
        normalized_interval = max(float(interval_seconds), 0.0)
        self.logger.info(
            "press_key %s presses=%s interval_seconds=%.2f",
            normalized_key,
            normalized_presses,
            normalized_interval,
        )
        pyautogui = self._require_pyautogui()
        pyautogui.press(
            normalized_key,
            presses=normalized_presses,
            interval=normalized_interval,
        )

    def maximize_window(
        self,
        window_title: str,
        timeout: float = 5.0,
        interval: float = 0.2,
    ) -> None:
        window = self._resolve_window(
            window_title,
            timeout=timeout,
            interval=interval,
        )
        if getattr(window, "isMinimized", False) and hasattr(window, "restore"):
            window.restore()
        if hasattr(window, "maximize"):
            window.maximize()
        self._force_foreground_window(window)
        self.logger.info("maximize_window %s", window_title)

    def activate_window(
        self,
        window_title: str,
        timeout: float = 5.0,
        interval: float = 0.2,
        settle_seconds: float = 0.2,
    ) -> bool:
        window = self._resolve_window(
            window_title,
            timeout=timeout,
            interval=interval,
        )
        return self._activate_window_instance(
            window,
            settle_seconds=settle_seconds,
            log_label=window_title,
            log_action="activate_window",
        )

    def activate_bound_window(self, settle_seconds: float = 0.2) -> bool:
        window = self._find_bound_window()
        if window is None:
            return False
        return self._activate_window_instance(
            window,
            settle_seconds=settle_seconds,
            log_label=str(getattr(window, "title", "")),
            log_action="activate_bound_window",
        )

    def sleep(self, seconds: float) -> None:
        self.logger.info("sleep %.2fs", seconds)
        time.sleep(seconds)

    def capture_frame(self) -> CaptureResult:
        capture_result = getattr(self.capture_engine, "capture_result", None)
        if callable(capture_result):
            result = capture_result()
            if isinstance(result, CaptureResult):
                return result
            image = getattr(result, "image", result)
            artifact_image = getattr(result, "artifact_image", image)
            coordinate_space = getattr(result, "coordinate_space", None)
            capture_region = getattr(result, "capture_region", None)
            source = getattr(result, "source", "")
            score = float(getattr(result, "score", 0.0))
            return CaptureResult(
                image=image,
                artifact_image=artifact_image,
                coordinate_space=coordinate_space,
                capture_region=capture_region,
                source=source,
                score=score,
            )

        image = self.capture_engine.capture()
        if isinstance(image, CaptureResult):
            return image

        get_last_capture_result = getattr(self.capture_engine, "get_last_capture_result", None)
        if callable(get_last_capture_result):
            result = get_last_capture_result()
            if isinstance(result, CaptureResult):
                return result

        capture_region = None
        resolve_region = getattr(self.capture_engine, "resolve_capture_region", None)
        if callable(resolve_region):
            capture_region = resolve_region()

        return CaptureResult(
            image=image,
            artifact_image=image,
            capture_region=capture_region,
        )

    def screenshot(self, save_path: str | None = None):
        capture_result = self.capture_frame()
        image = capture_result.image
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            artifact_image = capture_result.artifact_image or image
            artifact_image.save(path)
            self.logger.info("saved screenshot: %s", path)
        return image

    def capture_to_screen(
        self,
        x: int,
        y: int,
        capture_result: CaptureResult | object | None = None,
    ) -> tuple[int, int]:
        if capture_result is not None:
            to_screen = getattr(capture_result, "to_screen", None)
            if callable(to_screen):
                return to_screen(int(x), int(y))

            coordinate_space = getattr(capture_result, "coordinate_space", None)
            if coordinate_space is not None:
                coordinate_to_screen = getattr(coordinate_space, "to_screen", None)
                if callable(coordinate_to_screen):
                    return coordinate_to_screen(int(x), int(y))

            capture_region = getattr(capture_result, "capture_region", None)
            if capture_region is not None:
                return (int(capture_region.left + x), int(capture_region.top + y))

        get_last_capture_result = getattr(self.capture_engine, "get_last_capture_result", None)
        if callable(get_last_capture_result):
            result = get_last_capture_result()
            if result is not None:
                to_screen = getattr(result, "to_screen", None)
                if callable(to_screen):
                    return to_screen(int(x), int(y))

        resolve_region = getattr(self.capture_engine, "resolve_capture_region", None)
        if not callable(resolve_region):
            return (int(x), int(y))
        region = resolve_region()
        if region is None:
            return (int(x), int(y))
        return (int(region.left + x), int(region.top + y))

    @staticmethod
    def _require_pyautogui():
        try:
            import pyautogui
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DependencyMissingError(
                "pyautogui is not installed. Run: pip install pyautogui"
            ) from exc
        return pyautogui

    @staticmethod
    def _require_pygetwindow():
        try:
            import pygetwindow
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DependencyMissingError(
                "pygetwindow is not installed. Run: pip install pygetwindow"
            ) from exc
        return pygetwindow

    def _find_bound_window(self):
        handle = getattr(self.capture_engine, "get_bound_window_handle", lambda: None)()
        if handle is None:
            return None
        try:
            pygetwindow = self._require_pygetwindow()
        except DependencyMissingError:
            return None
        for window in pygetwindow.getAllWindows():
            if self._get_window_handle(window) == handle:
                return window
        return None

    def _resolve_window(self, window_title: str, *, timeout: float, interval: float):
        window = self._find_bound_window()
        if window is not None and not self._window_matches_title(window, window_title):
            window = None
        if window is None:
            window = self._wait_for_window(window_title, timeout=timeout, interval=interval)
        self._bind_window(window)
        return window

    def _activate_window_instance(
        self,
        window: object,
        *,
        settle_seconds: float,
        log_label: str,
        log_action: str,
    ) -> bool:
        if getattr(window, "isMinimized", False):
            self._try_window_operation(window, "restore")
        self._try_window_operation(window, "show")
        self._try_window_operation(window, "activate")
        self._force_foreground_window(window)
        self.logger.info("%s %s", log_action, log_label)
        self._sleep_if_needed(settle_seconds)
        return True

    @staticmethod
    def _sleep_if_needed(seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def _wait_for_window(self, window_title: str, timeout: float, interval: float):
        pygetwindow = self._require_pygetwindow()
        deadline = time.time() + max(timeout, 0.0)
        while True:
            windows = pygetwindow.getWindowsWithTitle(window_title)
            if windows:
                return windows[0]
            if time.time() >= deadline:
                break
            time.sleep(max(interval, 0.0))
        raise ActionExecutionError(f"No window found for title: {window_title}")

    def _bind_window(self, window: object) -> None:
        handle = self._get_window_handle(window)
        if handle is None:
            return
        bind_window_handle = getattr(self.capture_engine, "bind_window_handle", None)
        if callable(bind_window_handle):
            bind_window_handle(handle)

    @staticmethod
    def _get_window_handle(window: object) -> int | None:
        handle = getattr(window, "_hWnd", None)
        if handle is None:
            return None
        return int(handle)

    @staticmethod
    def _window_matches_title(window: object, title: str) -> bool:
        window_title = str(getattr(window, "title", "") or "")
        return title.casefold() in window_title.casefold()

    def _try_window_operation(self, window: object, operation: str) -> bool:
        method = getattr(window, operation, None)
        if not callable(method):
            return False
        try:
            method()
            return True
        except Exception as exc:
            self.logger.warning(
                "window %s failed for '%s': %s",
                operation,
                getattr(window, "title", ""),
                exc,
            )
            return False

    def _force_foreground_window(self, window: object) -> bool:
        handle = self._get_window_handle(window)
        if handle is None:
            return False
        return self._force_foreground_window_handle(handle)

    @staticmethod
    def _force_foreground_window_handle(window_handle: int | None) -> bool:
        if window_handle is None or not hasattr(ctypes, "windll"):
            return False
        user32 = getattr(ctypes.windll, "user32", None)
        kernel32 = getattr(ctypes.windll, "kernel32", None)
        if user32 is None:
            return False

        hwnd = int(window_handle)
        BaseActions._show_and_raise_window(user32, hwnd)
        BaseActions._set_foreground_window(user32, hwnd)
        if BaseActions._get_foreground_window_handle(user32) == hwnd:
            return True

        if kernel32 is None:
            return False
        attach_thread_input = getattr(user32, "AttachThreadInput", None)
        get_window_thread_process_id = getattr(user32, "GetWindowThreadProcessId", None)
        get_current_thread_id = getattr(kernel32, "GetCurrentThreadId", None)
        if not callable(attach_thread_input):
            return False
        if not callable(get_window_thread_process_id):
            return False
        if not callable(get_current_thread_id):
            return False

        foreground_handle = BaseActions._get_foreground_window_handle(user32)
        if foreground_handle is None or foreground_handle == hwnd:
            return foreground_handle == hwnd

        foreground_process_id = ctypes.c_ulong()
        target_process_id = ctypes.c_ulong()
        try:
            foreground_thread_id = int(
                get_window_thread_process_id(
                    int(foreground_handle),
                    ctypes.byref(foreground_process_id),
                )
            )
            target_thread_id = int(
                get_window_thread_process_id(
                    hwnd,
                    ctypes.byref(target_process_id),
                )
            )
            current_thread_id = int(get_current_thread_id())
        except Exception as exc:
            BaseActions._log_window_api_error("GetWindowThreadProcessId", exc)
            return False

        attached_foreground = False
        attached_target = False
        try:
            if foreground_thread_id and foreground_thread_id != current_thread_id:
                attached_foreground = bool(
                    attach_thread_input(
                        foreground_thread_id,
                        current_thread_id,
                        True,
                    )
                )
            if target_thread_id and target_thread_id != current_thread_id:
                attached_target = bool(
                    attach_thread_input(
                        target_thread_id,
                        current_thread_id,
                        True,
                    )
                )
            BaseActions._show_and_raise_window(user32, hwnd)
            BaseActions._set_foreground_window(user32, hwnd)
        finally:
            if attached_target:
                try:
                    attach_thread_input(target_thread_id, current_thread_id, False)
                except Exception as exc:
                    BaseActions._log_window_api_error("AttachThreadInput(False)", exc)
            if attached_foreground:
                try:
                    attach_thread_input(foreground_thread_id, current_thread_id, False)
                except Exception as exc:
                    BaseActions._log_window_api_error("AttachThreadInput(False)", exc)

        return BaseActions._get_foreground_window_handle(user32) == hwnd

    @staticmethod
    def _show_and_raise_window(user32: object, hwnd: int) -> None:
        show_window = getattr(user32, "ShowWindow", None)
        is_iconic = getattr(user32, "IsIconic", None)
        bring_to_top = getattr(user32, "BringWindowToTop", None)
        if callable(show_window):
            try:
                if callable(is_iconic) and bool(is_iconic(hwnd)):
                    show_window(hwnd, 9)
                else:
                    show_window(hwnd, 5)
            except Exception as exc:
                BaseActions._log_window_api_error("ShowWindow", exc)
        if callable(bring_to_top):
            try:
                bring_to_top(hwnd)
            except Exception as exc:
                BaseActions._log_window_api_error("BringWindowToTop", exc)

    @staticmethod
    def _set_foreground_window(user32: object, hwnd: int) -> None:
        for operation in ("SetForegroundWindow", "SetActiveWindow", "SetFocus"):
            method = getattr(user32, operation, None)
            if not callable(method):
                continue
            try:
                method(hwnd)
            except Exception as exc:
                BaseActions._log_window_api_error(operation, exc)

    @staticmethod
    def _get_foreground_window_handle(user32: object | None = None) -> int | None:
        if user32 is None:
            if not hasattr(ctypes, "windll"):
                return None
            user32 = getattr(ctypes.windll, "user32", None)
        if user32 is None:
            return None
        get_foreground_window = getattr(user32, "GetForegroundWindow", None)
        if not callable(get_foreground_window):
            return None
        try:
            handle = get_foreground_window()
        except Exception as exc:
            BaseActions._log_window_api_error("GetForegroundWindow", exc)
            return None
        if not handle:
            return None
        return int(handle)

    @staticmethod
    def _log_window_api_error(operation: str, exc: Exception) -> None:
        _WINDOW_API_LOGGER.debug("ignored window api error for %s: %s", operation, exc)
