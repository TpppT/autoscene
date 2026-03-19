from __future__ import annotations

from typing import Any, Protocol

from autoscene.core.models import ObjectLocateSpec, TextLocateSpec


class ScreenshotActionService(Protocol):
    def screenshot(self, path: str | None = None) -> Any: ...


class BaseActionService(ScreenshotActionService, Protocol):
    def click(self, x: int, y: int) -> None: ...

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300,
    ) -> None: ...

    def input_text(self, text: str) -> None: ...

    def press_key(
        self,
        key: str,
        presses: int = 1,
        interval_seconds: float = 0.0,
    ) -> None: ...

    def open_browser(
        self,
        url: str,
        browser: str = "chrome",
        browser_path: str | None = None,
        new_window: bool = True,
        args: list[str] | None = None,
    ) -> None: ...

    def maximize_window(
        self,
        window_title: str,
        timeout: float = 5.0,
        interval: float = 0.2,
    ) -> None: ...

    def activate_window(
        self,
        window_title: str,
        timeout: float = 5.0,
        interval: float = 0.2,
        settle_seconds: float = 0.2,
    ) -> bool: ...

    def sleep(self, seconds: float) -> None: ...


class TextActionService(Protocol):
    def click_text(
        self,
        locate: TextLocateSpec,
        debug_path: str | None = None,
        debug_crop_path: str | None = None,
    ) -> None: ...

    def click_relative_to_text(
        self,
        locate: TextLocateSpec,
        offset_x: int = 0,
        offset_y: int = 0,
        anchor: str = "center",
    ) -> None: ...

    def wait_for_text(
        self,
        locate: TextLocateSpec,
        timeout: float = 10.0,
        interval: float = 0.5,
    ) -> bool: ...

    def verify_text_exists(
        self,
        locate: TextLocateSpec,
    ) -> bool: ...


class ObjectActionService(Protocol):
    def click_object(
        self,
        locate: ObjectLocateSpec,
        debug_path: str | None = None,
    ) -> None: ...

    def drag_object_to_position(
        self,
        locate: ObjectLocateSpec,
        target_x: int,
        target_y: int,
        duration_ms: int = 500,
        debug_path: str | None = None,
    ) -> None: ...

    def drag_object_to_object(
        self,
        source: ObjectLocateSpec,
        target: ObjectLocateSpec,
        duration_ms: int = 500,
    ) -> None: ...

    def verify_object_exists(
        self,
        locate: ObjectLocateSpec,
    ) -> bool: ...


class LocateActionService(TextActionService, ObjectActionService, Protocol):
    text: TextActionService
    object: ObjectActionService
