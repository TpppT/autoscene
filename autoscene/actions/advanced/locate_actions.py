from __future__ import annotations

from autoscene.actions.advanced.object_actions import ObjectActions
from autoscene.actions.advanced.text_actions import TextActions
from autoscene.core.models import ObjectLocateSpec, TextLocateSpec


class LocateActions:
    """Unified facade for locate-driven actions.

    This keeps text and object implementations separate while giving callers a
    single entry point for higher-level locate workflows.
    """

    def __init__(
        self,
        *,
        text_actions: TextActions,
        object_actions: ObjectActions,
    ) -> None:
        self.text = text_actions
        self.object = object_actions

    def click_text(
        self,
        locate: TextLocateSpec,
        debug_path: str | None = None,
        debug_crop_path: str | None = None,
    ) -> None:
        self.text.click_text(
            locate,
            debug_path=debug_path,
            debug_crop_path=debug_crop_path,
        )

    def click_relative_to_text(
        self,
        locate: TextLocateSpec,
        offset_x: int = 0,
        offset_y: int = 0,
        anchor: str = "center",
    ) -> None:
        self.text.click_relative_to_text(
            locate,
            offset_x=offset_x,
            offset_y=offset_y,
            anchor=anchor,
        )

    def wait_for_text(
        self,
        locate: TextLocateSpec,
        timeout: float = 10.0,
        interval: float = 0.5,
    ) -> bool:
        return self.text.wait_for_text(
            locate,
            timeout=timeout,
            interval=interval,
        )

    def verify_text_exists(self, locate: TextLocateSpec) -> bool:
        return self.text.verify_text_exists(locate)

    def click_object(
        self,
        locate: ObjectLocateSpec,
        debug_path: str | None = None,
    ) -> None:
        self.object.click_object(
            locate,
            debug_path=debug_path,
        )

    def drag_object_to_position(
        self,
        locate: ObjectLocateSpec,
        target_x: int,
        target_y: int,
        duration_ms: int = 500,
        debug_path: str | None = None,
    ) -> None:
        self.object.drag_object_to_position(
            locate,
            target_x=target_x,
            target_y=target_y,
            duration_ms=duration_ms,
            debug_path=debug_path,
        )

    def drag_object_to_object(
        self,
        source: ObjectLocateSpec,
        target: ObjectLocateSpec,
        duration_ms: int = 500,
    ) -> None:
        self.object.drag_object_to_object(
            source,
            target,
            duration_ms=duration_ms,
        )

    def verify_object_exists(self, locate: ObjectLocateSpec) -> bool:
        return self.object.verify_object_exists(locate)
