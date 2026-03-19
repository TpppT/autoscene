from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoscene.actions.service_resolution import BaseActionService, LocateActionService
from autoscene.core.exceptions import ActionExecutionError
from autoscene.runner.registry import ActionRegistry
from autoscene.runner.step_specs import (
    ActivateWindowActionArgs,
    ClickActionArgs,
    ClickObjectActionArgs,
    ClickRelativeToTextActionArgs,
    ClickTextActionArgs,
    DragActionArgs,
    DragObjectToObjectActionArgs,
    DragObjectToPositionActionArgs,
    EmulatorCommandActionArgs,
    EmulatorSendActionArgs,
    InputTextActionArgs,
    MaximizeWindowActionArgs,
    OpenBrowserActionArgs,
    PressKeyActionArgs,
    ScreenshotActionArgs,
    SleepActionArgs,
    get_action_args_builder,
    parse_object_locate_spec,
    parse_text_locate_spec,
)


@dataclass(frozen=True)
class ActionDispatcherMetadata:
    logger: logging.Logger
    output_dir: Path


@dataclass(frozen=True)
class ActionDispatcherResources:
    base_actions: BaseActionService | None
    locate_actions: LocateActionService | None
    emulator: Any


@dataclass(frozen=True)
class _BuiltinActionRegistration:
    action_name: str
    context_handler_name: str
    typed_handler_name: str


_BUILTIN_ACTION_REGISTRATIONS = (
    _BuiltinActionRegistration("activate_window", "_handle_activate_window", "_handle_activate_window_typed"),
    _BuiltinActionRegistration("click", "_handle_click", "_handle_click_typed"),
    _BuiltinActionRegistration("click_object", "_handle_click_object", "_handle_click_object_typed"),
    _BuiltinActionRegistration(
        "click_relative_to_text",
        "_handle_click_relative_to_text",
        "_handle_click_relative_to_text_typed",
    ),
    _BuiltinActionRegistration("click_text", "_handle_click_text", "_handle_click_text_typed"),
    _BuiltinActionRegistration("drag", "_handle_drag", "_handle_drag_typed"),
    _BuiltinActionRegistration(
        "drag_object_to_object",
        "_handle_drag_object_to_object",
        "_handle_drag_object_to_object_typed",
    ),
    _BuiltinActionRegistration(
        "drag_object_to_position",
        "_handle_drag_object_to_position",
        "_handle_drag_object_to_position_typed",
    ),
    _BuiltinActionRegistration(
        "emulator_command",
        "_handle_emulator_command",
        "_handle_emulator_command_typed",
    ),
    _BuiltinActionRegistration(
        "emulator_launch",
        "_handle_emulator_launch",
        "_handle_emulator_launch_typed",
    ),
    _BuiltinActionRegistration("emulator_send", "_handle_emulator_send", "_handle_emulator_send_typed"),
    _BuiltinActionRegistration("emulator_stop", "_handle_emulator_stop", "_handle_emulator_stop_typed"),
    _BuiltinActionRegistration("input_text", "_handle_input_text", "_handle_input_text_typed"),
    _BuiltinActionRegistration(
        "maximize_window",
        "_handle_maximize_window",
        "_handle_maximize_window_typed",
    ),
    _BuiltinActionRegistration("open_browser", "_handle_open_browser", "_handle_open_browser_typed"),
    _BuiltinActionRegistration("press_key", "_handle_press_key", "_handle_press_key_typed"),
    _BuiltinActionRegistration("screenshot", "_handle_screenshot", "_handle_screenshot_typed"),
    _BuiltinActionRegistration("sleep", "_handle_sleep", "_handle_sleep_typed"),
)


class ActionDispatcher:
    def __init__(
        self,
        base_actions: object | None,
        emulator: Any,
        logger: logging.Logger,
        output_dir: str | Path,
        *,
        locate_actions: LocateActionService | None = None,
    ) -> None:
        self.metadata = ActionDispatcherMetadata(
            logger=logger,
            output_dir=Path(output_dir),
        )
        self.resources = ActionDispatcherResources(
            base_actions=base_actions,
            locate_actions=locate_actions,
            emulator=emulator,
        )
        self.registry = ActionRegistry(self)
        self._register_builtin_handlers()

    def _register_builtin_handlers(self) -> None:
        for registration in _BUILTIN_ACTION_REGISTRATIONS:
            self._register_builtin_handler(registration)

    def _register_builtin_handler(self, registration: _BuiltinActionRegistration) -> None:
        self.registry.register(
            registration.action_name,
            context_handler=getattr(self, registration.context_handler_name),
            typed_handler=getattr(self, registration.typed_handler_name),
            args_builder=get_action_args_builder(registration.action_name),
        )

    def register(self, action_name: str, **kwargs: Any) -> None:
        self.registry.register(action_name, **kwargs)

    def resolve(self, action_name: str):
        return self.registry.resolve(action_name)

    def dispatch_step(self, step) -> None:
        self.registry.dispatch_step(step)

    def _click_impl(self, x: Any, y: Any) -> None:
        self._require_base_actions().click(int(x), int(y))

    def _drag_impl(
        self,
        start_x: Any,
        start_y: Any,
        end_x: Any,
        end_y: Any,
        duration_ms: Any,
    ) -> None:
        self._require_base_actions().drag(
            int(start_x),
            int(start_y),
            int(end_x),
            int(end_y),
            int(duration_ms),
        )

    def _input_text_impl(self, text: Any) -> None:
        self._require_base_actions().input_text(str(text))

    def _press_key_impl(
        self,
        key: Any,
        *,
        presses: Any = 1,
        interval_seconds: Any = 0.0,
    ) -> None:
        self._require_base_actions().press_key(
            str(key),
            presses=int(presses),
            interval_seconds=float(interval_seconds),
        )

    def _open_browser_impl(
        self,
        *,
        url: Any,
        browser: Any = "chrome",
        browser_path: Any = None,
        new_window: Any = True,
        args: Any = None,
        wait_seconds: Any = 0.0,
    ) -> None:
        normalized_args = None if args is None else [str(value) for value in args]
        self._require_base_actions().open_browser(
            url=str(url),
            browser=str(browser),
            browser_path=None if browser_path is None else str(browser_path),
            new_window=bool(new_window),
            args=normalized_args,
        )
        self._sleep_if_needed(wait_seconds)

    def _maximize_window_impl(
        self,
        *,
        window_title: Any,
        timeout: Any = 5.0,
        interval: Any = 0.2,
    ) -> None:
        self._require_base_actions().maximize_window(
            window_title=str(window_title),
            timeout=float(timeout),
            interval=float(interval),
        )

    def _activate_window_impl(
        self,
        *,
        window_title: Any,
        timeout: Any = 5.0,
        interval: Any = 0.2,
        settle_seconds: Any = 0.2,
    ) -> None:
        self._require_base_actions().activate_window(
            window_title=str(window_title),
            timeout=float(timeout),
            interval=float(interval),
            settle_seconds=float(settle_seconds),
        )

    def _sleep_if_needed(self, seconds: Any) -> None:
        wait_seconds = float(seconds)
        if wait_seconds > 0:
            self._require_base_actions().sleep(wait_seconds)

    def _sleep_impl(self, seconds: Any = 1.0) -> None:
        self._require_base_actions().sleep(float(seconds))

    def _screenshot_output_path(
        self,
        *,
        filename: Any = None,
        default_name_source: object,
    ) -> str:
        resolved_filename = (
            str(filename)
            if filename is not None
            else f"screenshot_{id(default_name_source)}.png"
        )
        return str(self.metadata.output_dir / resolved_filename)

    def _click_text_impl(
        self,
        locate,
        *,
        debug_filename: Any = None,
        debug_crop_filename: Any = None,
    ) -> None:
        self._require_locate_actions().click_text(
            locate,
            debug_path=self._resolve_output_path(debug_filename),
            debug_crop_path=self._resolve_output_path(debug_crop_filename),
        )

    def _click_relative_to_text_impl(
        self,
        locate,
        *,
        offset_x: Any = 0,
        offset_y: Any = 0,
        anchor: Any = "center",
    ) -> None:
        self._require_locate_actions().click_relative_to_text(
            locate,
            offset_x=int(offset_x),
            offset_y=int(offset_y),
            anchor=str(anchor),
        )

    def _click_object_impl(self, locate, *, debug_filename: Any = None) -> None:
        self._require_locate_actions().click_object(
            locate,
            debug_path=self._resolve_output_path(debug_filename),
        )

    def _drag_object_to_position_impl(
        self,
        locate,
        *,
        target_x: Any,
        target_y: Any,
        duration_ms: Any = 500,
        debug_filename: Any = None,
    ) -> None:
        self._require_locate_actions().drag_object_to_position(
            locate,
            target_x=int(target_x),
            target_y=int(target_y),
            duration_ms=int(duration_ms),
            debug_path=self._resolve_output_path(debug_filename),
        )

    def _drag_object_to_object_impl(
        self,
        source,
        target,
        *,
        duration_ms: Any = 500,
    ) -> None:
        self._require_locate_actions().drag_object_to_object(
            source,
            target,
            duration_ms=int(duration_ms),
        )

    def _emulator_command_impl(self, command: Any) -> None:
        output = self.resources.emulator.execute(str(command))
        self.metadata.logger.info("emulator output: %s", output)

    def _emulator_send_impl(
        self,
        *,
        payload: Any = None,
        text: Any = None,
        endpoint: Any = None,
        method: Any = None,
        headers: Any = None,
    ) -> None:
        send_payload = self._resolve_emulator_send_payload(payload=payload, text=text)
        output = self.resources.emulator.send(
            payload=send_payload,
            endpoint=endpoint,
            method=method,
            headers=headers,
        )
        self.metadata.logger.info("emulator send response: %s", output)

    @staticmethod
    def _coerce_browser_args(value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ActionExecutionError("open_browser field 'args' must be a list.")
        return [str(item) for item in value]

    @staticmethod
    def _resolve_emulator_send_payload(*, payload: Any = None, text: Any = None) -> Any:
        if payload is not None:
            return payload
        if text is not None:
            return text
        raise ActionExecutionError("emulator_send requires 'payload' (or 'text') field.")

    def _handle_click(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del payload
        self._click_impl(params["x"], params["y"])

    def _handle_click_typed(self, args: ClickActionArgs) -> None:
        self._click_impl(args.x, args.y)

    def _handle_drag(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del payload
        self._drag_impl(
            params["start_x"],
            params["start_y"],
            params["end_x"],
            params["end_y"],
            params.get("duration_ms", 300),
        )

    def _handle_drag_typed(self, args: DragActionArgs) -> None:
        self._drag_impl(args.start_x, args.start_y, args.end_x, args.end_y, args.duration_ms)

    def _handle_input_text(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del payload
        self._input_text_impl(params["text"])

    def _handle_input_text_typed(self, args: InputTextActionArgs) -> None:
        self._input_text_impl(args.text)

    def _handle_press_key(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del payload
        self._press_key_impl(
            params["key"],
            presses=params.get("presses", 1),
            interval_seconds=params.get("interval_seconds", 0.0),
        )

    def _handle_press_key_typed(self, args: PressKeyActionArgs) -> None:
        self._press_key_impl(
            args.key,
            presses=args.presses,
            interval_seconds=args.interval_seconds,
        )

    def _handle_open_browser(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del payload
        self._open_browser_impl(
            url=params["url"],
            browser=params.get("browser", "chrome"),
            browser_path=params.get("browser_path"),
            new_window=params.get("new_window", True),
            args=self._coerce_browser_args(params.get("args")),
            wait_seconds=params.get("wait_seconds", 0),
        )

    def _handle_open_browser_typed(self, args: OpenBrowserActionArgs) -> None:
        self._open_browser_impl(
            url=args.url,
            browser=args.browser,
            browser_path=args.browser_path,
            new_window=args.new_window,
            args=args.args,
            wait_seconds=args.wait_seconds,
        )

    def _handle_maximize_window(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del payload
        self._maximize_window_impl(
            window_title=params["window_title"],
            timeout=params.get("timeout", 5.0),
            interval=params.get("interval", 0.2),
        )

    def _handle_maximize_window_typed(self, args: MaximizeWindowActionArgs) -> None:
        self._maximize_window_impl(
            window_title=args.window_title,
            timeout=args.timeout,
            interval=args.interval,
        )

    def _handle_activate_window(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del payload
        self._activate_window_impl(
            window_title=params["window_title"],
            timeout=params.get("timeout", 5.0),
            interval=params.get("interval", 0.2),
            settle_seconds=params.get("settle_seconds", 0.2),
        )

    def _handle_activate_window_typed(self, args: ActivateWindowActionArgs) -> None:
        self._activate_window_impl(
            window_title=args.window_title,
            timeout=args.timeout,
            interval=args.interval,
            settle_seconds=args.settle_seconds,
        )

    def _handle_sleep(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del payload
        self._sleep_impl(params.get("seconds", 1))

    def _handle_sleep_typed(self, args: SleepActionArgs) -> None:
        self._sleep_impl(args.seconds)

    def _handle_screenshot(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        self._require_base_actions().screenshot(
            self._screenshot_output_path(
                filename=params.get("filename"),
                default_name_source=payload,
            )
        )

    def _handle_screenshot_typed(self, args: ScreenshotActionArgs) -> None:
        self._require_base_actions().screenshot(
            self._screenshot_output_path(
                filename=args.filename,
                default_name_source=args,
            )
        )

    def _handle_click_text(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del payload
        self._click_text_impl(
            self._parse_text_locate(params.get("locate")),
            debug_filename=params.get("debug_filename"),
            debug_crop_filename=params.get("debug_crop_filename"),
        )

    def _handle_click_text_typed(self, args: ClickTextActionArgs) -> None:
        self._click_text_impl(
            args.locate,
            debug_filename=args.debug_filename,
            debug_crop_filename=args.debug_crop_filename,
        )

    def _handle_click_relative_to_text(
        self, params: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        del payload
        self._click_relative_to_text_impl(
            self._parse_text_locate(params.get("locate")),
            offset_x=params.get("offset_x", 0),
            offset_y=params.get("offset_y", 0),
            anchor=params.get("anchor", "center"),
        )

    def _handle_click_relative_to_text_typed(
        self,
        args: ClickRelativeToTextActionArgs,
    ) -> None:
        self._click_relative_to_text_impl(
            args.locate,
            offset_x=args.offset_x,
            offset_y=args.offset_y,
            anchor=args.anchor,
        )

    def _handle_click_object(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del payload
        self._click_object_impl(
            self._parse_object_locate(params.get("locate")),
            debug_filename=params.get("debug_filename"),
        )

    def _handle_click_object_typed(self, args: ClickObjectActionArgs) -> None:
        self._click_object_impl(
            args.locate,
            debug_filename=args.debug_filename,
        )

    def _handle_drag_object_to_position(
        self, params: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        del payload
        self._drag_object_to_position_impl(
            self._parse_object_locate(params.get("locate")),
            target_x=params["target_x"],
            target_y=params["target_y"],
            duration_ms=params.get("duration_ms", 500),
            debug_filename=params.get("debug_filename"),
        )

    def _handle_drag_object_to_position_typed(
        self,
        args: DragObjectToPositionActionArgs,
    ) -> None:
        self._drag_object_to_position_impl(
            args.locate,
            target_x=args.target_x,
            target_y=args.target_y,
            duration_ms=args.duration_ms,
            debug_filename=args.debug_filename,
        )

    def _handle_drag_object_to_object(
        self, params: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        del payload
        self._drag_object_to_object_impl(
            self._parse_object_locate(params.get("source"), field_name="source"),
            self._parse_object_locate(params.get("target"), field_name="target"),
            duration_ms=params.get("duration_ms", 500),
        )

    def _handle_drag_object_to_object_typed(
        self,
        args: DragObjectToObjectActionArgs,
    ) -> None:
        self._drag_object_to_object_impl(
            args.source,
            args.target,
            duration_ms=args.duration_ms,
        )

    def _handle_emulator_launch(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del params, payload
        self._invoke_emulator_operation("launch")

    def _handle_emulator_launch_typed(self, args) -> None:
        del args
        self._invoke_emulator_operation("launch")

    def _handle_emulator_stop(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del params, payload
        self._invoke_emulator_operation("stop")

    def _handle_emulator_stop_typed(self, args) -> None:
        del args
        self._invoke_emulator_operation("stop")

    def _handle_emulator_command(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del payload
        self._emulator_command_impl(params["command"])

    def _handle_emulator_command_typed(self, args: EmulatorCommandActionArgs) -> None:
        self._emulator_command_impl(args.command)

    def _handle_emulator_send(self, params: dict[str, Any], payload: dict[str, Any]) -> None:
        del payload
        self._emulator_send_impl(
            payload=params.get("payload"),
            text=params.get("text"),
            endpoint=params.get("endpoint"),
            method=params.get("method"),
            headers=params.get("headers"),
        )

    def _handle_emulator_send_typed(self, args: EmulatorSendActionArgs) -> None:
        self._emulator_send_impl(
            payload=args.payload,
            text=args.text,
            endpoint=args.endpoint,
            method=args.method,
            headers=args.headers,
        )

    def _resolve_output_path(self, filename: Any) -> str | None:
        if filename is None:
            return None
        return str(self.metadata.output_dir / str(filename))

    def _invoke_emulator_operation(self, operation: str) -> None:
        getattr(self.resources.emulator, operation)()

    def _require_base_actions(self):
        if self.resources.base_actions is None:
            raise ActionExecutionError("Primitive action service is not configured.")
        return self.resources.base_actions

    def _require_locate_actions(self):
        if self.resources.locate_actions is None:
            raise ActionExecutionError("Locate action service is not configured.")
        return self.resources.locate_actions

    @staticmethod
    def _parse_object_locate(value: Any, *, field_name: str = "locate"):
        return ActionDispatcher._parse_locate_spec(
            value,
            parser=parse_object_locate_spec,
            field_name=field_name,
        )

    @staticmethod
    def _parse_text_locate(value: Any):
        return ActionDispatcher._parse_locate_spec(
            value,
            parser=parse_text_locate_spec,
            field_name="locate",
        )

    @staticmethod
    def _parse_locate_spec(value: Any, *, parser, field_name: str):
        try:
            return parser(value, field_name=field_name)
        except ValueError as exc:
            raise ActionExecutionError(str(exc)) from exc
