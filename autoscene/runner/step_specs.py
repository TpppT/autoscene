from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

from autoscene.core.models import BoundingBox, ObjectLocateSpec, TextLocateSpec


def coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


@dataclass(frozen=True)
class RegionSpec:
    x1: int
    y1: int
    x2: int
    y2: int

    def to_payload(self) -> dict[str, int]:
        return {
            "x1": int(self.x1),
            "y1": int(self.y1),
            "x2": int(self.x2),
            "y2": int(self.y2),
        }


@dataclass(frozen=True)
class StepArgs:
    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if value is None:
                continue
            to_payload = getattr(value, "to_payload", None)
            if callable(to_payload):
                payload[field_info.name] = to_payload()
                continue
            if isinstance(value, tuple):
                payload[field_info.name] = list(value)
                continue
            payload[field_info.name] = value
        return payload


@dataclass(frozen=True)
class ClickActionArgs(StepArgs):
    x: int
    y: int


@dataclass(frozen=True)
class DragActionArgs(StepArgs):
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    duration_ms: int = 300


@dataclass(frozen=True)
class InputTextActionArgs(StepArgs):
    text: str


@dataclass(frozen=True)
class PressKeyActionArgs(StepArgs):
    key: str
    presses: int = 1
    interval_seconds: float = 0.0


@dataclass(frozen=True)
class OpenBrowserActionArgs(StepArgs):
    url: str
    browser: str = "chrome"
    browser_path: str | None = None
    new_window: bool = True
    args: tuple[str, ...] | None = None
    wait_seconds: float = 0.0


@dataclass(frozen=True)
class MaximizeWindowActionArgs(StepArgs):
    window_title: str
    timeout: float = 5.0
    interval: float = 0.2


@dataclass(frozen=True)
class ActivateWindowActionArgs(StepArgs):
    window_title: str
    timeout: float = 5.0
    interval: float = 0.2
    settle_seconds: float = 0.2


@dataclass(frozen=True)
class SleepActionArgs(StepArgs):
    seconds: float = 1.0


@dataclass(frozen=True)
class ScreenshotActionArgs(StepArgs):
    filename: str | None = None


@dataclass(frozen=True)
class ClickTextActionArgs(StepArgs):
    locate: TextLocateSpec
    debug_filename: str | None = None
    debug_crop_filename: str | None = None


@dataclass(frozen=True)
class ClickRelativeToTextActionArgs(StepArgs):
    locate: TextLocateSpec
    offset_x: int = 0
    offset_y: int = 0
    anchor: str = "center"


@dataclass(frozen=True)
class ClickObjectActionArgs(StepArgs):
    locate: ObjectLocateSpec
    debug_filename: str | None = None


@dataclass(frozen=True)
class DragObjectToPositionActionArgs(StepArgs):
    locate: ObjectLocateSpec
    target_x: int
    target_y: int
    duration_ms: int = 500
    debug_filename: str | None = None


@dataclass(frozen=True)
class DragObjectToObjectActionArgs(StepArgs):
    source: ObjectLocateSpec
    target: ObjectLocateSpec
    duration_ms: int = 500


@dataclass(frozen=True)
class NoArgs(StepArgs):
    pass


@dataclass(frozen=True)
class EmulatorCommandActionArgs(StepArgs):
    command: str


@dataclass(frozen=True)
class EmulatorSendActionArgs(StepArgs):
    payload: Any = None
    text: str | None = None
    endpoint: Any = None
    method: Any = None
    headers: Any = None


@dataclass(frozen=True)
class TextExistsCheckArgs(StepArgs):
    locate: TextLocateSpec


@dataclass(frozen=True)
class ObjectExistsCheckArgs(StepArgs):
    locate: ObjectLocateSpec


@dataclass(frozen=True)
class WaitForTextCheckArgs(StepArgs):
    locate: TextLocateSpec
    timeout: float = 10.0
    interval: float = 0.5


@dataclass(frozen=True)
class ReaderValueInRangeCheckArgs(StepArgs):
    reader: str | None = None
    query: Any = None
    image_path: str | None = None
    region: RegionSpec | None = None
    min_score: float = 0.0
    expected: float | None = None
    tolerance: float = 0.0
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class LogContainsCheckArgs(StepArgs):
    source: str | None = None
    contains: str | None = None
    text: str | None = None
    regex: str | None = None
    ignore_case: bool = False


@dataclass(frozen=True)
class WaitForLogCheckArgs(StepArgs):
    source: str | None = None
    contains: str | None = None
    text: str | None = None
    regex: str | None = None
    ignore_case: bool = False
    timeout: float = 10.0
    interval: float = 0.5


ArgsBuilder = Callable[[dict[str, Any]], StepArgs]
ArgsBuilderRegistry = dict[str, ArgsBuilder]


def build_action_args(action_name: str, params: dict[str, Any]) -> StepArgs | None:
    return _build_args(_ACTION_BUILDERS, action_name, params)


def build_check_args(check_name: str, params: dict[str, Any]) -> StepArgs | None:
    return _build_args(_CHECK_BUILDERS, check_name, params)


def _build_args(
    registry: ArgsBuilderRegistry,
    name: str,
    params: dict[str, Any],
) -> StepArgs | None:
    builder = registry.get(_normalize_builder_name(name))
    if builder is None:
        return None
    return builder(dict(params))


def get_action_args_builder(
    action_name: str,
) -> ArgsBuilder | None:
    return _ACTION_BUILDERS.get(_normalize_builder_name(action_name))


def get_check_args_builder(
    check_name: str,
) -> ArgsBuilder | None:
    return _CHECK_BUILDERS.get(_normalize_builder_name(check_name))


def _normalize_builder_name(name: str) -> str:
    return str(name).lower()


def _require(params: dict[str, Any], field_name: str) -> Any:
    if field_name not in params or params[field_name] is None:
        raise ValueError(f"field {field_name!r} is required.")
    return params[field_name]


def _as_str(params: dict[str, Any], field_name: str, default: Any = None) -> str | None:
    if field_name not in params or params[field_name] is None:
        if default is None:
            return None
        return str(default)
    return str(params[field_name])


def _as_int(params: dict[str, Any], field_name: str, default: Any = None) -> int:
    raw = params.get(field_name, default)
    if raw is None:
        raise ValueError(f"field {field_name!r} is required.")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field {field_name!r} must be an integer.") from exc


def _as_float(params: dict[str, Any], field_name: str, default: Any = None) -> float:
    raw = params.get(field_name, default)
    if raw is None:
        raise ValueError(f"field {field_name!r} is required.")
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field {field_name!r} must be numeric.") from exc


def _as_bool(params: dict[str, Any], field_name: str, default: bool = False) -> bool:
    return coerce_bool(params.get(field_name), default=default)


def _as_args_tuple(params: dict[str, Any], field_name: str) -> tuple[str, ...] | None:
    raw = params.get(field_name)
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"field {field_name!r} must be a list.")
    return tuple(str(value) for value in raw)


def _as_region_spec(params: dict[str, Any], field_name: str) -> RegionSpec | None:
    return _coerce_region_spec(params.get(field_name), field_name=field_name)


def _parse_object_locate_arg(
    params: dict[str, Any],
    *,
    field_name: str = "locate",
) -> ObjectLocateSpec:
    return parse_object_locate_spec(_require(params, field_name), field_name=field_name)


def _parse_text_locate_arg(
    params: dict[str, Any],
    *,
    field_name: str = "locate",
) -> TextLocateSpec:
    return parse_text_locate_spec(_require(params, field_name), field_name=field_name)


def parse_object_locate_spec(
    value: Any,
    *,
    field_name: str = "locate",
) -> ObjectLocateSpec:
    if value is None:
        raise ValueError(f"field {field_name!r} is required.")
    if not isinstance(value, dict):
        raise ValueError(f"field {field_name!r} must be a mapping.")
    return ObjectLocateSpec(
        label=str(_require(value, "label")),
        min_score=_as_float(value, "min_score", default=0.3),
        pick=_as_str(value, "pick", default="highest_score") or "highest_score",
        detector=_as_str(value, "detector"),
        region=_region_spec_to_bounding_box(
            _coerce_region_spec(value.get("region"), field_name=f"{field_name}.region")
        ),
    )


def parse_text_locate_spec(
    value: Any,
    *,
    field_name: str = "locate",
) -> TextLocateSpec:
    if value is None:
        raise ValueError(f"field {field_name!r} is required.")
    if not isinstance(value, dict):
        raise ValueError(f"field {field_name!r} must be a mapping.")
    return TextLocateSpec(
        text=str(_require(value, "text")),
        exact=_as_bool(value, "exact", default=False),
        region=_region_spec_to_bounding_box(
            _coerce_region_spec(value.get("region"), field_name=f"{field_name}.region")
        ),
        ocr=_as_mapping(value, "ocr"),
    )


def _coerce_region_spec(value: Any, *, field_name: str) -> RegionSpec | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"field {field_name!r} must be a mapping.")
    try:
        return RegionSpec(
            x1=int(value["x1"]),
            y1=int(value["y1"]),
            x2=int(value["x2"]),
            y2=int(value["y2"]),
        )
    except KeyError as exc:
        raise ValueError(f"field {field_name!r} requires x1, y1, x2, y2.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field {field_name!r} coordinates must be integers.") from exc


def _region_spec_to_bounding_box(value: RegionSpec | None) -> BoundingBox | None:
    if value is None:
        return None
    return BoundingBox(
        x1=int(value.x1),
        y1=int(value.y1),
        x2=int(value.x2),
        y2=int(value.y2),
    )


def _as_mapping(params: dict[str, Any], field_name: str) -> dict[str, Any] | None:
    raw = params.get(field_name)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"field {field_name!r} must be a mapping.")
    return dict(raw)


def _as_numeric_optional(params: dict[str, Any], field_name: str) -> float | None:
    raw = params.get(field_name)
    if raw is None:
        return None
    return _as_float(params, field_name)


def _validate_log_matcher(params: dict[str, Any]) -> None:
    if params.get("regex") is not None:
        return
    if params.get("contains") is not None:
        return
    if params.get("text") is not None:
        return
    raise ValueError("one of 'contains', 'text', or 'regex' is required.")


def _build_log_matcher_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    _validate_log_matcher(params)
    return {
        "source": _as_str(params, "source"),
        "contains": _as_str(params, "contains"),
        "text": _as_str(params, "text"),
        "regex": _as_str(params, "regex"),
        "ignore_case": _as_bool(params, "ignore_case", default=False),
    }


def _build_click_action(params: dict[str, Any]) -> StepArgs:
    return ClickActionArgs(x=_as_int(params, "x"), y=_as_int(params, "y"))


def _build_drag_action(params: dict[str, Any]) -> StepArgs:
    return DragActionArgs(
        start_x=_as_int(params, "start_x"),
        start_y=_as_int(params, "start_y"),
        end_x=_as_int(params, "end_x"),
        end_y=_as_int(params, "end_y"),
        duration_ms=_as_int(params, "duration_ms", default=300),
    )


def _build_input_text_action(params: dict[str, Any]) -> StepArgs:
    return InputTextActionArgs(text=str(_require(params, "text")))


def _build_press_key_action(params: dict[str, Any]) -> StepArgs:
    return PressKeyActionArgs(
        key=str(_require(params, "key")),
        presses=_as_int(params, "presses", default=1),
        interval_seconds=_as_float(params, "interval_seconds", default=0.0),
    )


def _build_open_browser_action(params: dict[str, Any]) -> StepArgs:
    return OpenBrowserActionArgs(
        url=str(_require(params, "url")),
        browser=_as_str(params, "browser", default="chrome") or "chrome",
        browser_path=_as_str(params, "browser_path"),
        new_window=_as_bool(params, "new_window", default=True),
        args=_as_args_tuple(params, "args"),
        wait_seconds=_as_float(params, "wait_seconds", default=0.0),
    )


def _build_maximize_window_action(params: dict[str, Any]) -> StepArgs:
    return MaximizeWindowActionArgs(
        window_title=str(_require(params, "window_title")),
        timeout=_as_float(params, "timeout", default=5.0),
        interval=_as_float(params, "interval", default=0.2),
    )


def _build_activate_window_action(params: dict[str, Any]) -> StepArgs:
    return ActivateWindowActionArgs(
        window_title=str(_require(params, "window_title")),
        timeout=_as_float(params, "timeout", default=5.0),
        interval=_as_float(params, "interval", default=0.2),
        settle_seconds=_as_float(params, "settle_seconds", default=0.2),
    )


def _build_sleep_action(params: dict[str, Any]) -> StepArgs:
    return SleepActionArgs(seconds=_as_float(params, "seconds", default=1.0))


def _build_screenshot_action(params: dict[str, Any]) -> StepArgs:
    return ScreenshotActionArgs(filename=_as_str(params, "filename"))


def _build_click_text_action(params: dict[str, Any]) -> StepArgs:
    return ClickTextActionArgs(
        locate=_parse_text_locate_arg(params),
        debug_filename=_as_str(params, "debug_filename"),
        debug_crop_filename=_as_str(params, "debug_crop_filename"),
    )


def _build_click_relative_to_text_action(params: dict[str, Any]) -> StepArgs:
    return ClickRelativeToTextActionArgs(
        locate=_parse_text_locate_arg(params),
        offset_x=_as_int(params, "offset_x", default=0),
        offset_y=_as_int(params, "offset_y", default=0),
        anchor=_as_str(params, "anchor", default="center") or "center",
    )


def _build_click_object_action(params: dict[str, Any]) -> StepArgs:
    return ClickObjectActionArgs(
        locate=_parse_object_locate_arg(params),
        debug_filename=_as_str(params, "debug_filename"),
    )


def _build_drag_object_to_position_action(params: dict[str, Any]) -> StepArgs:
    return DragObjectToPositionActionArgs(
        locate=_parse_object_locate_arg(params),
        target_x=_as_int(params, "target_x"),
        target_y=_as_int(params, "target_y"),
        duration_ms=_as_int(params, "duration_ms", default=500),
        debug_filename=_as_str(params, "debug_filename"),
    )


def _build_drag_object_to_object_action(params: dict[str, Any]) -> StepArgs:
    return DragObjectToObjectActionArgs(
        source=_parse_object_locate_arg(params, field_name="source"),
        target=_parse_object_locate_arg(params, field_name="target"),
        duration_ms=_as_int(params, "duration_ms", default=500),
    )


def _build_no_args(_: dict[str, Any]) -> StepArgs:
    return NoArgs()


def _build_emulator_command_action(params: dict[str, Any]) -> StepArgs:
    return EmulatorCommandActionArgs(command=str(_require(params, "command")))


def _build_emulator_send_action(params: dict[str, Any]) -> StepArgs:
    payload_value = params.get("payload")
    text_value = _as_str(params, "text")
    if payload_value is None and text_value is None:
        raise ValueError("field 'payload' (or 'text') is required.")
    return EmulatorSendActionArgs(
        payload=payload_value,
        text=text_value,
        endpoint=params.get("endpoint"),
        method=params.get("method"),
        headers=params.get("headers"),
    )


def _build_text_exists_check(params: dict[str, Any]) -> StepArgs:
    return TextExistsCheckArgs(
        locate=_parse_text_locate_arg(params),
    )


def _build_object_exists_check(params: dict[str, Any]) -> StepArgs:
    return ObjectExistsCheckArgs(
        locate=_parse_object_locate_arg(params),
    )


def _build_wait_for_text_check(params: dict[str, Any]) -> StepArgs:
    return WaitForTextCheckArgs(
        locate=_parse_text_locate_arg(params),
        timeout=_as_float(params, "timeout", default=10.0),
        interval=_as_float(params, "interval", default=0.5),
    )


def _build_reader_value_in_range_check(params: dict[str, Any]) -> StepArgs:
    expected = _as_numeric_optional(params, "expected")
    minimum = _as_numeric_optional(params, "min")
    maximum = _as_numeric_optional(params, "max")
    if expected is None and (minimum is None or maximum is None):
        raise ValueError("field 'expected' or both 'min' and 'max' are required.")
    return ReaderValueInRangeCheckArgs(
        reader=_as_str(params, "reader"),
        query=params.get("query"),
        image_path=_as_str(params, "image_path"),
        region=_as_region_spec(params, "region"),
        min_score=_as_float(params, "min_score", default=0.0),
        expected=expected,
        tolerance=_as_float(params, "tolerance", default=0.0),
        min=minimum,
        max=maximum,
    )


def _build_log_contains_check(params: dict[str, Any]) -> StepArgs:
    return LogContainsCheckArgs(**_build_log_matcher_kwargs(params))


def _build_wait_for_log_check(params: dict[str, Any]) -> StepArgs:
    return WaitForLogCheckArgs(
        **_build_log_matcher_kwargs(params),
        timeout=_as_float(params, "timeout", default=10.0),
        interval=_as_float(params, "interval", default=0.5),
    )


_ACTION_BUILDERS: ArgsBuilderRegistry = {
    "activate_window": _build_activate_window_action,
    "click": _build_click_action,
    "click_object": _build_click_object_action,
    "click_relative_to_text": _build_click_relative_to_text_action,
    "click_text": _build_click_text_action,
    "drag": _build_drag_action,
    "drag_object_to_object": _build_drag_object_to_object_action,
    "drag_object_to_position": _build_drag_object_to_position_action,
    "emulator_command": _build_emulator_command_action,
    "emulator_launch": _build_no_args,
    "emulator_send": _build_emulator_send_action,
    "emulator_stop": _build_no_args,
    "input_text": _build_input_text_action,
    "maximize_window": _build_maximize_window_action,
    "open_browser": _build_open_browser_action,
    "press_key": _build_press_key_action,
    "screenshot": _build_screenshot_action,
    "sleep": _build_sleep_action,
}


_CHECK_BUILDERS: ArgsBuilderRegistry = {
    "log_contains": _build_log_contains_check,
    "object_exists": _build_object_exists_check,
    "reader_value_in_range": _build_reader_value_in_range_check,
    "text_exists": _build_text_exists_check,
    "wait_for_log": _build_wait_for_log_check,
    "wait_for_text": _build_wait_for_text_check,
}
