from __future__ import annotations

from pathlib import Path

from autoscene.core.exceptions import DependencyMissingError
from autoscene.core.models import TestCase
from autoscene.runner.spec_registry import (
    build_registered_action_args,
    build_registered_check_args,
)

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


_VERIFICATION_CHECK_ONLY_MESSAGE = "does not support 'action'; use 'check'."


def _ensure_list(value, field_name: str) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Field '{field_name}' must be a list.")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Each '{field_name}' item must be an object/dict.")
    return value


def _ensure_mapping(value, field_name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Field '{field_name}' must be an object/dict.")
    return value


def _ensure_mapping_of_mappings(
    value, field_name: str, reserved_keys: set[str] | None = None
) -> dict[str, dict]:
    mapping = _ensure_mapping(value, field_name)
    reserved_keys = set(reserved_keys or ())
    for reserved_key in reserved_keys:
        if reserved_key in mapping:
            raise ValueError(
                f"Field '{field_name}' cannot contain reserved key {reserved_key!r}."
            )
    for key, item in mapping.items():
        if not isinstance(item, dict):
            raise ValueError(
                f"Each '{field_name}' entry must be an object/dict, got {key!r}."
            )
    return mapping


def _normalize_step_name(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text


def _validate_stage_items(
    items: list[dict],
    field_name: str,
    *,
    allow_checks: bool,
    verification_mode: bool = False,
) -> list[dict]:
    validated_items: list[dict] = []
    for index, item in enumerate(items, start=1):
        action_name = _normalize_step_name(item.get("action"))
        check_name = _normalize_step_name(item.get("check"))
        params = {key: value for key, value in item.items() if key not in {"action", "check"}}

        if verification_mode:
            if action_name is not None:
                raise ValueError(
                    f"{field_name}[{index}] {_VERIFICATION_CHECK_ONLY_MESSAGE}"
                )
            if check_name is None:
                raise ValueError(f"{field_name}[{index}] missing 'check': {item!r}")
            try:
                build_registered_check_args(check_name, params)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name}[{index}] invalid parameters for check {check_name!r}: {exc}"
                ) from exc
            validated_items.append(item)
            continue

        if action_name is not None:
            try:
                build_registered_action_args(action_name, params)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name}[{index}] invalid parameters for action {action_name!r}: {exc}"
                ) from exc
            validated_items.append(item)
            continue

        if allow_checks and check_name is not None:
            try:
                build_registered_check_args(check_name, params)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name}[{index}] invalid parameters for check {check_name!r}: {exc}"
                ) from exc
            validated_items.append(item)
            continue

        expected_fields = "'action' or 'check'" if allow_checks else "'action'"
        raise ValueError(f"{field_name}[{index}] missing {expected_fields}: {item!r}")

    return validated_items


def load_test_case(path: str | Path) -> TestCase:
    if yaml is None:
        raise DependencyMissingError("PyYAML is not installed. Run: pip install PyYAML")
    file_path = Path(path)
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML must be a mapping/object.")

    setup = _validate_stage_items(
        _ensure_list(raw.get("setup"), "setup"),
        "setup",
        allow_checks=True,
    )
    steps = _validate_stage_items(
        _ensure_list(raw.get("steps"), "steps"),
        "steps",
        allow_checks=True,
    )
    verification_setup = _validate_stage_items(
        _ensure_list(raw.get("verification_setup"), "verification_setup"),
        "verification_setup",
        allow_checks=True,
    )
    verification = _validate_stage_items(
        _ensure_list(raw.get("verification"), "verification"),
        "verification",
        allow_checks=True,
        verification_mode=True,
    )
    teardown = _validate_stage_items(
        _ensure_list(raw.get("teardown"), "teardown"),
        "teardown",
        allow_checks=True,
    )

    return TestCase(
        name=str(raw.get("name", file_path.stem)),
        emulator=_ensure_mapping(raw.get("emulator"), "emulator"),
        detector=_ensure_mapping(raw.get("detector"), "detector"),
        detectors=_ensure_mapping_of_mappings(
            raw.get("detectors"), "detectors", reserved_keys={"default"}
        ),
        readers=_ensure_mapping_of_mappings(raw.get("readers"), "readers"),
        log_sources=_ensure_mapping_of_mappings(raw.get("log_sources"), "log_sources"),
        ocr=_ensure_mapping(raw.get("ocr"), "ocr"),
        capture=_ensure_mapping(raw.get("capture"), "capture"),
        setup=setup,
        steps=steps,
        verification_setup=verification_setup,
        verification=verification,
        teardown=teardown,
    )
