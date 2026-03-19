from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from autoscene.core.models import TestCase
from autoscene.runner.runtime_models import (
    ActionStep,
    CheckStep,
    InvalidStep,
    RuntimeProfile,
    ScenarioPlan,
    Step,
)
from autoscene.runner.spec_registry import (
    build_registered_action_args,
    build_registered_check_args,
)
from autoscene.runner.step_specs import coerce_bool


_VERIFICATION_CHECK_ONLY_MESSAGE = "verification items do not support 'action'; use 'check'."


class StepCompiler:
    def compile_stage_item(
        self,
        item: Step | dict[str, Any],
        *,
        allow_checks: bool,
        profile: RuntimeProfile | None = None,
    ) -> Step:
        if isinstance(item, Step):
            return item
        payload = dict(item)
        metadata = self._step_metadata(payload)
        for source_key in self._allowed_stage_source_keys(allow_checks):
            step_name = payload.get(source_key)
            if step_name:
                return self._build_named_step(
                    step_type=self._step_type_for_source_key(source_key),
                    step_name=str(step_name),
                    payload=payload,
                    metadata=metadata,
                    profile=profile,
                    source_key=source_key,
                )

        expected_fields = "'action' or 'check'" if allow_checks else "'action'"
        return InvalidStep(raw=payload, expected_fields=expected_fields, **metadata)

    def compile_verification_item(
        self,
        item: Step | dict[str, Any],
        *,
        profile: RuntimeProfile | None = None,
    ) -> Step:
        if isinstance(item, CheckStep):
            return item
        if isinstance(item, ActionStep):
            return InvalidStep(
                raw=item.to_raw_payload(),
                expected_fields="'check'",
                validation_error=_VERIFICATION_CHECK_ONLY_MESSAGE,
                timeout=item.timeout,
                continue_on_failure=item.continue_on_failure,
                retry_count=item.retry_count,
                retry_interval_seconds=item.retry_interval_seconds,
                tags=item.tags,
            )
        if isinstance(item, InvalidStep):
            return item

        payload = dict(item)
        metadata = self._step_metadata(payload)
        if payload.get("action"):
            return InvalidStep(
                raw=payload,
                expected_fields="'check'",
                validation_error=_VERIFICATION_CHECK_ONLY_MESSAGE,
                **metadata,
            )

        step_name = payload.get("check")
        if step_name:
            return self._build_named_step(
                step_type="check",
                step_name=str(step_name),
                payload=payload,
                metadata=metadata,
                profile=profile,
                source_key="check",
            )
        return InvalidStep(raw=payload, expected_fields="'check'", **metadata)

    def _build_named_step(
        self,
        *,
        step_type: str,
        step_name: str,
        payload: dict[str, Any],
        metadata: dict[str, Any],
        profile: RuntimeProfile | None,
        source_key: str,
    ) -> Step:
        params = self._step_params(payload)
        invalid_step = self._build_invalid_named_step
        try:
            args_model = self._build_args_model(
                step_type=step_type,
                step_name=step_name,
                params=params,
                plugins=self._plugins_for(profile),
            )
        except ValueError as exc:
            return invalid_step(
                payload=payload,
                metadata=metadata,
                step_type=step_type,
                step_name=step_name,
                validation_error=str(exc),
            )
        step_params = args_model.to_payload() if args_model is not None else params
        if step_type == "action":
            return ActionStep(
                name=step_name,
                params=step_params,
                raw=payload,
                args_model=args_model,
                **metadata,
            )
        return CheckStep(
            name=step_name,
            params=step_params,
            raw=payload,
            args_model=args_model,
            source_key=source_key,
            **metadata,
        )

    @staticmethod
    def _allowed_stage_source_keys(allow_checks: bool) -> tuple[str, ...]:
        return ("action", "check") if allow_checks else ("action",)

    @staticmethod
    def _step_type_for_source_key(source_key: str) -> str:
        return "check" if source_key == "check" else "action"

    @staticmethod
    def _build_args_model(
        *,
        step_type: str,
        step_name: str,
        params: dict[str, Any],
        plugins: tuple[object, ...] | list[object] | None,
    ):
        builder = (
            build_registered_action_args
            if step_type == "action"
            else build_registered_check_args
        )
        return builder(
            step_name,
            params,
            plugins=plugins,
        )

    @staticmethod
    def _build_invalid_named_step(
        *,
        payload: dict[str, Any],
        metadata: dict[str, Any],
        step_type: str,
        step_name: str,
        validation_error: str,
    ) -> InvalidStep:
        return InvalidStep(
            raw=payload,
            expected_fields=f"parameters for {step_type} {step_name!r}",
            validation_error=validation_error,
            **metadata,
        )

    @staticmethod
    def _plugins_for(profile: RuntimeProfile | None) -> tuple[object, ...] | list[object] | None:
        if profile is None:
            return ()
        return profile.plugins

    @staticmethod
    def _step_params(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if key not in {"action", "check"}}

    @staticmethod
    def _extract_timeout(payload: dict[str, Any]) -> float | None:
        raw_timeout = payload.get("timeout")
        if raw_timeout is None:
            return None
        try:
            return float(raw_timeout)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_retry_count(payload: dict[str, Any]) -> int:
        raw_retry_count = payload.get("retry_count", 0)
        try:
            return max(int(raw_retry_count), 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _extract_retry_interval_seconds(payload: dict[str, Any]) -> float:
        raw_retry_interval = payload.get("retry_interval_seconds", 0.0)
        try:
            return max(float(raw_retry_interval), 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _extract_tags(payload: dict[str, Any]) -> tuple[str, ...]:
        raw_tags = payload.get("tags")
        if raw_tags is None:
            return ()
        if isinstance(raw_tags, (list, tuple, set)):
            return tuple(str(tag) for tag in raw_tags)
        return (str(raw_tags),)

    @classmethod
    def _step_metadata(cls, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "timeout": cls._extract_timeout(payload),
            "continue_on_failure": coerce_bool(payload.get("continue_on_failure"), default=False),
            "retry_count": cls._extract_retry_count(payload),
            "retry_interval_seconds": cls._extract_retry_interval_seconds(payload),
            "tags": cls._extract_tags(payload),
        }


class ScenarioPlanCompiler:
    def __init__(self, step_compiler: StepCompiler | None = None) -> None:
        self.step_compiler = step_compiler or StepCompiler()

    def compile(
        self,
        case: TestCase,
        profile: RuntimeProfile | None = None,
    ) -> ScenarioPlan:
        return ScenarioPlan(
            setup=self.compile_stage_items(case.setup, allow_checks=True, profile=profile),
            steps=self.compile_stage_items(case.steps, allow_checks=True, profile=profile),
            verification_setup=self.compile_stage_items(
                getattr(case, "verification_setup", []),
                allow_checks=True,
                profile=profile,
            ),
            verification=self.compile_verification_items(case.verification, profile=profile),
            teardown=self.compile_stage_items(
                getattr(case, "teardown", []),
                allow_checks=True,
                profile=profile,
            ),
        )

    def compile_stage_items(
        self,
        items: Sequence[Step | dict[str, Any]],
        *,
        allow_checks: bool,
        profile: RuntimeProfile | None = None,
    ) -> list[Step]:
        return [
            self.step_compiler.compile_stage_item(
                item,
                allow_checks=allow_checks,
                profile=profile,
            )
            for item in items
        ]

    def compile_verification_items(
        self,
        items: Sequence[Step | dict[str, Any]],
        *,
        profile: RuntimeProfile | None = None,
    ) -> list[Step]:
        return [
            self.step_compiler.compile_verification_item(item, profile=profile) for item in items
        ]


__all__ = [
    "ScenarioPlanCompiler",
    "StepCompiler",
]
