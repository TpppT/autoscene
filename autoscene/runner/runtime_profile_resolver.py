from __future__ import annotations

import inspect
from typing import Any, Callable

from autoscene.actions.services import ActionServices
from autoscene.capture.static_image_capture import create_static_image_capture
from autoscene.capture.video_stream_capture import create_video_stream_capture
from autoscene.capture.window_capture import WindowCapture
from autoscene.emulator.registry import create_emulator
from autoscene.logs.registry import create_log_source
from autoscene.runner.action_dispatcher import ActionDispatcher
from autoscene.runner.check_dispatcher import CheckDispatcher
from autoscene.runner.runtime_models import RuntimeProfile
from autoscene.runner.runtime_policies import HookBus
from autoscene.vision import (
    create_detector,
    create_ocr_engine,
    create_reader_adapter,
)


class RuntimeProfileResolver:
    def __init__(
        self,
        *,
        emulator_factory: Callable[..., object] | None = None,
        detector_factory: Callable[..., object] | None = None,
        reader_factory: Callable[..., object] | None = None,
        log_source_factory: Callable[..., object] | None = None,
        ocr_engine_factory: Callable[..., object] | None = None,
        capture_factory: Callable[..., object] | None = None,
        actions_factory: Callable[..., object] | None = None,
        action_dispatcher_factory: Callable[..., object] | None = None,
        check_dispatcher_factory: Callable[..., object] | None = None,
        hook_bus_factory: Callable[..., object] | None = None,
    ) -> None:
        self._emulator_factory = emulator_factory or create_emulator
        self._detector_factory = detector_factory or create_detector
        self._reader_factory = reader_factory or create_reader_adapter
        self._log_source_factory = log_source_factory or create_log_source
        self._ocr_engine_factory = ocr_engine_factory or create_ocr_engine
        self._capture_factory = capture_factory or default_capture_factory
        self._actions_factory = actions_factory or ActionServices
        self._action_dispatcher_factory = action_dispatcher_factory or ActionDispatcher
        self._check_dispatcher_factory = check_dispatcher_factory or CheckDispatcher
        self._hook_bus_factory = hook_bus_factory or HookBus

    def resolve(self, profile: RuntimeProfile | None = None) -> RuntimeProfile:
        override = profile or RuntimeProfile()
        plugins = tuple(override.plugins)
        return RuntimeProfile(
            name=override.name,
            environment=override.environment,
            plugins=plugins,
            emulator_factory=self._resolve_factory(
                override.emulator_factory,
                self._emulator_factory,
                plugins=plugins,
                plugin_aware=True,
            ),
            detector_factory=self._resolve_factory(
                override.detector_factory,
                self._detector_factory,
                plugins=plugins,
                plugin_aware=True,
            ),
            reader_factory=self._resolve_factory(
                override.reader_factory,
                self._reader_factory,
                plugins=plugins,
                plugin_aware=True,
            ),
            log_source_factory=override.log_source_factory or self._log_source_factory,
            ocr_engine_factory=self._resolve_factory(
                override.ocr_engine_factory,
                self._ocr_engine_factory,
                plugins=plugins,
                plugin_aware=True,
            ),
            capture_factory=override.capture_factory or self._capture_factory,
            actions_factory=override.actions_factory or self._actions_factory,
            action_dispatcher_factory=override.action_dispatcher_factory
            or self._action_dispatcher_factory,
            check_dispatcher_factory=override.check_dispatcher_factory
            or self._check_dispatcher_factory,
            hook_bus_factory=override.hook_bus_factory or self._hook_bus_factory,
            artifact_store_factory=override.artifact_store_factory,
            runner_retry_policy_factory=override.runner_retry_policy_factory,
            failure_policy_factory=override.failure_policy_factory,
            pipeline_factory=override.pipeline_factory,
        )

    @classmethod
    def _resolve_factory(
        cls,
        override_factory: Callable[..., object] | None,
        default_factory: Callable[..., object],
        *,
        plugins: tuple[object, ...],
        plugin_aware: bool = False,
    ) -> Callable[..., object]:
        if override_factory is not None:
            return override_factory
        if not plugin_aware:
            return default_factory
        return cls._bind_plugin_aware_factory(default_factory, plugins=plugins)

    @staticmethod
    def _bind_plugin_aware_factory(
        factory: Callable[..., object],
        *,
        plugins: tuple[object, ...],
    ) -> Callable[[dict[str, Any]], object]:
        accepts_plugins = RuntimeProfileResolver._factory_accepts_plugins(factory)

        def bound(config: dict[str, Any]):
            if accepts_plugins:
                return factory(config, plugins=plugins)
            return factory(config)

        return bound

    @staticmethod
    def _factory_accepts_plugins(factory: Callable[..., object]) -> bool:
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            return False

        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        return accepts_kwargs or "plugins" in signature.parameters


def default_capture_factory(capture_config: dict[str, object]):
    capture_type = str(capture_config.get("type", "window")).strip().lower()
    if capture_type == "video_stream":
        return create_video_stream_capture(capture_config)
    if capture_type in {"static_image", "image_file"}:
        return create_static_image_capture(capture_config)
    return WindowCapture(
        default_window_title=capture_config.get("window_title"),
        default_region=capture_config.get("region"),
    )


__all__ = [
    "RuntimeProfileResolver",
    "default_capture_factory",
]
