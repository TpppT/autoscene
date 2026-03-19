from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from autoscene.actions.service_resolution import (
    BaseActionService,
    LocateActionService,
    ScreenshotActionService,
)
from autoscene.core.models import TestCase
from autoscene.runner.protocols import CaptureProtocol, EmulatorProtocol
from autoscene.runner.registry import install_registry_plugins
from autoscene.runner.runtime_models import (
    RuntimeContext,
    RuntimeMetadata,
    RuntimeProfile,
    RuntimeResources,
    RuntimeServices,
)
from autoscene.runner.runtime_policies import (
    ArtifactStore,
    FailurePolicy,
    HookBus,
    RunnerRetryPolicy,
)
from autoscene.vision.interfaces import Detector, OCREngine, ReaderAdapter


_REQUIRED_FACTORY_FIELDS = (
    "emulator_factory",
    "detector_factory",
    "reader_factory",
    "log_source_factory",
    "ocr_engine_factory",
    "capture_factory",
    "actions_factory",
    "action_dispatcher_factory",
    "check_dispatcher_factory",
)


@dataclass(frozen=True)
class RuntimeFactorySet:
    emulator_factory: Callable[[dict[str, Any]], EmulatorProtocol]
    detector_factory: Callable[[dict[str, Any]], Detector]
    reader_factory: Callable[[dict[str, Any]], ReaderAdapter]
    log_source_factory: Callable[[dict[str, Any]], object]
    ocr_engine_factory: Callable[[dict[str, Any]], OCREngine]
    capture_factory: Callable[[dict[str, Any]], CaptureProtocol]
    actions_factory: Callable[..., object]
    action_dispatcher_factory: Callable[..., object]
    check_dispatcher_factory: Callable[..., object]


@dataclass(frozen=True)
class RuntimeActionBindings:
    base_actions: BaseActionService | None
    locate_actions: LocateActionService | None
    screenshot_actions: ScreenshotActionService | None


class RuntimeFactoryResolver:
    def resolve(self, profile: RuntimeProfile) -> RuntimeFactorySet:
        resolved_factories = {
            field_name: self._require_factory(profile, field_name)
            for field_name in _REQUIRED_FACTORY_FIELDS
        }
        return RuntimeFactorySet(**resolved_factories)

    @staticmethod
    def _require_factory(profile: RuntimeProfile, field_name: str) -> Callable[..., Any]:
        factory = getattr(profile, field_name)
        if factory is None:
            raise RuntimeError(f"RuntimeProfile.{field_name} is required.")
        return factory

class RuntimeResourceFactory:
    def build(
        self,
        *,
        case: TestCase,
        factories: RuntimeFactorySet,
    ) -> RuntimeResources:
        emulator = factories.emulator_factory(case.emulator)
        detector = factories.detector_factory(case.detector)
        detectors = self._build_named_resources(case.detectors, factories.detector_factory)
        readers = self._build_named_resources(case.readers, factories.reader_factory)
        log_sources = self._build_named_resources(
            case.log_sources,
            factories.log_source_factory,
        )
        ocr_engine = factories.ocr_engine_factory(case.ocr)
        capture = factories.capture_factory(case.capture)
        actions = factories.actions_factory(
            capture=capture,
            detector=detector,
            detectors=detectors,
            ocr=ocr_engine,
        )
        bindings = self._resolve_action_bindings(actions)
        return RuntimeResources(
            emulator=emulator,
            detector=detector,
            detectors=detectors,
            readers=readers,
            log_sources=log_sources,
            ocr_engine=ocr_engine,
            capture=capture,
            base_actions=bindings.base_actions,
            locate_actions=bindings.locate_actions,
            screenshot_actions=bindings.screenshot_actions,
        )

    @staticmethod
    def _build_named_resources(
        configs: Mapping[str, dict[str, Any]],
        factory: Callable[[dict[str, Any]], Any],
    ) -> dict[str, Any]:
        return {str(name): factory(config) for name, config in configs.items()}

    @staticmethod
    def _resolve_action_bindings(actions: object) -> RuntimeActionBindings:
        base_actions = cast(
            BaseActionService | None,
            getattr(actions, "base_actions", None),
        )
        locate_actions = cast(
            LocateActionService | None,
            getattr(actions, "locate_actions", None),
        )
        screenshot_actions = cast(
            ScreenshotActionService | None,
            getattr(actions, "screenshot_actions", base_actions),
        )
        return RuntimeActionBindings(
            base_actions=base_actions,
            locate_actions=locate_actions,
            screenshot_actions=screenshot_actions,
        )


class RuntimeServiceFactory:
    def build(
        self,
        *,
        profile: RuntimeProfile,
        factories: RuntimeFactorySet,
        resources: RuntimeResources,
        logger: logging.Logger,
        output_path: Path,
    ) -> RuntimeServices:
        action_dispatcher = invoke_factory(
            factories.action_dispatcher_factory,
            base_actions=resources.base_actions,
            locate_actions=resources.locate_actions,
            emulator=resources.emulator,
            logger=logger,
            output_dir=output_path,
        )
        check_dispatcher = invoke_factory(
            factories.check_dispatcher_factory,
            base_actions=resources.base_actions,
            locate_actions=resources.locate_actions,
            screenshot_actions=resources.screenshot_actions,
            readers=resources.readers,
            log_sources=resources.log_sources,
            logger=logger,
        )
        hook_bus = self._resolve_profile_component(
            profile.hook_bus_factory,
            HookBus,
        )
        artifact_store = self._resolve_profile_component(
            profile.artifact_store_factory,
            ArtifactStore,
            output_dir=output_path,
            logger=logger,
        )
        retry_policy = self._resolve_profile_component(
            profile.runner_retry_policy_factory,
            RunnerRetryPolicy,
            logger=logger,
        )
        failure_policy = self._resolve_profile_component(
            profile.failure_policy_factory,
            FailurePolicy,
            artifact_store=artifact_store,
            logger=logger,
        )

        action_registry = self._resolve_registry(action_dispatcher)
        check_registry = self._resolve_registry(check_dispatcher)
        self._install_registry_plugins(
            action_registry,
            check_registry,
            profile.plugins,
        )

        return RuntimeServices(
            action_dispatcher=action_dispatcher,
            check_dispatcher=check_dispatcher,
            action_registry=action_registry,
            check_registry=check_registry,
            hook_bus=hook_bus,
            artifact_store=artifact_store,
            retry_policy=retry_policy,
            failure_policy=failure_policy,
        )

    @staticmethod
    def _resolve_profile_component(
        factory: Callable[..., Any] | None,
        default_factory: Callable[..., Any],
        **kwargs: Any,
    ) -> Any:
        return invoke_factory(factory or default_factory, **kwargs)

    @staticmethod
    def _resolve_registry(dispatcher: object) -> object:
        return getattr(dispatcher, "registry", dispatcher)

    @staticmethod
    def _install_registry_plugins(
        action_registry: object,
        check_registry: object,
        plugins: tuple[object, ...],
    ) -> None:
        if hasattr(action_registry, "register") and hasattr(check_registry, "register"):
            install_registry_plugins(action_registry, check_registry, plugins)

class RuntimeContextFactory:
    def __init__(
        self,
        logger_name: str = "TestExecutor",
        *,
        factory_resolver: RuntimeFactoryResolver | None = None,
        resource_factory: RuntimeResourceFactory | None = None,
        service_factory: RuntimeServiceFactory | None = None,
    ) -> None:
        self._logger_name = logger_name
        self.factory_resolver = factory_resolver or RuntimeFactoryResolver()
        self.resource_factory = resource_factory or RuntimeResourceFactory()
        self.service_factory = service_factory or RuntimeServiceFactory()

    def build(
        self,
        profile: RuntimeProfile,
        case: TestCase,
        output_dir: str | Path = "outputs",
    ) -> RuntimeContext:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(self._logger_name)
        factories = self.factory_resolver.resolve(profile)
        resources = self.resource_factory.build(case=case, factories=factories)
        services = self.service_factory.build(
            profile=profile,
            factories=factories,
            resources=resources,
            logger=logger,
            output_path=output_path,
        )
        metadata = RuntimeMetadata(
            case=case,
            profile=profile,
            output_dir=output_path,
            logger=logger,
        )
        return RuntimeContext(
            metadata=metadata,
            resources=resources,
            services=services,
        )


def invoke_factory(factory: Callable[..., Any], **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return factory(**kwargs)

    filtered_kwargs = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return factory(**filtered_kwargs)


__all__ = [
    "RuntimeActionBindings",
    "RuntimeContextFactory",
    "RuntimeFactoryResolver",
    "RuntimeFactorySet",
    "RuntimeResourceFactory",
    "RuntimeServiceFactory",
    "invoke_factory",
]
