from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from autoscene.emulator.base import EmulatorAdapter
from autoscene.emulator.network_device import NetworkDeviceEmulatorAdapter
from autoscene.emulator.null import NullEmulatorAdapter
from autoscene.emulator.qt_drive_cluster import QtDriveClusterEmulatorAdapter

EmulatorFactory = Callable[..., EmulatorAdapter]


class EmulatorPlugin(Protocol):
    namespace: str | None
    override: bool

    def register_emulators(self, registry: "ScopedEmulatorRegistry") -> None: ...


@dataclass(frozen=True)
class _FactoryRegistration:
    name: str
    factory: EmulatorFactory


class EmulatorRegistry:
    registry_name = "emulator"

    def __init__(self) -> None:
        self._registrations: dict[str, _FactoryRegistration] = {}

    def clone(self) -> "EmulatorRegistry":
        cloned = EmulatorRegistry()
        cloned._registrations = dict(self._registrations)
        return cloned

    def register(
        self,
        name: str,
        factory: EmulatorFactory,
        *,
        namespace: str | None = None,
        override: bool = True,
    ) -> None:
        qualified_name = _qualify_registry_name(name, namespace=namespace)
        if not override and qualified_name in self._registrations:
            raise ValueError(f"Emulator already registered: {qualified_name}")
        self._registrations[qualified_name] = _FactoryRegistration(
            name=qualified_name,
            factory=factory,
        )

    def resolve(self, name: str) -> EmulatorFactory | None:
        registration = self._registrations.get(str(name).strip().lower())
        if registration is None:
            return None
        return registration.factory

    def create(self, config: dict[str, Any]) -> EmulatorAdapter:
        payload = dict(config or {})
        emulator_type = str(payload.pop("type", "none")).strip().lower()
        if not emulator_type:
            emulator_type = "none"
        factory = self.resolve(emulator_type)
        if factory is None:
            available = ", ".join(sorted(self._registrations))
            raise ValueError(f"Unknown emulator type '{emulator_type}'. Available: {available}")
        return _invoke_factory(factory, payload)


class ScopedEmulatorRegistry:
    def __init__(
        self,
        registry: EmulatorRegistry,
        *,
        namespace: str | None = None,
        override: bool = False,
    ) -> None:
        self._registry = registry
        self._namespace = _normalize_namespace(namespace)
        self._override = bool(override)

    def register(self, name: str, factory: EmulatorFactory, **kwargs: Any) -> None:
        kwargs.setdefault("namespace", self._namespace)
        kwargs.setdefault("override", self._override)
        self._registry.register(name, factory, **kwargs)


def _register_builtin_emulators(registry: EmulatorRegistry) -> None:
    registry.register("none", NullEmulatorAdapter)
    registry.register("network_device", NetworkDeviceEmulatorAdapter)
    registry.register("external_device", NetworkDeviceEmulatorAdapter)
    registry.register("qt_drive_cluster", QtDriveClusterEmulatorAdapter)
    registry.register("qt_cluster", QtDriveClusterEmulatorAdapter)


def build_emulator_registry(
    plugins: tuple[object, ...] | list[object] | None = None,
) -> EmulatorRegistry:
    registry = _DEFAULT_EMULATOR_REGISTRY.clone()
    install_emulator_plugins(registry, plugins)
    return registry


def install_emulator_plugins(
    registry: EmulatorRegistry,
    plugins: tuple[object, ...] | list[object] | None,
) -> EmulatorRegistry:
    for plugin in tuple(plugins or ()):
        register_emulators = getattr(plugin, "register_emulators", None)
        if not callable(register_emulators):
            continue
        namespace = _normalize_namespace(getattr(plugin, "namespace", None))
        override = bool(getattr(plugin, "override", False))
        register_emulators(
            ScopedEmulatorRegistry(
                registry,
                namespace=namespace,
                override=override,
            )
        )
    return registry


def register_emulator(name: str, factory: EmulatorFactory) -> None:
    _DEFAULT_EMULATOR_REGISTRY.register(name, factory)


def create_emulator(
    config: dict[str, Any],
    *,
    plugins: tuple[object, ...] | list[object] | None = None,
) -> EmulatorAdapter:
    if not plugins:
        return _DEFAULT_EMULATOR_REGISTRY.create(config)
    return build_emulator_registry(plugins=plugins).create(config)


def _invoke_factory(factory: EmulatorFactory, payload: dict[str, Any]) -> EmulatorAdapter:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**payload)

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return factory(**payload)

    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    filtered_payload = {key: value for key, value in payload.items() if key in accepted}
    return factory(**filtered_payload)


def _normalize_namespace(namespace: str | None) -> str | None:
    if namespace is None:
        return None
    text = str(namespace).strip().lower()
    return text or None


def _qualify_registry_name(name: str, *, namespace: str | None = None) -> str:
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("Emulator registration requires a non-empty name.")
    normalized_namespace = _normalize_namespace(namespace)
    if normalized_namespace:
        return f"{normalized_namespace}.{normalized_name}"
    return normalized_name


_DEFAULT_EMULATOR_REGISTRY = EmulatorRegistry()
_register_builtin_emulators(_DEFAULT_EMULATOR_REGISTRY)
