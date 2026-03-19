from __future__ import annotations

import logging
from types import SimpleNamespace

import autoscene.runner.runtime_assembly as assembly_mod
from autoscene.runner.runtime_models import RuntimeProfile, RuntimeResources


class FakeRegistry:
    def __init__(self, name: str) -> None:
        self.name = name
        self.registered: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def register(self, *args, **kwargs) -> None:
        self.registered.append((args, dict(kwargs)))


class DispatcherWithRegistry:
    def __init__(self, registry: FakeRegistry) -> None:
        self.registry = registry


def _make_factory_set(*, action_dispatcher_factory, check_dispatcher_factory):
    noop = lambda config: config
    return assembly_mod.RuntimeFactorySet(
        emulator_factory=noop,
        detector_factory=noop,
        reader_factory=noop,
        log_source_factory=noop,
        ocr_engine_factory=noop,
        capture_factory=noop,
        actions_factory=lambda **kwargs: kwargs,
        action_dispatcher_factory=action_dispatcher_factory,
        check_dispatcher_factory=check_dispatcher_factory,
    )


def _make_resources() -> RuntimeResources:
    return RuntimeResources(
        emulator=object(),
        detector=object(),
        detectors={},
        readers={"main": object()},
        log_sources={"backend": object()},
        ocr_engine=object(),
        capture=object(),
        base_actions=object(),
        locate_actions=object(),
        screenshot_actions=object(),
    )


def test_runtime_service_factory_builds_custom_services_and_installs_plugins(
    monkeypatch,
    tmp_path,
) -> None:
    action_registry = FakeRegistry("actions")
    check_registry = FakeRegistry("checks")
    calls: dict[str, object] = {}
    installed: dict[str, object] = {}
    hook_bus = object()
    resources = _make_resources()
    logger = logging.getLogger("runtime-service-factory")

    def action_dispatcher_factory(*, base_actions, logger):
        calls["action_dispatcher"] = {
            "base_actions": base_actions,
            "logger": logger,
        }
        return DispatcherWithRegistry(action_registry)

    def check_dispatcher_factory(*, screenshot_actions, readers):
        calls["check_dispatcher"] = {
            "screenshot_actions": screenshot_actions,
            "readers": readers,
        }
        return DispatcherWithRegistry(check_registry)

    def artifact_store_factory(output_dir):
        calls["artifact_store"] = output_dir
        return SimpleNamespace(output_dir=output_dir)

    def retry_policy_factory():
        calls["retry_policy"] = True
        return "retry-policy"

    def failure_policy_factory(artifact_store):
        calls["failure_policy"] = artifact_store
        return SimpleNamespace(artifact_store=artifact_store)

    monkeypatch.setattr(
        assembly_mod,
        "install_registry_plugins",
        lambda action_reg, check_reg, plugins: installed.update(
            {
                "action_registry": action_reg,
                "check_registry": check_reg,
                "plugins": plugins,
            }
        ),
    )

    profile = RuntimeProfile(
        plugins=("plugin",),
        hook_bus_factory=lambda: hook_bus,
        artifact_store_factory=artifact_store_factory,
        runner_retry_policy_factory=retry_policy_factory,
        failure_policy_factory=failure_policy_factory,
    )
    services = assembly_mod.RuntimeServiceFactory().build(
        profile=profile,
        factories=_make_factory_set(
            action_dispatcher_factory=action_dispatcher_factory,
            check_dispatcher_factory=check_dispatcher_factory,
        ),
        resources=resources,
        logger=logger,
        output_path=tmp_path,
    )

    assert services.action_registry is action_registry
    assert services.check_registry is check_registry
    assert services.hook_bus is hook_bus
    assert services.retry_policy == "retry-policy"
    assert services.failure_policy.artifact_store is services.artifact_store
    assert calls["action_dispatcher"] == {
        "base_actions": resources.base_actions,
        "logger": logger,
    }
    assert calls["check_dispatcher"] == {
        "screenshot_actions": resources.screenshot_actions,
        "readers": resources.readers,
    }
    assert calls["artifact_store"] == tmp_path
    assert installed == {
        "action_registry": action_registry,
        "check_registry": check_registry,
        "plugins": ("plugin",),
    }


def test_runtime_service_factory_uses_dispatchers_as_registries_and_default_policies(
    monkeypatch,
    tmp_path,
) -> None:
    action_dispatcher = FakeRegistry("actions")
    check_dispatcher = FakeRegistry("checks")
    installed: dict[str, object] = {}

    monkeypatch.setattr(
        assembly_mod,
        "install_registry_plugins",
        lambda action_reg, check_reg, plugins: installed.update(
            {
                "action_registry": action_reg,
                "check_registry": check_reg,
                "plugins": plugins,
            }
        ),
    )

    services = assembly_mod.RuntimeServiceFactory().build(
        profile=RuntimeProfile(plugins=("plugin",)),
        factories=_make_factory_set(
            action_dispatcher_factory=lambda **kwargs: action_dispatcher,
            check_dispatcher_factory=lambda **kwargs: check_dispatcher,
        ),
        resources=_make_resources(),
        logger=logging.getLogger("runtime-service-factory-defaults"),
        output_path=tmp_path,
    )

    assert services.action_registry is action_dispatcher
    assert services.check_registry is check_dispatcher
    assert isinstance(services.hook_bus, assembly_mod.HookBus)
    assert isinstance(services.artifact_store, assembly_mod.ArtifactStore)
    assert isinstance(services.retry_policy, assembly_mod.RunnerRetryPolicy)
    assert isinstance(services.failure_policy, assembly_mod.FailurePolicy)
    assert services.failure_policy.artifact_store is services.artifact_store
    assert installed == {
        "action_registry": action_dispatcher,
        "check_registry": check_dispatcher,
        "plugins": ("plugin",),
    }
