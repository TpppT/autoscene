from __future__ import annotations

import logging
from functools import lru_cache

from autoscene.runner.action_dispatcher import ActionDispatcher
from autoscene.runner.check_dispatcher import CheckDispatcher
from autoscene.runner.registry import (
    ActionRegistry,
    CheckRegistry,
    install_registry_plugins,
)
from autoscene.runner.step_specs import StepArgs


class _NullEmulator:
    def launch(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def execute(self, command: str) -> str:
        return str(command)

    def send(self, payload, endpoint=None, method=None, headers=None) -> str:
        del endpoint, method, headers
        return str(payload)


@lru_cache(maxsize=1)
def _default_action_spec_registry() -> ActionRegistry:
    return create_action_spec_registry()


@lru_cache(maxsize=1)
def _default_check_spec_registry() -> CheckRegistry:
    return create_check_spec_registry()


def _create_spec_dispatchers():
    action_dispatcher = ActionDispatcher(
        base_actions=None,
        emulator=_NullEmulator(),
        logger=logging.getLogger("ActionSpecRegistry"),
        output_dir=".",
        locate_actions=None,
    )
    check_dispatcher = CheckDispatcher(
        base_actions=None,
        screenshot_actions=None,
        readers={},
        log_sources={},
        logger=logging.getLogger("CheckSpecRegistry"),
        locate_actions=None,
    )
    return action_dispatcher, check_dispatcher


def create_action_spec_registry(
    plugins: tuple[object, ...] | list[object] | None = None,
) -> ActionRegistry:
    action_dispatcher, check_dispatcher = _create_spec_dispatchers()
    install_registry_plugins(
        action_dispatcher.registry,
        check_dispatcher.registry,
        plugins,
    )
    return action_dispatcher.registry


def create_check_spec_registry(
    plugins: tuple[object, ...] | list[object] | None = None,
) -> CheckRegistry:
    action_dispatcher, check_dispatcher = _create_spec_dispatchers()
    install_registry_plugins(
        action_dispatcher.registry,
        check_dispatcher.registry,
        plugins,
    )
    return check_dispatcher.registry


def build_registered_action_args(
    action_name: str,
    params: dict[str, object],
    *,
    plugins: tuple[object, ...] | list[object] | None = None,
) -> StepArgs | None:
    return _build_registered_args(
        action_name,
        params,
        plugins=plugins,
        default_registry_factory=_default_action_spec_registry,
        registry_factory=create_action_spec_registry,
    )


def build_registered_check_args(
    check_name: str,
    params: dict[str, object],
    *,
    plugins: tuple[object, ...] | list[object] | None = None,
) -> StepArgs | None:
    return _build_registered_args(
        check_name,
        params,
        plugins=plugins,
        default_registry_factory=_default_check_spec_registry,
        registry_factory=create_check_spec_registry,
    )


def _build_registered_args(
    name: str,
    params: dict[str, object],
    *,
    plugins: tuple[object, ...] | list[object] | None,
    default_registry_factory,
    registry_factory,
) -> StepArgs | None:
    registry = default_registry_factory() if not plugins else registry_factory(plugins)
    return registry.build_args(name, dict(params))
