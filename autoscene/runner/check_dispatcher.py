from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from autoscene.actions.service_resolution import (
    BaseActionService,
    LocateActionService,
    ScreenshotActionService,
)
from autoscene.logs.interfaces import LogSource
from autoscene.runner.registry import CheckRegistry
from autoscene.runner.checks.log_checks import LogChecks
from autoscene.runner.checks.ui import BasicUIChecks, ReaderUIChecks
from autoscene.runner.step_specs import get_check_args_builder
from autoscene.vision.interfaces import ReaderAdapter


@dataclass(frozen=True)
class CheckDispatcherMetadata:
    logger: logging.Logger


@dataclass(frozen=True)
class CheckDispatcherResources:
    base_actions: BaseActionService | None
    locate_actions: LocateActionService | None
    screenshot_actions: ScreenshotActionService | None
    readers: dict[str, ReaderAdapter]
    log_sources: dict[str, LogSource]


class CheckDispatcher:
    def __init__(
        self,
        base_actions: object | None = None,
        screenshot_actions: object | None = None,
        readers: dict[str, ReaderAdapter] | None = None,
        log_sources: dict[str, LogSource] | None = None,
        logger: logging.Logger | None = None,
        *,
        locate_actions: LocateActionService | None = None,
    ) -> None:
        self.metadata = CheckDispatcherMetadata(
            logger=logger or logging.getLogger(self.__class__.__name__)
        )
        self.resources = CheckDispatcherResources(
            base_actions=base_actions,
            locate_actions=locate_actions,
            screenshot_actions=screenshot_actions,
            readers=dict(readers or {}),
            log_sources=dict(log_sources or {}),
        )
        self._basic_checks = BasicUIChecks(
            locate_actions=self.resources.locate_actions,
        )
        self._reader_checks = ReaderUIChecks(
            screenshot_actions=self.resources.screenshot_actions,
            readers=self.resources.readers,
            logger=self.metadata.logger,
        )
        self._log_checks = LogChecks(
            log_sources=self.resources.log_sources,
            logger=self.metadata.logger,
        )
        self.registry = CheckRegistry(self)
        self._register_builtin_handlers()

    def _register_builtin_handlers(self) -> None:
        self._register_handler_group(
            handlers=self._basic_checks.handlers,
            typed_handlers=self._basic_checks.typed_handlers,
        )
        self._register_handler_group(
            handlers=self._reader_checks.handlers,
            typed_handlers=self._reader_checks.typed_handlers,
        )
        self._register_handler_group(
            handlers=self._log_checks.handlers,
            typed_handlers=self._log_checks.typed_handlers,
        )

    def _register_handler_group(
        self,
        *,
        handlers: Mapping[str, Any],
        typed_handlers: Mapping[str, Any],
    ) -> None:
        for name, handler in handlers.items():
            self.registry.register(
                name,
                context_handler=handler,
                typed_handler=typed_handlers.get(name),
                args_builder=get_check_args_builder(name),
            )

    def register(self, check_name: str, **kwargs: Any) -> None:
        self.registry.register(check_name, **kwargs)

    def resolve(self, check_name: str):
        return self.registry.resolve(check_name)

    def dispatch_step(self, step) -> bool:
        return self.registry.dispatch_step(step)
