from __future__ import annotations

from collections.abc import Callable
from typing import Any

from autoscene.logs.command_log_source import CommandLogSource
from autoscene.logs.file_log_source import FileLogSource
from autoscene.logs.interfaces import LogSource

LogSourceFactory = Callable[..., LogSource]

_LOG_SOURCE_REGISTRY: dict[str, LogSourceFactory] = {
    "file": FileLogSource,
    "command": CommandLogSource,
}


def register_log_source(name: str, factory: LogSourceFactory) -> None:
    _LOG_SOURCE_REGISTRY[name.lower()] = factory


def create_log_source(config: dict[str, Any]) -> LogSource:
    config = dict(config or {})
    source_type = str(config.pop("type", "file")).lower()
    if source_type not in _LOG_SOURCE_REGISTRY:
        available = ", ".join(sorted(_LOG_SOURCE_REGISTRY))
        raise ValueError(f"Unknown log source '{source_type}'. Available: {available}")
    return _LOG_SOURCE_REGISTRY[source_type](**config)
