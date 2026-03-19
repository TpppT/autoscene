from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError

from autoscene.emulator.base import EmulatorAdapter
from autoscene.emulator.network_device import NetworkDeviceEmulatorAdapter


@dataclass(frozen=True)
class QtDriveClusterCommand:
    name: str
    endpoint: str
    method: str


@dataclass(frozen=True)
class QtDriveClusterProcessConfig:
    executable_path: str | None
    launch_args: tuple[str, ...]
    working_dir: str | None
    ready_timeout_seconds: float
    ready_interval_seconds: float
    stop_demo_on_launch: bool
    terminate_on_stop: bool


class QtDriveClusterProcessController:
    def __init__(
        self,
        config: QtDriveClusterProcessConfig,
        *,
        logger: logging.Logger,
    ) -> None:
        self._config = config
        self._logger = logger
        self._process: subprocess.Popen[str] | None = None

    def launch(self, *, on_ready: Callable[[], None], after_ready: Callable[[], None] | None) -> None:
        if not self._config.executable_path:
            return
        if self._process is not None and self._process.poll() is None:
            on_ready()
            return

        executable = Path(self._config.executable_path)
        command = [str(executable), *self._config.launch_args]
        self._logger.info("launch qt cluster process: %s", command)
        self._process = subprocess.Popen(
            command,
            cwd=self._config.working_dir or (str(executable.parent) if executable.parent else None),
        )
        on_ready()
        if self._config.stop_demo_on_launch and after_ready is not None:
            after_ready()

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is not None:
            self._process = None
            return
        if self._config.terminate_on_stop:
            self._logger.info("terminate qt cluster process")
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        self._process = None

    def poll(self) -> int | None:
        if self._process is None:
            return None
        return self._process.poll()


class QtDriveClusterStateNormalizer:
    _FIELD_ALIASES: dict[str, str] = {
        "temperature": "coolant_temp",
        "temp": "coolant_temp",
        "mode_name": "mode",
        "left_indicator": "turn_left",
        "right_indicator": "turn_right",
        "fuel_percent": "fuel",
        "battery_percent": "battery",
    }

    def __init__(self, *, state_defaults: Mapping[str, Any] | None = None) -> None:
        self._state_defaults = self.normalize_mapping(state_defaults or {})

    def normalize_state_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(self._state_defaults)
        normalized.update(self.normalize_mapping(payload))
        return normalized

    @classmethod
    def normalize_mapping(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            target_key = cls._FIELD_ALIASES.get(str(key), str(key))
            normalized_value = cls._normalize_value(target_key, value)
            normalized[target_key] = normalized_value
            if target_key == "mode" and "mode_index" not in normalized and "mode_index" not in payload:
                mode_index = cls._mode_to_index(normalized_value)
                if mode_index is not None:
                    normalized["mode_index"] = mode_index
        return normalized

    @staticmethod
    def _normalize_value(key: str, value: Any) -> Any:
        if key == "mode" and isinstance(value, str):
            return value.strip().upper()
        if key == "gear" and isinstance(value, str):
            return value.strip().upper()
        if key == "playlist" and isinstance(value, tuple):
            return list(value)
        return value

    @staticmethod
    def _mode_to_index(value: Any) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return max(0, min(int(value), 2))
        text = str(value).strip().upper()
        if text == "SPORT":
            return 1
        if text == "ECO":
            return 2
        if text == "NORMAL":
            return 0
        return None


class QtDriveClusterEmulatorAdapter(EmulatorAdapter):
    """
    Modern qt-drive-cluster adapter built from:
    - a generic HTTP transport
    - a domain payload normalizer
    - an optional local process controller
    """

    _COMMANDS: dict[str, QtDriveClusterCommand] = {
        "state": QtDriveClusterCommand("state", "/state", "GET"),
        "demo_start": QtDriveClusterCommand("demo_start", "/demo/start", "POST"),
        "demo_stop": QtDriveClusterCommand("demo_stop", "/demo/stop", "POST"),
    }

    def __init__(
        self,
        base_url: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        scheme: str = "http",
        default_endpoint: str = "/state",
        timeout_seconds: float = 8.0,
        default_headers: dict[str, str] | None = None,
        state_defaults: dict[str, Any] | None = None,
        executable_path: str | None = None,
        launch_args: Sequence[str] | None = None,
        working_dir: str | None = None,
        ready_timeout_seconds: float = 10.0,
        ready_interval_seconds: float = 0.25,
        stop_demo_on_launch: bool = False,
        terminate_on_stop: bool = True,
    ) -> None:
        self._transport = NetworkDeviceEmulatorAdapter(
            base_url=base_url,
            host=host,
            port=port,
            scheme=scheme,
            default_endpoint=default_endpoint,
            timeout_seconds=timeout_seconds,
            default_headers=default_headers,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self._normalizer = QtDriveClusterStateNormalizer(state_defaults=state_defaults)
        self._process_controller = QtDriveClusterProcessController(
            QtDriveClusterProcessConfig(
                executable_path=executable_path,
                launch_args=tuple(str(arg) for arg in (launch_args or ())),
                working_dir=None if working_dir is None else str(working_dir),
                ready_timeout_seconds=max(float(ready_timeout_seconds), 0.0),
                ready_interval_seconds=max(float(ready_interval_seconds), 0.05),
                stop_demo_on_launch=bool(stop_demo_on_launch),
                terminate_on_stop=bool(terminate_on_stop),
            ),
            logger=self.logger,
        )

    def launch(self) -> None:
        self._process_controller.launch(
            on_ready=self._wait_until_ready,
            after_ready=lambda: self.execute("demo_stop"),
        )

    def stop(self) -> None:
        self._process_controller.stop()

    def execute(self, command: str) -> str:
        resolved = self._resolve_command(command)
        return self._transport.send(
            payload=None,
            endpoint=resolved.endpoint,
            method=resolved.method,
        )

    def send(
        self,
        payload: Any,
        endpoint: str | None = None,
        method: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        if isinstance(payload, Mapping):
            command_name = self._extract_command(payload)
            if command_name is not None:
                resolved = self._resolve_command(command_name)
                return self._transport.send(
                    payload=None,
                    endpoint=resolved.endpoint,
                    method=resolved.method,
                    headers=headers,
                )
            payload = self._normalizer.normalize_state_payload(payload)
        return self._transport.send(
            payload=payload,
            endpoint=endpoint,
            method=method,
            headers=headers,
        )

    def _wait_until_ready(self) -> None:
        config = self._process_controller._config
        deadline = time.time() + config.ready_timeout_seconds
        last_error: Exception | None = None
        while time.time() <= deadline:
            if self._process_controller.poll() is not None:
                raise RuntimeError(
                    f"qt cluster process exited early with code {self._process_controller.poll()}"
                )
            try:
                self._transport.send(payload=None, endpoint="/", method="GET")
                return
            except RuntimeError as exc:
                last_error = exc
            except URLError as exc:  # pragma: no cover - network adapter already wraps this
                last_error = exc
            time.sleep(config.ready_interval_seconds)
        raise RuntimeError(f"qt cluster API did not become ready: {last_error}")

    @classmethod
    def _resolve_command(cls, command: str) -> QtDriveClusterCommand:
        normalized = str(command).strip().casefold().replace("-", "_").replace(" ", "_")
        resolved = cls._COMMANDS.get(normalized)
        if resolved is None:
            available = ", ".join(sorted(cls._COMMANDS))
            raise ValueError(
                f"Unsupported qt_drive_cluster command '{command}'. Available: {available}"
            )
        return resolved

    @staticmethod
    def _extract_command(payload: Mapping[str, Any]) -> str | None:
        keys = set(payload)
        if keys == {"command"} and payload.get("command") is not None:
            return str(payload["command"])
        return None
