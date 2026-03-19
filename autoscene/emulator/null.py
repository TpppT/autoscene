from __future__ import annotations

from typing import Any

from autoscene.emulator.base import EmulatorAdapter


class NullEmulatorAdapter(EmulatorAdapter):
    def execute(self, command: str) -> str:
        raise RuntimeError(
            "No emulator configured. Add 'emulator' config in testcase YAML "
            "or avoid emulator_* actions."
        )

    def send(
        self,
        payload: Any,
        endpoint: str | None = None,
        method: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        del payload, endpoint, method, headers
        raise RuntimeError(
            "No emulator configured. Add 'emulator' config in testcase YAML "
            "before using 'emulator_send'."
        )

