from __future__ import annotations

from abc import ABC
from typing import Any


class EmulatorAdapter(ABC):
    """
    Logical device emulator used by testcase actions like `emulator_send`.
    """

    def launch(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def execute(self, command: str) -> str:
        raise NotImplementedError("execute is not implemented for this emulator adapter")

    def send(
        self,
        payload: Any,
        endpoint: str | None = None,
        method: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        del endpoint, method, headers
        if isinstance(payload, str):
            return self.execute(payload)
        return self.execute(str(payload))

