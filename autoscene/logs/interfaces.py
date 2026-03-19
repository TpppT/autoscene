from __future__ import annotations

from abc import ABC, abstractmethod


class LogSource(ABC):
    @abstractmethod
    def read_text(self) -> str:
        raise NotImplementedError
