from __future__ import annotations

from pathlib import Path

from autoscene.logs.interfaces import LogSource


class FileLogSource(LogSource):
    def __init__(
        self,
        path: str,
        encoding: str = "utf-8",
        errors: str = "replace",
        missing_ok: bool = True,
    ) -> None:
        self.path = Path(path)
        self.encoding = encoding
        self.errors = errors
        self.missing_ok = bool(missing_ok)

    def read_text(self) -> str:
        if not self.path.exists():
            if self.missing_ok:
                return ""
            raise FileNotFoundError(f"Log file does not exist: {self.path}")
        return self.path.read_text(encoding=self.encoding, errors=self.errors)
