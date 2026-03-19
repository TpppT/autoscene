from __future__ import annotations

import subprocess
from pathlib import Path

from autoscene.logs.interfaces import LogSource


class CommandLogSource(LogSource):
    def __init__(
        self,
        command: str,
        workdir: str | None = None,
        timeout: float = 10.0,
        shell: bool = True,
        include_stderr: bool = True,
        encoding: str = "utf-8",
        errors: str = "replace",
    ) -> None:
        self.command = str(command)
        self.workdir = None if workdir is None else str(Path(workdir))
        self.timeout = float(timeout)
        self.shell = bool(shell)
        self.include_stderr = bool(include_stderr)
        self.encoding = encoding
        self.errors = errors

    def read_text(self) -> str:
        completed = subprocess.run(
            self.command,
            capture_output=True,
            text=True,
            shell=self.shell,
            cwd=self.workdir,
            timeout=max(self.timeout, 0.0),
            encoding=self.encoding,
            errors=self.errors,
            check=False,
        )
        output = completed.stdout or ""
        if self.include_stderr and completed.stderr:
            if output:
                return f"{output}\n{completed.stderr}"
            return completed.stderr
        return output
