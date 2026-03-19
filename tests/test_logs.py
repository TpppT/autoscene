import subprocess
import types

import pytest

from autoscene.core.exceptions import VerificationError
from autoscene.logs.command_log_source import CommandLogSource
from autoscene.logs.file_log_source import FileLogSource
from autoscene.logs.interfaces import LogSource
from autoscene.logs.registry import create_log_source, register_log_source
from autoscene.runner.checks.log_checks import LogChecks


def test_file_log_source_reads_existing_file(tmp_path) -> None:
    path = tmp_path / "app.log"
    path.write_text("hello\nworld", encoding="utf-8")
    source = FileLogSource(str(path))
    assert source.read_text() == "hello\nworld"


def test_file_log_source_returns_empty_for_missing_file(tmp_path) -> None:
    source = FileLogSource(str(tmp_path / "missing.log"), missing_ok=True)
    assert source.read_text() == ""


def test_command_log_source_reads_stdout_and_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        return types.SimpleNamespace(stdout="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)
    source = CommandLogSource("echo hi")
    assert source.read_text() == "out\nerr"


def test_log_source_registry_create_and_register() -> None:
    class DemoLogSource(LogSource):
        def read_text(self) -> str:
            return "demo"

    register_log_source("demo", DemoLogSource)
    source = create_log_source({"type": "demo"})
    assert source.read_text() == "demo"


def test_log_checks_contains_and_regex() -> None:
    source = types.SimpleNamespace(read_text=lambda: "order created\npayment ok")
    checks = LogChecks(log_sources={"backend": source})
    assert checks.handlers["log_contains"]({"source": "backend", "contains": "created"}) is True
    assert checks.handlers["log_contains"]({"source": "backend", "regex": r"payment\s+ok"}) is True


def test_log_checks_wait_for_log(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter(["booting", "booting", "ready"])
    source = types.SimpleNamespace(read_text=lambda: next(values))
    checks = LogChecks(log_sources={"backend": source})
    timeline = iter([0.0, 0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr("time.time", lambda: next(timeline))
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda interval: sleep_calls.append(interval))
    ok = checks.handlers["wait_for_log"](
        {"source": "backend", "contains": "ready", "timeout": 1.0, "interval": 0.1}
    )
    assert ok is True
    assert sleep_calls == [0.1, 0.1]


def test_log_checks_require_source_when_multiple_configs() -> None:
    checks = LogChecks(
        log_sources={
            "backend": types.SimpleNamespace(read_text=lambda: "a"),
            "worker": types.SimpleNamespace(read_text=lambda: "b"),
        }
    )
    with pytest.raises(VerificationError, match="require 'source'"):
        checks.handlers["log_contains"]({"contains": "a"})
