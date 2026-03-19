import argparse
from pathlib import Path

import pytest

import run_tests


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_tests.py", "examples/sample_case.yaml"])
    args = run_tests.parse_args()
    assert args.case == "examples/sample_case.yaml"
    assert args.output_dir == "outputs"
    assert args.log_level == "INFO"


def test_parse_args_with_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_tests.py",
            "case.yaml",
            "--output-dir",
            "artifacts",
            "--log-level",
            "DEBUG",
        ],
    )
    args = run_tests.parse_args()
    assert args.case == "case.yaml"
    assert args.output_dir == "artifacts"
    assert args.log_level == "DEBUG"


def test_main_success(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(case="demo.yaml", output_dir="out", log_level="WARNING")
    calls = {
        "configure_dpi": 0,
        "configure_logging": None,
        "load_case": None,
        "resolve_profile": 0,
        "runner_init": None,
        "runner_run": 0,
    }
    resolved_profile = object()

    class FakeRunner:
        def __init__(self, case, profile, output_dir):
            calls["runner_init"] = (case, profile, output_dir)

        def run(self):
            calls["runner_run"] += 1

    class FakeProfileResolver:
        def resolve(self):
            calls["resolve_profile"] += 1
            return resolved_profile

    monkeypatch.setattr(run_tests, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_tests,
        "configure_windows_dpi_awareness",
        lambda: calls.__setitem__("configure_dpi", calls["configure_dpi"] + 1),
    )
    monkeypatch.setattr(
        run_tests,
        "configure_logging",
        lambda output_dir, case_path, log_level: calls.__setitem__(
            "configure_logging",
            (output_dir, case_path, log_level),
        )
        or Path(output_dir) / "demo.log",
    )
    monkeypatch.setattr(
        run_tests,
        "load_test_case",
        lambda path: calls.__setitem__("load_case", path) or {"name": "demo"},
    )
    monkeypatch.setattr(run_tests, "RuntimeProfileResolver", FakeProfileResolver)
    monkeypatch.setattr(run_tests, "TestExecutor", FakeRunner)

    assert run_tests.main() == 0
    assert calls["configure_dpi"] == 1
    assert calls["load_case"] == "demo.yaml"
    assert calls["configure_logging"] == ("out", "demo.yaml", "WARNING")
    assert calls["resolve_profile"] == 1
    assert calls["runner_init"] == ({"name": "demo"}, resolved_profile, "out")
    assert calls["runner_run"] == 1
