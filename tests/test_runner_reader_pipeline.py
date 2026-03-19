from pathlib import Path
import types

import pytest

import autoscene.runner.executor as exec_mod
from autoscene.core.models import TestCase as CaseModel


class FakeEmulator:
    def launch(self):
        return None

    def stop(self):
        return None


class FakeReader:
    def __init__(self):
        self.calls = []

    def read(self, image, query=None, region=None):
        self.calls.append(("read", image, query, region))
        return types.SimpleNamespace(value=126, score=0.9)


class FakeActions:
    def __init__(self, capture, detector, ocr, detectors=None):
        self.capture = capture
        self.detector = detector
        self.ocr = ocr
        self.detectors = detectors or {}
        self.calls = []
        self.screenshot_actions = self

    def screenshot(self, path=None):
        self.calls.append(("screenshot", path))
        return "frame"


def _make_profile_resolver() -> exec_mod.RuntimeProfileResolver:
    def capture_factory(capture_config: dict[str, object]):
        capture_type = str(capture_config.get("type", "window")).strip().lower()
        if capture_type == "video_stream":
            return exec_mod.create_video_stream_capture(capture_config)
        return exec_mod.WindowCapture(
            default_window_title=capture_config.get("window_title"),
            default_region=capture_config.get("region"),
        )

    return exec_mod.RuntimeProfileResolver(
        emulator_factory=exec_mod.create_emulator,
        detector_factory=exec_mod.create_detector,
        reader_factory=exec_mod.create_reader_adapter,
        log_source_factory=exec_mod.create_log_source,
        ocr_engine_factory=exec_mod.create_ocr_engine,
        capture_factory=capture_factory,
        actions_factory=exec_mod.ActionServices,
        action_dispatcher_factory=exec_mod.ActionDispatcher,
        check_dispatcher_factory=exec_mod.CheckDispatcher,
        hook_bus_factory=exec_mod.HookBus,
    )


def test_executor_runs_reader_verification_in_dedicated_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_reader = FakeReader()
    reader_calls = []

    monkeypatch.setattr(exec_mod, "create_emulator", lambda cfg: FakeEmulator())
    monkeypatch.setattr(exec_mod, "create_detector", lambda cfg: {"config": cfg})
    monkeypatch.setattr(
        exec_mod,
        "create_reader_adapter",
        lambda cfg: reader_calls.append(cfg) or fake_reader,
    )
    monkeypatch.setattr(exec_mod, "create_ocr_engine", lambda cfg: object())
    monkeypatch.setattr(exec_mod, "WindowCapture", lambda **kwargs: object())
    monkeypatch.setattr(exec_mod, "ActionServices", FakeActions)

    case = CaseModel(
        name="reader-only",
        detector={"type": "mock"},
        readers={"gauges": {"type": "opencv_qt_cluster_static"}},
        verification=[
            {
                "check": "reader_value_in_range",
                "reader": "gauges",
                "query": "speed",
                "expected": 126,
                "tolerance": 1,
            }
        ],
    )

    executor = exec_mod.TestExecutor(
        case=case,
        profile=_make_profile_resolver().resolve(),
        output_dir=Path("outputs_test"),
    )
    executor.run()

    assert reader_calls == [{"type": "opencv_qt_cluster_static"}]
    assert executor.context.resources.screenshot_actions.calls == [("screenshot", None)]
    assert fake_reader.calls == [("read", "frame", "speed", None)]
