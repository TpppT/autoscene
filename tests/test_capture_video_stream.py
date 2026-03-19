from pathlib import Path

from autoscene.actions.base import BaseActions
from autoscene.capture.video_stream_capture import (
    VideoStreamCapture,
    create_video_stream_capture,
)
from autoscene.runner.runtime_profile_resolver import default_capture_factory


class FakeImage:
    def __init__(self, size=(320, 240), *, name="frame") -> None:
        self.size = size
        self.name = name
        self.saved_to: str | None = None

    def crop(self, box):
        left, top, right, bottom = box
        return FakeImage(
            size=(int(right - left), int(bottom - top)),
            name=f"{self.name}:crop",
        )

    def save(self, path) -> None:
        self.saved_to = str(path)


class FakeFrameProvider:
    def __init__(self, frame=None, *, source_name="fake://stream") -> None:
        self._frame = frame or FakeImage()
        self._source_name = source_name
        self.closed = False

    @property
    def source_name(self) -> str:
        return self._source_name

    def read_frame(self):
        return self._frame

    def close(self) -> None:
        self.closed = True


def test_video_stream_capture_returns_capture_result_with_mapping() -> None:
    provider = FakeFrameProvider(frame=FakeImage(size=(640, 480)))
    capture = VideoStreamCapture(
        provider,
        default_region={"left": 10, "top": 20, "width": 80, "height": 40},
        coordinate_region={"left": 100, "top": 200, "width": 80, "height": 40},
    )

    result = capture.capture_result()

    assert result.image.size == (80, 40)
    assert result.artifact_image is provider.read_frame()
    assert result.source == "video_stream:fake://stream"
    assert result.to_screen(5, 6) == (105, 206)
    assert capture.get_last_capture_result() is result


def test_base_actions_screenshot_accepts_video_stream_capture(tmp_path: Path) -> None:
    frame = FakeImage(size=(400, 300))
    capture = VideoStreamCapture(
        FakeFrameProvider(frame=frame),
        default_region={"left": 5, "top": 6, "width": 40, "height": 30},
        coordinate_region={"left": 50, "top": 60, "width": 40, "height": 30},
    )
    actions = BaseActions(capture=capture)

    image = actions.screenshot(str(tmp_path / "frame.png"))

    assert image.size == (40, 30)
    assert frame.saved_to == str(tmp_path / "frame.png")

    capture_result = capture.get_last_capture_result()
    assert capture_result is not None
    assert actions.capture_to_screen(7, 9, capture_result=capture_result) == (57, 69)


def test_create_video_stream_capture_supports_screen_region_alias(monkeypatch) -> None:
    created = {}

    class FakeOpenCVProvider:
        def __init__(self, source, *, api_preference=None) -> None:
            created["source"] = source
            created["api_preference"] = api_preference
            self._source_name = f"provider:{source}"

        @property
        def source_name(self) -> str:
            return self._source_name

        def read_frame(self):
            return FakeImage()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "autoscene.capture.video_stream_capture.OpenCVVideoStreamProvider",
        FakeOpenCVProvider,
    )

    capture = create_video_stream_capture(
        {
            "type": "video_stream",
            "source": "rtsp://demo",
            "region": {"left": 1, "top": 2, "width": 30, "height": 20},
            "screen_region": {"left": 101, "top": 102, "width": 30, "height": 20},
        }
    )

    result = capture.capture_result()

    assert created == {"source": "rtsp://demo", "api_preference": None}
    assert result.to_screen(3, 4) == (104, 106)


def test_default_capture_factory_builds_video_stream_capture(monkeypatch) -> None:
    created = {}

    def fake_create_video_stream_capture(config):
        created["config"] = dict(config)
        return "video-capture"

    monkeypatch.setattr(
        "autoscene.runner.runtime_profile_resolver.create_video_stream_capture",
        fake_create_video_stream_capture,
    )

    capture = default_capture_factory({"type": "video_stream", "source": "demo"})

    assert capture == "video-capture"
    assert created["config"] == {"type": "video_stream", "source": "demo"}
