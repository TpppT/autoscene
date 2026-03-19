from pathlib import Path

from PIL import Image

from autoscene.actions.base import BaseActions
from autoscene.capture.static_image_capture import (
    StaticImageCapture,
    create_static_image_capture,
)
from autoscene.runner.runtime_profile_resolver import default_capture_factory


def _write_image(path: Path, *, size: tuple[int, int] = (160, 120)) -> None:
    image = Image.new("RGB", size, color=(32, 96, 160))
    image.save(path)


def test_static_image_capture_returns_capture_result_with_mapping(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    _write_image(image_path, size=(320, 240))
    capture = StaticImageCapture(
        image_path,
        default_region={"left": 10, "top": 20, "width": 80, "height": 40},
        coordinate_region={"left": 100, "top": 200, "width": 80, "height": 40},
    )

    result = capture.capture_result()

    assert result.image.size == (80, 40)
    assert result.artifact_image.size == (320, 240)
    assert result.source == f"static_image:{image_path.as_posix()}"
    assert result.to_screen(5, 6) == (105, 206)
    assert capture.get_last_capture_result() is result


def test_base_actions_screenshot_accepts_static_image_capture(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    _write_image(image_path, size=(400, 300))
    capture = StaticImageCapture(
        image_path,
        default_region={"left": 5, "top": 6, "width": 40, "height": 30},
        coordinate_region={"left": 50, "top": 60, "width": 40, "height": 30},
    )
    actions = BaseActions(capture=capture)

    image = actions.screenshot(str(tmp_path / "captured.png"))

    assert image.size == (40, 30)
    assert (tmp_path / "captured.png").exists()

    capture_result = capture.get_last_capture_result()
    assert capture_result is not None
    assert actions.capture_to_screen(7, 9, capture_result=capture_result) == (57, 69)


def test_create_static_image_capture_supports_image_path_alias(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    _write_image(image_path)

    capture = create_static_image_capture(
        {
            "type": "static_image",
            "image_path": str(image_path),
            "region": {"left": 1, "top": 2, "width": 30, "height": 20},
            "screen_region": {"left": 101, "top": 102, "width": 30, "height": 20},
        }
    )
    result = capture.capture_result()

    assert result.image.size == (30, 20)
    assert result.to_screen(3, 4) == (104, 106)


def test_default_capture_factory_builds_static_image_capture(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    _write_image(image_path)

    capture = default_capture_factory(
        {"type": "static_image", "path": str(image_path)}
    )

    assert isinstance(capture, StaticImageCapture)
