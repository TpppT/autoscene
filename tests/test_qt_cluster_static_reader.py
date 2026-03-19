import numpy as np
import pytest

from autoscene.vision import create_reader_adapter
from autoscene.imaging.opencv.base import OpenCVAdapterBase
from autoscene.vision.opencv.readers.qt_cluster_static_reader import (
    OpenCVQtClusterStaticReader,
)


def test_qt_cluster_static_reader_registered() -> None:
    reader = create_reader_adapter({"type": "opencv_qt_cluster_static"})
    assert isinstance(reader, OpenCVQtClusterStaticReader)


def test_qt_cluster_static_reader_maps_overlay_angle_to_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = OpenCVQtClusterStaticReader()
    monkeypatch.setattr(
        reader,
        "clip_region",
        lambda image, region=None: (np.zeros((480, 800, 3), dtype=np.uint8), (0, 0)),
    )
    monkeypatch.setattr(
        reader,
        "_detect_overlay",
        lambda frame, spec: {
            "center_xy": [190, 214],
            "angle_deg": 170.875,
            "score": 72.0,
            "gauge_radius": 145,
        }
        if spec.name == "speed"
        else None,
    )

    result = reader.read("frame", query="speed")

    assert result.value == 82
    assert result.score == pytest.approx(0.9)
    assert result.label == "speed"
    assert result.source == "opencv_qt_cluster_static"


def test_opencv_adapter_base_caches_image_path_reads(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"frame")
    adapter = OpenCVAdapterBase()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    read_calls = []

    class FakeCV2:
        @staticmethod
        def imread(path):
            read_calls.append(path)
            return frame.copy()

    monkeypatch.setattr(adapter, "require_cv2", lambda: FakeCV2)

    first = adapter.to_ndarray(str(image_path))
    second = adapter.to_ndarray(str(image_path))
    first[0, 0, 0] = 255

    assert read_calls == [str(image_path)]
    assert second[0, 0, 0] == 0
