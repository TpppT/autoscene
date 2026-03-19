import io
import types

import pytest

import autoscene.capture.window_capture as wc
from autoscene.capture.window_capture import (
    CaptureCoordinateSpace,
    CaptureResult,
    CaptureRegion,
    CaptureScorer,
    WindowCapture,
    WindowLocator,
)
from autoscene.core.exceptions import DependencyMissingError


def test_resolve_region_from_dict() -> None:
    capture = WindowCapture()
    region = capture._resolve_region(None, {"left": 1, "top": 2, "width": 3, "height": 4})
    assert isinstance(region, CaptureRegion)
    assert (region.left, region.top, region.width, region.height) == (1, 2, 3, 4)


def test_resolve_region_from_default_region() -> None:
    capture = WindowCapture(default_region={"left": 5, "top": 6, "width": 7, "height": 8})
    region = capture._resolve_region(None, None)
    assert region == CaptureRegion(left=5, top=6, width=7, height=8)


def test_resolve_region_from_window_title(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = WindowCapture(default_window_title="Demo App")
    monkeypatch.setattr(
        WindowLocator,
        "_find_window_region",
        lambda self, title: CaptureRegion(left=9, top=10, width=11, height=12),
    )
    region = capture._resolve_region(None, None)
    assert region == CaptureRegion(left=9, top=10, width=11, height=12)


def test_resolve_region_from_bound_window_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = WindowCapture(default_window_title="Demo App")
    capture.bind_window_handle(123)
    monkeypatch.setattr(
        WindowLocator,
        "_find_window_region_by_handle",
        lambda self, handle: CaptureRegion(left=1, top=2, width=3, height=4),
    )
    region = capture._resolve_region(None, None)
    assert region == CaptureRegion(left=1, top=2, width=3, height=4)


def test_resolve_region_refreshes_bound_window_region(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = WindowCapture(default_window_title="Demo App")
    capture.bind_window_handle(123)
    calls = []
    regions = [
        CaptureRegion(left=1, top=2, width=3, height=4),
        CaptureRegion(left=10, top=20, width=30, height=40),
    ]

    def fake_region(handle):
        calls.append(handle)
        return regions[len(calls) - 1]

    monkeypatch.setattr(
        WindowLocator,
        "_find_window_region_by_handle",
        lambda self, handle: fake_region(handle),
    )

    assert capture._resolve_region(None, None) == CaptureRegion(left=1, top=2, width=3, height=4)
    assert capture._resolve_region(None, None) == CaptureRegion(left=10, top=20, width=30, height=40)
    assert calls == [123, 123]


def test_resolve_region_clears_stale_bound_window_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = WindowCapture(default_window_title="Demo App")
    capture.bind_window_handle(123)
    monkeypatch.setattr(
        WindowLocator,
        "_find_window_region_by_handle",
        lambda self, handle: (_ for _ in ()).throw(RuntimeError("missing")),
    )
    monkeypatch.setattr(
        WindowLocator,
        "_find_window_region",
        lambda self, title: CaptureRegion(left=5, top=6, width=7, height=8),
    )
    region = capture._resolve_region(None, None)
    assert region == CaptureRegion(left=5, top=6, width=7, height=8)
    assert capture.get_bound_window_handle() is None


def test_find_window_region_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wc, "pygetwindow", None)
    with pytest.raises(DependencyMissingError):
        WindowLocator()._find_window_region("any")


def test_find_window_region_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pg = types.SimpleNamespace(getWindowsWithTitle=lambda _: [])
    monkeypatch.setattr(wc, "pygetwindow", fake_pg)
    with pytest.raises(RuntimeError, match="No window found"):
        WindowLocator()._find_window_region("missing")


def test_find_window_region_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_window = types.SimpleNamespace(left=1, top=2, width=100, height=200)
    fake_pg = types.SimpleNamespace(getWindowsWithTitle=lambda _: [fake_window])
    monkeypatch.setattr(wc, "pygetwindow", fake_pg)
    region = WindowLocator()._find_window_region("ok")
    assert region == CaptureRegion(left=1, top=2, width=100, height=200)


def test_find_window_region_by_handle_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_window = types.SimpleNamespace(_hWnd=55, left=1, top=2, width=100, height=200)
    fake_pg = types.SimpleNamespace(getAllWindows=lambda: [fake_window])
    monkeypatch.setattr(wc, "pygetwindow", fake_pg)
    monkeypatch.setattr(
        WindowLocator,
        "_outer_region_from_handle",
        staticmethod(lambda handle: None),
    )
    monkeypatch.setattr(
        WindowLocator,
        "_client_region_from_handle",
        staticmethod(lambda handle: None),
    )
    region = WindowLocator()._find_window_region_by_handle(55)
    assert region == CaptureRegion(left=1, top=2, width=100, height=200)


def test_find_window_region_by_handle_prefers_client_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        WindowLocator,
        "_client_region_from_handle",
        staticmethod(lambda handle: CaptureRegion(left=10, top=20, width=30, height=40)),
    )
    monkeypatch.setattr(
        WindowLocator,
        "_outer_region_from_handle",
        staticmethod(lambda handle: CaptureRegion(left=9, top=19, width=31, height=41)),
    )
    monkeypatch.setattr(wc, "pygetwindow", None)

    region = WindowLocator()._find_window_region_by_handle(55)

    assert region == CaptureRegion(left=10, top=20, width=30, height=40)


def test_find_window_region_by_handle_prefers_outer_region_when_client_is_too_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        WindowLocator,
        "_client_region_from_handle",
        staticmethod(lambda handle: CaptureRegion(left=127, top=112, width=1281, height=721)),
    )
    monkeypatch.setattr(
        WindowLocator,
        "_outer_region_from_handle",
        staticmethod(lambda handle: CaptureRegion(left=150, top=102, width=1620, height=949)),
    )
    monkeypatch.setattr(wc, "pygetwindow", None)

    region = WindowLocator()._find_window_region_by_handle(55)

    assert region == CaptureRegion(left=150, top=102, width=1620, height=949)


def test_find_window_region_by_handle_skips_pygetwindow_without_user32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        WindowLocator,
        "_client_region_from_handle",
        staticmethod(lambda handle: None),
    )
    monkeypatch.setattr(
        WindowLocator,
        "_outer_region_from_handle",
        staticmethod(lambda handle: None),
    )
    monkeypatch.setattr(
        wc,
        "pygetwindow",
        types.SimpleNamespace(getAllWindows=lambda: pytest.fail("should not query pygetwindow")),
    )
    monkeypatch.setattr(wc.ctypes, "windll", object(), raising=False)

    with pytest.raises(RuntimeError, match="Cannot query window handles through pygetwindow"):
        WindowLocator()._find_window_region_by_handle(55)


def test_capture_missing_mss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wc, "mss", None)
    with pytest.raises(DependencyMissingError, match="mss is not installed"):
        WindowCapture().capture()


def test_capture_missing_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMSSContext:
        monitors = [None, {"left": 0, "top": 0, "width": 2, "height": 1}]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def grab(self, monitor):
            return types.SimpleNamespace(size=(2, 1), rgb=b"abcdef")

    monkeypatch.setattr(wc, "mss", types.SimpleNamespace(mss=lambda: FakeMSSContext()))
    monkeypatch.setitem(__import__("sys").modules, "PIL", None)
    with pytest.raises(DependencyMissingError, match="Pillow is not installed"):
        WindowCapture().capture()


def test_capture_success_full_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_monitor = {}

    class FakeMSSContext:
        monitors = [None, {"left": 100, "top": 200, "width": 300, "height": 400}]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def grab(self, monitor):
            captured_monitor["value"] = monitor
            return types.SimpleNamespace(size=(2, 1), rgb=b"abcdef")

    class FakeImageModule:
        @staticmethod
        def frombytes(mode, size, rgb):
            return types.SimpleNamespace(mode=mode, size=size, rgb=rgb)

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = FakeImageModule

    monkeypatch.setattr(wc, "mss", types.SimpleNamespace(mss=lambda: FakeMSSContext()))
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)

    image = WindowCapture().capture()
    assert captured_monitor["value"] == {"left": 100, "top": 200, "width": 300, "height": 400}
    assert image.size == (2, 1)


def test_capture_success_explicit_region(monkeypatch: pytest.MonkeyPatch) -> None:
    used_monitor = {}

    class FakeMSSContext:
        monitors = [None, {"left": 0, "top": 0, "width": 1, "height": 1}]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def grab(self, monitor):
            used_monitor["value"] = monitor
            return types.SimpleNamespace(size=(7, 8), rgb=b"a" * (7 * 8 * 3))

    class FakeImageModule:
        @staticmethod
        def frombytes(mode, size, rgb):
            return io.BytesIO(rgb)

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = FakeImageModule
    monkeypatch.setattr(wc, "mss", types.SimpleNamespace(mss=lambda: FakeMSSContext()))
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)

    WindowCapture().capture(region={"left": 1, "top": 2, "width": 3, "height": 4})
    assert used_monitor["value"] == {"left": 1, "top": 2, "width": 3, "height": 4}


def test_capture_prefers_imagegrab_for_bound_window(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = WindowCapture(default_window_title="Demo App")
    capture.bind_window_handle(321)
    fake_image = object()

    class FakeImageGrab:
        calls = []

        @staticmethod
        def grab(window=None):
            FakeImageGrab.calls.append(window)
            return fake_image

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = object()
    fake_pil.ImageGrab = FakeImageGrab

    monkeypatch.setattr(wc, "mss", None)
    monkeypatch.setattr(wc.ctypes, "windll", object(), raising=False)
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)
    monkeypatch.setattr(
        CaptureScorer,
        "_image_capture_score",
        staticmethod(lambda image: 15.0),
    )

    assert capture.capture() is fake_image
    assert FakeImageGrab.calls == [321]


def test_capture_falls_back_to_mss_when_imagegrab_window_capture_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = WindowCapture(default_window_title="Demo App")
    capture.bind_window_handle(321)
    used_monitor = {}

    class FakeMSSContext:
        monitors = [None, {"left": 0, "top": 0, "width": 1, "height": 1}]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def grab(self, monitor):
            used_monitor["value"] = monitor
            return types.SimpleNamespace(size=(1, 1), rgb=b"abc")

    class FakeImageModule:
        @staticmethod
        def frombytes(mode, size, rgb):
            return {"mode": mode, "size": size, "rgb": rgb}

    class FakeImageGrab:
        @staticmethod
        def grab(window=None):
            raise OSError("window capture failed")

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = FakeImageModule
    fake_pil.ImageGrab = FakeImageGrab

    monkeypatch.setattr(wc, "mss", types.SimpleNamespace(mss=lambda: FakeMSSContext()))
    monkeypatch.setattr(wc.ctypes, "windll", object(), raising=False)
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)
    monkeypatch.setattr(
        WindowLocator,
        "_find_window_region_by_handle",
        lambda self, handle: CaptureRegion(left=5, top=6, width=7, height=8),
    )

    image = capture.capture()
    assert used_monitor["value"] == {"left": 5, "top": 6, "width": 7, "height": 8}
    assert image["size"] == (1, 1)


def test_capture_imagegrab_retries_until_best_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = WindowCapture(default_window_title="Demo App")
    capture.bind_window_handle(321)
    frames = [object(), object(), object()]
    scores = {id(frames[0]): 2.0, id(frames[1]): 3.0, id(frames[2]): 15.0}

    class FakeImageGrab:
        calls = []

        @staticmethod
        def grab(window=None):
            FakeImageGrab.calls.append(window)
            return frames[len(FakeImageGrab.calls) - 1]

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = object()
    fake_pil.ImageGrab = FakeImageGrab

    monkeypatch.setattr(wc, "mss", None)
    monkeypatch.setattr(wc.ctypes, "windll", object(), raising=False)
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)
    monkeypatch.setattr(
        CaptureScorer,
        "_image_capture_score",
        staticmethod(lambda image: scores[id(image)]),
    )
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

    assert capture.capture() is frames[2]
    assert FakeImageGrab.calls == [321, 321, 321]
    assert sleeps == [0.2, 0.2]


def test_capture_prefers_imagegrab_when_bound_window_region_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = WindowCapture(default_window_title="Demo App")
    capture.bind_window_handle(321)
    used_monitor = {}

    class FakeMSSContext:
        monitors = [None, {"left": 0, "top": 0, "width": 1, "height": 1}]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def grab(self, monitor):
            used_monitor["value"] = monitor
            return types.SimpleNamespace(size=(7, 8), rgb=b"a" * (7 * 8 * 3))

    class FakeImageModule:
        @staticmethod
        def frombytes(mode, size, rgb):
            return types.SimpleNamespace(mode=mode, size=size, rgb=rgb)

    class FakeWindowImage:
        def __init__(self):
            self.size = (70, 80)
            self.resize_calls = []

        def resize(self, size, resample=None):
            self.resize_calls.append((size, resample))
            return types.SimpleNamespace(size=size)

    fake_window_image = FakeWindowImage()

    class FakeImageGrab:
        calls = []

        @staticmethod
        def grab(window=None):
            FakeImageGrab.calls.append(window)
            return fake_window_image

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = FakeImageModule
    fake_pil.ImageGrab = FakeImageGrab
    fake_pil.Image.Resampling = types.SimpleNamespace(LANCZOS="lanczos")

    monkeypatch.setattr(wc, "mss", types.SimpleNamespace(mss=lambda: FakeMSSContext()))
    monkeypatch.setattr(wc.ctypes, "windll", object(), raising=False)
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)
    monkeypatch.setattr(
        CaptureScorer,
        "_image_capture_score",
        staticmethod(lambda image: 20.0),
    )
    monkeypatch.setattr(
        WindowLocator,
        "_find_window_region_by_handle",
        lambda self, handle: CaptureRegion(left=5, top=6, width=7, height=8),
    )
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    image = capture.capture()

    assert FakeImageGrab.calls == [321]
    assert "value" not in used_monitor
    assert image.size == (7, 8)
    assert fake_window_image.resize_calls == [((7, 8), "lanczos")]


def test_capture_falls_back_to_mss_when_imagegrab_window_size_is_not_uniformly_scaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = WindowCapture(default_window_title="Demo App")
    capture.bind_window_handle(321)
    used_monitor = {}

    class FakeMSSContext:
        monitors = [None, {"left": 0, "top": 0, "width": 1, "height": 1}]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def grab(self, monitor):
            used_monitor["value"] = monitor
            return types.SimpleNamespace(size=(7, 8), rgb=b"a" * (7 * 8 * 3))

    class FakeImageModule:
        @staticmethod
        def frombytes(mode, size, rgb):
            return types.SimpleNamespace(mode=mode, size=size, rgb=rgb)

    class FakeImage:
        size = (70, 90)

    class FakeImageGrab:
        calls = []

        @staticmethod
        def grab(window=None):
            FakeImageGrab.calls.append(window)
            return FakeImage()

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = FakeImageModule
    fake_pil.ImageGrab = FakeImageGrab

    monkeypatch.setattr(wc, "mss", types.SimpleNamespace(mss=lambda: FakeMSSContext()))
    monkeypatch.setattr(wc.ctypes, "windll", object(), raising=False)
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)
    monkeypatch.setattr(
        WindowLocator,
        "_find_window_region_by_handle",
        lambda self, handle: CaptureRegion(left=5, top=6, width=7, height=8),
    )
    monkeypatch.setattr(
        CaptureScorer,
        "_image_capture_score",
        staticmethod(lambda image: 20.0),
    )
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    image = capture.capture()

    assert FakeImageGrab.calls == [321] * 6
    assert used_monitor["value"] == {"left": 5, "top": 6, "width": 7, "height": 8}
    assert image.size == (7, 8)


def test_resolve_capture_region_exposes_public_region_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = WindowCapture(default_region={"left": 11, "top": 12, "width": 13, "height": 14})
    assert capture.resolve_capture_region() == CaptureRegion(
        left=11,
        top=12,
        width=13,
        height=14,
    )


def test_capture_result_exposes_last_capture_space_for_mss(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMSSContext:
        monitors = [None, {"left": 100, "top": 200, "width": 300, "height": 400}]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def grab(self, monitor):
            return types.SimpleNamespace(size=(300, 400), rgb=b"x" * (300 * 400 * 3))

    class FakeImageModule:
        @staticmethod
        def frombytes(mode, size, rgb):
            return types.SimpleNamespace(mode=mode, size=size, rgb=rgb)

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = FakeImageModule

    monkeypatch.setattr(wc, "mss", types.SimpleNamespace(mss=lambda: FakeMSSContext()))
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)

    capture = WindowCapture()
    capture.capture()
    result = capture.get_last_capture_result()
    assert result is not None
    capture_space = result.coordinate_space

    assert capture_space == CaptureCoordinateSpace(
        region=CaptureRegion(left=100, top=200, width=300, height=400),
        image_width=300,
        image_height=400,
    )


def test_capture_result_exposes_last_artifact_image(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMSSContext:
        monitors = [None, {"left": 100, "top": 200, "width": 300, "height": 400}]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def grab(self, monitor):
            return types.SimpleNamespace(size=(300, 400), rgb=b"x" * (300 * 400 * 3))

    class FakeImageModule:
        @staticmethod
        def frombytes(mode, size, rgb):
            return types.SimpleNamespace(mode=mode, size=size, rgb=rgb)

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = FakeImageModule

    monkeypatch.setattr(wc, "mss", types.SimpleNamespace(mss=lambda: FakeMSSContext()))
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)

    capture = WindowCapture()
    image = capture.capture()
    result = capture.get_last_capture_result()
    assert result is not None
    artifact = result.artifact_image

    assert artifact is image


def test_capture_result_exposes_explicit_context_for_mss(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMSSContext:
        monitors = [None, {"left": 10, "top": 20, "width": 30, "height": 40}]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def grab(self, monitor):
            return types.SimpleNamespace(size=(30, 40), rgb=b"x" * (30 * 40 * 3))

    class FakeImageModule:
        @staticmethod
        def frombytes(mode, size, rgb):
            return types.SimpleNamespace(mode=mode, size=size, rgb=rgb)

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = FakeImageModule

    monkeypatch.setattr(wc, "mss", types.SimpleNamespace(mss=lambda: FakeMSSContext()))
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)

    result = WindowCapture().capture_result()

    assert isinstance(result, CaptureResult)
    assert result.image.size == (30, 40)
    assert result.artifact_image is result.image
    assert result.coordinate_space == CaptureCoordinateSpace(
        region=CaptureRegion(left=10, top=20, width=30, height=40),
        image_width=30,
        image_height=40,
    )
    assert result.to_screen(5, 6) == (15, 26)


def test_capture_result_uses_imagegrab_source_region_for_last_capture_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = WindowCapture(default_window_title="Demo App")
    capture.bind_window_handle(321)

    class FakeWindowImage:
        def __init__(self):
            self.size = (70, 80)

        def resize(self, size, resample=None):
            return types.SimpleNamespace(size=size)

    fake_window_image = FakeWindowImage()

    class FakeImageGrab:
        @staticmethod
        def grab(window=None):
            return fake_window_image

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = types.SimpleNamespace(
        Resampling=types.SimpleNamespace(LANCZOS="lanczos")
    )
    fake_pil.ImageGrab = FakeImageGrab

    monkeypatch.setattr(wc, "mss", None)
    monkeypatch.setattr(wc.ctypes, "windll", object(), raising=False)
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)
    monkeypatch.setattr(
        WindowLocator,
        "_find_window_region_by_handle",
        lambda self, handle: CaptureRegion(left=110, top=220, width=7, height=8),
    )
    monkeypatch.setattr(
        WindowLocator,
        "_outer_region_from_handle",
        staticmethod(lambda handle: CaptureRegion(left=100, top=200, width=70, height=80)),
    )
    monkeypatch.setattr(
        WindowLocator,
        "_client_region_from_handle",
        staticmethod(lambda handle: CaptureRegion(left=110, top=220, width=7, height=8)),
    )
    monkeypatch.setattr(
        CaptureScorer,
        "_image_capture_score",
        staticmethod(lambda image: 20.0),
    )

    capture.capture()
    result = capture.get_last_capture_result()
    assert result is not None
    capture_space = result.coordinate_space

    assert capture_space == CaptureCoordinateSpace(
        region=CaptureRegion(left=100, top=200, width=70, height=80),
        image_width=7,
        image_height=8,
    )
    assert capture_space.to_screen(3, 4) == (130, 240)
