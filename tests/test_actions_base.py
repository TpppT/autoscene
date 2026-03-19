from pathlib import Path

import pytest

from autoscene.actions.base import BaseActions
from autoscene.actions.vision_runtime import ActionVisionRuntime
from autoscene.capture.window_capture import CaptureResult
from autoscene.core.exceptions import ActionExecutionError, DependencyMissingError


class FakeCapture:
    def __init__(self, image, artifact_image=None, last_capture_space=None):
        self._image = image
        self._artifact_image = artifact_image
        self._last_capture_space = last_capture_space
        self.bound_window_handle = None
        self.region = None

    def capture(self):
        return self._image

    def get_last_capture_result(self):
        if self._artifact_image is None and self._last_capture_space is None:
            return None
        return CaptureResult(
            image=self._image,
            artifact_image=self._artifact_image or self._image,
            coordinate_space=self._last_capture_space,
        )

    def bind_window_handle(self, handle):
        self.bound_window_handle = handle

    def get_bound_window_handle(self):
        return self.bound_window_handle

    def resolve_capture_region(self):
        return self.region


class FakeDetector:
    def detect(self, image, labels=None):
        return []


class FakeOCR:
    def read(self, image):
        return []


class FakeImage:
    def __init__(self):
        self.saved_to = None

    def save(self, path):
        self.saved_to = str(path)


class FakePyAutoGUI:
    def __init__(self):
        self.calls = []

    def click(self, **kwargs):
        self.calls.append(("click", kwargs))

    def moveTo(self, x, y):
        self.calls.append(("moveTo", {"x": x, "y": y}))

    def mouseDown(self):
        self.calls.append(("mouseDown", {}))

    def mouseUp(self):
        self.calls.append(("mouseUp", {}))

    def write(self, text):
        self.calls.append(("write", {"text": text}))

    def press(self, key, presses=1, interval=0.0):
        self.calls.append(
            ("press", {"key": key, "presses": presses, "interval": interval})
        )


def _build_actions(image=None):
    if image is None:
        image = FakeImage()
    return BaseActions(capture=FakeCapture(image))


def test_click_calls_pyautogui(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _build_actions()
    fake_gui = FakePyAutoGUI()
    monkeypatch.setattr(BaseActions, "_require_pyautogui", staticmethod(lambda: fake_gui))
    actions.click(1.9, 2.2)
    assert fake_gui.calls == [("click", {"x": 1, "y": 2})]


def test_drag_calls_pyautogui(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _build_actions()
    fake_gui = FakePyAutoGUI()
    sleep_calls = []
    monkeypatch.setattr(BaseActions, "_require_pyautogui", staticmethod(lambda: fake_gui))
    monkeypatch.setattr("time.sleep", lambda seconds: sleep_calls.append(seconds))
    actions.drag(10, 20, 30, 40, duration_ms=500)
    assert fake_gui.calls[0] == ("moveTo", {"x": 10, "y": 20})
    assert fake_gui.calls[1] == ("mouseDown", {})
    assert fake_gui.calls[-2] == ("moveTo", {"x": 30, "y": 40})
    assert fake_gui.calls[-1] == ("mouseUp", {})
    assert sleep_calls[0] == pytest.approx(0.15)
    assert sleep_calls[1:] == [pytest.approx(0.05)] * 9


def test_input_text_calls_pyautogui(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _build_actions()
    fake_gui = FakePyAutoGUI()
    monkeypatch.setattr(BaseActions, "_require_pyautogui", staticmethod(lambda: fake_gui))
    actions.input_text("hello")
    assert fake_gui.calls == [("write", {"text": "hello"})]


def test_press_key_calls_pyautogui(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _build_actions()
    fake_gui = FakePyAutoGUI()
    monkeypatch.setattr(BaseActions, "_require_pyautogui", staticmethod(lambda: fake_gui))
    actions.press_key("tab", presses=2, interval_seconds=0.1)
    assert fake_gui.calls == [
        ("press", {"key": "tab", "presses": 2, "interval": 0.1})
    ]


def test_open_browser_launches_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _build_actions()
    launches = []
    monkeypatch.setattr(
        BaseActions,
        "_resolve_browser_command",
        staticmethod(lambda browser, browser_path=None: "chrome.exe"),
    )
    monkeypatch.setattr(
        BaseActions,
        "_launch_process",
        staticmethod(lambda command: launches.append(list(command))),
    )
    actions.open_browser("https://example.com", browser="chrome")
    assert launches == [["chrome.exe", "--new-window", "https://example.com"]]


def test_open_browser_passes_custom_args(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _build_actions()
    launches = []
    monkeypatch.setattr(
        BaseActions,
        "_resolve_browser_command",
        staticmethod(lambda browser, browser_path=None: "chrome.exe"),
    )
    monkeypatch.setattr(
        BaseActions,
        "_launch_process",
        staticmethod(lambda command: launches.append(list(command))),
    )
    actions.open_browser(
        "https://example.com",
        browser="chrome",
        new_window=False,
        args=["--start-maximized"],
    )
    assert launches == [["chrome.exe", "--start-maximized", "https://example.com"]]


def test_open_browser_binds_new_window(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _build_actions()
    launches = []

    class FakeWindow:
        def __init__(self, handle):
            self._hWnd = handle

    windows = [[FakeWindow(1)], [FakeWindow(1), FakeWindow(2)]]

    class FakePyGetWindow:
        @staticmethod
        def getAllWindows():
            return windows.pop(0)

        @staticmethod
        def getActiveWindow():
            return None

    monkeypatch.setattr(
        BaseActions,
        "_resolve_browser_command",
        staticmethod(lambda browser, browser_path=None: "chrome.exe"),
    )
    monkeypatch.setattr(
        BaseActions,
        "_launch_process",
        staticmethod(lambda command: launches.append(list(command))),
    )
    monkeypatch.setattr(
        BaseActions,
        "_require_pygetwindow",
        staticmethod(lambda: FakePyGetWindow),
    )
    actions.open_browser("https://example.com", browser="chrome")
    assert launches == [["chrome.exe", "--new-window", "https://example.com"]]
    assert actions.capture_engine.bound_window_handle == 2


def test_open_browser_does_not_bind_active_window_when_no_new_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _build_actions()
    launches = []

    class FakeWindow:
        def __init__(self, handle):
            self._hWnd = handle

    existing = [FakeWindow(1)]

    class FakePyGetWindow:
        @staticmethod
        def getAllWindows():
            return existing

        @staticmethod
        def getActiveWindow():
            return FakeWindow(99)

    monkeypatch.setattr(
        BaseActions,
        "_resolve_browser_command",
        staticmethod(lambda browser, browser_path=None: "chrome.exe"),
    )
    monkeypatch.setattr(
        BaseActions,
        "_launch_process",
        staticmethod(lambda command: launches.append(list(command))),
    )
    monkeypatch.setattr(
        BaseActions,
        "_require_pygetwindow",
        staticmethod(lambda: FakePyGetWindow),
    )
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    timeline = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    monkeypatch.setattr("time.time", lambda: next(timeline))
    actions.open_browser("https://example.com", browser="chrome")
    assert launches == [["chrome.exe", "--new-window", "https://example.com"]]
    assert actions.capture_engine.bound_window_handle is None


def test_maximize_window(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _build_actions()
    events = []
    forced = []

    class FakeWindow:
        _hWnd = 3
        isMinimized = False

        def maximize(self):
            events.append("maximize")

    class FakePyGetWindow:
        @staticmethod
        def getWindowsWithTitle(title):
            events.append(("lookup", title))
            return [FakeWindow()]

    monkeypatch.setattr(
        BaseActions,
        "_require_pygetwindow",
        staticmethod(lambda: FakePyGetWindow),
    )
    monkeypatch.setattr(
        BaseActions,
        "_force_foreground_window",
        lambda self, window: forced.append(getattr(window, "_hWnd", None)) or True,
    )
    actions.maximize_window("Google Chrome")
    assert events == [("lookup", "Google Chrome"), "maximize"]
    assert actions.capture_engine.bound_window_handle == 3
    assert forced == [3]


def test_activate_window_restores_and_activates(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _build_actions()
    events = []
    forced = []

    class FakeWindow:
        _hWnd = 4
        isMinimized = True

        def restore(self):
            events.append("restore")

        def show(self):
            events.append("show")

        def activate(self):
            events.append("activate")

    class FakePyGetWindow:
        @staticmethod
        def getWindowsWithTitle(title):
            events.append(("lookup", title))
            return [FakeWindow()]

    monkeypatch.setattr(
        BaseActions,
        "_require_pygetwindow",
        staticmethod(lambda: FakePyGetWindow),
    )
    monkeypatch.setattr(
        BaseActions,
        "_force_foreground_window",
        lambda self, window: forced.append(getattr(window, "_hWnd", None)) or True,
    )
    monkeypatch.setattr("time.sleep", lambda seconds: events.append(("sleep", seconds)))
    assert actions.activate_window("Google Chrome", settle_seconds=0.1) is True
    assert events == [("lookup", "Google Chrome"), "restore", "show", "activate", ("sleep", 0.1)]
    assert actions.capture_engine.bound_window_handle == 4
    assert forced == [4]


def test_maximize_window_ignores_bound_window_with_wrong_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _build_actions()
    actions.capture_engine.bind_window_handle(1)
    events = []

    class BoundWindow:
        _hWnd = 1
        title = "PyCharm"
        isMinimized = False

        def maximize(self):
            events.append("bound-maximize")

    class ChromeWindow:
        _hWnd = 2
        title = "Test Store - Google Chrome"
        isMinimized = False

        def maximize(self):
            events.append("chrome-maximize")

    class FakePyGetWindow:
        @staticmethod
        def getAllWindows():
            return [BoundWindow(), ChromeWindow()]

        @staticmethod
        def getWindowsWithTitle(title):
            events.append(("lookup", title))
            return [ChromeWindow()]

    monkeypatch.setattr(
        BaseActions,
        "_require_pygetwindow",
        staticmethod(lambda: FakePyGetWindow),
    )
    actions.maximize_window("Google Chrome")
    assert events == [("lookup", "Google Chrome"), "chrome-maximize"]
    assert actions.capture_engine.bound_window_handle == 2


def test_activate_bound_window_ignores_activate_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _build_actions()
    actions.capture_engine.bind_window_handle(5)
    events = []

    class FakeWindow:
        _hWnd = 5
        title = "Test Store - Google Chrome"
        isMinimized = False

        def show(self):
            events.append("show")

        def activate(self):
            events.append("activate")
            raise RuntimeError("activate failed")

    class FakePyGetWindow:
        @staticmethod
        def getAllWindows():
            return [FakeWindow()]

    monkeypatch.setattr(
        BaseActions,
        "_require_pygetwindow",
        staticmethod(lambda: FakePyGetWindow),
    )
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    assert actions.activate_bound_window() is True
    assert events == ["show", "activate"]


def test_sleep_uses_time_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _build_actions()
    calls = []
    monkeypatch.setattr("time.sleep", lambda seconds: calls.append(seconds))
    actions.sleep(0.25)
    assert calls == [0.25]


def test_screenshot_returns_image() -> None:
    image = FakeImage()
    actions = _build_actions(image=image)
    assert actions.screenshot() is image


def test_screenshot_saves_file_path(tmp_path: Path) -> None:
    image = FakeImage()
    actions = _build_actions(image=image)
    target = tmp_path / "nested" / "screen.png"
    actions.screenshot(str(target))
    assert image.saved_to == str(target)
    assert target.parent.exists()


def test_screenshot_saves_artifact_image_when_available(tmp_path: Path) -> None:
    image = FakeImage()
    artifact = FakeImage()
    capture = FakeCapture(image=image, artifact_image=artifact)
    actions = BaseActions(capture=capture)
    target = tmp_path / "screen.png"

    actions.screenshot(str(target))

    assert image.saved_to is None
    assert artifact.saved_to == str(target)


def test_capture_to_screen_uses_capture_region_offset() -> None:
    actions = _build_actions()
    actions.capture_engine.region = type("Region", (), {"left": 100, "top": 200})()
    assert actions.capture_to_screen(10, 20) == (110, 220)


def test_capture_to_screen_prefers_last_capture_space() -> None:
    class FakeCaptureSpace:
        def to_screen(self, x, y):
            return (x + 300, y + 400)

    actions = BaseActions(
        capture=FakeCapture(FakeImage(), last_capture_space=FakeCaptureSpace()),
    )
    actions.capture_engine.region = type("Region", (), {"left": 100, "top": 200})()

    assert actions.capture_to_screen(10, 20) == (310, 420)


def test_capture_frame_prefers_explicit_capture_result() -> None:
    image = FakeImage()
    artifact = FakeImage()

    class ResultCapture(FakeCapture):
        def capture_result(self):
            return CaptureResult(
                image=image,
                artifact_image=artifact,
                capture_region=type("Region", (), {"left": 10, "top": 20})(),
            )

    actions = BaseActions(capture=ResultCapture(image))

    result = actions.capture_frame()

    assert result.image is image
    assert result.artifact_image is artifact
    assert result.to_screen(3, 4) == (13, 24)


def test_capture_to_screen_accepts_explicit_capture_result() -> None:
    actions = _build_actions()
    explicit = CaptureResult(
        image=FakeImage(),
        capture_region=type("Region", (), {"left": 500, "top": 600})(),
    )

    assert actions.capture_to_screen(10, 20, capture_result=explicit) == (510, 620)


def test_action_vision_runtime_resolves_named_detector() -> None:
    primary = FakeDetector()
    secondary = FakeDetector()
    runtime = ActionVisionRuntime(
        detector=primary,
        detectors={"icons": secondary},
        ocr=FakeOCR(),
    )
    assert runtime.resolve_detector() is primary
    assert runtime.resolve_detector("default") is primary
    assert runtime.resolve_detector("icons") is secondary


def test_action_vision_runtime_raises_for_unknown_alias() -> None:
    runtime = ActionVisionRuntime(
        detector=FakeDetector(),
        ocr=FakeOCR(),
    )
    with pytest.raises(ActionExecutionError, match="Unknown detector alias"):
        runtime.resolve_detector("missing")


def test_require_pyautogui_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "pyautogui", None)
    with pytest.raises(DependencyMissingError, match="pyautogui is not installed"):
        BaseActions._require_pyautogui()


def test_resolve_browser_command_errors_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("autoscene.actions.browser.shutil.which", lambda name: None)
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.delenv("LocalAppData", raising=False)
    with pytest.raises(ActionExecutionError, match="Google Chrome was not found"):
        BaseActions._resolve_browser_command("chrome")


def test_maximize_window_raises_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _build_actions()

    class FakePyGetWindow:
        @staticmethod
        def getWindowsWithTitle(title):
            return []

    monkeypatch.setattr(
        BaseActions,
        "_require_pygetwindow",
        staticmethod(lambda: FakePyGetWindow),
    )
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    timeline = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr("time.time", lambda: next(timeline))
    with pytest.raises(ActionExecutionError, match="No window found for title"):
        actions.maximize_window("Missing", timeout=1.0, interval=0.1)
