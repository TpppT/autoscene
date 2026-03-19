from pathlib import Path

import pytest
from PIL import Image

from autoscene.actions import ActionServices
from autoscene.capture.window_capture import CaptureResult
from autoscene.core.models import BoundingBox, OCRText, ObjectLocateSpec, TextLocateSpec
from autoscene.vision.pipeline import utils as pipeline_utils


class FakeCapture:
    def __init__(self, region=None, artifact_image=None):
        self.region = region
        self.artifact_image = artifact_image

    def capture(self):
        return "image"

    def resolve_capture_region(self):
        return self.region

    def get_last_capture_result(self):
        if self.artifact_image is None and self.region is None:
            return None
        return CaptureResult(
            image=self.capture(),
            artifact_image=self.artifact_image or self.capture(),
            capture_region=self.region,
        )


class FakeDetector:
    def __init__(self, boxes_by_label=None):
        self.boxes_by_label = boxes_by_label or {}

    def detect(self, image, labels=None):
        if not labels:
            return []
        output = []
        for label in labels:
            output.extend(self.boxes_by_label.get(label, []))
        return output


class FakeOCR:
    def __init__(self, entries=None):
        self.entries = entries or []

    def read(self, image):
        return self.entries


def _actions(detector=None, ocr=None, detectors=None):
    if detector is None:
        detector = FakeDetector()
    if ocr is None:
        ocr = FakeOCR()
    return ActionServices(capture=FakeCapture(), detector=detector, detectors=detectors, ocr=ocr)


def _text_locate(text: str, *, exact: bool = False, ocr: dict | None = None):
    return TextLocateSpec(text=text, exact=exact, ocr=ocr)


def test_click_text_partial_match(monkeypatch: pytest.MonkeyPatch) -> None:
    ocr = FakeOCR(
        [OCRText(text="Click Login", bbox=BoundingBox(10, 20, 30, 40), score=1.0)]
    )
    actions = _actions(ocr=ocr)
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    actions.text_actions.click_text(_text_locate("Login"))
    assert calls == [(20, 30)]


def test_locate_actions_exposes_underlying_text_and_object_actions() -> None:
    actions = _actions()

    assert actions.locate_actions.text is actions.text_actions
    assert actions.locate_actions.object is actions.object_actions


def test_click_text_exact_match(monkeypatch: pytest.MonkeyPatch) -> None:
    ocr = FakeOCR(
        [
            OCRText(text="Login Button", bbox=BoundingBox(1, 1, 11, 11)),
            OCRText(text="Login", bbox=BoundingBox(10, 10, 20, 20)),
        ]
    )
    actions = _actions(ocr=ocr)
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    actions.text_actions.click_text(_text_locate("Login", exact=True))
    assert calls == [(15, 15)]


def test_locate_actions_can_click_text(monkeypatch: pytest.MonkeyPatch) -> None:
    ocr = FakeOCR([OCRText(text="Login", bbox=BoundingBox(10, 20, 30, 40), score=1.0)])
    actions = _actions(ocr=ocr)
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))

    actions.locate_actions.click_text(_text_locate("Login"))

    assert calls == [(20, 30)]


def test_click_text_not_found() -> None:
    actions = _actions(ocr=FakeOCR([]))
    with pytest.raises(AssertionError, match="Text not found"):
        actions.text_actions.click_text(_text_locate("Missing"))


def test_click_text_requires_text_locate_spec() -> None:
    actions = _actions(ocr=FakeOCR([]))
    with pytest.raises(TypeError, match="TextLocateSpec"):
        actions.text_actions.click_text("Missing")  # type: ignore[arg-type]


def test_click_text_matches_multi_word_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    ocr = FakeOCR(
        [
            OCRText(text="Hummingbird", bbox=BoundingBox(10, 20, 80, 40), score=0.9),
            OCRText(text="Printed", bbox=BoundingBox(90, 20, 150, 40), score=0.9),
            OCRText(text="Sweater", bbox=BoundingBox(160, 20, 230, 40), score=0.9),
        ]
    )
    actions = _actions(ocr=ocr)
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    actions.text_actions.click_text(_text_locate("Hummingbird printed sweater"))
    assert calls == [(120, 30)]


def test_pipeline_text_match_normalizes_each_entry_once_for_phrase_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        OCRText(text="Hummingbird", bbox=BoundingBox(10, 20, 80, 40), score=0.9),
        OCRText(text="Printed", bbox=BoundingBox(90, 20, 150, 40), score=0.9),
        OCRText(text="Sweater", bbox=BoundingBox(160, 20, 230, 40), score=0.9),
    ]
    original_normalize = pipeline_utils.normalize_ocr_text
    normalized_inputs = []

    def counting_normalize(text: str) -> str:
        normalized_inputs.append(text)
        return original_normalize(text)

    monkeypatch.setattr(pipeline_utils, "normalize_ocr_text", counting_normalize)

    matched = pipeline_utils.find_ocr_text_match(entries, "Hummingbird printed sweater")

    assert matched is not None
    assert len(normalized_inputs) == 4
    assert normalized_inputs.count("Hummingbird printed sweater") == 1
    assert normalized_inputs.count("Hummingbird") == 1
    assert normalized_inputs.count("Printed") == 1
    assert normalized_inputs.count("Sweater") == 1


def test_pipeline_text_match_can_reuse_pre_normalized_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        OCRText(text="Hummingbird", bbox=BoundingBox(10, 20, 80, 40), score=0.9),
        OCRText(text="Printed", bbox=BoundingBox(90, 20, 150, 40), score=0.9),
        OCRText(text="Sweater", bbox=BoundingBox(160, 20, 230, 40), score=0.9),
    ]
    original_normalize = pipeline_utils.normalize_ocr_text
    normalized_inputs = []

    def counting_normalize(text: str) -> str:
        normalized_inputs.append(text)
        return original_normalize(text)

    monkeypatch.setattr(pipeline_utils, "normalize_ocr_text", counting_normalize)

    matched = pipeline_utils.find_ocr_text_match(
        entries,
        "Hummingbird printed sweater",
        normalized_text="hummingbird printed sweater",
    )

    assert matched is not None
    assert len(normalized_inputs) == 3
    assert "Hummingbird printed sweater" not in normalized_inputs
    assert normalized_inputs.count("Hummingbird") == 1
    assert normalized_inputs.count("Printed") == 1
    assert normalized_inputs.count("Sweater") == 1


def test_text_actions_no_longer_expose_phrase_matching_helpers() -> None:
    actions = _actions()

    assert not hasattr(actions.text_actions, "find_text_match")
    assert not hasattr(actions.text_actions, "group_ocr_lines")
    assert not hasattr(actions.text_actions, "match_phrase_in_line")
    assert not hasattr(actions.text_actions, "merge_ocr_entries")
    assert not hasattr(actions.text_actions, "normalize_text")


def test_click_text_uses_screen_coordinates_from_capture_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture(region=type("Region", (), {"left": 300, "top": 400})())
    ocr = FakeOCR([OCRText(text="Login", bbox=BoundingBox(10, 20, 70, 40), score=0.9)])
    actions = ActionServices(capture=capture, detector=FakeDetector(), ocr=ocr)
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: True,
    )
    actions.text_actions.click_text(_text_locate("Login"))
    assert calls == [(340, 430)]


def test_click_text_phrase_does_not_include_intermediate_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ocr = FakeOCR(
        [
            OCRText(text="Hummingbird", bbox=BoundingBox(10, 20, 80, 40), score=0.9),
            OCRText(text="Printed", bbox=BoundingBox(90, 20, 150, 40), score=0.9),
            OCRText(text="T-Shirt", bbox=BoundingBox(160, 20, 220, 40), score=0.9),
            OCRText(text="Hummingbird", bbox=BoundingBox(230, 20, 300, 40), score=0.9),
            OCRText(text="Printed", bbox=BoundingBox(310, 20, 370, 40), score=0.9),
            OCRText(text="Sweater", bbox=BoundingBox(380, 20, 450, 40), score=0.9),
        ]
    )
    actions = _actions(ocr=ocr)
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    actions.text_actions.click_text(_text_locate("Hummingbird printed sweater"))
    assert calls == [(340, 30)]


def test_group_ocr_lines_groups_entries_by_row_and_sorts_each_line() -> None:
    lines = pipeline_utils.group_ocr_lines(
        [
            OCRText(text="Bottom Right", bbox=BoundingBox(80, 58, 120, 92), score=0.9),
            OCRText(text="Top Right", bbox=BoundingBox(120, 14, 160, 38), score=0.9),
            OCRText(text="Bottom Left", bbox=BoundingBox(10, 60, 40, 90), score=0.9),
            OCRText(text="Top Left", bbox=BoundingBox(10, 10, 50, 30), score=0.9),
        ]
    )

    assert [[entry.text for entry in line] for line in lines] == [
        ["Top Left", "Top Right"],
        ["Bottom Left", "Bottom Right"],
    ]


def test_click_text_saves_debug_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class ImageCapture:
        def capture(self):
            return Image.new("RGB", (200, 100), "white")

    ocr = FakeOCR([OCRText(text="Login", bbox=BoundingBox(10, 20, 80, 40), score=0.9)])
    actions = ActionServices(capture=ImageCapture(), detector=FakeDetector(), ocr=ocr)
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: True,
    )

    debug_path = tmp_path / "full.png"
    crop_path = tmp_path / "crop.png"
    actions.text_actions.click_text(
        _text_locate("Login"),
        debug_path=str(debug_path),
        debug_crop_path=str(crop_path),
    )

    assert calls == [(45, 30)]
    assert debug_path.exists()
    assert crop_path.exists()


def test_click_object_uses_highest_score(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = FakeDetector(
        {
            "icon": [
                BoundingBox(0, 0, 10, 10, score=0.4, label="icon"),
                BoundingBox(10, 10, 30, 30, score=0.9, label="icon"),
            ]
        }
    )
    actions = _actions(detector=detector)
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    actions.object_actions.click_object(ObjectLocateSpec(label="icon", min_score=0.3))
    assert calls == [(20, 20)]


def test_locate_actions_can_click_object(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = FakeDetector({"icon": [BoundingBox(10, 10, 30, 30, score=0.9, label="icon")]})
    actions = _actions(detector=detector)
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))

    actions.locate_actions.click_object(ObjectLocateSpec(label="icon", min_score=0.3))

    assert calls == [(20, 20)]


def test_click_object_uses_named_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    default_detector = FakeDetector({"icon": [BoundingBox(0, 0, 10, 10, score=0.4, label="icon")]})
    secondary_detector = FakeDetector(
        {"icon": [BoundingBox(20, 20, 40, 40, score=0.9, label="icon")]}
    )
    actions = _actions(detector=default_detector, detectors={"secondary": secondary_detector})
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    actions.object_actions.click_object(
        ObjectLocateSpec(label="icon", min_score=0.3, detector="secondary")
    )
    assert calls == [(30, 30)]


def test_click_relative_to_text_uses_anchor_and_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = FakeCapture(region=type("Region", (), {"left": 100, "top": 200})())
    ocr = FakeOCR([OCRText(text="Quantity", bbox=BoundingBox(10, 20, 70, 40), score=0.9)])
    actions = ActionServices(capture=capture, detector=FakeDetector(), ocr=ocr)
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: True,
    )
    actions.text_actions.click_relative_to_text(
        _text_locate("Quantity"),
        offset_x=15,
        offset_y=5,
        anchor="bottom_left",
    )
    assert calls == [(125, 245)]


def test_click_relative_to_text_uses_read_ocr_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture(region=type("Region", (), {"left": 10, "top": 20})())
    actions = ActionServices(capture=capture, detector=FakeDetector(), ocr=FakeOCR())
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: True,
    )
    monkeypatch.setattr(
        actions.text_actions,
        "read_ocr",
        lambda image, ocr_options=None: [
            OCRText(text="Quantity", bbox=BoundingBox(10, 20, 70, 40), score=0.9)
        ],
    )
    monkeypatch.setattr(
        actions.vision_runtime.ocr,
        "read",
        lambda image: (_ for _ in ()).throw(AssertionError("direct ocr.read should not be used")),
    )

    actions.text_actions.click_relative_to_text(_text_locate("Quantity"))

    assert calls == [(50, 50)]


def test_click_object_uses_search_region_and_pick(monkeypatch: pytest.MonkeyPatch) -> None:
    class ImageCapture(FakeCapture):
        def capture(self):
            return Image.new("RGB", (200, 200), "white")

    capture = ImageCapture(region=type("Region", (), {"left": 100, "top": 200})())

    class DetectorWithRegion(FakeDetector):
        def detect(self, image, labels=None):
            assert tuple(labels or ()) == ("quantity_up",)
            assert image.size == (200, 200)
            return [
                BoundingBox(20, 8, 40, 28, score=0.06, label="quantity_up"),
                BoundingBox(70, 80, 90, 100, score=0.05, label="quantity_up"),
                BoundingBox(80, 70, 100, 90, score=0.07, label="quantity_up"),
            ]

    actions = ActionServices(capture=capture, detector=DetectorWithRegion(), ocr=FakeOCR())
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: True,
    )
    actions.object_actions.click_object(
        ObjectLocateSpec(
            label="quantity_up",
            min_score=0.03,
            pick="topmost",
            region=BoundingBox(x1=50, y1=60, x2=130, y2=160),
        ),
    )
    assert calls == [(190, 280)]


def test_drag_object_debug_uses_artifact_image_scale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    logical_image = Image.new("RGB", (100, 50), "white")
    artifact_image = Image.new("RGB", (200, 100), "white")

    class ImageCapture(FakeCapture):
        def capture(self):
            return logical_image

    capture = ImageCapture(artifact_image=artifact_image)
    detector = FakeDetector({"icon": [BoundingBox(10, 10, 30, 20, score=0.9, label="icon")]})
    actions = ActionServices(capture=capture, detector=detector, ocr=FakeOCR())
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: True,
    )
    monkeypatch.setattr(actions.base_actions, "drag", lambda sx, sy, tx, ty, duration_ms: None)

    debug_path = tmp_path / "debug.png"
    actions.object_actions.drag_object_to_position(
        ObjectLocateSpec(label="icon"),
        target_x=80,
        target_y=40,
        duration_ms=300,
        debug_path=str(debug_path),
    )

    saved = Image.open(debug_path)
    assert saved.size == (200, 100)


def test_click_object_uses_screen_coordinates_from_capture_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture(region=type("Region", (), {"left": 50, "top": 60})())
    detector = FakeDetector({"icon": [BoundingBox(10, 10, 30, 30, score=0.9, label="icon")]})
    actions = ActionServices(capture=capture, detector=detector, ocr=FakeOCR())
    calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: True,
    )
    actions.object_actions.click_object(ObjectLocateSpec(label="icon", min_score=0.3))
    assert calls == [(70, 80)]


def test_click_object_not_found() -> None:
    actions = _actions(detector=FakeDetector({"x": [BoundingBox(0, 0, 1, 1, score=0.2)]}))
    with pytest.raises(AssertionError, match="Object not found"):
        actions.object_actions.click_object(ObjectLocateSpec(label="x", min_score=0.3))


def test_click_object_retries_detector_before_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    class FlakyDetector(FakeDetector):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def detect(self, image, labels=None):
            self.calls += 1
            if self.calls < 2:
                return []
            return [BoundingBox(10, 10, 30, 30, score=0.9, label="icon")]

    detector = FlakyDetector()
    actions = _actions(detector=detector)
    calls = []
    sleep_calls = []
    monkeypatch.setattr(actions.base_actions, "click", lambda x, y: calls.append((x, y)))
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: True,
    )
    monkeypatch.setattr("time.sleep", lambda interval: sleep_calls.append(interval))

    actions.object_actions.click_object(ObjectLocateSpec(label="icon", min_score=0.3))

    assert calls == [(20, 20)]
    assert detector.calls == 2
    assert sleep_calls == [0.3]


def test_drag_object_to_position_uses_screen_coordinates_from_capture_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture(region=type("Region", (), {"left": 100, "top": 200})())
    detector = FakeDetector(
        {
            "audio_balance": [
                BoundingBox(10, 10, 30, 30, score=0.9, label="audio_balance")
            ]
        }
    )
    actions = ActionServices(capture=capture, detector=detector, ocr=FakeOCR())
    calls = []
    monkeypatch.setattr(
        actions.base_actions,
        "drag",
        lambda sx, sy, tx, ty, duration_ms: calls.append((sx, sy, tx, ty, duration_ms)),
    )
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: True,
    )

    actions.object_actions.drag_object_to_position(
        ObjectLocateSpec(label="audio_balance"),
        target_x=80,
        target_y=90,
        duration_ms=650,
    )

    assert calls == [(120, 220, 180, 290, 650)]


def test_drag_object_to_position_retries_detector_before_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FlakyDetector(FakeDetector):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def detect(self, image, labels=None):
            self.calls += 1
            if self.calls < 3:
                return []
            return [BoundingBox(10, 10, 30, 30, score=0.9, label="audio_balance")]

    detector = FlakyDetector()
    actions = _actions(detector=detector)
    calls = []
    sleep_calls = []
    monkeypatch.setattr(
        actions.base_actions,
        "drag",
        lambda sx, sy, tx, ty, duration_ms: calls.append((sx, sy, tx, ty, duration_ms)),
    )
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: True,
    )
    monkeypatch.setattr("time.sleep", lambda interval: sleep_calls.append(interval))

    actions.object_actions.drag_object_to_position(
        ObjectLocateSpec(label="audio_balance"),
        target_x=80,
        target_y=90,
        duration_ms=650,
    )

    assert calls == [(20, 20, 80, 90, 650)]
    assert detector.calls == 3
    assert sleep_calls == [0.3, 0.3]


def test_drag_object_to_object(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = FakeDetector(
        {
            "src": [BoundingBox(0, 0, 10, 10, score=0.7, label="src")],
            "dst": [BoundingBox(10, 20, 30, 40, score=0.8, label="dst")],
        }
    )
    actions = _actions(detector=detector)
    calls = []
    monkeypatch.setattr(
        actions.base_actions,
        "drag",
        lambda sx, sy, tx, ty, duration_ms: calls.append((sx, sy, tx, ty, duration_ms)),
    )
    actions.object_actions.drag_object_to_object(
        ObjectLocateSpec(label="src"),
        ObjectLocateSpec(label="dst"),
        duration_ms=600,
    )
    assert calls == [(5, 5, 20, 30, 600)]


def test_drag_object_to_object_detects_source_and_target_in_single_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingDetector(FakeDetector):
        def __init__(self, boxes_by_label=None):
            super().__init__(boxes_by_label=boxes_by_label)
            self.calls = []

        def detect(self, image, labels=None):
            self.calls.append(labels)
            return super().detect(image, labels=labels)

    detector = CountingDetector(
        {
            "src": [BoundingBox(0, 0, 10, 10, score=0.7, label="src")],
            "dst": [BoundingBox(10, 20, 30, 40, score=0.8, label="dst")],
        }
    )
    actions = _actions(detector=detector)
    monkeypatch.setattr(actions.base_actions, "drag", lambda sx, sy, tx, ty, duration_ms: None)

    actions.object_actions.drag_object_to_object(
        ObjectLocateSpec(label="src"),
        ObjectLocateSpec(label="dst"),
        duration_ms=600,
    )

    assert detector.calls == [["src", "dst"]]


def test_drag_object_to_object_uses_screen_coordinates_from_capture_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture(region=type("Region", (), {"left": 100, "top": 200})())
    detector = FakeDetector(
        {
            "src": [BoundingBox(0, 0, 10, 10, score=0.7, label="src")],
            "dst": [BoundingBox(10, 20, 30, 40, score=0.8, label="dst")],
        }
    )
    actions = ActionServices(capture=capture, detector=detector, ocr=FakeOCR())
    calls = []
    monkeypatch.setattr(
        actions.base_actions,
        "drag",
        lambda sx, sy, tx, ty, duration_ms: calls.append((sx, sy, tx, ty, duration_ms)),
    )
    actions.object_actions.drag_object_to_object(
        ObjectLocateSpec(label="src"),
        ObjectLocateSpec(label="dst"),
        duration_ms=600,
    )
    assert calls == [(105, 205, 120, 230, 600)]


def test_drag_object_to_object_missing_source() -> None:
    detector = FakeDetector({"dst": [BoundingBox(0, 0, 1, 1, score=1.0)]})
    actions = _actions(detector=detector)
    with pytest.raises(AssertionError, match="Source object not found"):
        actions.object_actions.drag_object_to_object(
            ObjectLocateSpec(label="src"),
            ObjectLocateSpec(label="dst"),
        )


def test_drag_object_to_object_requires_object_locate_specs() -> None:
    actions = _actions(detector=FakeDetector({}))
    with pytest.raises(TypeError, match="ObjectLocateSpec"):
        actions.object_actions.drag_object_to_object(
            "src",  # type: ignore[arg-type]
            ObjectLocateSpec(label="dst"),
        )


def test_verify_text_exists() -> None:
    ocr = FakeOCR([OCRText(text="Welcome Home", bbox=BoundingBox(0, 0, 1, 1))])
    actions = _actions(ocr=ocr)
    assert actions.text_actions.verify_text_exists(_text_locate("Home")) is True
    assert actions.text_actions.verify_text_exists(_text_locate("Welcome Home", exact=True)) is True
    assert actions.text_actions.verify_text_exists(_text_locate("Missing")) is False


def test_verify_text_exists_activates_bound_window(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _actions(ocr=FakeOCR([OCRText(text="Ready", bbox=BoundingBox(0, 0, 1, 1))]))
    events = []
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: events.append("activate") or True,
    )

    assert actions.text_actions.verify_text_exists(_text_locate("Ready")) is True
    assert events == ["activate"]


def test_verify_text_exists_uses_read_ocr_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _actions(ocr=FakeOCR())
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: True,
    )
    monkeypatch.setattr(
        actions.text_actions,
        "read_ocr",
        lambda image, ocr_options=None: [
            OCRText(text="Ready", bbox=BoundingBox(0, 0, 10, 10), score=1.0)
        ],
    )
    monkeypatch.setattr(
        actions.vision_runtime.ocr,
        "read",
        lambda image: (_ for _ in ()).throw(AssertionError("direct ocr.read should not be used")),
    )

    assert actions.text_actions.verify_text_exists(_text_locate("Ready")) is True


def test_verify_text_exists_for_phrase_spanning_multiple_ocr_boxes() -> None:
    ocr = FakeOCR(
        [
            OCRText(text="Hummingbird", bbox=BoundingBox(10, 20, 80, 40), score=0.9),
            OCRText(text="Printed", bbox=BoundingBox(90, 20, 150, 40), score=0.9),
            OCRText(text="Sweater", bbox=BoundingBox(160, 20, 230, 40), score=0.9),
        ]
    )
    actions = _actions(ocr=ocr)
    assert actions.text_actions.verify_text_exists(
        _text_locate("Hummingbird printed sweater")
    ) is True


def test_verify_object_exists() -> None:
    detector = FakeDetector({"item": [BoundingBox(0, 0, 1, 1, score=0.5, label="item")]})
    actions = _actions(detector=detector)
    assert actions.object_actions.verify_object_exists(
        ObjectLocateSpec(label="item", min_score=0.3)
    ) is True
    assert actions.object_actions.verify_object_exists(
        ObjectLocateSpec(label="item", min_score=0.8)
    ) is False


def test_verify_object_exists_can_filter_by_search_region() -> None:
    class ImageCapture(FakeCapture):
        def capture(self):
            return Image.new("RGB", (240, 240), "white")

    detector = FakeDetector(
        {
            "audio_balance": [
                BoundingBox(10, 10, 30, 30, score=0.7, label="audio_balance"),
                BoundingBox(150, 150, 190, 190, score=0.8, label="audio_balance"),
            ]
        }
    )
    actions = ActionServices(capture=ImageCapture(), detector=detector, ocr=FakeOCR())

    assert (
        actions.object_actions.verify_object_exists(
            ObjectLocateSpec(
                label="audio_balance",
                min_score=0.3,
                region=BoundingBox(x1=0, y1=0, x2=50, y2=50),
            ),
        )
        is True
    )
    assert (
        actions.object_actions.verify_object_exists(
            ObjectLocateSpec(
                label="audio_balance",
                min_score=0.3,
                region=BoundingBox(x1=40, y1=40, x2=120, y2=120),
            ),
        )
        is False
    )


def test_verify_object_exists_uses_named_detector() -> None:
    actions = _actions(
        detector=FakeDetector(),
        detectors={"icons": FakeDetector({"item": [BoundingBox(0, 0, 1, 1, score=0.6, label="item")]})},
    )
    assert (
        actions.object_actions.verify_object_exists(
            ObjectLocateSpec(label="item", min_score=0.3, detector="icons")
        )
        is True
    )


def test_verify_object_exists_retries_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    class FlakyDetector(FakeDetector):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def detect(self, image, labels=None):
            self.calls += 1
            if self.calls < 3:
                return []
            return [BoundingBox(10, 10, 30, 30, score=0.9, label="item")]

    detector = FlakyDetector()
    actions = _actions(detector=detector)
    sleep_calls = []
    monkeypatch.setattr(
        actions.base_actions,
        "activate_bound_window",
        lambda settle_seconds=0.2: True,
    )
    monkeypatch.setattr("time.sleep", lambda interval: sleep_calls.append(interval))

    assert actions.object_actions.verify_object_exists(
        ObjectLocateSpec(label="item", min_score=0.3)
    ) is True
    assert detector.calls == 3
    assert sleep_calls == [0.3, 0.3]


def test_wait_for_text_success(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _actions()
    values = iter([False, False, True])
    monkeypatch.setattr(actions.text_actions, "verify_text_exists", lambda text: next(values))
    timeline = iter([0.0, 0.2, 0.4, 0.6])
    monkeypatch.setattr("time.time", lambda: next(timeline))
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda interval: sleep_calls.append(interval))
    assert actions.text_actions.wait_for_text(
        _text_locate("x"),
        timeout=1.0,
        interval=0.1,
    ) is True
    assert sleep_calls == [0.1, 0.1]


def test_wait_for_text_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = _actions()
    monkeypatch.setattr(actions.text_actions, "verify_text_exists", lambda text: False)
    timeline = iter([0.0, 0.6, 1.2])
    monkeypatch.setattr("time.time", lambda: next(timeline))
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda interval: sleep_calls.append(interval))
    assert actions.text_actions.wait_for_text(
        _text_locate("x"),
        timeout=1.0,
        interval=0.5,
    ) is False
    assert sleep_calls == [0.5]
