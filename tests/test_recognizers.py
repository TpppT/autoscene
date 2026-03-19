import types

import pytest
from PIL import Image, ImageDraw

import autoscene.vision.algorithms.opencv.template_matcher as tmatch
import autoscene.vision.detectors.yolo_detector as ydet
import autoscene.vision.omni.omniparser_detector as odet
import autoscene.vision.ocr.tesseract_ocr as tocr
from autoscene.core.exceptions import DependencyMissingError
from autoscene.core.models import BoundingBox, OCRText, ObjectLocateSpec, TextLocateSpec
from autoscene.vision import (
    Detector,
    OCREngine,
    DetectorRegionStage,
    MatcherClassificationStage,
    NodeFilterStage,
    OperatorStage,
    TextLocateStage,
    VisionNode,
    VisionOperator,
    VisionOperatorOutput,
    VisionPipeline,
    VisionPipelineDetector,
    VisionPipelineStage,
    VisionStageBuildContext,
    build_vision_stage_registry,
    build_vision_registry_bundle,
    create_detector as create_vision_detector,
    create_matcher_adapter,
    create_operator,
    create_ocr_engine as create_vision_ocr_engine,
    create_detector,
    create_ocr_engine,
    register_comparator_adapter,
    register_detector,
    register_operator,
    register_ocr_engine,
    register_reader_adapter,
    run_object_locate_pipeline,
    run_text_locate_pipeline,
)
from autoscene.vision.detectors.cascade_detector import CascadeDetector
from autoscene.vision.detectors.mock_detector import MockDetector
from autoscene.vision.interfaces import (
    Detector as VisionDetector,
    OCREngine as VisionOCREngine,
)
from autoscene.vision.models import CompareResult, ReadResult
from autoscene.vision.ocr.mock_ocr import MockOCREngine


def test_mock_detector_filters_labels() -> None:
    detector = MockDetector(
        fixtures=[
            {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "label": "a", "score": 0.9},
            {"x1": 1, "y1": 1, "x2": 11, "y2": 11, "label": "b", "score": 0.8},
        ]
    )
    all_boxes = detector.detect("img")
    filtered = detector.detect("img", labels=["b"])
    assert len(all_boxes) == 2
    assert len(filtered) == 1
    assert filtered[0].label == "b"


def test_mock_ocr_returns_entries() -> None:
    engine = MockOCREngine(
        fixtures=[{"text": "hello", "x1": 1, "y1": 2, "x2": 3, "y2": 4, "score": 0.5}]
    )
    items = engine.read("img")
    assert len(items) == 1
    assert items[0].text == "hello"
    assert items[0].bbox.center == (2, 3)


def test_registry_create_mock_instances() -> None:
    assert isinstance(create_detector({"type": "mock"}), MockDetector)
    assert isinstance(create_ocr_engine({"type": "mock"}), MockOCREngine)


def test_registry_unknown_type_errors() -> None:
    with pytest.raises(ValueError, match="Unknown detector"):
        create_detector({"type": "nope"})
    with pytest.raises(ValueError, match="Unknown OCR engine"):
        create_ocr_engine({"type": "nope"})


def test_registry_register_custom_types() -> None:
    class CustomDetector(Detector):
        def detect(self, image, labels=None):
            return [BoundingBox(0, 0, 1, 1)]

    class CustomOCR(OCREngine):
        def read(self, image):
            return []

    register_detector("custom_detector", CustomDetector)
    register_ocr_engine("custom_ocr", CustomOCR)
    assert isinstance(create_detector({"type": "custom_detector"}), CustomDetector)
    assert isinstance(create_ocr_engine({"type": "custom_ocr"}), CustomOCR)


def test_registry_register_custom_operator_type() -> None:
    class FixedOperator(VisionOperator):
        @property
        def backend(self) -> str:
            return "fixed_operator"

        def run(self, image, nodes, *, context, params=None):
            del image, context
            label = str((params or {}).get("label", "operator_box"))
            base_region = nodes[0].region if nodes else BoundingBox(0, 0, 6, 6)
            return VisionOperatorOutput(
                nodes=[
                    VisionNode(
                        region=base_region,
                        label=label,
                        score=0.87,
                        source="fixed_operator",
                    )
                ],
                metadata={"kind": "fixed"},
            )

    register_operator("custom_operator", FixedOperator)

    operator = create_operator({"type": "custom_operator"})

    assert isinstance(operator, FixedOperator)
    result = operator.run("img", [], context=None, params={"label": "custom"})
    assert isinstance(result, VisionOperatorOutput)
    assert result.nodes[0].label == "custom"


def test_vision_exports_match_interfaces() -> None:
    assert Detector is VisionDetector
    assert OCREngine is VisionOCREngine


def test_vision_registry_exposes_detector_and_ocr_factories() -> None:
    assert isinstance(create_vision_detector({"type": "mock"}), MockDetector)
    assert isinstance(create_vision_ocr_engine({"type": "mock"}), MockOCREngine)


def test_vision_registry_supports_namespaced_detector_plugins() -> None:
    class PluginDetector(Detector):
        def detect(self, image, labels=None):
            return [BoundingBox(0, 0, 5, 5, score=0.9, label="plugin")]

    class DetectorPlugin:
        namespace = "lab"
        override = False

        def register_detectors(self, registry):
            registry.register("demo", PluginDetector)

    detector = create_vision_detector(
        {"type": "lab.demo"},
        plugins=(DetectorPlugin(),),
    )

    assert isinstance(detector, PluginDetector)
    assert detector.detect("img")[0].label == "plugin"


def test_vision_registry_supports_namespaced_operator_plugins() -> None:
    class PluginOperator(VisionOperator):
        @property
        def backend(self) -> str:
            return "plugin_operator"

        def run(self, image, nodes, *, context, params=None):
            del image, context, params
            return [
                VisionNode(
                    region=nodes[0].region if nodes else BoundingBox(0, 0, 8, 8),
                    label="anchor",
                    score=0.82,
                    source="plugin_operator",
                )
            ]

    class OperatorPlugin:
        namespace = "lab"
        override = False

        def register_operators(self, registry):
            registry.register("refine", PluginOperator)

    operator = create_operator(
        {"type": "lab.refine"},
        plugins=(OperatorPlugin(),),
    )

    assert isinstance(operator, PluginOperator)
    nodes = operator.run("img", [], context=None, params={})
    assert isinstance(nodes, list)
    assert nodes[0].source == "plugin_operator"


def test_vision_registry_plugin_override_requires_opt_in() -> None:
    class OverridePlugin:
        namespace = None
        override = False

        def register_detectors(self, registry):
            registry.register("mock", MockDetector)

    with pytest.raises(ValueError, match="Detector already registered: mock"):
        build_vision_registry_bundle(plugins=(OverridePlugin(),))


def test_yolo_detector_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ydet, "YOLO", None)
    with pytest.raises(DependencyMissingError, match="ultralytics is not installed"):
        ydet.YoloDetector(model_path="model.pt")


def test_yolo_detector_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    class Scalar:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class XY:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

    class Box:
        def __init__(self, cls_idx, score, xyxy):
            self.cls = [Scalar(cls_idx)]
            self.conf = [Scalar(score)]
            self.xyxy = [XY(xyxy)]

    result = types.SimpleNamespace(
        names={0: "button", 1: "icon"},
        boxes=[
            Box(0, 0.95, [1, 2, 11, 12]),
            Box(1, 0.50, [3, 4, 13, 14]),
        ],
    )

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

        def predict(self, image, conf, verbose):
            assert conf == 0.4
            assert verbose is False
            return [result]

    monkeypatch.setattr(ydet, "YOLO", FakeYOLO)
    detector = ydet.YoloDetector(model_path="m.pt", confidence=0.4)
    boxes = detector.detect("img", labels=["button"])
    assert len(boxes) == 1
    assert boxes[0].label == "button"
    assert boxes[0].score == pytest.approx(0.95)


def test_yolo_detector_detect_uses_inference_region(monkeypatch: pytest.MonkeyPatch) -> None:
    class Scalar:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class XY:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

    class Box:
        def __init__(self, cls_idx, score, xyxy):
            self.cls = [Scalar(cls_idx)]
            self.conf = [Scalar(score)]
            self.xyxy = [XY(xyxy)]

    result = types.SimpleNamespace(
        names={0: "pink_point"},
        boxes=[Box(0, 0.91, [10, 20, 110, 120])],
    )

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

        def predict(self, image, conf, verbose):
            assert conf == 0.4
            assert verbose is False
            assert image.size == (832, 720)
            return [result]

    monkeypatch.setattr(ydet, "YOLO", FakeYOLO)
    detector = ydet.YoloDetector(
        model_path="m.pt",
        confidence=0.4,
        inference_region={"x1": 448, "y1": 0, "x2": 1280, "y2": 720},
    )
    boxes = detector.detect(Image.new("RGB", (1280, 720), "white"))

    assert len(boxes) == 1
    assert boxes[0].label == "pink_point"
    assert boxes[0].score == pytest.approx(0.91)
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (458, 20, 558, 120)


def test_template_matcher_matches_icon_templates(tmp_path) -> None:
    templates_dir = tmp_path / "templates"
    settings_dir = templates_dir / "settings"
    settings_dir.mkdir(parents=True)
    home_path = templates_dir / "home.png"
    settings_path = settings_dir / "default.png"

    _icon_image("square").save(home_path)
    _icon_image("circle").save(settings_path)

    matcher = tmatch.TemplateMatcher(templates_dir=templates_dir, match_size=32)

    home_match = matcher.match(_icon_image("square").convert("RGB"))
    settings_match = matcher.match(_icon_image("circle").convert("RGB"))

    assert home_match is not None
    assert home_match.label == "home"
    assert home_match.score > 0.95
    assert settings_match is not None
    assert settings_match.label == "settings"
    assert settings_match.score > 0.95


def test_vision_template_matcher_matches_compatibility_module(tmp_path) -> None:
    templates_dir = tmp_path / "templates"
    (templates_dir / "search").mkdir(parents=True)
    _icon_image("square").save(templates_dir / "search" / "default.png")

    compat_matcher = tmatch.TemplateMatcher(templates_dir=templates_dir, match_size=32)
    vision_matcher = tmatch.TemplateMatcher(templates_dir=templates_dir, match_size=32)

    compat = compat_matcher.match(_icon_image("square").convert("RGB"))
    direct = vision_matcher.match(_icon_image("square").convert("RGB"))

    assert compat is not None
    assert direct is not None
    assert compat.label == direct.label
    assert compat.score == pytest.approx(direct.score)


def test_omniparser_detector_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(odet, "YOLO", None)
    with pytest.raises(DependencyMissingError, match="ultralytics is not installed"):
        odet.OmniParserDetector(model_path="model.pt")


def test_omniparser_detector_detects_and_classifies(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    templates_dir = tmp_path / "templates"
    (templates_dir / "search").mkdir(parents=True)
    (templates_dir / "settings").mkdir(parents=True)
    _icon_image("square").save(templates_dir / "search" / "default.png")
    _icon_image("circle").save(templates_dir / "settings" / "default.png")

    class Scalar:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class XY:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

    class Box:
        def __init__(self, score, xyxy):
            self.conf = [Scalar(score)]
            self.xyxy = [XY(xyxy)]

    result = types.SimpleNamespace(
        boxes=[
            Box(0.9, [2, 2, 26, 26]),
            Box(0.85, [34, 2, 58, 26]),
        ]
    )

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

        def predict(self, image, conf, iou, verbose, imgsz=None):
            assert conf == 0.05
            assert iou == 0.1
            assert verbose is False
            assert imgsz is None
            assert image.size == (60, 30)
            return [result]

    monkeypatch.setattr(odet, "YOLO", FakeYOLO)

    screen = Image.new("RGB", (60, 30), "white")
    left = _icon_image("square")
    right = _icon_image("circle")
    screen.paste(left, (2, 2), mask=left)
    screen.paste(right, (34, 2), mask=right)

    detector = odet.OmniParserDetector(
        model_path="omniparser.pt",
        templates_dir=templates_dir,
        template_match_threshold=0.7,
    )

    boxes = detector.detect(screen)
    filtered = detector.detect(screen, labels=["settings"])

    assert [box.label for box in boxes] == ["search", "settings"]
    assert boxes[0].score > 0.7
    assert boxes[1].score > 0.65
    assert len(filtered) == 1
    assert filtered[0].label == "settings"


def test_omniparser_detector_passes_allowed_labels_to_template_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Scalar:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class XY:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

    class Box:
        def __init__(self, score, xyxy):
            self.conf = [Scalar(score)]
            self.xyxy = [XY(xyxy)]

    result = types.SimpleNamespace(boxes=[Box(0.9, [2, 2, 26, 26])])

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

        def predict(self, image, conf, iou, verbose, imgsz=None):
            return [result]

    class FakeMatcher:
        def __init__(self):
            self.calls = []
            self.labels = {"search", "settings"}

        def match(self, crop, labels=None):
            self.calls.append(labels)
            return types.SimpleNamespace(label="settings", score=0.95)

    monkeypatch.setattr(odet, "YOLO", FakeYOLO)

    detector = odet.OmniParserDetector(model_path="omniparser.pt")
    matcher = FakeMatcher()
    detector._matcher = matcher

    screen = Image.new("RGB", (60, 30), "white")
    filtered = detector.detect(screen, labels=["settings"])

    assert len(filtered) == 1
    assert filtered[0].label == "settings"
    assert matcher.calls == [{"settings"}, None]


def test_registry_create_omniparser_instance(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    template_path = tmp_path / "icon.png"
    _icon_image("square").save(template_path)

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

        def predict(self, image, conf, iou, verbose, imgsz=None):
            return [types.SimpleNamespace(boxes=[])]

    monkeypatch.setattr(odet, "YOLO", FakeYOLO)

    detector = create_detector(
        {
            "type": "omniparser",
            "model_path": "omniparser.pt",
            "template_paths": {"home": str(template_path)},
        }
    )
    assert isinstance(detector, odet.OmniParserDetector)


def test_cascade_detector_can_chain_layout_and_template_matching(tmp_path) -> None:
    templates_dir = tmp_path / "templates"
    (templates_dir / "play").mkdir(parents=True)
    _icon_image("square").save(templates_dir / "play" / "default.png")

    class FixedRegionDetector(Detector):
        def detect(self, image, labels=None):
            return [BoundingBox(2, 2, 26, 26, score=0.8, label="panel")]

    register_detector("test_fixed_region", FixedRegionDetector)

    screen = Image.new("RGB", (40, 40), "white")
    icon = _icon_image("square")
    screen.paste(icon, (2, 2), mask=icon)

    detector = create_detector(
        {
            "type": "cascade",
            "region_detector": {"type": "test_fixed_region"},
            "detail_matcher": {
                "type": "template_matcher",
                "templates_dir": str(templates_dir),
                "match_size": 32,
            },
            "match_threshold": 0.5,
        }
    )

    boxes = detector.detect(screen, labels=["play"])

    assert len(boxes) == 1
    assert boxes[0].label == "play"
    assert boxes[0].score > 0.5
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (2, 2, 26, 26)


def test_vision_pipeline_records_stage_trace(tmp_path) -> None:
    templates_dir = tmp_path / "templates"
    (templates_dir / "play").mkdir(parents=True)
    _icon_image("square").save(templates_dir / "play" / "default.png")

    class FixedRegionDetector(Detector):
        def detect(self, image, labels=None):
            return [BoundingBox(2, 2, 26, 26, score=0.8, label="panel")]

    screen = Image.new("RGB", (40, 40), "white")
    icon = _icon_image("square")
    screen.paste(icon, (2, 2), mask=icon)

    pipeline = VisionPipeline(
        [
            DetectorRegionStage(FixedRegionDetector(), labels=["panel"], max_regions=1),
            MatcherClassificationStage(
                create_matcher_adapter(
                    {
                        "type": "template_matcher",
                        "templates_dir": str(templates_dir),
                        "match_size": 32,
                    }
                ),
                match_threshold=0.5,
            ),
        ]
    )

    result = pipeline.run(screen, labels=["play"])

    assert len(result.nodes) == 1
    assert len(result.boxes) == 1
    assert result.boxes[0].label == "play"
    assert result.nodes[0].label == "play"
    assert result.nodes[0].source.endswith("default.png")
    assert [entry.stage_name for entry in result.nodes[0].trace] == [
        "region_detector",
        "detail_matcher",
    ]
    assert [entry.stage_kind for entry in result.trace] == [
        "detector_region",
        "matcher_classification",
    ]
    assert result.trace[0].input_count == 0
    assert result.trace[0].output_count == 1
    assert result.trace[1].input_count == 1
    assert result.trace[1].output_count == 1


def test_node_filter_stage_can_filter_by_score_and_region() -> None:
    class FixedDetector(Detector):
        def detect(self, image, labels=None):
            del image, labels
            return [
                BoundingBox(2, 2, 8, 8, score=0.8, label="play"),
                BoundingBox(20, 20, 28, 28, score=0.9, label="play"),
                BoundingBox(4, 4, 10, 10, score=0.2, label="play"),
            ]

    pipeline = VisionPipeline(
        [
            DetectorRegionStage(FixedDetector(), labels=["play"]),
            NodeFilterStage(
                min_score=0.5,
                region=BoundingBox(0, 0, 12, 12),
            ),
        ]
    )

    result = pipeline.run(Image.new("RGB", (32, 32), "white"), labels=["play"])

    assert len(result.nodes) == 1
    assert result.nodes[0].label == "play"
    assert result.boxes[0].center == (5, 5)
    assert [entry.stage_kind for entry in result.trace] == [
        "detector_region",
        "node_filter",
    ]


def test_detector_region_stage_preserves_nested_pipeline_nodes() -> None:
    class FixedOCR(OCREngine):
        def read(self, image):
            del image
            return [OCRText(text="Play", bbox=BoundingBox(2, 2, 8, 8), score=0.9)]

    register_ocr_engine("test_nested_trace_ocr", FixedOCR)

    detector = create_detector(
        {
            "type": "pipeline",
            "stages": [
                {
                    "type": "ocr_classification",
                    "ocr": {"type": "test_nested_trace_ocr"},
                    "match_mode": "contains",
                    "min_score": 0.5,
                }
            ],
        }
    )

    pipeline = VisionPipeline(
        [
            DetectorRegionStage(detector, labels=["play"]),
            NodeFilterStage(min_score=0.5),
        ]
    )

    result = pipeline.run(Image.new("RGB", (20, 20), "white"), labels=["play"])

    assert len(result.nodes) == 1
    assert result.nodes[0].text == "Play"
    assert [entry.stage_kind for entry in result.nodes[0].trace] == [
        "ocr_classification",
        "detector_region",
        "node_filter",
    ]


def test_run_object_locate_pipeline_filters_detector_results() -> None:
    class FixedDetector(Detector):
        def detect(self, image, labels=None):
            del image
            assert tuple(labels or ()) == ("play",)
            return [
                BoundingBox(2, 2, 8, 8, score=0.8, label="play"),
                BoundingBox(20, 20, 28, 28, score=0.95, label="play"),
            ]

    result = run_object_locate_pipeline(
        Image.new("RGB", (32, 32), "white"),
        detector=FixedDetector(),
        locate=ObjectLocateSpec(
            label="play",
            min_score=0.5,
            region=BoundingBox(0, 0, 12, 12),
        ),
    )

    assert len(result.boxes) == 1
    assert result.boxes[0].center == (5, 5)
    assert result.region is not None
    assert [entry.stage_kind for entry in result.pipeline_result.trace] == [
        "detector_region",
        "node_filter",
    ]


def test_text_locate_stage_matches_phrase_across_ocr_boxes() -> None:
    class FixedOCR(OCREngine):
        def read(self, image):
            del image
            return [
                OCRText(text="Hummingbird", bbox=BoundingBox(2, 4, 20, 14), score=0.9),
                OCRText(text="Printed", bbox=BoundingBox(24, 4, 42, 14), score=0.85),
                OCRText(text="Sweater", bbox=BoundingBox(46, 4, 64, 14), score=0.88),
            ]

    pipeline = VisionPipeline(
        [
            TextLocateStage(
                FixedOCR(),
                query="Hummingbird printed sweater",
            )
        ]
    )

    result = pipeline.run(Image.new("RGB", (80, 30), "white"))

    assert len(result.nodes) == 1
    assert result.nodes[0].text == "Hummingbird Printed Sweater"
    assert result.nodes[0].bbox is not None
    assert result.nodes[0].bbox.center == (33, 9)
    assert [entry.stage_kind for entry in result.nodes[0].trace] == ["text_locate"]


def test_run_text_locate_pipeline_respects_region_filter() -> None:
    class FixedOCR(OCREngine):
        def read(self, image):
            assert image.size == (30, 20)
            return [
                OCRText(text="Ready", bbox=BoundingBox(3, 5, 15, 15), score=0.92),
            ]

    result = run_text_locate_pipeline(
        Image.new("RGB", (100, 50), "white"),
        ocr_engine=FixedOCR(),
        locate=TextLocateSpec(
            text="Ready",
            region=BoundingBox(40, 10, 70, 30),
        ),
    )

    assert len(result.nodes) == 1
    assert result.match is not None
    assert result.match.text == "Ready"
    assert result.match.bbox.center == (49, 20)
    assert result.region is not None
    assert [entry.stage_kind for entry in result.pipeline_result.trace] == ["text_locate"]


def test_cascade_detector_exposes_last_pipeline_result(tmp_path) -> None:
    templates_dir = tmp_path / "templates"
    (templates_dir / "play").mkdir(parents=True)
    _icon_image("square").save(templates_dir / "play" / "default.png")

    class FixedRegionDetector(Detector):
        def detect(self, image, labels=None):
            return [BoundingBox(2, 2, 26, 26, score=0.8, label="panel")]

    register_detector("test_trace_region", FixedRegionDetector)

    screen = Image.new("RGB", (40, 40), "white")
    icon = _icon_image("square")
    screen.paste(icon, (2, 2), mask=icon)

    detector = create_detector(
        {
            "type": "cascade",
            "region_detector": {"type": "test_trace_region"},
            "detail_matcher": {
                "type": "template_matcher",
                "templates_dir": str(templates_dir),
                "match_size": 32,
            },
            "match_threshold": 0.5,
        }
    )

    assert isinstance(detector, CascadeDetector)

    boxes = detector.detect(screen, labels=["play"])

    assert len(boxes) == 1
    assert detector.last_pipeline_result is not None
    assert [entry.stage_name for entry in detector.last_pipeline_result.trace] == [
        "region_detector",
        "detail_matcher",
    ]
    assert detector.last_pipeline_result.nodes[0].label == "play"


def test_pipeline_detector_config_can_chain_layout_and_template_matching(tmp_path) -> None:
    templates_dir = tmp_path / "templates"
    (templates_dir / "play").mkdir(parents=True)
    _icon_image("square").save(templates_dir / "play" / "default.png")

    class FixedRegionDetector(Detector):
        def detect(self, image, labels=None):
            return [BoundingBox(2, 2, 26, 26, score=0.8, label="panel")]

    register_detector("test_pipeline_region", FixedRegionDetector)

    screen = Image.new("RGB", (40, 40), "white")
    icon = _icon_image("square")
    screen.paste(icon, (2, 2), mask=icon)

    detector = create_detector(
        {
            "type": "pipeline",
            "stages": [
                {
                    "type": "detector_region",
                    "detector": {"type": "test_pipeline_region"},
                    "labels": ["panel"],
                    "max_regions": 1,
                },
                {
                    "type": "matcher_classification",
                    "matcher": {
                        "type": "template_matcher",
                        "templates_dir": str(templates_dir),
                        "match_size": 32,
                    },
                    "match_threshold": 0.5,
                },
            ],
        }
    )

    assert isinstance(detector, VisionPipelineDetector)

    boxes = detector.detect(screen, labels=["play"])

    assert len(boxes) == 1
    assert boxes[0].label == "play"
    assert detector.last_pipeline_result is not None
    assert detector.last_pipeline_result.nodes[0].source.endswith("default.png")
    assert [entry.stage_kind for entry in detector.last_pipeline_result.trace] == [
        "detector_region",
        "matcher_classification",
    ]


def test_pipeline_detector_config_supports_ocr_stage() -> None:
    class FixedOCR(OCREngine):
        def read(self, image):
            return [OCRText(text="Play", bbox=BoundingBox(2, 2, 8, 8), score=0.9)]

    register_ocr_engine("test_fixed_ocr", FixedOCR)

    detector = create_detector(
        {
            "type": "pipeline",
            "stages": [
                {
                    "type": "ocr_classification",
                    "ocr": {"type": "test_fixed_ocr"},
                    "match_mode": "contains",
                    "min_score": 0.5,
                }
            ],
        }
    )

    boxes = detector.detect(Image.new("RGB", (20, 20), "white"), labels=["play"])

    assert len(boxes) == 1
    assert boxes[0].label == "play"
    assert boxes[0].score == pytest.approx(0.9)
    assert detector.last_pipeline_result is not None
    assert detector.last_pipeline_result.nodes[0].text == "Play"
    assert detector.last_pipeline_result.nodes[0].source == "FixedOCR"


def test_pipeline_detector_config_supports_detector_refinement_stage() -> None:
    nested_crop_sizes = []

    class FixedOuterDetector(Detector):
        def detect(self, image, labels=None):
            del image, labels
            return [BoundingBox(-5, -5, 10, 10, score=0.8, label="panel")]

    class FixedInnerDetector(Detector):
        def detect(self, image, labels=None):
            del labels
            nested_crop_sizes.append(image.size)
            return [BoundingBox(1, 1, 4, 4, score=0.9, label="play")]

    register_detector("test_fixed_outer_detector", FixedOuterDetector)
    register_detector("test_fixed_inner_detector", FixedInnerDetector)

    detector = create_detector(
        {
            "type": "pipeline",
            "stages": [
                {
                    "type": "detector_region",
                    "detector": {"type": "test_fixed_outer_detector"},
                },
                {
                    "type": "detector_refinement",
                    "detector": {"type": "test_fixed_inner_detector"},
                },
            ],
        }
    )

    boxes = detector.detect(Image.new("RGB", (20, 20), "white"), labels=["play"])

    assert nested_crop_sizes == [(10, 10)]
    assert len(boxes) == 1
    assert boxes[0].label == "play"
    assert boxes[0].score == pytest.approx(0.72)
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (1, 1, 4, 4)


def test_pipeline_detector_config_supports_reader_stage() -> None:
    class FixedReader:
        @property
        def backend(self):
            return "fixed"

        def read(self, image, query=None, region=None):
            del image, query
            return ReadResult(
                value=72,
                score=0.8,
                label="speed",
                source="fixed_reader",
                region=region,
            )

    register_reader_adapter("test_fixed_reader", FixedReader)

    detector = create_detector(
        {
            "type": "pipeline",
            "stages": [
                {
                    "type": "reader_classification",
                    "reader": {"type": "test_fixed_reader"},
                    "label_source": "label",
                    "min_score": 0.5,
                }
            ],
        }
    )

    boxes = detector.detect(Image.new("RGB", (20, 20), "white"), labels=["speed"])

    assert len(boxes) == 1
    assert boxes[0].label == "speed"
    assert boxes[0].score == pytest.approx(0.8)
    assert detector.last_pipeline_result is not None
    assert detector.last_pipeline_result.nodes[0].value == 72
    assert detector.last_pipeline_result.nodes[0].source == "fixed_reader"


def test_pipeline_detector_config_supports_comparator_stage() -> None:
    class FixedComparator:
        @property
        def backend(self):
            return "fixed"

        def compare(self, image, expected, region=None):
            del image, expected, region
            return CompareResult(passed=True, score=0.85, source="fixed_comparator")

    register_comparator_adapter("test_fixed_comparator", FixedComparator)

    detector = create_detector(
        {
            "type": "pipeline",
            "stages": [
                {
                    "type": "comparator_filter",
                    "comparator": {"type": "test_fixed_comparator"},
                    "expected": {"id": "reference"},
                    "pass_label": "matched_panel",
                }
            ],
        }
    )

    boxes = detector.detect(Image.new("RGB", (20, 20), "white"), labels=["matched_panel"])

    assert len(boxes) == 1
    assert boxes[0].label == "matched_panel"
    assert boxes[0].score == pytest.approx(0.85)
    assert detector.last_pipeline_result is not None
    assert detector.last_pipeline_result.nodes[0].metadata["passed"] is True
    assert detector.last_pipeline_result.nodes[0].source == "fixed_comparator"


def test_pipeline_detector_config_supports_operator_stage() -> None:
    class FixedRegionDetector(Detector):
        def detect(self, image, labels=None):
            del image, labels
            return [BoundingBox(2, 2, 18, 18, score=0.9, label="panel")]

    class OffsetOperator(VisionOperator):
        @property
        def backend(self) -> str:
            return "offset_operator"

        def run(self, image, nodes, *, context, params=None):
            del image, context
            delta = int((params or {}).get("delta", 0))
            label = str((params or {}).get("label", "offset_box"))
            region = nodes[0].to_bounding_box()
            return VisionOperatorOutput(
                nodes=[
                    VisionNode(
                        region=BoundingBox(
                            region.x1 + delta,
                            region.y1 + delta,
                            region.x2 + delta,
                            region.y2 + delta,
                            score=0.88,
                            label=label,
                        ),
                        label=label,
                        score=0.88,
                        source="offset_operator",
                        metadata={"delta": delta},
                    )
                ],
                metadata={"mode": "offset"},
            )

    register_detector("test_operator_region", FixedRegionDetector)
    register_operator("test_offset_operator", OffsetOperator)

    detector = create_detector(
        {
            "type": "pipeline",
            "stages": [
                {
                    "type": "detector_region",
                    "detector": {"type": "test_operator_region"},
                },
                {
                    "type": "operator",
                    "operator": {"type": "test_offset_operator"},
                    "delta": 2,
                    "label": "operator_target",
                },
            ],
        }
    )

    boxes = detector.detect(Image.new("RGB", (30, 30), "white"), labels=["operator_target"])

    assert len(boxes) == 1
    assert boxes[0].label == "operator_target"
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (4, 4, 20, 20)
    assert detector.last_pipeline_result is not None
    assert detector.last_pipeline_result.nodes[0].metadata["delta"] == 2
    assert detector.last_pipeline_result.trace[1].metadata["backend"] == "offset_operator"
    assert detector.last_pipeline_result.trace[1].metadata["mode"] == "offset"


def test_custom_stage_registry_can_build_pipeline_without_core_changes() -> None:
    class ConstantNodeStage(VisionPipelineStage):
        stage_kind = "custom_constant"

        def __init__(self, *, label: str, score: float = 0.7) -> None:
            super().__init__(name="constant_stage")
            self._label = label
            self._score = score

        def run(self, *, image, nodes, context):
            del image, nodes, context
            return [
                VisionNode(
                    region=BoundingBox(1, 2, 5, 6, score=self._score, label=self._label),
                    label=self._label,
                    score=self._score,
                    source="custom_stage",
                )
            ]

    def build_constant_stage(
        payload: dict[str, object],
        build_context: VisionStageBuildContext,
    ) -> VisionPipelineStage:
        del build_context
        return ConstantNodeStage(
            label=str(payload.get("label", "constant")),
            score=float(payload.get("score", 0.7)),
        )

    stage_registry = build_vision_stage_registry()
    stage_registry.register("constant_node", build_constant_stage, override=False)

    detector = VisionPipelineDetector(
        stages=[{"type": "constant_node", "label": "custom_box", "score": 0.91}],
        stage_registry=stage_registry,
    )

    boxes = detector.detect(Image.new("RGB", (20, 20), "white"), labels=["custom_box"])

    assert len(boxes) == 1
    assert boxes[0].label == "custom_box"
    assert boxes[0].score == pytest.approx(0.91)


def test_stage_plugin_can_extend_pipeline_registry() -> None:
    class ConstantNodeStage(VisionPipelineStage):
        stage_kind = "custom_plugin"

        def __init__(self, *, label: str) -> None:
            super().__init__(name="plugin_stage")
            self._label = label

        def run(self, *, image, nodes, context):
            del image, nodes, context
            return [
                VisionNode(
                    region=BoundingBox(0, 0, 3, 3, score=0.8, label=self._label),
                    label=self._label,
                    score=0.8,
                    source="plugin_stage",
                )
            ]

    class StagePlugin:
        namespace = "sample"
        override = False

        def register_pipeline_stages(self, registry):
            def build_stage(payload, build_context):
                del build_context
                return ConstantNodeStage(label=str(payload.get("label", "plugin_label")))

            registry.register("constant_node", build_stage)

    bundle = build_vision_registry_bundle(plugins=(StagePlugin(),))
    detector = create_vision_detector(
        {
            "type": "pipeline",
            "stages": [{"type": "sample.constant_node", "label": "plugin_box"}],
        },
        registry_bundle=bundle,
    )

    boxes = detector.detect(Image.new("RGB", (20, 20), "white"), labels=["plugin_box"])

    assert len(boxes) == 1
    assert boxes[0].label == "plugin_box"
    assert detector.last_pipeline_result is not None
    assert detector.last_pipeline_result.nodes[0].source == "plugin_stage"


def test_tesseract_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tocr, "pytesseract", None)
    with pytest.raises(DependencyMissingError, match="pytesseract is not installed"):
        tocr.TesseractOCREngine()


def test_tesseract_read_filters_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOutput:
        DICT = "DICT"

    fake_runtime = types.SimpleNamespace(tesseract_cmd=None)

    class FakeTesseract:
        pytesseract = fake_runtime
        Output = FakeOutput
        TesseractNotFoundError = FileNotFoundError

        @staticmethod
        def image_to_data(image, lang, output_type, config=None):
            assert lang == "eng"
            assert output_type == "DICT"
            assert config is None
            return {
                "text": ["", "low", "ok"],
                "conf": ["90", "20", "80"],
                "left": [0, 1, 2],
                "top": [0, 1, 2],
                "width": [1, 2, 3],
                "height": [1, 2, 3],
            }

    monkeypatch.setattr(tocr, "pytesseract", FakeTesseract)
    monkeypatch.setattr(
        tocr.TesseractOCREngine,
        "_find_tesseract_cmd",
        staticmethod(lambda configured: "C:\\Tesseract-OCR\\tesseract.exe"),
    )
    engine = tocr.TesseractOCREngine(lang="eng", min_confidence=40)
    result = engine.read("img")
    assert fake_runtime.tesseract_cmd == "C:\\Tesseract-OCR\\tesseract.exe"
    assert len(result) == 1
    assert result[0].text == "ok"
    assert result[0].bbox == BoundingBox(x1=2, y1=2, x2=5, y2=5, score=0.8, label="")


def test_tesseract_read_applies_preprocess_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOutput:
        DICT = "DICT"

    fake_runtime = types.SimpleNamespace(tesseract_cmd=None)
    calls = {}

    class FakeTesseract:
        pytesseract = fake_runtime
        Output = FakeOutput
        TesseractNotFoundError = FileNotFoundError

        @staticmethod
        def image_to_data(image, lang, output_type, config=None):
            calls["size"] = getattr(image, "size", None)
            calls["mode"] = getattr(image, "mode", None)
            calls["config"] = config
            assert lang == "eng"
            assert output_type == "DICT"
            return {
                "text": ["placeholder"],
                "conf": ["88"],
                "left": [4],
                "top": [5],
                "width": [20],
                "height": [6],
            }

    monkeypatch.setattr(tocr, "pytesseract", FakeTesseract)
    monkeypatch.setattr(
        tocr.TesseractOCREngine,
        "_find_tesseract_cmd",
        staticmethod(lambda configured: "C:\\Tesseract-OCR\\tesseract.exe"),
    )

    engine = tocr.TesseractOCREngine(
        lang="eng",
        min_confidence=40,
        preprocess={"enabled": True, "scale": 2.0, "threshold": "none"},
        tesseract_config="--psm 6",
    )
    image = Image.new("RGB", (30, 12), (245, 245, 245))
    result = engine.read(image)

    assert fake_runtime.tesseract_cmd == "C:\\Tesseract-OCR\\tesseract.exe"
    assert calls["size"] == (60, 24)
    assert calls["mode"] in {"L", "RGB"}
    assert calls["config"] == "--psm 6"
    assert len(result) == 1
    assert result[0].text == "placeholder"
    assert result[0].bbox == BoundingBox(x1=2, y1=2, x2=12, y2=5, score=0.88, label="")


def test_tesseract_preprocess_rejects_unknown_options() -> None:
    with pytest.raises(ValueError, match="Unknown OCR preprocess options"):
        tocr.TesseractOCREngine._normalize_preprocess({"bogus": True})


def test_tesseract_read_with_overrides_uses_override_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOutput:
        DICT = "DICT"

    fake_runtime = types.SimpleNamespace(tesseract_cmd=None)
    calls = {}

    class FakeTesseract:
        pytesseract = fake_runtime
        Output = FakeOutput
        TesseractNotFoundError = FileNotFoundError

        @staticmethod
        def image_to_data(image, lang, output_type, config=None):
            calls["lang"] = lang
            calls["config"] = config
            calls["size"] = getattr(image, "size", None)
            return {
                "text": ["20"],
                "conf": ["95"],
                "left": [30],
                "top": [18],
                "width": [18],
                "height": [12],
            }

    monkeypatch.setattr(tocr, "pytesseract", FakeTesseract)
    monkeypatch.setattr(
        tocr.TesseractOCREngine,
        "_find_tesseract_cmd",
        staticmethod(lambda configured: "C:\\Tesseract-OCR\\tesseract.exe"),
    )
    engine = tocr.TesseractOCREngine(lang="eng", min_confidence=40)

    result = engine.read_with_overrides(
        Image.new("RGB", (20, 10), "white"),
        overrides={
            "lang": "eng",
            "min_confidence": 50,
            "tesseract_config": "--psm 11",
            "preprocess": {"enabled": True, "scale": 3.0, "threshold": "none"},
        },
    )

    assert calls["lang"] == "eng"
    assert calls["config"] == "--psm 11"
    assert calls["size"] == (60, 30)
    assert result[0].bbox == BoundingBox(x1=10, y1=6, x2=16, y2=10, score=0.95, label="")


def test_tesseract_read_with_overrides_merges_preprocess_with_base_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOutput:
        DICT = "DICT"

    fake_runtime = types.SimpleNamespace(tesseract_cmd=None)
    calls = {}

    class FakeTesseract:
        pytesseract = fake_runtime
        Output = FakeOutput
        TesseractNotFoundError = FileNotFoundError

        @staticmethod
        def image_to_data(image, lang, output_type, config=None):
            calls["lang"] = lang
            calls["config"] = config
            calls["size"] = getattr(image, "size", None)
            return {
                "text": ["20"],
                "conf": ["95"],
                "left": [30],
                "top": [18],
                "width": [18],
                "height": [12],
            }

    monkeypatch.setattr(tocr, "pytesseract", FakeTesseract)
    monkeypatch.setattr(
        tocr.TesseractOCREngine,
        "_find_tesseract_cmd",
        staticmethod(lambda configured: "C:\\Tesseract-OCR\\tesseract.exe"),
    )
    engine = tocr.TesseractOCREngine(
        lang="eng",
        min_confidence=40,
        preprocess={"enabled": True, "scale": 3.0, "threshold": "adaptive_gaussian"},
        tesseract_config="--psm 6",
    )

    result = engine.read_with_overrides(
        Image.new("RGB", (20, 10), "white"),
        overrides={
            "tesseract_config": "--psm 11",
            "preprocess": {"threshold": "none"},
        },
    )

    assert calls["lang"] == "eng"
    assert calls["config"] == "--psm 11"
    assert calls["size"] == (60, 30)
    assert result[0].bbox == BoundingBox(x1=10, y1=6, x2=16, y2=10, score=0.95, label="")


def test_tesseract_read_with_overrides_rejects_unknown_options() -> None:
    with pytest.raises(ValueError, match="Unknown OCR override options"):
        tocr.TesseractOCREngine._normalize_overrides({"bad": True})


def test_tesseract_read_raises_clear_error_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runtime = types.SimpleNamespace(tesseract_cmd=None)

    class FakeOutput:
        DICT = "DICT"

    class FakeNotFoundError(Exception):
        pass

    class FakeTesseract:
        pytesseract = fake_runtime
        Output = FakeOutput
        TesseractNotFoundError = FakeNotFoundError

        @staticmethod
        def image_to_data(image, lang, output_type, config=None):
            raise FakeNotFoundError("tesseract is not installed or it's not in your PATH")

    monkeypatch.setattr(tocr, "pytesseract", FakeTesseract)
    monkeypatch.setattr(
        tocr.TesseractOCREngine,
        "_find_tesseract_cmd",
        staticmethod(lambda configured: None),
    )
    engine = tocr.TesseractOCREngine(lang="eng", min_confidence=40)
    with pytest.raises(DependencyMissingError, match="Tesseract OCR executable was not found"):
        engine.read("img")


def _icon_image(kind: str, size: tuple[int, int] = (24, 24)) -> Image.Image:
    image = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    if kind == "square":
        draw.rectangle((4, 4, size[0] - 4, size[1] - 4), fill=(0, 0, 0, 255))
        return image
    if kind == "circle":
        draw.ellipse((4, 4, size[0] - 4, size[1] - 4), fill=(0, 0, 0, 255))
        return image
    raise ValueError(f"Unsupported icon kind: {kind}")
