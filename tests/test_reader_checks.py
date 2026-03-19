import types
import logging
from pathlib import Path

import pytest

from autoscene.core.exceptions import VerificationError
from autoscene.core.models import BoundingBox
from autoscene.runner.checks.ui.reader_checks import ReaderUIChecks


class FakeActions:
    def __init__(self):
        self.calls = []

    def screenshot(self, path=None):
        self.calls.append(("screenshot", path))
        return "frame"


class FakeReader:
    def __init__(self, value=1.0, score=1.0):
        self.value = value
        self.score = score
        self.calls = []

    def read(self, image, query=None, region=None):
        self.calls.append(("read", image, query, region))
        return types.SimpleNamespace(value=self.value, score=self.score)


class FakeReadAllReader(FakeReader):
    def __init__(self, values_by_query=None, score=1.0):
        super().__init__(value=1.0, score=score)
        self.values_by_query = dict(values_by_query or {})

    def read_all(self, image, region=None):
        self.calls.append(("read_all", image, region))
        return {
            str(query): types.SimpleNamespace(value=value, score=self.score)
            for query, value in self.values_by_query.items()
        }


def test_reader_checks_match_expected_value_with_tolerance() -> None:
    actions = FakeActions()
    reader = FakeReader(value=126, score=0.8)
    checks = ReaderUIChecks(screenshot_actions=actions, readers={"cluster": reader})

    ok = checks.handlers["reader_value_in_range"](
        {"reader": "cluster", "query": "speed", "expected": 120, "tolerance": 10}
    )

    assert ok is True
    assert actions.calls == [("screenshot", None)]
    assert reader.calls == [("read", "frame", "speed", None)]


def test_reader_checks_support_min_max_and_region() -> None:
    actions = FakeActions()
    reader = FakeReader(value=95, score=0.9)
    checks = ReaderUIChecks(screenshot_actions=actions, readers={"cluster": reader})

    ok = checks.handlers["reader_value_in_range"](
        {
            "reader": "cluster",
            "query": "temp",
            "min": 90,
            "max": 100,
            "region": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
        }
    )

    assert ok is True
    assert reader.calls == [
        ("read", "frame", "temp", BoundingBox(x1=1, y1=2, x2=3, y2=4, score=1.0, label=""))
    ]


def test_reader_checks_return_false_when_score_too_low() -> None:
    actions = FakeActions()
    reader = FakeReader(value=126, score=0.1)
    checks = ReaderUIChecks(screenshot_actions=actions, readers={"cluster": reader})

    ok = checks.handlers["reader_value_in_range"](
        {"reader": "cluster", "query": "speed", "expected": 126, "min_score": 0.2}
    )

    assert ok is False


def test_reader_checks_emit_runtime_logs(caplog: pytest.LogCaptureFixture) -> None:
    actions = FakeActions()
    reader = FakeReader(value=126, score=0.8)
    checks = ReaderUIChecks(screenshot_actions=actions, readers={"cluster": reader})

    with caplog.at_level(logging.INFO):
        ok = checks.handlers["reader_value_in_range"](
            {"reader": "cluster", "query": "speed", "expected": 120, "tolerance": 10}
        )

    assert ok is True
    assert "raw_value=126" in caplog.text
    assert "actual=126.000" in caplog.text
    assert "result=passed" in caplog.text


def test_reader_checks_support_image_path(tmp_path: Path) -> None:
    actions = FakeActions()
    reader = FakeReader(value=70, score=0.8)
    checks = ReaderUIChecks(screenshot_actions=actions, readers={"cluster": reader})
    image_path = tmp_path / "frame.png"
    image_path.write_text("not-a-real-image", encoding="utf-8")

    ok = checks.handlers["reader_value_in_range"](
        {
            "reader": "cluster",
            "query": "speed",
            "expected": 70,
            "image_path": str(image_path),
        }
    )

    assert ok is True
    assert actions.calls == []
    assert reader.calls == [("read", str(image_path), "speed", None)]


def test_reader_checks_accept_dedicated_screenshot_actions() -> None:
    actions = FakeActions()
    reader = FakeReader(value=70, score=0.8)
    checks = ReaderUIChecks(screenshot_actions=actions, readers={"cluster": reader})

    ok = checks.handlers["reader_value_in_range"](
        {
            "reader": "cluster",
            "query": "speed",
            "expected": 70,
            "tolerance": 0.0,
        }
    )

    assert ok is True
    assert actions.calls == [("screenshot", None)]
    assert reader.calls == [("read", "frame", "speed", None)]


def test_reader_checks_reuse_read_all_results_for_same_image_path(tmp_path: Path) -> None:
    actions = FakeActions()
    reader = FakeReadAllReader(values_by_query={"speed": 70, "rpm": 3200}, score=0.8)
    checks = ReaderUIChecks(screenshot_actions=actions, readers={"cluster": reader})
    image_path = tmp_path / "frame.png"
    image_path.write_text("not-a-real-image", encoding="utf-8")

    speed_ok = checks.handlers["reader_value_in_range"](
        {
            "reader": "cluster",
            "query": "speed",
            "expected": 70,
            "image_path": str(image_path),
        }
    )
    rpm_ok = checks.handlers["reader_value_in_range"](
        {
            "reader": "cluster",
            "query": "rpm",
            "expected": 3200,
            "image_path": str(image_path),
        }
    )

    assert speed_ok is True
    assert rpm_ok is True
    assert actions.calls == []
    assert reader.calls == [("read_all", str(image_path), None)]


def test_reader_checks_require_reader_name_when_multiple_configs() -> None:
    checks = ReaderUIChecks(
        screenshot_actions=FakeActions(),
        readers={"a": FakeReader(), "b": FakeReader()},
    )

    with pytest.raises(VerificationError, match="require 'reader'"):
        checks.handlers["reader_value_in_range"]({"expected": 1, "tolerance": 0.1})


def test_reader_checks_raise_when_bounds_missing() -> None:
    checks = ReaderUIChecks(screenshot_actions=FakeActions(), readers={"cluster": FakeReader()})

    with pytest.raises(VerificationError, match="requires either 'expected' or both 'min' and 'max'"):
        checks.handlers["reader_value_in_range"]({"reader": "cluster"})


def test_reader_checks_raise_for_non_numeric_reader_value() -> None:
    checks = ReaderUIChecks(
        screenshot_actions=FakeActions(),
        readers={"cluster": FakeReader(value="bad")},
    )

    with pytest.raises(VerificationError, match="reader result value"):
        checks.handlers["reader_value_in_range"](
            {"reader": "cluster", "expected": 1, "tolerance": 0.1}
        )
