from autoscene.core.exceptions import (
    ActionExecutionError,
    DependencyMissingError,
    FrameworkError,
    VerificationError,
)
from autoscene.core.models import BoundingBox, TestCase as CaseModel


def test_exception_hierarchy() -> None:
    assert issubclass(DependencyMissingError, FrameworkError)
    assert issubclass(ActionExecutionError, FrameworkError)
    assert issubclass(VerificationError, FrameworkError)


def test_bounding_box_center() -> None:
    box = BoundingBox(x1=10, y1=20, x2=14, y2=28)
    assert box.center == (12, 24)


def test_test_case_defaults_are_initialized() -> None:
    case = CaseModel(name="demo")
    assert case.emulator == {}
    assert case.detector == {}
    assert case.detectors == {}
    assert case.log_sources == {}
    assert case.ocr == {}
    assert case.capture == {}
    assert case.setup == []
    assert case.steps == []
    assert case.verification_setup == []
    assert case.verification == []
    assert case.teardown == []
