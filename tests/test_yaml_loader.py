import types

import pytest

import autoscene.yamlcase.loader as loader
from autoscene.core.exceptions import DependencyMissingError
from autoscene.core.models import TestCase as CaseModel


def test_ensure_list_behaviors() -> None:
    assert loader._ensure_list(None, "x") == []
    assert loader._ensure_list([{"a": 1}], "x") == [{"a": 1}]
    with pytest.raises(ValueError, match="must be a list"):
        loader._ensure_list("bad", "x")
    with pytest.raises(ValueError, match="must be an object/dict"):
        loader._ensure_list([1], "x")


def test_load_test_case_missing_yaml_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    file_path = tmp_path / "case.yaml"
    file_path.write_text("name: demo", encoding="utf-8")
    monkeypatch.setattr(loader, "yaml", None)
    with pytest.raises(DependencyMissingError, match="PyYAML is not installed"):
        loader.load_test_case(file_path)


def test_load_test_case_top_level_must_be_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    file_path = tmp_path / "case.yaml"
    file_path.write_text("irrelevant", encoding="utf-8")
    fake_yaml = types.SimpleNamespace(safe_load=lambda text: ["not", "mapping"])
    monkeypatch.setattr(loader, "yaml", fake_yaml)
    with pytest.raises(ValueError, match="Top-level YAML must be a mapping/object"):
        loader.load_test_case(file_path)


def test_load_test_case_success(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    file_path = tmp_path / "case.yaml"
    file_path.write_text("anything", encoding="utf-8")
    raw = {
        "name": "demo",
        "emulator": {"type": "network_device"},
        "detector": {"type": "mock"},
        "detectors": {"icons": {"type": "mock"}, "cards": {"type": "mock"}},
        "readers": {"gauges": {"type": "opencv_qt_cluster_static"}},
        "log_sources": {"backend": {"type": "file", "path": "logs/app.log"}},
        "ocr": {"type": "mock"},
        "capture": {"window_title": "x"},
        "setup": [{"action": "sleep"}],
        "steps": [{"action": "click", "x": 10, "y": 20}],
        "verification_setup": [{"action": "activate_window", "window_title": "x"}],
        "verification": [{"check": "text_exists", "locate": {"text": "ok"}}],
        "teardown": [{"action": "emulator_stop"}],
    }
    fake_yaml = types.SimpleNamespace(safe_load=lambda text: raw)
    monkeypatch.setattr(loader, "yaml", fake_yaml)
    case = loader.load_test_case(file_path)
    assert isinstance(case, CaseModel)
    assert case.name == "demo"
    assert case.emulator["type"] == "network_device"
    assert sorted(case.detectors) == ["cards", "icons"]
    assert sorted(case.readers) == ["gauges"]
    assert sorted(case.log_sources) == ["backend"]
    assert len(case.setup) == 1
    assert len(case.steps) == 1
    assert len(case.verification_setup) == 1
    assert len(case.verification) == 1
    assert len(case.teardown) == 1


def test_load_test_case_rejects_non_mapping_detectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    file_path = tmp_path / "case.yaml"
    file_path.write_text("anything", encoding="utf-8")
    raw = {"detectors": {"icons": "bad"}}
    fake_yaml = types.SimpleNamespace(safe_load=lambda text: raw)
    monkeypatch.setattr(loader, "yaml", fake_yaml)
    with pytest.raises(ValueError, match="Each 'detectors' entry must be an object/dict"):
        loader.load_test_case(file_path)


def test_load_test_case_rejects_reserved_default_detector_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    file_path = tmp_path / "case.yaml"
    file_path.write_text("anything", encoding="utf-8")
    raw = {"detectors": {"default": {"type": "mock"}}}
    fake_yaml = types.SimpleNamespace(safe_load=lambda text: raw)
    monkeypatch.setattr(loader, "yaml", fake_yaml)
    with pytest.raises(ValueError, match="reserved key 'default'"):
        loader.load_test_case(file_path)


def test_load_test_case_rejects_stage_item_missing_action_or_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    file_path = tmp_path / "case.yaml"
    file_path.write_text("anything", encoding="utf-8")
    fake_yaml = types.SimpleNamespace(safe_load=lambda text: {"steps": [{}]})
    monkeypatch.setattr(loader, "yaml", fake_yaml)

    with pytest.raises(ValueError, match="steps\\[1\\] missing 'action' or 'check'"):
        loader.load_test_case(file_path)


def test_load_test_case_rejects_invalid_known_action_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    file_path = tmp_path / "case.yaml"
    file_path.write_text("anything", encoding="utf-8")
    raw = {
        "steps": [
            {
                "action": "open_browser",
                "url": "https://example.com",
                "args": "--bad",
            }
        ]
    }
    fake_yaml = types.SimpleNamespace(safe_load=lambda text: raw)
    monkeypatch.setattr(loader, "yaml", fake_yaml)

    with pytest.raises(
        ValueError,
        match=r"steps\[1\] invalid parameters for action 'open_browser': field 'args' must be a list",
    ):
        loader.load_test_case(file_path)


def test_load_test_case_rejects_invalid_known_check_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    file_path = tmp_path / "case.yaml"
    file_path.write_text("anything", encoding="utf-8")
    raw = {
        "verification": [
            {
                "check": "log_contains",
                "source": "backend",
            }
        ]
    }
    fake_yaml = types.SimpleNamespace(safe_load=lambda text: raw)
    monkeypatch.setattr(loader, "yaml", fake_yaml)

    with pytest.raises(
        ValueError,
        match=r"verification\[1\] invalid parameters for check 'log_contains': one of 'contains', 'text', or 'regex' is required",
    ):
        loader.load_test_case(file_path)


def test_load_test_case_rejects_verification_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    file_path = tmp_path / "case.yaml"
    file_path.write_text("anything", encoding="utf-8")
    raw = {
        "verification": [
            {
                "action": "wait_for_text",
                "locate": {"text": "ready"},
            }
        ]
    }
    fake_yaml = types.SimpleNamespace(safe_load=lambda text: raw)
    monkeypatch.setattr(loader, "yaml", fake_yaml)

    with pytest.raises(
        ValueError,
        match=r"verification\[1\] does not support 'action'; use 'check'",
    ):
        loader.load_test_case(file_path)


def test_load_test_case_allows_unknown_plugin_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    file_path = tmp_path / "case.yaml"
    file_path.write_text("anything", encoding="utf-8")
    raw = {
        "steps": [{"action": "custom_plugin_action", "foo": "bar"}],
        "verification_setup": [{"action": "custom_plugin_prepare", "foo": "bar"}],
        "verification": [{"check": "custom_plugin_check", "answer": 42}],
    }
    fake_yaml = types.SimpleNamespace(safe_load=lambda text: raw)
    monkeypatch.setattr(loader, "yaml", fake_yaml)

    case = loader.load_test_case(file_path)

    assert case.steps == [{"action": "custom_plugin_action", "foo": "bar"}]
    assert case.verification_setup == [{"action": "custom_plugin_prepare", "foo": "bar"}]
    assert case.verification == [{"check": "custom_plugin_check", "answer": 42}]
