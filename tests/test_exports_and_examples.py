from pathlib import Path

import autoscene
import autoscene.actions as actions_pkg
import autoscene.capture as capture_pkg
import autoscene.core as core_pkg
import autoscene.emulator as emulator_pkg
import autoscene.runner as runner_pkg
import autoscene.vision as vision_pkg
import autoscene.yamlcase as yamlcase_pkg


def test_package_exports() -> None:
    assert autoscene.__doc__
    assert "BaseActions" in actions_pkg.__all__
    assert "ActionServices" in actions_pkg.__all__
    assert "WindowCapture" in capture_pkg.__all__
    assert core_pkg.__doc__ is not None
    assert "create_emulator" in emulator_pkg.__all__
    assert "register_emulator" in emulator_pkg.__all__
    assert "create_detector" in vision_pkg.__all__
    assert "create_operator" in vision_pkg.__all__
    assert "create_ocr_engine" in vision_pkg.__all__
    assert "TestExecutor" in runner_pkg.__all__
    assert "load_test_case" in yamlcase_pkg.__all__


def test_example_yaml_files_exist_and_contain_sections() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = root / "examples" / "sample_case.yaml"
    mock_case = root / "examples" / "mock_case.yaml"
    network_case = root / "examples" / "network_device_case.yaml"

    for file_path in [sample, mock_case, network_case]:
        assert file_path.exists(), f"{file_path} should exist"
        text = file_path.read_text(encoding="utf-8")
        assert "name:" in text
        assert "setup:" in text
        assert "steps:" in text
        assert "verification:" in text


def test_readme_mentions_network_device_flow() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text(encoding="utf-8")
    assert "network_device_case.yaml" in text
    assert "emulator_send" in text


def test_requirements_split_yolo_out_of_core_install() -> None:
    root = Path(__file__).resolve().parents[1]
    core_requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    yolo_requirements = (root / "requirements-yolo.txt").read_text(encoding="utf-8")
    assert "ultralytics" not in core_requirements
    assert "ultralytics" in yolo_requirements


def test_readme_mentions_optional_yolo_install_and_license_notice() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text(encoding="utf-8")
    assert "requirements-yolo.txt" in text
    assert "THIRD_PARTY_NOTICES.md" in text
