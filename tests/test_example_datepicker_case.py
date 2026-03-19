from pathlib import Path

from autoscene.yamlcase.loader import load_test_case


def test_datepicker_basic_datetime_example_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    case = load_test_case(root / "examples" / "datepicker_basic_datetime_case.yaml")
    placeholder_step = next(
        step
        for step in case.steps
        if step.get("action") == "click_text"
        and isinstance(step.get("locate"), dict)
        and step["locate"].get("text") == "Please select Date Time"
    )
    day_step = next(
        step
        for step in case.steps
        if step.get("action") == "click_text"
        and isinstance(step.get("locate"), dict)
        and step["locate"].get("text") == "20"
    )
    year_step = next(
        step
        for step in case.steps
        if step.get("action") == "click_text"
        and isinstance(step.get("locate"), dict)
        and step["locate"].get("text") == "2026"
    )

    assert case.name == "datepicker_basic_datetime"
    assert case.ocr["preprocess"]["enabled"] is True
    assert placeholder_step["locate"].get("ocr") is None
    assert year_step["locate"].get("exact") is True
    assert day_step["locate"].get("exact") is True
    assert day_step["locate"]["ocr"]["tesseract_config"] == "--psm 11"
    assert day_step["locate"]["ocr"]["preprocess"]["threshold"] == "none"
    assert "scale" not in day_step["locate"]["ocr"]["preprocess"]
    assert any(
        step.get("action") == "input_text" and step.get("text") == "2027"
        for step in case.steps
    )
    assert any(
        item.get("check") == "text_exists"
        and isinstance(item.get("locate"), dict)
        and item["locate"].get("text") == "15:30"
        for item in case.verification
    )
