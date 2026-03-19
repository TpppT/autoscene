from pathlib import Path

import autoscene.runner.executor as executor_mod
from autoscene.runner.runtime import RuntimeProfileResolver
from autoscene.yamlcase.loader import load_test_case


def test_mock_case_runs_end_to_end(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    case = load_test_case(root / "examples" / "mock_case.yaml")

    executor = executor_mod.TestExecutor(
        case,
        profile=RuntimeProfileResolver().resolve(),
        output_dir=tmp_path,
    )
    session = executor.run()

    assert session.status == "passed"
    assert (tmp_path / "mock_frame.png").exists()
