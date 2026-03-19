from __future__ import annotations

import argparse
import ctypes
import logging
import sys
from pathlib import Path

from autoscene.runner.executor import TestExecutor
from autoscene.runner.runtime import RuntimeProfileResolver
from autoscene.yamlcase.loader import load_test_case

_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
_PROCESS_PER_MONITOR_DPI_AWARE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YAML-based UI automation test case.")
    parser.add_argument("case", help="Path to YAML test case file.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save runtime artifacts (screenshots/logs).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def configure_logging(output_dir: str, case_path: str, log_level: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / f"{Path(case_path).stem}.log"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return log_file


def configure_windows_dpi_awareness() -> None:
    if not hasattr(ctypes, "windll"):
        return
    user32 = getattr(ctypes.windll, "user32", None)
    shcore = getattr(ctypes.windll, "shcore", None)

    set_context = getattr(user32, "SetProcessDpiAwarenessContext", None)
    if callable(set_context):
        try:
            if set_context(ctypes.c_void_p(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)):
                return
        except Exception:
            pass

    set_awareness = getattr(shcore, "SetProcessDpiAwareness", None)
    if callable(set_awareness):
        try:
            if int(set_awareness(_PROCESS_PER_MONITOR_DPI_AWARE)) == 0:
                return
        except Exception:
            pass

    set_legacy = getattr(user32, "SetProcessDPIAware", None)
    if callable(set_legacy):
        try:
            set_legacy()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    configure_windows_dpi_awareness()
    log_file = configure_logging(
        output_dir=args.output_dir,
        case_path=args.case,
        log_level=args.log_level,
    )
    logging.getLogger(__name__).info("Runtime log file: %s", log_file)
    case = load_test_case(args.case)
    profile = RuntimeProfileResolver().resolve()
    runner = TestExecutor(case, profile=profile, output_dir=args.output_dir)
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
