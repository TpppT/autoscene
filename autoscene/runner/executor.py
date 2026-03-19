from __future__ import annotations

from pathlib import Path

from autoscene.actions.services import ActionServices
from autoscene.capture.video_stream_capture import create_video_stream_capture
from autoscene.capture.window_capture import WindowCapture
from autoscene.core.models import TestCase
from autoscene.emulator.registry import create_emulator
from autoscene.logs.registry import create_log_source
from autoscene.runner.action_dispatcher import ActionDispatcher
from autoscene.runner.check_dispatcher import CheckDispatcher
from autoscene.runner.runtime import (
    RuntimeBuilder,
    RuntimeContext,
    RuntimeProfile,
    RuntimeProfileResolver,
    RunSession,
    ScenarioPlan,
    ScenarioRunnerFactory,
    ScenarioRunner,
)
from autoscene.runner.runtime_policies import HookBus
from autoscene.vision import (
    create_detector,
    create_ocr_engine,
    create_reader_adapter,
)


class TestExecutor:
    def __init__(
        self,
        case: TestCase,
        profile: RuntimeProfile,
        output_dir: str | Path = "outputs",
        builder: RuntimeBuilder | None = None,
        runner_factory: ScenarioRunnerFactory | None = None,
        runner: ScenarioRunner | None = None,
    ) -> None:
        self.case = case
        self.profile = profile
        self.builder = builder or RuntimeBuilder(logger_name=self.__class__.__name__)
        self.runner_factory = runner_factory or ScenarioRunnerFactory()
        self.plan: ScenarioPlan = self.builder.compile(case, profile=self.profile)
        self.context: RuntimeContext = self.builder.build(
            profile=self.profile,
            case=case,
            output_dir=output_dir,
        )
        self.runner = self.runner_factory.create(
            profile=self.profile,
            runner=runner,
        )
        self.last_session: RunSession | None = None

    def run(self) -> RunSession:
        self.last_session = self.runner.run(self.plan, self.context)
        return self.last_session
