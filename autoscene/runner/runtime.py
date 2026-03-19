from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autoscene.core.models import TestCase
from autoscene.runner.runtime_assembly import (
    RuntimeActionBindings,
    RuntimeContextFactory,
    RuntimeFactoryResolver,
    RuntimeFactorySet,
    RuntimeResourceFactory,
    RuntimeServiceFactory,
    invoke_factory,
)
from autoscene.runner.runtime_compile import ScenarioPlanCompiler, StepCompiler
from autoscene.runner.runtime_events import (
    AFTER_STAGE,
    AFTER_STEP,
    ALL_RUNTIME_EVENTS,
    BEFORE_STAGE,
    BEFORE_STEP,
    BEFORE_STEP_ATTEMPT,
    CASE_FAILED,
    CASE_FINISHED,
    CASE_LIFECYCLE_EVENTS,
    CASE_STARTED,
    RETRY_EVENTS,
    STAGE_EVENTS,
    STAGE_FAILED,
    STEP_ATTEMPT_FAILED,
    STEP_EVENTS,
    STEP_FAILED,
)
from autoscene.runner.runtime_execution import (
    ActionStepRunner,
    CheckStepRunner,
    RunSessionLifecycle,
    ScenarioPipeline,
    ScenarioRunnerFactory,
    ScenarioRunner,
    StageDefinition,
    StageRunner,
    StageStepOperation,
    StepOperation,
    StepExecutor,
    default_stage_definitions,
    setup_stage_definition,
    steps_stage_definition,
    teardown_stage_definition,
    verification_setup_stage_definition,
    verification_stage_definition,
    VerificationStepOperation,
)
from autoscene.runner.runtime_models import (
    ActionStep,
    CheckStep,
    InvalidStep,
    RetryExecutionOutcome,
    RunSession,
    RuntimeContext,
    RuntimeMetadata,
    RuntimeProfile,
    RuntimeResources,
    RuntimeServices,
    ScenarioPlan,
    Step,
    StepResult,
)
from autoscene.runner.runtime_profile_resolver import (
    RuntimeProfileResolver,
    default_capture_factory,
)
from autoscene.runner.runtime_policies import (
    ArtifactStore,
    FailurePolicy,
    HookBus,
    RunnerRetryPolicy,
)


class RuntimeBuilder:
    def __init__(
        self,
        logger_name: str = "TestExecutor",
        *,
        plan_compiler: ScenarioPlanCompiler | None = None,
        context_factory: RuntimeContextFactory | None = None,
    ) -> None:
        self.plan_compiler = plan_compiler or ScenarioPlanCompiler()
        self.context_factory = context_factory or RuntimeContextFactory(logger_name=logger_name)

    def compile(
        self,
        case: TestCase,
        profile: RuntimeProfile | None = None,
    ) -> ScenarioPlan:
        return self.plan_compiler.compile(case, profile=profile)

    def compile_stage_items(
        self,
        items: Sequence[Step | dict[str, Any]],
        *,
        stage_name: str,
        allow_checks: bool,
        profile: RuntimeProfile | None = None,
    ) -> list[Step]:
        del stage_name
        return self.plan_compiler.compile_stage_items(
            items,
            allow_checks=allow_checks,
            profile=profile,
        )

    def compile_verification_items(
        self,
        items: Sequence[Step | dict[str, Any]],
        profile: RuntimeProfile | None = None,
    ) -> list[Step]:
        return self.plan_compiler.compile_verification_items(items, profile=profile)

    def build(
        self,
        profile: RuntimeProfile,
        case: TestCase,
        output_dir: str | Path = "outputs",
    ) -> RuntimeContext:
        return self.context_factory.build(
            profile=profile,
            case=case,
            output_dir=output_dir,
        )


__all__ = [
    "ActionStep",
    "ActionStepRunner",
    "AFTER_STAGE",
    "AFTER_STEP",
    "ALL_RUNTIME_EVENTS",
    "ArtifactStore",
    "BEFORE_STAGE",
    "BEFORE_STEP",
    "BEFORE_STEP_ATTEMPT",
    "CASE_FAILED",
    "CASE_FINISHED",
    "CASE_LIFECYCLE_EVENTS",
    "CASE_STARTED",
    "CheckStep",
    "CheckStepRunner",
    "FailurePolicy",
    "HookBus",
    "InvalidStep",
    "RETRY_EVENTS",
    "RetryExecutionOutcome",
    "RunSession",
    "RunSessionLifecycle",
    "RuntimeBuilder",
    "RuntimeActionBindings",
    "RuntimeContext",
    "RuntimeContextFactory",
    "RuntimeFactoryResolver",
    "RuntimeFactorySet",
    "RuntimeMetadata",
    "RuntimeProfile",
    "RuntimeProfileResolver",
    "RuntimeResourceFactory",
    "RuntimeResources",
    "RuntimeServiceFactory",
    "RuntimeServices",
    "RunnerRetryPolicy",
    "ScenarioPlan",
    "ScenarioPlanCompiler",
    "ScenarioPipeline",
    "ScenarioRunnerFactory",
    "ScenarioRunner",
    "StageDefinition",
    "StageRunner",
    "STAGE_EVENTS",
    "STAGE_FAILED",
    "StageStepOperation",
    "Step",
    "STEP_ATTEMPT_FAILED",
    "STEP_EVENTS",
    "STEP_FAILED",
    "StepOperation",
    "StepCompiler",
    "StepExecutor",
    "StepResult",
    "default_stage_definitions",
    "VerificationStepOperation",
    "default_capture_factory",
    "invoke_factory",
    "setup_stage_definition",
    "steps_stage_definition",
    "teardown_stage_definition",
    "verification_setup_stage_definition",
    "verification_stage_definition",
]
