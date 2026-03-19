from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

from autoscene.core.exceptions import ActionExecutionError, VerificationError
from autoscene.runner.runtime_events import (
    AFTER_STAGE,
    AFTER_STEP,
    BEFORE_STAGE,
    BEFORE_STEP,
    CASE_FAILED,
    CASE_FINISHED,
    CASE_STARTED,
    STAGE_FAILED,
)
from autoscene.runner.runtime_models import (
    ActionStep,
    CheckStep,
    InvalidStep,
    RetryExecutionOutcome,
    RunSession,
    RuntimeContext,
    RuntimeProfile,
    ScenarioPlan,
    Step,
    StepResult,
)


def _step_reference(stage_name: str, index: int) -> str:
    return f"{stage_name}[{index}]"


def _raise_step_configuration_error(
    *,
    error_type: type[Exception],
    stage_name: str,
    index: int,
    step: Step,
    missing_expected_fields: str,
    invalid_expected_fields: str | None = None,
    use_step_expected_fields_for_missing: bool = True,
) -> None:
    step_reference = _step_reference(stage_name, index)
    if isinstance(step, InvalidStep):
        if step.validation_error:
            expected_fields = invalid_expected_fields or step.expected_fields
            raise error_type(
                f"{step_reference} invalid {expected_fields}: "
                f"{step.validation_error} payload={step.to_raw_payload()!r}"
            )
        if use_step_expected_fields_for_missing:
            missing_expected_fields = step.expected_fields

    raise error_type(
        f"{step_reference} missing {missing_expected_fields}: {step.to_payload()!r}"
    )


class ActionStepRunner:
    def execute(
        self,
        *,
        context: RuntimeContext,
        stage_name: str,
        index: int,
        step: ActionStep,
    ) -> tuple[str, str]:
        context.metadata.logger.info("[%s] #%s action=%s", stage_name, index, step.name)
        context.services.action_registry.dispatch_step(step)
        return (step.name, "action")


class CheckStepRunner:
    def execute(
        self,
        *,
        context: RuntimeContext,
        stage_name: str,
        index: int,
        step: CheckStep,
        failure_message: str,
    ) -> tuple[str, str]:
        context.metadata.logger.info("[%s] #%s check=%s", stage_name, index, step.name)
        ok = bool(context.services.check_registry.dispatch_step(step))
        context.metadata.logger.info(
            "[%s] #%s result=%s",
            stage_name,
            index,
            "passed" if ok else "failed",
        )
        if not ok:
            raise VerificationError(failure_message)
        return (step.name, "check")


class StepOperation:
    def execute(
        self,
        *,
        context: RuntimeContext,
        stage_name: str,
        index: int,
        step: Step,
    ) -> tuple[str, str]:
        raise NotImplementedError


class StageStepOperation(StepOperation):
    def __init__(
        self,
        *,
        allow_checks: bool = False,
        action_runner: ActionStepRunner | None = None,
        check_runner: CheckStepRunner | None = None,
    ) -> None:
        self.allow_checks = allow_checks
        self.action_runner = action_runner or ActionStepRunner()
        self.check_runner = check_runner or CheckStepRunner()

    def execute(
        self,
        *,
        context: RuntimeContext,
        stage_name: str,
        index: int,
        step: Step,
    ) -> tuple[str, str]:
        if isinstance(step, ActionStep):
            return self.action_runner.execute(
                context=context,
                stage_name=stage_name,
                index=index,
                step=step,
            )

        if self.allow_checks and isinstance(step, CheckStep):
            payload = step.to_payload()
            return self.check_runner.execute(
                context=context,
                stage_name=stage_name,
                index=index,
                step=step,
                failure_message=f"Verification failed at {stage_name} #{index}: {payload!r}",
            )

        self._raise_invalid_step(
            stage_name=stage_name,
            index=index,
            step=step,
            allow_checks=self.allow_checks,
        )

    @staticmethod
    def _raise_invalid_step(
        *,
        stage_name: str,
        index: int,
        step: Step,
        allow_checks: bool,
    ) -> None:
        expected_fields = "'action' or 'check'" if allow_checks else "'action'"
        _raise_step_configuration_error(
            error_type=ActionExecutionError,
            stage_name=stage_name,
            index=index,
            step=step,
            missing_expected_fields=expected_fields,
        )


class VerificationStepOperation(StepOperation):
    def __init__(self, check_runner: CheckStepRunner | None = None) -> None:
        self.check_runner = check_runner or CheckStepRunner()

    def execute(
        self,
        *,
        context: RuntimeContext,
        stage_name: str,
        index: int,
        step: Step,
    ) -> tuple[str, str]:
        if not isinstance(step, CheckStep):
            self._raise_invalid_step(
                stage_name=stage_name,
                index=index,
                step=step,
            )
        payload = step.to_payload()
        return self.check_runner.execute(
            context=context,
            stage_name=stage_name,
            index=index,
            step=step,
            failure_message=f"Verification failed at #{index}: {payload!r}",
        )

    @staticmethod
    def _raise_invalid_step(
        *,
        stage_name: str,
        index: int,
        step: Step,
    ) -> None:
        _raise_step_configuration_error(
            error_type=VerificationError,
            stage_name=stage_name,
            index=index,
            step=step,
            missing_expected_fields="'check'",
            invalid_expected_fields="'check'",
            use_step_expected_fields_for_missing=False,
        )


@dataclass(frozen=True)
class StageDefinition:
    name: str
    step_operation: StepOperation
    plan_field: str | None = None
    items_getter: Callable[[ScenarioPlan], Sequence[Step]] | None = None

    def get_items(self, plan: ScenarioPlan) -> list[Step]:
        if self.items_getter is not None:
            return list(self.items_getter(plan))
        if self.plan_field is None:
            return []
        return list(getattr(plan, self.plan_field))


def setup_stage_definition() -> StageDefinition:
    return StageDefinition(
        name="setup",
        plan_field="setup",
        step_operation=StageStepOperation(allow_checks=True),
    )


def steps_stage_definition() -> StageDefinition:
    return StageDefinition(
        name="steps",
        plan_field="steps",
        step_operation=StageStepOperation(allow_checks=True),
    )


def verification_setup_stage_definition() -> StageDefinition:
    return StageDefinition(
        name="verification_setup",
        plan_field="verification_setup",
        step_operation=StageStepOperation(allow_checks=True),
    )


def verification_stage_definition() -> StageDefinition:
    return StageDefinition(
        name="verification",
        plan_field="verification",
        step_operation=VerificationStepOperation(),
    )


def teardown_stage_definition() -> StageDefinition:
    return StageDefinition(
        name="teardown",
        plan_field="teardown",
        step_operation=StageStepOperation(allow_checks=True),
    )


def default_stage_definitions() -> list[StageDefinition]:
    return [
        setup_stage_definition(),
        steps_stage_definition(),
        verification_setup_stage_definition(),
        verification_stage_definition(),
        teardown_stage_definition(),
    ]


class StepExecutor:
    def execute(
        self,
        *,
        stage_name: str,
        index: int,
        step: Step,
        step_operation: StepOperation,
        context: RuntimeContext,
        session: RunSession,
    ) -> StepResult:
        started = time.perf_counter()
        payload = step.to_payload()
        context.services.hook_bus.emit(
            BEFORE_STEP,
            context,
            session,
            self._before_step_payload(
                stage_name=stage_name,
                index=index,
                step=step,
                payload=payload,
            ),
        )

        outcome = context.services.retry_policy.execute(
            stage_name=stage_name,
            index=index,
            step=step,
            context=context,
            session=session,
            operation=lambda: step_operation.execute(
                context=context,
                stage_name=stage_name,
                index=index,
                step=step,
            ),
        )

        if outcome.passed:
            result = self._build_passed_result(
                stage_name=stage_name,
                index=index,
                step=step,
                payload=payload,
                outcome=outcome,
                started=started,
            )
            session.step_results.append(result)
            context.services.hook_bus.emit(
                AFTER_STEP,
                context,
                session,
                self._after_step_payload(
                    stage_name=stage_name,
                    index=index,
                    outcome=outcome,
                    result=result,
                ),
            )
            return result

        assert outcome.error is not None
        return context.services.failure_policy.handle_step_failure(
            stage_name=stage_name,
            index=index,
            step=step,
            context=context,
            session=session,
            payload=payload,
            duration_ms=self._elapsed_ms(started),
            attempts=outcome.attempts,
            error=outcome.error,
            step_name=outcome.step_name,
            step_type=outcome.step_type,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int(round((time.perf_counter() - started) * 1000.0))

    @staticmethod
    def _before_step_payload(
        *,
        stage_name: str,
        index: int,
        step: Step,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return {
            "stage_name": stage_name,
            "index": index,
            "step_name": step.name,
            "item": payload,
            "raw_item": step.to_raw_payload(),
            "step": step,
            "retry_count": step.retry_count,
        }

    @staticmethod
    def _after_step_payload(
        *,
        stage_name: str,
        index: int,
        outcome: RetryExecutionOutcome,
        result: StepResult,
    ) -> dict[str, Any]:
        return {
            "stage_name": stage_name,
            "index": index,
            "attempt": outcome.attempts,
            "result": result,
        }

    def _build_passed_result(
        self,
        *,
        stage_name: str,
        index: int,
        step: Step,
        payload: dict[str, Any],
        outcome: RetryExecutionOutcome,
        started: float,
    ) -> StepResult:
        return StepResult(
            stage_name=stage_name,
            step_index=index,
            step_name=outcome.step_name,
            step_type=outcome.step_type,
            passed=True,
            duration_ms=self._elapsed_ms(started),
            attempts=outcome.attempts,
            tags=step.tags,
            payload=payload,
        )


class StageRunner:
    def __init__(self, definition: StageDefinition) -> None:
        self.definition = definition

    @property
    def stage_name(self) -> str:
        return self.definition.name

    @property
    def step_operation(self) -> StepOperation:
        return self.definition.step_operation

    def get_items(self, plan: ScenarioPlan) -> list[Step]:
        return self.definition.get_items(plan)

    def run(
        self,
        *,
        plan: ScenarioPlan,
        context: RuntimeContext,
        session: RunSession,
        step_executor: StepExecutor,
    ) -> None:
        items = self.get_items(plan)
        context.services.hook_bus.emit(
            BEFORE_STAGE,
            context,
            session,
            self._stage_event_payload(items),
        )
        try:
            for index, item in enumerate(items, start=1):
                step_executor.execute(
                    stage_name=self.stage_name,
                    index=index,
                    step=item,
                    step_operation=self.step_operation,
                    context=context,
                    session=session,
                )
        except Exception as exc:
            context.services.hook_bus.emit(
                STAGE_FAILED,
                context,
                session,
                {
                    "stage_name": self.stage_name,
                    "error": exc,
                },
            )
            raise
        context.services.hook_bus.emit(
            AFTER_STAGE,
            context,
            session,
            self._stage_event_payload(items),
        )

    def _stage_event_payload(self, items: Sequence[Step]) -> dict[str, object]:
        return {
            "stage_name": self.stage_name,
            "size": len(items),
        }


class ScenarioPipeline:
    def __init__(
        self,
        stages: Sequence[StageRunner | StageDefinition] | None = None,
        step_executor: StepExecutor | None = None,
    ) -> None:
        stage_entries = list(stages) if stages is not None else default_stage_definitions()
        self.stages = [self._coerce_stage(stage) for stage in stage_entries]
        self.step_executor = step_executor or StepExecutor()

    @staticmethod
    def _coerce_stage(stage: StageRunner | StageDefinition) -> StageRunner:
        if isinstance(stage, StageRunner):
            return stage
        return StageRunner(stage)

    def run(
        self,
        *,
        plan: ScenarioPlan,
        context: RuntimeContext,
        session: RunSession,
    ) -> None:
        for stage in self.stages:
            stage.run(
                plan=plan,
                context=context,
                session=session,
                step_executor=self.step_executor,
            )


class RunSessionLifecycle:
    def start(self, context: RuntimeContext) -> RunSession:
        session = RunSession(case_name=context.metadata.case.name, status="running")
        context.metadata.logger.info("Start case: %s", context.metadata.case.name)
        context.services.hook_bus.emit(
            CASE_STARTED,
            context,
            session,
            {
                "case_name": context.metadata.case.name,
                "profile": context.metadata.profile.name,
            },
        )
        return session

    def fail(self, *, context: RuntimeContext, session: RunSession, error: Exception) -> None:
        session.status = "failed"
        context.services.hook_bus.emit(
            CASE_FAILED,
            context,
            session,
            {
                "error": error,
            },
        )
        context.services.hook_bus.emit(
            CASE_FINISHED,
            context,
            session,
            {"status": session.status},
        )

    def finish(self, *, context: RuntimeContext, session: RunSession) -> None:
        session.status = "passed_with_failures" if session.failures else "passed"
        if session.failures:
            context.metadata.logger.warning(
                "Case completed with failures: %s failures=%s",
                context.metadata.case.name,
                len(session.failures),
            )
        else:
            context.metadata.logger.info("Case passed: %s", context.metadata.case.name)
        context.services.hook_bus.emit(
            CASE_FINISHED,
            context,
            session,
            {"status": session.status},
        )


class ScenarioRunnerFactory:
    def __init__(self, session_lifecycle: RunSessionLifecycle | None = None) -> None:
        self.session_lifecycle = session_lifecycle or RunSessionLifecycle()

    def create(
        self,
        *,
        profile: RuntimeProfile,
        runner: ScenarioRunner | None = None,
    ) -> ScenarioRunner:
        if runner is not None:
            return runner
        pipeline = profile.pipeline_factory() if profile.pipeline_factory else None
        return ScenarioRunner(
            pipeline=pipeline,
            session_lifecycle=self.session_lifecycle,
        )


class ScenarioRunner:
    def __init__(
        self,
        pipeline: ScenarioPipeline | None = None,
        session_lifecycle: RunSessionLifecycle | None = None,
    ) -> None:
        self.pipeline = pipeline or ScenarioPipeline()
        self.session_lifecycle = session_lifecycle or RunSessionLifecycle()

    def run(self, plan: ScenarioPlan, context: RuntimeContext) -> RunSession:
        session = self.session_lifecycle.start(context)
        try:
            self.pipeline.run(plan=plan, context=context, session=session)
        except Exception as exc:
            self.session_lifecycle.fail(
                context=context,
                session=session,
                error=exc,
            )
            raise
        self.session_lifecycle.finish(
            context=context,
            session=session,
        )
        return session


__all__ = [
    "ActionStepRunner",
    "CheckStepRunner",
    "RunSessionLifecycle",
    "ScenarioPipeline",
    "ScenarioRunnerFactory",
    "ScenarioRunner",
    "StageDefinition",
    "StageRunner",
    "StageStepOperation",
    "StepExecutor",
    "StepOperation",
    "default_stage_definitions",
    "setup_stage_definition",
    "steps_stage_definition",
    "teardown_stage_definition",
    "verification_setup_stage_definition",
    "verification_stage_definition",
    "VerificationStepOperation",
]
