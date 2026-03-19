from __future__ import annotations

from types import SimpleNamespace

import pytest

import autoscene.runner.runtime as runtime_mod
import autoscene.runner.runtime_events as event_mod
from autoscene.core.exceptions import ActionExecutionError, VerificationError


class RecordingLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []

    def info(self, message: str, *args) -> None:
        self.info_messages.append(message % args if args else message)

    def warning(self, message: str, *args) -> None:
        self.warning_messages.append(message % args if args else message)


def _make_runtime_context():
    logger = RecordingLogger()
    hook_bus = runtime_mod.HookBus()
    events: list[tuple[str, dict[str, object]]] = []
    for event_name in event_mod.CASE_LIFECYCLE_EVENTS:
        hook_bus.register(
            event_name,
            lambda context, session, payload, event_name=event_name: events.append(
                (event_name, dict(payload))
            ),
        )
    context = SimpleNamespace(
        metadata=SimpleNamespace(
            case=SimpleNamespace(name="demo-case"),
            profile=SimpleNamespace(name="demo-profile"),
            logger=logger,
        ),
        services=SimpleNamespace(hook_bus=hook_bus),
    )
    return context, logger, events


def _make_step_context(*, check_result: bool = True):
    logger = RecordingLogger()
    action_calls: list[object] = []
    check_calls: list[object] = []
    context = SimpleNamespace(
        metadata=SimpleNamespace(logger=logger),
        services=SimpleNamespace(
            action_registry=SimpleNamespace(
                dispatch_step=lambda step: action_calls.append(step)
            ),
            check_registry=SimpleNamespace(
                dispatch_step=lambda step: check_calls.append(step) or check_result
            ),
        ),
    )
    return context, logger, action_calls, check_calls


def test_scenario_runner_factory_uses_profile_pipeline_factory() -> None:
    pipeline = runtime_mod.ScenarioPipeline(stages=[])
    profile = runtime_mod.RuntimeProfile(pipeline_factory=lambda: pipeline)

    runner = runtime_mod.ScenarioRunnerFactory().create(profile=profile)

    assert runner.pipeline is pipeline
    assert isinstance(runner.session_lifecycle, runtime_mod.RunSessionLifecycle)


def test_default_stage_definitions_describe_runtime_pipeline() -> None:
    definitions = runtime_mod.default_stage_definitions()

    assert [definition.name for definition in definitions] == [
        "setup",
        "steps",
        "verification_setup",
        "verification",
        "teardown",
    ]
    assert [definition.plan_field for definition in definitions] == [
        "setup",
        "steps",
        "verification_setup",
        "verification",
        "teardown",
    ]
    assert isinstance(definitions[0].step_operation, runtime_mod.StageStepOperation)
    assert isinstance(definitions[2].step_operation, runtime_mod.StageStepOperation)
    assert isinstance(definitions[3].step_operation, runtime_mod.VerificationStepOperation)


def test_stage_definition_supports_custom_item_getter() -> None:
    plan = runtime_mod.ScenarioPlan(
        setup=[runtime_mod.ActionStep(name="prepare")],
        teardown=[runtime_mod.ActionStep(name="cleanup")],
    )
    definition = runtime_mod.StageDefinition(
        name="custom",
        step_operation=runtime_mod.StageStepOperation(allow_checks=False),
        items_getter=lambda current_plan: current_plan.teardown,
    )

    runner = runtime_mod.StageRunner(definition)

    assert runner.stage_name == "custom"
    assert runner.get_items(plan) == plan.teardown


def test_scenario_runner_factory_preserves_explicit_runner() -> None:
    explicit_runner = runtime_mod.ScenarioRunner()

    runner = runtime_mod.ScenarioRunnerFactory().create(
        profile=runtime_mod.RuntimeProfile(),
        runner=explicit_runner,
    )

    assert runner is explicit_runner


def test_scenario_runner_marks_passed_session_via_lifecycle() -> None:
    context, logger, events = _make_runtime_context()

    class SuccessPipeline:
        def run(self, *, plan, context, session) -> None:
            session.variables["pipeline"] = "ran"

    runner = runtime_mod.ScenarioRunner(pipeline=SuccessPipeline())

    session = runner.run(runtime_mod.ScenarioPlan(), context)

    assert session.status == "passed"
    assert session.variables == {"pipeline": "ran"}
    assert [event_name for event_name, _ in events] == [
        event_mod.CASE_STARTED,
        event_mod.CASE_FINISHED,
    ]
    assert logger.info_messages[0] == "Start case: demo-case"
    assert logger.info_messages[-1] == "Case passed: demo-case"


def test_scenario_runner_marks_failed_session_via_lifecycle() -> None:
    context, logger, events = _make_runtime_context()

    class FailingPipeline:
        def run(self, *, plan, context, session) -> None:
            raise RuntimeError("boom")

    runner = runtime_mod.ScenarioRunner(pipeline=FailingPipeline())

    with pytest.raises(RuntimeError, match="boom"):
        runner.run(runtime_mod.ScenarioPlan(), context)

    assert logger.warning_messages == []
    assert [event_name for event_name, _ in events] == [
        event_mod.CASE_STARTED,
        event_mod.CASE_FAILED,
        event_mod.CASE_FINISHED,
    ]
    assert events[-1][1]["status"] == "failed"


def test_scenario_pipeline_coerces_stage_definitions_into_stage_runners() -> None:
    pipeline = runtime_mod.ScenarioPipeline(stages=runtime_mod.default_stage_definitions())

    assert [stage.stage_name for stage in pipeline.stages] == [
        "setup",
        "steps",
        "verification_setup",
        "verification",
        "teardown",
    ]
    assert all(isinstance(stage, runtime_mod.StageRunner) for stage in pipeline.stages)


def test_stage_step_operation_dispatches_action_steps() -> None:
    context, logger, action_calls, _ = _make_step_context()
    step = runtime_mod.ActionStep(name="click", params={"x": 1, "y": 2})

    outcome = runtime_mod.StageStepOperation(allow_checks=True).execute(
        context=context,
        stage_name="steps",
        index=1,
        step=step,
    )

    assert outcome == ("click", "action")
    assert action_calls == [step]
    assert logger.info_messages == ["[steps] #1 action=click"]


def test_stage_step_operation_rejects_invalid_stage_items() -> None:
    context, _, _, _ = _make_step_context()
    step = runtime_mod.InvalidStep(raw={}, expected_fields="'action'")

    with pytest.raises(ActionExecutionError, match=r"setup\[1\] missing 'action'"):
        runtime_mod.StageStepOperation().execute(
            context=context,
            stage_name="setup",
            index=1,
            step=step,
        )


def test_verification_step_operation_raises_verification_error_on_false_result() -> None:
    context, logger, _, check_calls = _make_step_context(check_result=False)
    step = runtime_mod.CheckStep(name="ready", params={"locate": {"text": "ready"}})

    with pytest.raises(VerificationError, match=r"Verification failed at #2"):
        runtime_mod.VerificationStepOperation().execute(
            context=context,
            stage_name="verification",
            index=2,
            step=step,
        )

    assert check_calls == [step]
    assert logger.info_messages == [
        "[verification] #2 check=ready",
        "[verification] #2 result=failed",
    ]


def test_failure_policy_records_artifact_write_errors_consistently() -> None:
    logger = RecordingLogger()
    hook_bus = runtime_mod.HookBus()
    events: list[dict[str, object]] = []
    hook_bus.register(
        event_mod.STEP_FAILED,
        lambda context, session, payload: events.append(dict(payload)),
    )

    class BrokenArtifactStore:
        def record_step_failure(self, **kwargs):
            raise OSError("disk full")

    context = SimpleNamespace(
        metadata=SimpleNamespace(logger=logger),
        services=SimpleNamespace(hook_bus=hook_bus),
    )
    session = runtime_mod.RunSession(case_name="failure-case")
    step = runtime_mod.ActionStep(
        name="click",
        params={"x": 1, "y": 2},
        continue_on_failure=True,
    )
    error = ActionExecutionError("boom")

    result = runtime_mod.FailurePolicy(
        artifact_store=BrokenArtifactStore()
    ).handle_step_failure(
        stage_name="steps",
        index=1,
        step=step,
        context=context,
        session=session,
        payload=step.to_payload(),
        duration_ms=12,
        attempts=2,
        error=error,
        step_name="click",
        step_type="action",
    )

    assert result.details["error"] == "boom"
    assert result.details["artifact_error"] == "disk full"
    assert "artifact_path" not in result.details
    assert session.step_results == [result]
    assert session.failures == ["boom"]
    assert logger.warning_messages == [
        "[steps] #1 failed to persist failure artifact: disk full",
        "[steps] #1 continued after failure: boom",
    ]
    assert events == [
        {
            "stage_name": "steps",
            "index": 1,
            "result": result,
            "error": error,
            "artifact_path": None,
        }
    ]
