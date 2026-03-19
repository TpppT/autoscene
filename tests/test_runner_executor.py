import json
from pathlib import Path

import pytest

import autoscene.runner.executor as exec_mod
import autoscene.runner.runtime_events as event_mod
import autoscene.runner.runtime as runtime_mod
import autoscene.runner.step_specs as step_specs_mod
from autoscene.actions.advanced.locate_actions import LocateActions
from autoscene.core.exceptions import ActionExecutionError, VerificationError
from autoscene.core.models import (
    ObjectLocateSpec,
    TestCase as CaseModel,
    TextLocateSpec,
)


class FakeEmulator:
    def __init__(self):
        self.calls = []

    def launch(self):
        self.calls.append(("launch",))

    def stop(self):
        self.calls.append(("stop",))

    def execute(self, command):
        self.calls.append(("execute", command))
        return f"exec:{command}"

    def send(self, payload, endpoint=None, method=None, headers=None):
        self.calls.append(("send", payload, endpoint, method, headers))
        return "sent"


class FakeLogSource:
    def __init__(self, text=""):
        self.text = text
        self.calls = []

    def read_text(self):
        self.calls.append(("read_text",))
        return self.text


class FakeActions:
    def __init__(self, capture, detector, ocr, detectors=None):
        self.calls = []
        self.detectors = detectors or {}
        self.text_exists_result = True
        self.object_exists_result = True
        self.wait_result = True
        self.base_actions = self
        self.locate_actions = self
        self.screenshot_actions = self

    def click(self, x, y):
        self.calls.append(("click", x, y))

    def drag(self, sx, sy, ex, ey, duration_ms):
        self.calls.append(("drag", sx, sy, ex, ey, duration_ms))

    def input_text(self, text):
        self.calls.append(("input_text", text))

    def press_key(self, key, presses=1, interval_seconds=0.0):
        self.calls.append(("press_key", key, presses, interval_seconds))

    def open_browser(self, url, browser="chrome", browser_path=None, new_window=True, args=None):
        self.calls.append(("open_browser", url, browser, browser_path, new_window, args))

    def maximize_window(self, window_title, timeout=5.0, interval=0.2):
        self.calls.append(("maximize_window", window_title, timeout, interval))

    def activate_window(self, window_title, timeout=5.0, interval=0.2, settle_seconds=0.2):
        self.calls.append(("activate_window", window_title, timeout, interval, settle_seconds))

    def sleep(self, seconds):
        self.calls.append(("sleep", seconds))

    def screenshot(self, path=None):
        self.calls.append(("screenshot", path))
        return "fake_frame"

    def click_text(
        self,
        locate,
        debug_path=None,
        debug_crop_path=None,
    ):
        self.calls.append(("click_text", locate, debug_path, debug_crop_path))

    def click_object(
        self,
        locate,
        debug_path=None,
    ):
        self.calls.append(("click_object", locate, debug_path))

    def click_relative_to_text(
        self,
        locate,
        offset_x=0,
        offset_y=0,
        anchor="center",
    ):
        self.calls.append(("click_relative_to_text", locate, offset_x, offset_y, anchor))

    def drag_object_to_position(
        self,
        locate,
        target_x,
        target_y,
        duration_ms=500,
        debug_path=None,
    ):
        self.calls.append(
            (
                "drag_object_to_position",
                locate,
                target_x,
                target_y,
                duration_ms,
                debug_path,
            )
        )

    def drag_object_to_object(
        self,
        source,
        target,
        duration_ms=500,
    ):
        self.calls.append(("drag_object_to_object", source, target, duration_ms))

    def verify_text_exists(self, locate):
        self.calls.append(("verify_text_exists", locate))
        return self.text_exists_result

    def verify_object_exists(self, locate):
        self.calls.append(("verify_object_exists", locate))
        return self.object_exists_result

    def wait_for_text(self, locate, timeout=10.0, interval=0.5):
        self.calls.append(("wait_for_text", locate, timeout, interval))
        return self.wait_result


class SplitBaseActions:
    def __init__(self):
        self.calls = []

    def click(self, x, y):
        self.calls.append(("click", x, y))

    def drag(self, sx, sy, ex, ey, duration_ms=300):
        self.calls.append(("drag", sx, sy, ex, ey, duration_ms))

    def input_text(self, text):
        self.calls.append(("input_text", text))

    def press_key(self, key, presses=1, interval_seconds=0.0):
        self.calls.append(("press_key", key, presses, interval_seconds))

    def open_browser(self, url, browser="chrome", browser_path=None, new_window=True, args=None):
        self.calls.append(("open_browser", url, browser, browser_path, new_window, args))

    def maximize_window(self, window_title, timeout=5.0, interval=0.2):
        self.calls.append(("maximize_window", window_title, timeout, interval))

    def activate_window(self, window_title, timeout=5.0, interval=0.2, settle_seconds=0.2):
        self.calls.append(("activate_window", window_title, timeout, interval, settle_seconds))
        return True

    def sleep(self, seconds):
        self.calls.append(("sleep", seconds))

    def screenshot(self, path=None):
        self.calls.append(("screenshot", path))
        return "split-frame"


class SplitTextActions:
    def __init__(self):
        self.calls = []

    def click_text(
        self,
        locate,
        debug_path=None,
        debug_crop_path=None,
    ):
        self.calls.append(("click_text", locate, debug_path, debug_crop_path))

    def click_relative_to_text(
        self,
        locate,
        offset_x=0,
        offset_y=0,
        anchor="center",
    ):
        self.calls.append(("click_relative_to_text", locate, offset_x, offset_y, anchor))

    def wait_for_text(self, locate, timeout=10.0, interval=0.5):
        self.calls.append(("wait_for_text", locate, timeout, interval))
        return True

    def verify_text_exists(self, locate):
        self.calls.append(("verify_text_exists", locate))
        return True


class SplitObjectActions:
    def __init__(self):
        self.calls = []

    def click_object(
        self,
        locate,
        debug_path=None,
    ):
        self.calls.append(("click_object", locate, debug_path))

    def drag_object_to_position(
        self,
        locate,
        target_x,
        target_y,
        duration_ms=500,
        debug_path=None,
    ):
        self.calls.append(
            (
                "drag_object_to_position",
                locate,
                target_x,
                target_y,
                duration_ms,
                debug_path,
            )
        )

    def drag_object_to_object(
        self,
        source,
        target,
        duration_ms=500,
    ):
        self.calls.append(("drag_object_to_object", source, target, duration_ms))

    def verify_object_exists(self, locate):
        self.calls.append(("verify_object_exists", locate))
        return True


class SplitActions:
    def __init__(self, capture, detector, ocr, detectors=None):
        self.capture = capture
        self.detector = detector
        self.ocr = ocr
        self.detectors = detectors or {}
        self.base_actions = SplitBaseActions()
        self.screenshot_actions = self.base_actions
        self.text_actions = SplitTextActions()
        self.object_actions = SplitObjectActions()
        self.locate_actions = LocateActions(
            text_actions=self.text_actions,
            object_actions=self.object_actions,
        )


def _make_profile_resolver() -> exec_mod.RuntimeProfileResolver:
    def capture_factory(capture_config: dict[str, object]):
        capture_type = str(capture_config.get("type", "window")).strip().lower()
        if capture_type == "video_stream":
            return exec_mod.create_video_stream_capture(capture_config)
        return exec_mod.WindowCapture(
            default_window_title=capture_config.get("window_title"),
            default_region=capture_config.get("region"),
        )

    return exec_mod.RuntimeProfileResolver(
        emulator_factory=exec_mod.create_emulator,
        detector_factory=exec_mod.create_detector,
        reader_factory=exec_mod.create_reader_adapter,
        log_source_factory=exec_mod.create_log_source,
        ocr_engine_factory=exec_mod.create_ocr_engine,
        capture_factory=capture_factory,
        actions_factory=exec_mod.ActionServices,
        action_dispatcher_factory=exec_mod.ActionDispatcher,
        check_dispatcher_factory=exec_mod.CheckDispatcher,
        hook_bus_factory=exec_mod.HookBus,
    )


def _resolve_profile(profile: exec_mod.RuntimeProfile | None = None) -> exec_mod.RuntimeProfile:
    return _make_profile_resolver().resolve(profile)


def _make_executor(
    monkeypatch: pytest.MonkeyPatch,
    case: CaseModel,
    profile: exec_mod.RuntimeProfile | None = None,
    output_dir: Path | None = None,
):
    fake_emulator = FakeEmulator()
    fake_capture = object()
    fake_detector = object()
    fake_ocr = object()
    detector_calls = []
    log_source_calls = []
    monkeypatch.setattr(exec_mod, "create_emulator", lambda cfg: fake_emulator)
    monkeypatch.setattr(
        exec_mod,
        "create_detector",
        lambda cfg: detector_calls.append(cfg) or {"config": cfg},
    )
    monkeypatch.setattr(
        exec_mod,
        "create_log_source",
        lambda cfg: log_source_calls.append(cfg) or FakeLogSource(text="order created"),
    )
    monkeypatch.setattr(exec_mod, "create_ocr_engine", lambda cfg: fake_ocr)
    monkeypatch.setattr(exec_mod, "WindowCapture", lambda **kwargs: fake_capture)
    monkeypatch.setattr(exec_mod, "ActionServices", FakeActions)
    resolved_profile = _resolve_profile(profile)
    executor = exec_mod.TestExecutor(
        case=case,
        profile=resolved_profile,
        output_dir=output_dir or Path("outputs_test"),
    )
    resources = executor.context.resources
    action_entry = resources.base_actions or resources.locate_actions
    if action_entry is None:
        action_entry = resources.screenshot_actions
    return (
        executor,
        fake_emulator,
        action_entry,
        detector_calls,
        log_source_calls,
    )


def test_run_success_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CaseModel(
        name="ok",
        emulator={"type": "none"},
        detector={"type": "mock"},
        detectors={"icons": {"type": "mock"}, "cards": {"type": "mock"}},
        log_sources={"backend": {"type": "file", "path": "logs/app.log"}},
        setup=[{"action": "emulator_launch"}, {"action": "sleep", "seconds": 0.1}],
        steps=[
            {"action": "open_browser", "url": "https://example.com", "wait_seconds": 0.1},
            {"action": "activate_window", "window_title": "Google Chrome"},
            {"action": "maximize_window", "window_title": "Google Chrome"},
            {"action": "click", "x": 1, "y": 2},
            {"action": "drag", "start_x": 1, "start_y": 2, "end_x": 3, "end_y": 4},
            {"action": "input_text", "text": "abc"},
            {"action": "press_key", "key": "tab", "presses": 2, "interval_seconds": 0.1},
            {"action": "screenshot", "filename": "x.png"},
            {"action": "click_text", "locate": {"text": "Login"}},
            {"action": "click_relative_to_text", "locate": {"text": "Quantity"}, "offset_x": 10},
            {
                "action": "click_object",
                "locate": {
                    "label": "quantity_up",
                    "detector": "icons",
                    "pick": "topmost",
                    "region": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
                },
            },
            {"action": "click_object", "locate": {"label": "icon"}},
            {
                "action": "drag_object_to_position",
                "locate": {
                    "label": "audio_balance",
                    "detector": "icons",
                    "region": {"x1": 100, "y1": 100, "x2": 500, "y2": 260},
                },
                "target_x": 320,
                "target_y": 180,
                "duration_ms": 700,
            },
            {
                "action": "drag_object_to_object",
                "source": {
                    "label": "a",
                    "detector": "cards",
                },
                "target": {
                    "label": "b",
                    "detector": "cards",
                },
            },
            {"action": "emulator_command", "command": "status"},
            {"action": "emulator_send", "payload": {"v": 1}},
            {"action": "emulator_stop"},
        ],
        verification=[
            {"check": "text_exists", "locate": {"text": "done"}},
            {
                "check": "object_exists",
                "locate": {
                    "label": "ok",
                    "detector": "icons",
                    "region": {"x1": 300, "y1": 140, "x2": 360, "y2": 220},
                },
            },
            {"check": "log_contains", "source": "backend", "contains": "order created"},
            {"check": "wait_for_text", "locate": {"text": "ready"}},
        ],
    )
    (
        executor,
        emulator,
        actions,
        detector_calls,
        log_source_calls,
    ) = _make_executor(
        monkeypatch, case
    )
    executor.run()
    assert ("launch",) in emulator.calls
    assert ("execute", "status") in emulator.calls
    assert any(call[0] == "send" for call in emulator.calls)
    assert detector_calls == [
        {"type": "mock"},
        {"type": "mock"},
        {"type": "mock"},
    ]
    assert log_source_calls == [{"type": "file", "path": "logs/app.log"}]
    assert any(call[0] == "open_browser" for call in actions.calls)
    assert any(call[0] == "activate_window" for call in actions.calls)
    assert any(call[0] == "maximize_window" for call in actions.calls)
    assert any(
        call[0] == "press_key" and call[1:] == ("tab", 2, 0.1) for call in actions.calls
    )
    assert any(call[0] == "click_text" for call in actions.calls)
    assert any(call[0] == "click_relative_to_text" for call in actions.calls)
    assert any(
        call[0] == "click_object"
        and isinstance(call[1], ObjectLocateSpec)
        and call[1].label == "quantity_up"
        and call[1].detector == "icons"
        for call in actions.calls
    )
    assert any(
        call[0] == "drag_object_to_position"
        and isinstance(call[1], ObjectLocateSpec)
        and call[1].label == "audio_balance"
        and call[1].detector == "icons"
        for call in actions.calls
    )
    assert any(
        call[0] == "drag_object_to_object"
        and isinstance(call[1], ObjectLocateSpec)
        and isinstance(call[2], ObjectLocateSpec)
        and call[1].detector == "cards"
        and call[2].detector == "cards"
        for call in actions.calls
    )
    assert any(
        call[0] == "verify_object_exists"
        and isinstance(call[1], ObjectLocateSpec)
        and call[1].detector == "icons"
        and call[1].region is not None
        for call in actions.calls
    )
    assert any(call[0] == "verify_text_exists" for call in actions.calls)


def test_run_stage_missing_action(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CaseModel(name="bad", setup=[{}])
    executor, _, _, _, _ = _make_executor(monkeypatch, case)
    with pytest.raises(ActionExecutionError, match="missing 'action'"):
        executor.run()


def test_verification_missing_check(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CaseModel(name="bad", verification=[{}])
    executor, _, _, _, _ = _make_executor(monkeypatch, case)
    with pytest.raises(VerificationError, match="missing 'check'"):
        executor.run()


def test_verification_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CaseModel(name="bad", verification=[{"check": "text_exists", "locate": {"text": "x"}}])
    executor, _, actions, _, _ = _make_executor(monkeypatch, case)
    actions.text_exists_result = False
    with pytest.raises(VerificationError, match="Verification failed"):
        executor.run()


def test_steps_can_interleave_actions_and_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CaseModel(
        name="mixed",
        steps=[
            {"action": "click", "x": 1, "y": 2},
            {"check": "text_exists", "locate": {"text": "ok"}},
            {"action": "input_text", "text": "abc"},
            {"check": "wait_for_text", "locate": {"text": "ready"}},
        ],
    )
    executor, _, actions, _, _ = _make_executor(monkeypatch, case)

    executor.run()

    assert actions.calls == [
        ("click", 1, 2),
        ("verify_text_exists", TextLocateSpec(text="ok")),
        ("input_text", "abc"),
        ("wait_for_text", TextLocateSpec(text="ready"), 10.0, 0.5),
    ]


def test_steps_interleaved_check_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CaseModel(
        name="mixed-bad",
        steps=[
            {"action": "click", "x": 1, "y": 2},
            {"check": "text_exists", "locate": {"text": "missing"}},
        ],
    )
    executor, _, actions, _, _ = _make_executor(monkeypatch, case)
    actions.text_exists_result = False

    with pytest.raises(VerificationError, match="Verification failed at steps #2"):
        executor.run()


def test_runtime_builder_compiles_typed_steps() -> None:
    builder = runtime_mod.RuntimeBuilder()
    case = CaseModel(
        name="typed",
        setup=[{"action": "sleep", "seconds": "0.1"}],
        steps=[
            {
                "check": "text_exists",
                "locate": {"text": "ok"},
                "continue_on_failure": "true",
                "retry_count": "2",
                "retry_interval_seconds": "0.25",
                "tags": ["smoke", "ui"],
            },
            {},
        ],
        verification_setup=[
            {"action": "activate_window", "window_title": "Google Chrome", "timeout": 7},
            {"check": "text_exists", "locate": {"text": "preview"}},
        ],
        verification=[{"check": "wait_for_text", "locate": {"text": "ready"}, "timeout": 5}],
        teardown=[{"action": "emulator_stop"}],
    )

    plan = builder.compile(case)

    assert isinstance(plan.setup[0], runtime_mod.ActionStep)
    assert isinstance(plan.setup[0].args_model, step_specs_mod.SleepActionArgs)
    assert plan.setup[0].params["seconds"] == 0.1
    assert isinstance(plan.steps[0], runtime_mod.CheckStep)
    assert isinstance(plan.steps[0].args_model, step_specs_mod.TextExistsCheckArgs)
    assert isinstance(plan.steps[1], runtime_mod.InvalidStep)
    assert isinstance(plan.verification_setup[0], runtime_mod.ActionStep)
    assert isinstance(plan.verification_setup[1], runtime_mod.CheckStep)
    assert plan.verification_setup[0].timeout == 7.0
    assert isinstance(plan.verification[0], runtime_mod.CheckStep)
    assert isinstance(plan.verification[0].args_model, step_specs_mod.WaitForTextCheckArgs)
    assert plan.verification[0].source_key == "check"
    assert plan.verification[0].timeout == 5.0
    assert plan.steps[0].continue_on_failure is True
    assert plan.steps[0].retry_count == 2
    assert plan.steps[0].retry_interval_seconds == 0.25
    assert plan.steps[0].tags == ("smoke", "ui")
    assert isinstance(plan.teardown[0], runtime_mod.ActionStep)


def test_runtime_builder_rejects_verification_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(
        name="bad-verification-action",
        verification=[{"action": "wait_for_text", "locate": {"text": "ready"}, "timeout": 5}],
    )
    executor, _, _, _, _ = _make_executor(monkeypatch, case)

    assert isinstance(executor.plan.verification[0], runtime_mod.InvalidStep)
    assert executor.plan.verification[0].expected_fields == "'check'"
    assert "do not support 'action'" in str(executor.plan.verification[0].validation_error)

    with pytest.raises(VerificationError, match="invalid 'check'"):
        executor.run()


def test_verification_setup_runs_before_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(
        name="verification-setup",
        verification_setup=[
            {"action": "activate_window", "window_title": "Google Chrome"},
            {"check": "wait_for_text", "locate": {"text": "ready"}},
        ],
        verification=[{"check": "text_exists", "locate": {"text": "done"}}],
    )
    executor, _, actions, _, _ = _make_executor(monkeypatch, case)

    executor.run()

    assert actions.calls == [
        ("activate_window", "Google Chrome", 5.0, 0.2, 0.2),
        ("wait_for_text", TextLocateSpec(text="ready"), 10.0, 0.5),
        ("verify_text_exists", TextLocateSpec(text="done")),
    ]
    assert [
        (result.stage_name, result.step_name, result.passed)
        for result in executor.last_session.step_results
    ] == [
        ("verification_setup", "activate_window", True),
        ("verification_setup", "wait_for_text", True),
        ("verification", "text_exists", True),
    ]


def test_invalid_steps_are_compiled_but_fail_during_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(name="bad", setup=[{}])
    executor, _, _, _, _ = _make_executor(monkeypatch, case)

    assert isinstance(executor.plan.setup[0], runtime_mod.InvalidStep)

    with pytest.raises(ActionExecutionError, match="missing 'action' or 'check'"):
        executor.run()


def test_known_action_validation_failure_compiles_invalid_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(
        name="bad-open-browser",
        steps=[{"action": "open_browser", "url": "https://example.com", "args": "--bad"}],
    )
    executor, _, _, _, _ = _make_executor(monkeypatch, case)

    assert isinstance(executor.plan.steps[0], runtime_mod.InvalidStep)
    assert "must be a list" in str(executor.plan.steps[0].validation_error)

    with pytest.raises(ActionExecutionError, match="invalid parameters for action"):
        executor.run()


def test_step_continue_on_failure_keeps_pipeline_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(
        name="continue",
        steps=[
            {"action": "flaky_boom", "continue_on_failure": True, "tags": ["non_blocking"]},
            {"action": "click", "x": 1, "y": 2},
        ],
    )
    executor, _, actions, _, _ = _make_executor(monkeypatch, case)

    def fail_handler(params, payload):
        raise ActionExecutionError("boom")

    executor.context.services.action_registry.register("flaky_boom", context_handler=fail_handler)

    executor.run()

    assert actions.calls == [("click", 1, 2)]
    assert executor.last_session is not None
    assert executor.last_session.status == "passed_with_failures"
    assert executor.last_session.step_results[0].passed is False
    assert executor.last_session.step_results[0].continued_on_failure is True
    assert executor.last_session.step_results[0].tags == ("non_blocking",)
    assert executor.last_session.step_results[1].passed is True


def test_step_retry_count_retries_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CaseModel(
        name="retry",
        steps=[{"action": "flaky_click", "retry_count": 2, "retry_interval_seconds": 0.0}],
    )
    executor, _, _, _, _ = _make_executor(monkeypatch, case)
    calls = {"count": 0}

    def flaky_handler(params, payload):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ActionExecutionError(f"attempt {calls['count']} failed")

    executor.context.services.action_registry.register("flaky_click", context_handler=flaky_handler)

    executor.run()

    assert calls["count"] == 3
    assert executor.last_session is not None
    assert executor.last_session.status == "passed"
    assert executor.last_session.step_results[0].passed is True
    assert executor.last_session.step_results[0].attempts == 3


def test_step_failure_writes_failure_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = CaseModel(
        name="artifacted",
        steps=[{"action": "always_boom", "continue_on_failure": True}],
    )
    executor, _, _, _, _ = _make_executor(
        monkeypatch,
        case,
        output_dir=tmp_path,
    )

    def fail_handler(params, payload):
        raise ActionExecutionError("boom artifact")

    executor.context.services.action_registry.register("always_boom", context_handler=fail_handler)

    executor.run()

    assert executor.last_session is not None
    result = executor.last_session.step_results[0]
    artifact_path = Path(result.details["artifact_path"])
    assert artifact_path.exists()
    assert artifact_path in executor.last_session.artifacts

    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["case_name"] == "artifacted"
    assert artifact_payload["stage_name"] == "steps"
    assert artifact_payload["error"] == "boom artifact"
    assert artifact_payload["payload"]["action"] == "always_boom"


def test_runtime_profile_custom_runner_retry_policy_factory_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(
        name="custom-retry",
        steps=[{"action": "flaky_click", "retry_count": 2, "retry_interval_seconds": 0.0}],
    )
    used = {"called": False}

    class RecordingRetryPolicy(runtime_mod.RunnerRetryPolicy):
        def execute(self, **kwargs):
            used["called"] = True
            return super().execute(**kwargs)

    profile = exec_mod.RuntimeProfile(runner_retry_policy_factory=RecordingRetryPolicy)
    executor, _, _, _, _ = _make_executor(monkeypatch, case, profile=profile)
    calls = {"count": 0}

    def flaky_handler(params, payload):
        calls["count"] += 1
        if calls["count"] < 2:
            raise ActionExecutionError("fail once")

    executor.context.services.action_registry.register("flaky_click", context_handler=flaky_handler)

    executor.run()

    assert used["called"] is True
    assert isinstance(executor.context.services.retry_policy, RecordingRetryPolicy)
    assert calls["count"] == 2


def test_runtime_profile_custom_failure_policy_factory_is_used(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = CaseModel(
        name="custom-failure-policy",
        steps=[{"action": "always_boom", "continue_on_failure": True}],
    )
    used = {"called": False}

    class RecordingFailurePolicy(runtime_mod.FailurePolicy):
        def handle_step_failure(self, **kwargs):
            used["called"] = True
            return super().handle_step_failure(**kwargs)

    profile = exec_mod.RuntimeProfile(failure_policy_factory=RecordingFailurePolicy)
    executor, _, _, _, _ = _make_executor(
        monkeypatch,
        case,
        profile=profile,
        output_dir=tmp_path,
    )

    def fail_handler(params, payload):
        raise ActionExecutionError("policy boom")

    executor.context.services.action_registry.register("always_boom", context_handler=fail_handler)

    executor.run()

    assert used["called"] is True
    assert isinstance(executor.context.services.failure_policy, RecordingFailurePolicy)
    assert executor.last_session is not None
    assert executor.last_session.step_results[0].details["error"] == "policy boom"


def test_default_dispatchers_assemble_separate_registries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(name="registries", steps=[{"action": "click", "x": 1, "y": 2}])
    executor, _, _, _, _ = _make_executor(monkeypatch, case)

    services = executor.context.services

    assert services.action_registry is services.action_dispatcher.registry
    assert services.check_registry is services.check_dispatcher.registry
    assert services.action_registry is not services.action_dispatcher
    assert services.check_registry is not services.check_dispatcher
    assert services.action_registry.resolve("click").args_builder is not None
    assert services.check_registry.resolve("text_exists").args_builder is not None


def test_test_executor_exposes_runtime_through_context_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(name="grouped-context", steps=[{"action": "click", "x": 1, "y": 2}])
    executor, _, _, _, _ = _make_executor(monkeypatch, case)

    assert executor.context.metadata.case is case
    assert executor.context.metadata.output_dir == Path("outputs_test")
    assert executor.context.resources.base_actions is not None
    assert executor.context.resources.locate_actions is not None
    assert executor.context.services.action_registry is not None
    assert executor.context.services.hook_bus is not None
    assert not hasattr(executor.context, "base_actions")
    assert not hasattr(executor.context, "output_dir")
    assert not hasattr(executor.context, "action_registry")
    assert not hasattr(executor.context.resources, "text_actions")
    assert not hasattr(executor.context.resources, "object_actions")
    assert not hasattr(executor, "output_dir")
    assert not hasattr(executor, "base_actions")
    assert not hasattr(executor, "action_registry")


def test_action_registry_supports_context_aware_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(name="plugin-action", steps=[{"action": "plugin_click", "x": 9, "y": 7}])
    executor, _, actions, _, _ = _make_executor(monkeypatch, case)
    captured = {}

    def plugin_handler(context, params, payload):
        captured["payload"] = dict(payload)
        assert context.resources.base_actions is executor.context.resources.base_actions
        assert context.metadata.output_dir == executor.context.metadata.output_dir
        assert not hasattr(context, "base_actions")
        assert not hasattr(context, "output_dir")
        context.resources.base_actions.click(int(params["x"]), int(params["y"]))

    executor.context.services.action_registry.register(
        "plugin_click",
        context_handler=plugin_handler,
    )

    executor.run()

    assert captured["payload"]["action"] == "plugin_click"
    assert actions.calls == [("click", 9, 7)]


def test_check_registry_supports_context_aware_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(
        name="plugin-check",
        verification=[{"check": "custom_ready", "locate": {"text": "ready"}}],
    )
    executor, _, actions, _, _ = _make_executor(monkeypatch, case)

    def plugin_check(context, params):
        assert context.resources.locate_actions is executor.context.resources.locate_actions
        assert not hasattr(context, "text_actions")
        return context.resources.locate_actions.verify_text_exists(
            step_specs_mod.parse_text_locate_spec(params["locate"])
        )

    executor.context.services.check_registry.register(
        "custom_ready",
        context_handler=plugin_check,
    )

    executor.run()

    assert actions.calls == [("verify_text_exists", TextLocateSpec(text="ready"))]


def test_runtime_profile_plugins_extend_compile_and_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UiPlugin:
        @staticmethod
        def register_actions(registry) -> None:
            registry.register(
                "plugin_click",
                context_handler=lambda context, params, payload: context.resources.base_actions.click(
                    int(params["x"]), int(params["y"])
                ),
                args_builder=lambda params: step_specs_mod.ClickActionArgs(
                    x=int(params["x"]),
                    y=int(params["y"]),
                ),
            )

        @staticmethod
        def register_checks(registry) -> None:
            registry.register(
                "plugin_text_ready",
                context_handler=lambda context, params: context.resources.locate_actions.verify_text_exists(
                    step_specs_mod.parse_text_locate_spec(params["locate"]),
                ),
                args_builder=lambda params: step_specs_mod.TextExistsCheckArgs(
                    locate=step_specs_mod.parse_text_locate_spec(params["locate"]),
                ),
            )

    case = CaseModel(
        name="plugin-profile",
        steps=[{"action": "plugin_click", "x": "5", "y": "6"}],
        verification=[{"check": "plugin_text_ready", "locate": {"text": "ready"}}],
    )
    profile = exec_mod.RuntimeProfile(plugins=(UiPlugin(),))
    executor, _, actions, _, _ = _make_executor(monkeypatch, case, profile=profile)

    assert isinstance(executor.plan.steps[0].args_model, step_specs_mod.ClickActionArgs)
    assert isinstance(executor.plan.verification[0].args_model, step_specs_mod.TextExistsCheckArgs)

    executor.run()

    assert actions.calls == [
        ("click", 5, 6),
        ("verify_text_exists", TextLocateSpec(text="ready")),
    ]


def test_runtime_profile_plugins_support_namespaced_actions_and_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NamespacedPlugin:
        namespace = "sample"
        override = False

        @staticmethod
        def register_actions(registry) -> None:
            registry.register(
                "click_xy",
                context_handler=lambda context, params, payload: context.resources.base_actions.click(
                    int(params["x"]),
                    int(params["y"]),
                ),
                args_builder=lambda params: step_specs_mod.ClickActionArgs(
                    x=int(params["x"]),
                    y=int(params["y"]),
                ),
            )

        @staticmethod
        def register_checks(registry) -> None:
            registry.register(
                "ready",
                context_handler=lambda context, params: context.resources.locate_actions.verify_text_exists(
                    step_specs_mod.parse_text_locate_spec(params["locate"]),
                ),
                args_builder=lambda params: step_specs_mod.TextExistsCheckArgs(
                    locate=step_specs_mod.parse_text_locate_spec(params["locate"]),
                ),
            )

    case = CaseModel(
        name="plugin-namespace",
        steps=[{"action": "sample.click_xy", "x": 7, "y": 8}],
        verification=[{"check": "sample.ready", "locate": {"text": "go"}}],
    )
    profile = exec_mod.RuntimeProfile(plugins=(NamespacedPlugin(),))
    executor, _, actions, _, _ = _make_executor(monkeypatch, case, profile=profile)

    assert isinstance(executor.plan.steps[0].args_model, step_specs_mod.ClickActionArgs)
    assert isinstance(executor.plan.verification[0].args_model, step_specs_mod.TextExistsCheckArgs)

    executor.run()

    assert actions.calls == [
        ("click", 7, 8),
        ("verify_text_exists", TextLocateSpec(text="go")),
    ]


def test_runtime_profile_plugins_can_override_builtin_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OverrideClickPlugin:
        override = True

        @staticmethod
        def register_actions(registry) -> None:
            registry.register(
                "click",
                context_handler=lambda context, params, payload: context.resources.base_actions.click(
                    99,
                    100,
                ),
                args_builder=step_specs_mod.get_action_args_builder("click"),
            )

    case = CaseModel(name="plugin-override", steps=[{"action": "click", "x": 1, "y": 2}])
    profile = exec_mod.RuntimeProfile(plugins=(OverrideClickPlugin(),))
    executor, _, actions, _, _ = _make_executor(monkeypatch, case, profile=profile)

    executor.run()

    assert actions.calls == [("click", 99, 100)]


def test_runtime_profile_plugins_reject_builtin_collisions_without_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConflictingClickPlugin:
        override = False

        @staticmethod
        def register_actions(registry) -> None:
            registry.register(
                "click",
                context_handler=lambda context, params, payload: None,
                args_builder=step_specs_mod.get_action_args_builder("click"),
            )

    case = CaseModel(name="plugin-conflict", steps=[{"action": "click", "x": 1, "y": 2}])
    profile = exec_mod.RuntimeProfile(plugins=(ConflictingClickPlugin(),))

    with pytest.raises(ValueError, match="Action handler already registered: click"):
        _make_executor(monkeypatch, case, profile=profile)


def test_runtime_profile_plugins_can_extend_emulator_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emulator_calls = []

    class PluginEmulator:
        def launch(self):
            emulator_calls.append(("launch",))

        def stop(self):
            emulator_calls.append(("stop",))

        def execute(self, command):
            emulator_calls.append(("execute", command))
            return f"plugin:{command}"

        def send(self, payload, endpoint=None, method=None, headers=None):
            emulator_calls.append(("send", payload, endpoint, method, headers))
            return "plugin-send"

    class EmulatorPlugin:
        namespace = "sample"
        override = False

        @staticmethod
        def register_emulators(registry) -> None:
            registry.register("device", PluginEmulator)

    monkeypatch.setattr(exec_mod, "create_detector", lambda cfg: {"config": cfg})
    monkeypatch.setattr(exec_mod, "create_log_source", lambda cfg: FakeLogSource(text="ok"))
    monkeypatch.setattr(exec_mod, "create_ocr_engine", lambda cfg: object())
    monkeypatch.setattr(exec_mod, "WindowCapture", lambda **kwargs: object())
    monkeypatch.setattr(exec_mod, "ActionServices", FakeActions)

    case = CaseModel(
        name="plugin-emulator",
        emulator={"type": "sample.device"},
        detector={"type": "mock"},
        setup=[{"action": "emulator_launch"}],
        steps=[{"action": "emulator_command", "command": "status"}],
        teardown=[{"action": "emulator_stop"}],
    )
    profile = exec_mod.RuntimeProfile(plugins=(EmulatorPlugin(),))

    executor = exec_mod.TestExecutor(
        case=case,
        profile=_resolve_profile(profile),
        output_dir=Path("outputs_test"),
    )
    executor.run()

    assert emulator_calls == [
        ("launch",),
        ("execute", "status"),
        ("stop",),
    ]


def test_executor_prefers_dispatch_step_for_typed_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(
        name="typed-dispatch",
        steps=[{"action": "click", "x": "1", "y": "2"}],
        verification=[{"check": "text_exists", "locate": {"text": "ok"}}],
    )
    executor, _, actions, _, _ = _make_executor(monkeypatch, case)

    executor.run()

    assert actions.calls == [
        ("click", 1, 2),
        ("verify_text_exists", TextLocateSpec(text="ok")),
    ]


def test_executor_supports_split_action_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_emulator = FakeEmulator()

    monkeypatch.setattr(exec_mod, "create_emulator", lambda cfg: fake_emulator)
    monkeypatch.setattr(exec_mod, "create_detector", lambda cfg: {"config": cfg})
    monkeypatch.setattr(exec_mod, "create_log_source", lambda cfg: FakeLogSource(text="ok"))
    monkeypatch.setattr(exec_mod, "create_ocr_engine", lambda cfg: object())
    monkeypatch.setattr(exec_mod, "WindowCapture", lambda **kwargs: object())

    case = CaseModel(
        name="split-services",
        detector={"type": "mock"},
        steps=[
            {"action": "click", "x": 1, "y": 2},
            {"action": "click_text", "locate": {"text": "Login"}},
            {"action": "click_object", "locate": {"label": "icon"}},
        ],
        verification=[
            {"check": "text_exists", "locate": {"text": "ready"}},
            {"check": "object_exists", "locate": {"label": "icon"}},
        ],
    )
    profile = exec_mod.RuntimeProfile(actions_factory=SplitActions)

    executor = exec_mod.TestExecutor(
        case=case,
        profile=_resolve_profile(profile),
        output_dir=Path("outputs_test"),
    )
    executor.run()

    resources = executor.context.resources
    services = executor.context.services

    assert resources.screenshot_actions is resources.base_actions
    assert resources.locate_actions is not None
    assert services.action_dispatcher.resources.locate_actions is resources.locate_actions
    assert services.check_dispatcher.resources.locate_actions is resources.locate_actions
    assert services.action_dispatcher.resources.base_actions is resources.base_actions
    assert not hasattr(services.action_dispatcher, "base_actions")
    assert not hasattr(services.action_dispatcher.resources, "text_actions")
    assert not hasattr(services.action_dispatcher.resources, "object_actions")
    assert not hasattr(services.check_dispatcher.resources, "text_actions")
    assert not hasattr(services.check_dispatcher.resources, "object_actions")
    assert resources.base_actions.calls == [("click", 1, 2)]
    assert resources.locate_actions.text.calls == [
        ("click_text", TextLocateSpec(text="Login"), None, None),
        ("verify_text_exists", TextLocateSpec(text="ready")),
    ]
    assert resources.locate_actions.object.calls == [
        ("click_object", ObjectLocateSpec(label="icon"), None),
        ("verify_object_exists", ObjectLocateSpec(label="icon")),
    ]


def test_emulator_send_requires_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CaseModel(name="bad")
    executor, _, _, _, _ = _make_executor(monkeypatch, case)
    step = runtime_mod.ActionStep(name="emulator_send", params={})
    with pytest.raises(ActionExecutionError, match="requires 'payload'"):
        executor.context.services.action_registry.dispatch_step(step)


def test_dispatch_action_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CaseModel(name="bad")
    executor, _, _, _, _ = _make_executor(monkeypatch, case)
    step = runtime_mod.ActionStep(name="unknown", params={})
    with pytest.raises(ActionExecutionError, match="Unsupported action"):
        executor.context.services.action_registry.dispatch_step(step)


def test_open_browser_args_must_be_list(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CaseModel(name="bad")
    executor, _, _, _, _ = _make_executor(monkeypatch, case)
    step = runtime_mod.ActionStep(
        name="open_browser",
        params={"url": "https://example.com", "args": "--bad"},
    )
    with pytest.raises(ActionExecutionError, match="field 'args' must be a list"):
        executor.context.services.action_registry.dispatch_step(step)


def test_screenshot_without_filename_saves_default_path_for_untyped_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(name="screenshot-untyped")
    executor, _, actions, _, _ = _make_executor(monkeypatch, case)
    step = runtime_mod.ActionStep(name="screenshot", params={})

    executor.context.services.action_registry.dispatch_step(step)

    assert len(actions.calls) == 1
    action_name, path = actions.calls[0]
    assert action_name == "screenshot"
    assert isinstance(path, str)
    assert path.startswith(str(Path("outputs_test")))
    assert Path(path).name.startswith("screenshot_")
    assert Path(path).suffix == ".png"


def test_screenshot_without_filename_saves_default_path_for_typed_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CaseModel(name="screenshot-typed")
    executor, _, actions, _, _ = _make_executor(monkeypatch, case)
    step = runtime_mod.ActionStep(
        name="screenshot",
        params={},
        args_model=step_specs_mod.ScreenshotActionArgs(),
    )

    executor.context.services.action_registry.dispatch_step(step)

    assert len(actions.calls) == 1
    action_name, path = actions.calls[0]
    assert action_name == "screenshot"
    assert isinstance(path, str)
    assert path.startswith(str(Path("outputs_test")))
    assert Path(path).name.startswith("screenshot_")
    assert Path(path).suffix == ".png"


def test_dispatch_check_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CaseModel(name="bad")
    executor, _, _, _, _ = _make_executor(monkeypatch, case)
    step = runtime_mod.CheckStep(name="unknown", params={})
    with pytest.raises(VerificationError, match="Unsupported verification check"):
        executor.context.services.check_registry.dispatch_step(step)


def test_executor_records_last_session(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CaseModel(name="session", steps=[{"action": "click", "x": 1, "y": 2}])
    executor, _, _, _, _ = _make_executor(monkeypatch, case)

    executor.run()

    assert executor.last_session is not None
    assert executor.last_session.status == "passed"
    assert [
        (result.stage_name, result.step_name, result.passed)
        for result in executor.last_session.step_results
    ] == [("steps", "click", True)]


def test_executor_profile_hook_bus_receives_runtime_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    hook_bus = exec_mod.HookBus()
    for event_name in [
        event_mod.CASE_STARTED,
        event_mod.BEFORE_STAGE,
        event_mod.BEFORE_STEP,
        event_mod.AFTER_STEP,
        event_mod.AFTER_STAGE,
        event_mod.CASE_FINISHED,
    ]:
        hook_bus.register(
            event_name,
            lambda context, session, payload, event_name=event_name: events.append(
                (event_name, payload.copy())
            ),
        )

    case = CaseModel(name="hooks", steps=[{"action": "click", "x": 1, "y": 2}])
    profile = exec_mod.RuntimeProfile(hook_bus_factory=lambda: hook_bus)
    executor, _, _, _, _ = _make_executor(monkeypatch, case, profile=profile)

    executor.run()

    assert executor.context.services.hook_bus is hook_bus
    assert [name for name, _ in events] == [
        event_mod.CASE_STARTED,
        event_mod.BEFORE_STAGE,
        event_mod.AFTER_STAGE,
        event_mod.BEFORE_STAGE,
        event_mod.BEFORE_STEP,
        event_mod.AFTER_STEP,
        event_mod.AFTER_STAGE,
        event_mod.BEFORE_STAGE,
        event_mod.AFTER_STAGE,
        event_mod.BEFORE_STAGE,
        event_mod.AFTER_STAGE,
        event_mod.BEFORE_STAGE,
        event_mod.AFTER_STAGE,
        event_mod.CASE_FINISHED,
    ]
