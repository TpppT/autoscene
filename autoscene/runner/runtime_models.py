from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from autoscene.actions.service_resolution import (
    BaseActionService,
    LocateActionService,
    ScreenshotActionService,
)
from autoscene.core.models import TestCase
from autoscene.logs.interfaces import LogSource
from autoscene.runner.protocols import (
    ActionRegistryProtocol,
    ArtifactStoreProtocol,
    CaptureProtocol,
    CheckRegistryProtocol,
    EmulatorProtocol,
    FailurePolicyProtocol,
    HookBusProtocol,
    RunnerRetryPolicyProtocol,
)
from autoscene.runner.step_specs import StepArgs
from autoscene.vision.interfaces import Detector, OCREngine, ReaderAdapter


@dataclass(frozen=True)
class Step:
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    args_model: StepArgs | None = None
    source_key: str | None = None
    timeout: float | None = None
    continue_on_failure: bool = False
    retry_count: int = 0
    retry_interval_seconds: float = 0.0
    tags: tuple[str, ...] = ()

    @property
    def step_type(self) -> str:
        raise NotImplementedError

    @property
    def payload_key(self) -> str:
        raise NotImplementedError

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.params)
        if self.name:
            payload[self.payload_key] = self.name
        return payload

    def to_raw_payload(self) -> dict[str, Any]:
        if self.raw:
            return dict(self.raw)
        return self.to_payload()


@dataclass(frozen=True)
class ActionStep(Step):
    @property
    def step_type(self) -> str:
        return "action"

    @property
    def payload_key(self) -> str:
        return "action"


@dataclass(frozen=True)
class CheckStep(Step):
    @property
    def step_type(self) -> str:
        return "check"

    @property
    def payload_key(self) -> str:
        return self.source_key or "check"


@dataclass(frozen=True)
class InvalidStep(Step):
    expected_fields: str = "'action'"
    validation_error: str | None = None

    @property
    def step_type(self) -> str:
        return "invalid"

    @property
    def payload_key(self) -> str:
        return ""

    def to_payload(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class ScenarioPlan:
    setup: list[Step] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    verification_setup: list[Step] = field(default_factory=list)
    verification: list[Step] = field(default_factory=list)
    teardown: list[Step] = field(default_factory=list)


@dataclass
class StepResult:
    stage_name: str
    step_index: int
    step_name: str
    step_type: str
    passed: bool
    duration_ms: int
    attempts: int = 1
    continued_on_failure: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)
    payload: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSession:
    case_name: str
    variables: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    status: str = "pending"


@dataclass(frozen=True)
class RuntimeProfile:
    name: str = "default"
    environment: str = "default"
    plugins: tuple[object, ...] = ()
    emulator_factory: Callable[[dict[str, Any]], EmulatorProtocol] | None = None
    detector_factory: Callable[[dict[str, Any]], Detector] | None = None
    reader_factory: Callable[[dict[str, Any]], ReaderAdapter] | None = None
    log_source_factory: Callable[[dict[str, Any]], LogSource] | None = None
    ocr_engine_factory: Callable[[dict[str, Any]], OCREngine] | None = None
    capture_factory: Callable[[dict[str, Any]], CaptureProtocol] | None = None
    actions_factory: Callable[..., object] | None = None
    action_dispatcher_factory: Callable[..., object] | None = None
    check_dispatcher_factory: Callable[..., object] | None = None
    hook_bus_factory: Callable[[], "HookBus"] | None = None
    artifact_store_factory: Callable[..., "ArtifactStore"] | None = None
    runner_retry_policy_factory: Callable[..., "RunnerRetryPolicy"] | None = None
    failure_policy_factory: Callable[..., "FailurePolicy"] | None = None
    pipeline_factory: Callable[[], "ScenarioPipeline"] | None = None


@dataclass(frozen=True)
class RuntimeMetadata:
    case: TestCase
    profile: RuntimeProfile
    output_dir: Path
    logger: logging.Logger


@dataclass(frozen=True)
class RuntimeResources:
    emulator: EmulatorProtocol
    detector: Detector
    detectors: dict[str, Detector]
    readers: dict[str, ReaderAdapter]
    log_sources: dict[str, LogSource]
    ocr_engine: OCREngine
    capture: CaptureProtocol
    base_actions: BaseActionService | None
    locate_actions: LocateActionService | None
    screenshot_actions: ScreenshotActionService | None


@dataclass(frozen=True)
class RuntimeServices:
    action_dispatcher: ActionRegistryProtocol
    check_dispatcher: CheckRegistryProtocol
    action_registry: ActionRegistryProtocol
    check_registry: CheckRegistryProtocol
    hook_bus: HookBusProtocol
    artifact_store: ArtifactStoreProtocol
    retry_policy: RunnerRetryPolicyProtocol
    failure_policy: FailurePolicyProtocol


@dataclass
class RuntimeContext:
    metadata: RuntimeMetadata
    resources: RuntimeResources
    services: RuntimeServices


@dataclass(frozen=True)
class RetryExecutionOutcome:
    step_name: str
    step_type: str
    attempts: int
    error: Exception | None = None

    @property
    def passed(self) -> bool:
        return self.error is None


__all__ = [
    "ActionStep",
    "CheckStep",
    "InvalidStep",
    "RetryExecutionOutcome",
    "RunSession",
    "RuntimeContext",
    "RuntimeMetadata",
    "RuntimeProfile",
    "RuntimeResources",
    "RuntimeServices",
    "ScenarioPlan",
    "Step",
    "StepResult",
]
