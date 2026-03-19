from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from autoscene.runner.runtime_events import (
    BEFORE_STEP_ATTEMPT,
    STEP_ATTEMPT_FAILED,
    STEP_FAILED,
)
from autoscene.runner.runtime_models import (
    RetryExecutionOutcome,
    RunSession,
    RuntimeContext,
    Step,
    StepResult,
)


class HookBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., None]]] = {}

    def register(self, event_name: str, handler: Callable[..., None]) -> None:
        self._handlers.setdefault(str(event_name), []).append(handler)

    def emit(
        self,
        event_name: str,
        context: RuntimeContext,
        session: RunSession | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        for handler in self._handlers.get(str(event_name), []):
            handler(
                context=context,
                session=session,
                payload=self._copy_payload(payload),
            )

    @staticmethod
    def _copy_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
        return dict(payload or {})


class ArtifactStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_step_failure(
        self,
        *,
        context: RuntimeContext,
        session: RunSession,
        step: Step,
        result: StepResult,
        error: Exception,
    ) -> Path:
        artifact_path = self._build_failure_artifact_path(session=session, step=step, result=result)
        artifact_payload = self._build_failure_artifact_payload(
            context=context,
            session=session,
            step=step,
            result=result,
            error=error,
        )
        artifact_path.write_text(
            json.dumps(artifact_payload, indent=2, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        session.artifacts.append(artifact_path)
        return artifact_path

    def _build_failure_artifact_path(
        self,
        *,
        session: RunSession,
        step: Step,
        result: StepResult,
    ) -> Path:
        safe_case_name = self._slugify(session.case_name)
        safe_stage_name = self._slugify(result.stage_name)
        safe_step_name = self._slugify(result.step_name or step.step_type)
        filename = (
            f"{safe_case_name}_{safe_stage_name}_{int(result.step_index):03d}_{safe_step_name}"
            "_failure.json"
        )
        return self.output_dir / filename

    @staticmethod
    def _build_failure_artifact_payload(
        *,
        context: RuntimeContext,
        session: RunSession,
        step: Step,
        result: StepResult,
        error: Exception,
    ) -> dict[str, Any]:
        return {
            "case_name": session.case_name,
            "profile": context.metadata.profile.name,
            "stage_name": result.stage_name,
            "step_index": result.step_index,
            "step_name": result.step_name,
            "step_type": result.step_type,
            "attempts": result.attempts,
            "continued_on_failure": result.continued_on_failure,
            "tags": list(result.tags),
            "payload": dict(result.payload),
            "raw_payload": step.to_raw_payload(),
            "details": dict(result.details),
            "error_type": type(error).__name__,
            "error": str(error),
        }

    @staticmethod
    def _slugify(value: str) -> str:
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
        text = text.strip("._")
        return text or "step"


class RunnerRetryPolicy:
    def max_attempts(self, step: Step) -> int:
        return max(int(step.retry_count) + 1, 1)

    def should_retry(self, *, step: Step, attempt: int, error: Exception) -> bool:
        return attempt < self.max_attempts(step)

    def wait_before_retry(self, *, step: Step, attempt: int, error: Exception) -> None:
        interval_seconds = max(float(step.retry_interval_seconds), 0.0)
        if interval_seconds > 0.0:
            time.sleep(interval_seconds)

    def execute(
        self,
        *,
        stage_name: str,
        index: int,
        step: Step,
        context: RuntimeContext,
        session: RunSession,
        operation: Callable[[], tuple[str, str]],
    ) -> RetryExecutionOutcome:
        max_attempts = self.max_attempts(step)
        last_error: Exception | None = None
        step_name, step_type = self._step_identity(step)
        for attempt in range(1, max_attempts + 1):
            try:
                context.services.hook_bus.emit(
                    BEFORE_STEP_ATTEMPT,
                    context,
                    session,
                    self._attempt_payload(
                        stage_name=stage_name,
                        index=index,
                        attempt=attempt,
                        step=step,
                    ),
                )
                step_name, step_type = operation()
                return self._build_outcome(
                    step_name=step_name,
                    step_type=step_type,
                    attempts=attempt,
                )
            except Exception as exc:
                last_error = exc
                will_retry = self.should_retry(step=step, attempt=attempt, error=exc)
                context.services.hook_bus.emit(
                    STEP_ATTEMPT_FAILED,
                    context,
                    session,
                    self._attempt_failed_payload(
                        stage_name=stage_name,
                        index=index,
                        attempt=attempt,
                        error=exc,
                        will_retry=will_retry,
                    ),
                )
                if not will_retry:
                    break
                self.wait_before_retry(step=step, attempt=attempt, error=exc)

        assert last_error is not None
        return self._build_outcome(
            step_name=step_name,
            step_type=step_type,
            attempts=max_attempts,
            error=last_error,
        )

    @staticmethod
    def _step_identity(step: Step) -> tuple[str, str]:
        return (step.name, step.step_type)

    @staticmethod
    def _build_outcome(
        *,
        step_name: str,
        step_type: str,
        attempts: int,
        error: Exception | None = None,
    ) -> RetryExecutionOutcome:
        return RetryExecutionOutcome(
            step_name=step_name,
            step_type=step_type,
            attempts=attempts,
            error=error,
        )

    @staticmethod
    def _attempt_payload(
        *,
        stage_name: str,
        index: int,
        attempt: int,
        step: Step,
    ) -> dict[str, Any]:
        return {
            "stage_name": stage_name,
            "index": index,
            "attempt": attempt,
            "step": step,
        }

    @staticmethod
    def _attempt_failed_payload(
        *,
        stage_name: str,
        index: int,
        attempt: int,
        error: Exception,
        will_retry: bool,
    ) -> dict[str, Any]:
        return {
            "stage_name": stage_name,
            "index": index,
            "attempt": attempt,
            "error": error,
            "will_retry": will_retry,
        }


class FailurePolicy:
    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store

    def handle_step_failure(
        self,
        *,
        stage_name: str,
        index: int,
        step: Step,
        context: RuntimeContext,
        session: RunSession,
        payload: dict[str, Any],
        duration_ms: int,
        attempts: int,
        error: Exception,
        step_name: str,
        step_type: str,
    ) -> StepResult:
        result = self._build_failed_result(
            stage_name=stage_name,
            index=index,
            step_name=step_name,
            step_type=step_type,
            duration_ms=duration_ms,
            attempts=attempts,
            payload=payload,
            step=step,
            error=error,
        )
        artifact_path, artifact_error = self._record_artifact(
            context=context,
            session=session,
            step=step,
            result=result,
            error=error,
        )
        self._attach_artifact_details(
            result=result,
            artifact_path=artifact_path,
            artifact_error=artifact_error,
        )
        self._record_failure(session=session, result=result, error=error)
        context.services.hook_bus.emit(
            STEP_FAILED,
            context,
            session,
            self._step_failed_payload(
                stage_name=stage_name,
                index=index,
                result=result,
                error=error,
                artifact_path=artifact_path,
            ),
        )
        if step.continue_on_failure:
            context.metadata.logger.warning(
                "[%s] #%s continued after failure: %s",
                stage_name,
                index,
                error,
            )
            return result
        raise error

    @staticmethod
    def _build_failed_result(
        *,
        stage_name: str,
        index: int,
        step_name: str,
        step_type: str,
        duration_ms: int,
        attempts: int,
        payload: dict[str, Any],
        step: Step,
        error: Exception,
    ) -> StepResult:
        return StepResult(
            stage_name=stage_name,
            step_index=index,
            step_name=step_name,
            step_type=step_type,
            passed=False,
            duration_ms=duration_ms,
            attempts=attempts,
            continued_on_failure=bool(step.continue_on_failure),
            tags=step.tags,
            payload=payload,
            details={"error": str(error)},
        )

    def _record_artifact(
        self,
        *,
        context: RuntimeContext,
        session: RunSession,
        step: Step,
        result: StepResult,
        error: Exception,
    ) -> tuple[Path | None, str | None]:
        if self.artifact_store is None:
            return (None, None)

        try:
            artifact_path = self.artifact_store.record_step_failure(
                context=context,
                session=session,
                step=step,
                result=result,
                error=error,
            )
        except OSError as exc:
            context.metadata.logger.warning(
                "[%s] #%s failed to persist failure artifact: %s",
                result.stage_name,
                result.step_index,
                exc,
            )
            return (None, str(exc))
        return (artifact_path, None)

    @staticmethod
    def _attach_artifact_details(
        *,
        result: StepResult,
        artifact_path: Path | None,
        artifact_error: str | None,
    ) -> None:
        if artifact_path is not None:
            result.details["artifact_path"] = str(artifact_path)
        if artifact_error is not None:
            result.details["artifact_error"] = artifact_error

    @staticmethod
    def _record_failure(
        *,
        session: RunSession,
        result: StepResult,
        error: Exception,
    ) -> None:
        session.step_results.append(result)
        session.failures.append(str(error))

    @staticmethod
    def _step_failed_payload(
        *,
        stage_name: str,
        index: int,
        result: StepResult,
        error: Exception,
        artifact_path: Path | None,
    ) -> dict[str, Any]:
        return {
            "stage_name": stage_name,
            "index": index,
            "result": result,
            "error": error,
            "artifact_path": artifact_path,
        }


__all__ = [
    "ArtifactStore",
    "FailurePolicy",
    "HookBus",
    "RunnerRetryPolicy",
]
