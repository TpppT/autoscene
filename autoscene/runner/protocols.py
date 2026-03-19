from __future__ import annotations

from typing import Any, Callable, Protocol

class EmulatorProtocol(Protocol):
    def launch(self) -> None: ...

    def stop(self) -> None: ...

    def execute(self, command: str) -> str: ...

    def send(
        self,
        payload: Any,
        endpoint: str | None = None,
        method: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> str: ...


class CaptureProtocol(Protocol):
    def capture(self, *args: Any, **kwargs: Any) -> Any: ...

    def capture_result(self, *args: Any, **kwargs: Any) -> Any: ...

    def resolve_capture_region(self, *args: Any, **kwargs: Any) -> Any: ...

    def bind_window_handle(self, window_handle: int) -> None: ...

    def get_last_capture_result(self) -> Any: ...


class HookBusProtocol(Protocol):
    def register(self, event_name: str, handler) -> None: ...

    def emit(
        self,
        event_name: str,
        context: Any,
        session: Any,
        payload: dict[str, Any] | None = None,
    ) -> None: ...


class ArtifactStoreProtocol(Protocol):
    def record_step_failure(
        self,
        *,
        context: Any,
        session: Any,
        step: Any,
        result: Any,
        error: Exception,
    ) -> Path: ...


class RunnerRetryPolicyProtocol(Protocol):
    def execute(
        self,
        *,
        stage_name: str,
        index: int,
        step: Any,
        context: Any,
        session: Any,
        operation: Callable[[], tuple[str, str]],
    ) -> Any: ...


class FailurePolicyProtocol(Protocol):
    def handle_step_failure(
        self,
        *,
        stage_name: str,
        index: int,
        step: Any,
        context: Any,
        session: Any,
        payload: dict[str, Any],
        duration_ms: int,
        attempts: int,
        error: Exception,
        step_name: str,
        step_type: str,
    ) -> Any: ...


class ActionHandlerContextProtocol(Protocol):
    metadata: Any
    resources: Any


class CheckHandlerContextProtocol(Protocol):
    metadata: Any
    resources: Any


class ActionRegistryProtocol(Protocol):
    def register(self, action_name: str, **kwargs: Any) -> None: ...

    def resolve(self, action_name: str) -> Any: ...

    def dispatch_step(self, step: Any) -> None: ...


class CheckRegistryProtocol(Protocol):
    def register(self, check_name: str, **kwargs: Any) -> None: ...

    def resolve(self, check_name: str) -> Any: ...

    def dispatch_step(self, step: Any) -> bool: ...
