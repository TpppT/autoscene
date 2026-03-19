from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from autoscene.core.exceptions import ActionExecutionError, VerificationError
from autoscene.runner.protocols import (
    ActionHandlerContextProtocol,
    CheckHandlerContextProtocol,
)
from autoscene.runner.step_specs import StepArgs


class ActionHandler(Protocol):
    def __call__(
        self,
        context: ActionHandlerContextProtocol,
        params: dict[str, Any],
        payload: dict[str, Any],
    ) -> None: ...


class TypedActionHandler(Protocol):
    def __call__(
        self,
        context: ActionHandlerContextProtocol,
        args: StepArgs,
    ) -> None: ...


class CheckHandler(Protocol):
    def __call__(
        self,
        context: CheckHandlerContextProtocol,
        params: dict[str, Any],
    ) -> bool: ...


class TypedCheckHandler(Protocol):
    def __call__(
        self,
        context: CheckHandlerContextProtocol,
        args: StepArgs,
    ) -> bool: ...


class ActionPlugin(Protocol):
    namespace: str | None
    override: bool

    def register_actions(self, registry: "ScopedActionRegistry") -> None: ...


class CheckPlugin(Protocol):
    namespace: str | None
    override: bool

    def register_checks(self, registry: "ScopedCheckRegistry") -> None: ...


class RegistryPlugin(ActionPlugin, CheckPlugin, Protocol):
    pass


@dataclass(frozen=True)
class ActionRegistration:
    name: str
    context_handler: ActionHandler | None = None
    typed_handler: TypedActionHandler | None = None
    args_builder: Callable[[dict[str, Any]], StepArgs | None] | None = None


@dataclass(frozen=True)
class CheckRegistration:
    name: str
    context_handler: CheckHandler | None = None
    typed_handler: TypedCheckHandler | None = None
    args_builder: Callable[[dict[str, Any]], StepArgs | None] | None = None


class ActionRegistry:
    def __init__(self, context: ActionHandlerContextProtocol) -> None:
        self._context = context
        self._registrations: dict[str, ActionRegistration] = {}

    def register(
        self,
        action_name: str,
        *,
        context_handler: ActionHandler | None = None,
        typed_handler: TypedActionHandler | None = None,
        args_builder: Callable[[dict[str, Any]], StepArgs | None] | None = None,
        namespace: str | None = None,
        override: bool = True,
    ) -> None:
        name = _qualify_registry_name(action_name, namespace=namespace)
        if not override and name in self._registrations:
            raise ValueError(f"Action handler already registered: {name}")
        self._registrations[name] = ActionRegistration(
            name=name,
            context_handler=context_handler,
            typed_handler=typed_handler,
            args_builder=args_builder,
        )

    def resolve(self, action_name: str) -> ActionRegistration | None:
        return self._registrations.get(str(action_name).lower())

    def build_args(self, action_name: str, params: dict[str, Any]) -> StepArgs | None:
        registration = self.resolve(action_name)
        if registration is None or registration.args_builder is None:
            return None
        return registration.args_builder(dict(params))

    def dispatch_step(self, step: Any) -> None:
        name = str(getattr(step, "name", "")).lower()
        args_model = getattr(step, "args_model", None)
        registration = self._require_registration(name)
        if args_model is not None and registration.typed_handler is not None:
            _invoke_typed_action_handler(registration.typed_handler, self._context, args_model)
            return
        if registration.context_handler is not None:
            payload = step.to_payload()
            params = {k: v for k, v in payload.items() if k not in {"action", "check"}}
            _invoke_context_action_handler(
                registration.context_handler,
                self._context,
                params,
                payload,
            )
            return
        raise ActionExecutionError(f"Unsupported action: {name}")

    def _require_registration(self, action_name: str) -> ActionRegistration:
        registration = self.resolve(action_name)
        if registration is None:
            raise ActionExecutionError(f"Unsupported action: {action_name}")
        return registration


class CheckRegistry:
    def __init__(self, context: CheckHandlerContextProtocol) -> None:
        self._context = context
        self._registrations: dict[str, CheckRegistration] = {}

    def register(
        self,
        check_name: str,
        *,
        context_handler: CheckHandler | None = None,
        typed_handler: TypedCheckHandler | None = None,
        args_builder: Callable[[dict[str, Any]], StepArgs | None] | None = None,
        namespace: str | None = None,
        override: bool = True,
    ) -> None:
        name = _qualify_registry_name(check_name, namespace=namespace)
        if not override and name in self._registrations:
            raise ValueError(f"Check handler already registered: {name}")
        self._registrations[name] = CheckRegistration(
            name=name,
            context_handler=context_handler,
            typed_handler=typed_handler,
            args_builder=args_builder,
        )

    def resolve(self, check_name: str) -> CheckRegistration | None:
        return self._registrations.get(str(check_name).lower())

    def build_args(self, check_name: str, params: dict[str, Any]) -> StepArgs | None:
        registration = self.resolve(check_name)
        if registration is None or registration.args_builder is None:
            return None
        return registration.args_builder(dict(params))

    def dispatch_step(self, step: Any) -> bool:
        name = str(getattr(step, "name", "")).lower()
        args_model = getattr(step, "args_model", None)
        registration = self._require_registration(name)
        if args_model is not None and registration.typed_handler is not None:
            return bool(
                _invoke_typed_check_handler(registration.typed_handler, self._context, args_model)
            )
        if registration.context_handler is not None:
            payload = step.to_payload()
            params = {k: v for k, v in payload.items() if k not in {"action", "check"}}
            return bool(
                _invoke_context_check_handler(
                    registration.context_handler,
                    self._context,
                    params,
                )
            )
        raise VerificationError(f"Unsupported verification check: {name}")

    def _require_registration(self, check_name: str) -> CheckRegistration:
        registration = self.resolve(check_name)
        if registration is None:
            raise VerificationError(f"Unsupported verification check: {check_name}")
        return registration


class ScopedActionRegistry:
    def __init__(
        self,
        registry: ActionRegistry,
        *,
        namespace: str | None = None,
        override: bool = False,
    ) -> None:
        self._registry = registry
        self._namespace = _normalize_namespace(namespace)
        self._override = bool(override)

    def register(self, action_name: str, **kwargs: Any) -> None:
        kwargs.setdefault("namespace", self._namespace)
        kwargs.setdefault("override", self._override)
        self._registry.register(action_name, **kwargs)


class ScopedCheckRegistry:
    def __init__(
        self,
        registry: CheckRegistry,
        *,
        namespace: str | None = None,
        override: bool = False,
    ) -> None:
        self._registry = registry
        self._namespace = _normalize_namespace(namespace)
        self._override = bool(override)

    def register(self, check_name: str, **kwargs: Any) -> None:
        kwargs.setdefault("namespace", self._namespace)
        kwargs.setdefault("override", self._override)
        self._registry.register(check_name, **kwargs)


def install_registry_plugins(
    action_registry: ActionRegistry,
    check_registry: CheckRegistry,
    plugins: tuple[object, ...] | list[object] | None,
) -> None:
    for plugin in tuple(plugins or ()):
        namespace = _normalize_namespace(getattr(plugin, "namespace", None))
        override = bool(getattr(plugin, "override", False))
        register_actions = getattr(plugin, "register_actions", None)
        if callable(register_actions):
            register_actions(
                ScopedActionRegistry(
                    action_registry,
                    namespace=namespace,
                    override=override,
                )
            )
        register_checks = getattr(plugin, "register_checks", None)
        if callable(register_checks):
            register_checks(
                ScopedCheckRegistry(
                    check_registry,
                    namespace=namespace,
                    override=override,
                )
            )


def _invoke_typed_action_handler(
    handler: TypedActionHandler,
    context: ActionHandlerContextProtocol,
    args: StepArgs,
) -> None:
    param_count = _callable_param_count(handler)
    if param_count == 1:
        handler(args)
        return
    handler(context, args)


def _invoke_typed_check_handler(
    handler: TypedCheckHandler,
    context: CheckHandlerContextProtocol,
    args: StepArgs,
) -> bool:
    param_count = _callable_param_count(handler)
    if param_count == 1:
        return bool(handler(args))
    return bool(handler(context, args))


def _callable_param_count(handler: Callable[..., Any]) -> int:
    try:
        return len(inspect.signature(handler).parameters)
    except (TypeError, ValueError):
        return 2


def _invoke_context_action_handler(
    handler: ActionHandler,
    context: ActionHandlerContextProtocol,
    params: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    param_count = _callable_param_count(handler)
    if param_count == 2:
        handler(params, payload)
        return
    handler(context, params, payload)


def _invoke_context_check_handler(
    handler: CheckHandler,
    context: CheckHandlerContextProtocol,
    params: dict[str, Any],
) -> bool:
    param_count = _callable_param_count(handler)
    if param_count == 1:
        return bool(handler(params))
    return bool(handler(context, params))


def _normalize_namespace(namespace: str | None) -> str | None:
    if namespace is None:
        return None
    text = str(namespace).strip().strip(".").lower()
    return text or None


def _qualify_registry_name(name: str, *, namespace: str | None = None) -> str:
    normalized_name = str(name).strip().lower()
    normalized_namespace = _normalize_namespace(namespace)
    if normalized_namespace and "." not in normalized_name:
        return f"{normalized_namespace}.{normalized_name}"
    return normalized_name
