from __future__ import annotations

from typing import Any

from autoscene.core.exceptions import VerificationError
from autoscene.runner.step_specs import (
    ObjectExistsCheckArgs,
    TextExistsCheckArgs,
    WaitForTextCheckArgs,
    parse_object_locate_spec,
    parse_text_locate_spec,
)


class BasicUIChecks:
    def __init__(
        self,
        locate_actions: object | None = None,
    ) -> None:
        self.locate_actions = locate_actions
        self.handlers = {
            "text_exists": self._handle_text_exists,
            "object_exists": self._handle_object_exists,
            "wait_for_text": self._handle_wait_for_text,
        }
        self.typed_handlers = {
            "text_exists": self._handle_text_exists_typed,
            "object_exists": self._handle_object_exists_typed,
            "wait_for_text": self._handle_wait_for_text_typed,
        }

    def _handle_text_exists(self, params: dict[str, Any]) -> bool:
        return self._require_locate_actions().verify_text_exists(
            self._parse_text_locate(params.get("locate"))
        )

    def _handle_text_exists_typed(self, args: TextExistsCheckArgs) -> bool:
        return self._require_locate_actions().verify_text_exists(args.locate)

    def _handle_object_exists(self, params: dict[str, Any]) -> bool:
        return self._require_locate_actions().verify_object_exists(
            self._parse_object_locate(params.get("locate"))
        )

    def _handle_object_exists_typed(self, args: ObjectExistsCheckArgs) -> bool:
        return self._require_locate_actions().verify_object_exists(args.locate)

    def _handle_wait_for_text(self, params: dict[str, Any]) -> bool:
        return self._require_locate_actions().wait_for_text(
            self._parse_text_locate(params.get("locate")),
            timeout=float(params.get("timeout", 10.0)),
            interval=float(params.get("interval", 0.5)),
        )

    def _handle_wait_for_text_typed(self, args: WaitForTextCheckArgs) -> bool:
        return self._require_locate_actions().wait_for_text(
            args.locate,
            timeout=float(args.timeout),
            interval=float(args.interval),
        )

    def _require_locate_actions(self):
        if self.locate_actions is None:
            raise VerificationError("Locate check service is not configured.")
        return self.locate_actions

    @staticmethod
    def _parse_object_locate(value: Any):
        try:
            return parse_object_locate_spec(value, field_name="locate")
        except ValueError as exc:
            raise VerificationError(str(exc)) from exc

    @staticmethod
    def _parse_text_locate(value: Any):
        try:
            return parse_text_locate_spec(value, field_name="locate")
        except ValueError as exc:
            raise VerificationError(str(exc)) from exc
