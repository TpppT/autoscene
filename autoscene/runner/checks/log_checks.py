from __future__ import annotations

import logging
import re
import time
from typing import Any

from autoscene.core.exceptions import VerificationError
from autoscene.logs.interfaces import LogSource
from autoscene.runner.step_specs import LogContainsCheckArgs, WaitForLogCheckArgs


class LogChecks:
    def __init__(
        self,
        log_sources: dict[str, LogSource] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.log_sources = dict(log_sources or {})
        self.logger = (
            logging.getLogger(self.__class__.__name__)
            if logger is None
            else logger.getChild(self.__class__.__name__)
        )
        self.handlers = {
            "log_contains": self._handle_log_contains,
            "wait_for_log": self._handle_wait_for_log,
        }
        self.typed_handlers = {
            "log_contains": self._handle_log_contains_typed,
            "wait_for_log": self._handle_wait_for_log_typed,
        }

    def _handle_log_contains(self, params: dict[str, Any]) -> bool:
        source_name, source = self._resolve_source(params)
        content = source.read_text()
        matched = self._matches(content, params)
        self.logger.info(
            "log_contains source=%s matcher=%s content_length=%s result=%s",
            source_name,
            self._describe_match(params),
            len(content),
            "passed" if matched else "failed",
        )
        return matched

    def _handle_log_contains_typed(self, args: LogContainsCheckArgs) -> bool:
        params = args.to_payload()
        source_name, source = self._resolve_source({"source": args.source})
        content = source.read_text()
        matched = self._matches(content, params)
        self.logger.info(
            "log_contains source=%s matcher=%s content_length=%s result=%s",
            source_name,
            self._describe_match(params),
            len(content),
            "passed" if matched else "failed",
        )
        return matched

    def _handle_wait_for_log(self, params: dict[str, Any]) -> bool:
        deadline = time.time() + float(params.get("timeout", 10.0))
        interval = float(params.get("interval", 0.5))
        source_name, _ = self._resolve_source(params)
        self.logger.info(
            "wait_for_log source=%s matcher=%s timeout=%.3f interval=%.3f",
            source_name,
            self._describe_match(params),
            float(params.get("timeout", 10.0)),
            interval,
        )
        while time.time() < deadline:
            if self._handle_log_contains(params):
                return True
            time.sleep(interval)
        self.logger.info(
            "wait_for_log source=%s matcher=%s result=failed",
            source_name,
            self._describe_match(params),
        )
        return False

    def _handle_wait_for_log_typed(self, args: WaitForLogCheckArgs) -> bool:
        params = args.to_payload()
        deadline = time.time() + float(args.timeout)
        interval = float(args.interval)
        source_name, _ = self._resolve_source({"source": args.source})
        self.logger.info(
            "wait_for_log source=%s matcher=%s timeout=%.3f interval=%.3f",
            source_name,
            self._describe_match(params),
            float(args.timeout),
            interval,
        )
        while time.time() < deadline:
            if self._handle_log_contains_typed(
                LogContainsCheckArgs(
                    source=args.source,
                    contains=args.contains,
                    text=args.text,
                    regex=args.regex,
                    ignore_case=args.ignore_case,
                )
            ):
                return True
            time.sleep(interval)
        self.logger.info(
            "wait_for_log source=%s matcher=%s result=failed",
            source_name,
            self._describe_match(params),
        )
        return False

    def _resolve_source(self, params: dict[str, Any]) -> tuple[str, LogSource]:
        if not self.log_sources:
            raise VerificationError(
                "No log sources configured. Add 'log_sources' to the test case."
            )
        raw_source = params.get("source")
        if raw_source is None:
            if len(self.log_sources) == 1:
                return next(iter(self.log_sources.items()))
            raise VerificationError(
                "Log checks require 'source' when multiple log_sources are configured."
            )
        source_name = str(raw_source)
        source = self.log_sources.get(source_name)
        if source is None:
            available = ", ".join(sorted(self.log_sources))
            raise VerificationError(
                f"Unknown log source '{source_name}'. Available: {available}"
            )
        return source_name, source

    @staticmethod
    def _describe_match(params: dict[str, Any]) -> str:
        if "regex" in params and params.get("regex") is not None:
            return f"regex={params['regex']!r}"
        if "contains" in params and params.get("contains") is not None:
            return f"contains={params['contains']!r}"
        if "text" in params and params.get("text") is not None:
            return f"text={params['text']!r}"
        return "unknown"

    @staticmethod
    def _matches(content: str, params: dict[str, Any]) -> bool:
        if "regex" in params and params.get("regex") is not None:
            flags = re.IGNORECASE if bool(params.get("ignore_case", False)) else 0
            return re.search(str(params["regex"]), content, flags=flags) is not None

        expected = params.get("contains", params.get("text"))
        if expected is None:
            raise VerificationError(
                "Log checks require 'contains', 'text', or 'regex'."
            )
        haystack = str(content)
        needle = str(expected)
        if bool(params.get("ignore_case", False)):
            haystack = haystack.casefold()
            needle = needle.casefold()
        return needle in haystack
