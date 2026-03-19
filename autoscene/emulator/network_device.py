from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from autoscene.emulator.base import EmulatorAdapter


class NetworkDeviceEmulatorAdapter(EmulatorAdapter):
    """
    Simulates an external device that pushes events to a backend service.
    """

    def __init__(
        self,
        base_url: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        scheme: str = "http",
        default_endpoint: str = "/device/event",
        timeout_seconds: float = 8.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = (base_url or f"{scheme}://{host}:{port}").rstrip("/")
        self._default_endpoint = default_endpoint
        self._timeout_seconds = timeout_seconds
        self._default_headers = default_headers or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, command: str) -> str:
        # Compatibility for emulator_command action.
        return self.send({"command": command})

    def send(
        self,
        payload: Any,
        endpoint: str | None = None,
        method: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        target_endpoint = endpoint or self._default_endpoint
        target_url = urljoin(f"{self._base_url}/", target_endpoint.lstrip("/"))
        request_method = (method or "POST").upper()
        merged_headers = {**self._default_headers, **(headers or {})}

        if isinstance(payload, (dict, list)):
            body = json.dumps(payload).encode("utf-8")
            merged_headers.setdefault("Content-Type", "application/json")
        elif payload is None:
            body = b""
        else:
            body = str(payload).encode("utf-8")
            merged_headers.setdefault("Content-Type", "text/plain; charset=utf-8")

        req = Request(
            url=target_url,
            data=body,
            headers=merged_headers,
            method=request_method,
        )
        self.logger.info(
            "send device event method=%s url=%s payload=%s",
            request_method,
            target_url,
            payload,
        )
        try:
            with urlopen(req, timeout=self._timeout_seconds) as resp:
                content = resp.read().decode("utf-8", errors="replace")
                self.logger.info("server response status=%s body=%s", resp.status, content)
                return content
        except HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP error while sending device event: {exc.code} {body_text}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Network error while sending device event: {exc}") from exc

