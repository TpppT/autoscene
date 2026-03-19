from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from autoscene.core.exceptions import ActionExecutionError, DependencyMissingError


class BrowserActionsMixin:
    def open_browser(
        self,
        url: str,
        browser: str = "chrome",
        browser_path: str | None = None,
        new_window: bool = True,
        args: Sequence[str] | None = None,
    ) -> None:
        command = self._resolve_browser_command(browser=browser, browser_path=browser_path)
        known_handles = self._list_window_handles()
        launch_args: list[str] = []
        browser_name = browser.lower()
        if new_window and browser_name in {"chrome", "google-chrome", "chromium"}:
            launch_args.append("--new-window")
        if args:
            launch_args.extend(str(arg) for arg in args)
        launch_args.append(url)
        self.logger.info("open_browser %s %s", browser_name, url)
        self._launch_process([command, *launch_args])
        self._bind_new_window(known_handles)

    @staticmethod
    def _launch_process(command: Sequence[str]) -> None:
        subprocess.Popen(list(command))

    def _list_window_handles(self) -> set[int]:
        try:
            pygetwindow = self._require_pygetwindow()
        except DependencyMissingError:
            return set()
        handles: set[int] = set()
        for window in pygetwindow.getAllWindows():
            handle = self._get_window_handle(window)
            if handle is not None:
                handles.add(handle)
        return handles

    def _bind_new_window(
        self,
        known_handles: set[int],
        timeout: float = 5.0,
        interval: float = 0.2,
    ) -> None:
        if not known_handles and not hasattr(self.capture_engine, "bind_window_handle"):
            return
        try:
            pygetwindow = self._require_pygetwindow()
        except DependencyMissingError:
            return
        deadline = time.time() + max(timeout, 0.0)
        while True:
            for window in pygetwindow.getAllWindows():
                handle = self._get_window_handle(window)
                if handle is None or handle in known_handles:
                    continue
                self._bind_window(window)
                return
            if time.time() >= deadline:
                break
            time.sleep(max(interval, 0.0))

    @staticmethod
    def _resolve_browser_command(browser: str, browser_path: str | None = None) -> str:
        if browser_path:
            resolved = shutil.which(browser_path)
            if resolved:
                return resolved
            candidate = Path(browser_path)
            if candidate.exists():
                return str(candidate)
            raise ActionExecutionError(f"Browser executable not found: {browser_path}")

        browser_name = browser.lower()
        if browser_name == "chrome":
            candidates = [
                "chrome",
                "chrome.exe",
                "google-chrome",
                "google-chrome-stable",
                "chromium",
                "chromium-browser",
            ]
            env_paths = [
                Path(os.environ[key]) / "Google/Chrome/Application/chrome.exe"
                for key in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData")
                if os.environ.get(key)
            ]
            for candidate in candidates:
                resolved = shutil.which(candidate)
                if resolved:
                    return resolved
            for candidate in env_paths:
                if candidate.exists():
                    return str(candidate)
            raise ActionExecutionError(
                "Google Chrome was not found. Set 'browser_path' in the action."
            )

        resolved = shutil.which(browser_name)
        if resolved:
            return resolved
        raise ActionExecutionError(
            f"Browser '{browser}' was not found. Set 'browser_path' in the action."
        )
