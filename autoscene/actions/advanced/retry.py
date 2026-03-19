from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


class RetryPolicy:
    def run_with_retry(
        self,
        operation: Callable[[], T],
        *,
        attempts: int = 1,
        retry_interval_seconds: float = 0.0,
        should_retry: Callable[[T], bool] | None = None,
    ) -> T:
        total_attempts = max(int(attempts), 1)
        retry_delay = max(float(retry_interval_seconds), 0.0)
        has_result = False
        last_result: T | None = None
        last_error: Exception | None = None

        for attempt in range(total_attempts):
            try:
                result = operation()
            except Exception as exc:
                last_error = exc
                if attempt < total_attempts - 1 and retry_delay > 0.0:
                    time.sleep(retry_delay)
                else:
                    raise
                continue

            has_result = True
            last_result = result
            if should_retry is None or not should_retry(result):
                return result
            if attempt < total_attempts - 1 and retry_delay > 0.0:
                time.sleep(retry_delay)

        if has_result:
            return last_result  # type: ignore[return-value]
        if last_error is not None:
            raise last_error
        raise AssertionError("Retry policy exhausted without a result.")
