from __future__ import annotations

CASE_STARTED = "case_started"
CASE_FAILED = "case_failed"
CASE_FINISHED = "case_finished"

BEFORE_STAGE = "before_stage"
AFTER_STAGE = "after_stage"
STAGE_FAILED = "stage_failed"

BEFORE_STEP = "before_step"
AFTER_STEP = "after_step"
STEP_FAILED = "step_failed"

BEFORE_STEP_ATTEMPT = "before_step_attempt"
STEP_ATTEMPT_FAILED = "step_attempt_failed"

CASE_LIFECYCLE_EVENTS = (
    CASE_STARTED,
    CASE_FAILED,
    CASE_FINISHED,
)

STAGE_EVENTS = (
    BEFORE_STAGE,
    AFTER_STAGE,
    STAGE_FAILED,
)

STEP_EVENTS = (
    BEFORE_STEP,
    AFTER_STEP,
    STEP_FAILED,
)

RETRY_EVENTS = (
    BEFORE_STEP_ATTEMPT,
    STEP_ATTEMPT_FAILED,
)

ALL_RUNTIME_EVENTS = (
    *CASE_LIFECYCLE_EVENTS,
    *STAGE_EVENTS,
    *STEP_EVENTS,
    *RETRY_EVENTS,
)

__all__ = [
    "AFTER_STAGE",
    "AFTER_STEP",
    "ALL_RUNTIME_EVENTS",
    "BEFORE_STAGE",
    "BEFORE_STEP",
    "BEFORE_STEP_ATTEMPT",
    "CASE_FAILED",
    "CASE_FINISHED",
    "CASE_LIFECYCLE_EVENTS",
    "CASE_STARTED",
    "RETRY_EVENTS",
    "STAGE_EVENTS",
    "STAGE_FAILED",
    "STEP_ATTEMPT_FAILED",
    "STEP_EVENTS",
    "STEP_FAILED",
]
