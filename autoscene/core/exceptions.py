class FrameworkError(Exception):
    """Base framework exception."""


class DependencyMissingError(FrameworkError):
    """Raised when an optional dependency is not installed."""


class ActionExecutionError(FrameworkError):
    """Raised when an action cannot be executed."""


class VerificationError(FrameworkError):
    """Raised when a verification step fails."""

