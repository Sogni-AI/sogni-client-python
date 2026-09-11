"""Public exception and error helpers."""

from __future__ import annotations

from typing import Any

SUBSCRIPTION_ERROR_CODES = {
    "NOT_ENTITLED": 4078,
    "QUEUE_CAP": 4079,
    "GRACE_RETRY": 4080,
    "SUBSCRIPTION_FEATURE_REQUIRES_UPGRADE": 4081,
}

# Error types for an LLM request that did not complete because the connection to
# Sogni was interrupted, not because of the request itself:
#
# - ``server_restarting``: the socket server restarted (a platform release) and
#   refunded the request, or refused it while shutting down.
# - ``transport_lost``: the request could not be sent, or it was in flight when
#   the socket dropped and the server no longer had it after reconnecting.
#
# Send the request again as a new request. The SDK waits for the reconnect before
# sending, so an immediate retry is fine.
RETRYABLE_CHAT_ERROR_TYPES: tuple[str, ...] = ("server_restarting", "transport_lost")


class SogniError(Exception):
    """Base exception for the Python SDK."""


class ApiError(SogniError):
    """A non-successful HTTP response from a Sogni endpoint."""

    def __init__(self, status: int, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        self.status = status
        self.payload = payload
        self.error_code = payload.get("errorCode", payload.get("error_code", status))
        self.errorCode = self.error_code
        super().__init__(str(payload.get("message") or f"HTTP {status}"))


class ProjectError(SogniError):
    """A generation project failed after it was accepted."""

    def __init__(self, error: dict[str, Any]) -> None:
        self.error = error
        self.code = error.get("code")
        super().__init__(str(error.get("message") or "Project failed"))


class ChatJobError(SogniError):
    """Chat failure preserving the structured socket/REST error contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str | int | None = None,
        error_type: str | None = None,
        job_id: str | None = None,
        status: int | None = None,
        payload: Any = None,
        subscription_limit: bool | None = None,
        required_plans: list[str] | None = None,
        feature: str | None = None,
        limitation: str | None = None,
    ) -> None:
        self.code = str(code) if code is not None else None
        self.error_code = self.code
        self.errorCode = self.code
        self.error_type = error_type
        self.errorType = error_type
        self.job_id = job_id
        self.jobID = job_id
        self.status = status
        self.payload = payload
        self.subscription_limit = subscription_limit
        self.subscriptionLimit = subscription_limit
        self.required_plans = required_plans
        self.requiredPlans = required_plans
        self.feature = feature
        self.limitation = limitation
        super().__init__(message)

    @property
    def retryable(self) -> bool:
        """``True`` when the connection interrupted the request rather than
        rejecting it, so sending it again is expected to work. See
        :data:`RETRYABLE_CHAT_ERROR_TYPES`."""

        return bool(self.error_type) and self.error_type in RETRYABLE_CHAT_ERROR_TYPES

    @property
    def subscription_error_code(self) -> int | None:
        if self.code is None:
            return None
        try:
            value = int(self.code)
        except ValueError:
            return None
        return value if value in SUBSCRIPTION_ERROR_CODES.values() else None

    @property
    def subscriptionErrorCode(self) -> int | None:
        return self.subscription_error_code


def is_retryable_chat_error(error: Any) -> bool:
    """Whether ``error`` is a chat/LLM failure caused by the connection (see
    :data:`RETRYABLE_CHAT_ERROR_TYPES`) that is safe to send again."""

    if isinstance(error, ChatJobError):
        return error.retryable
    if isinstance(error, dict):
        error_type = error.get("error_type", error.get("errorType"))
    else:
        error_type = getattr(error, "error_type", getattr(error, "errorType", None))
    return isinstance(error_type, str) and error_type in RETRYABLE_CHAT_ERROR_TYPES


isRetryableChatError = is_retryable_chat_error


def extract_chat_job_error_fields(payload: Any) -> dict[str, Any] | None:
    """Recognize OpenAI-style and socket-style chat error payloads."""

    if not isinstance(payload, dict):
        return None

    def structured(source: dict[str, Any]) -> dict[str, Any]:
        return {
            "subscription_limit": source.get("subscriptionLimit") is True,
            "required_plans": [p for p in source.get("requiredPlans", []) if isinstance(p, str)]
            or None,
            "feature": source.get("feature") if isinstance(source.get("feature"), str) else None,
            "limitation": (
                source.get("limitation") if isinstance(source.get("limitation"), str) else None
            ),
        }

    envelope = payload.get("error")
    if isinstance(envelope, dict):
        extra = structured(envelope.get("subscription", {}))
        code = envelope.get("code")
        error_type = envelope.get("type")
        if code is not None or error_type is not None or extra["subscription_limit"]:
            return {
                "code": code,
                "error_type": error_type,
                "message": envelope.get("message"),
                **extra,
            }
        return None

    extra = structured(payload)
    code = payload.get("error_code")
    error_type = payload.get("error")
    message = payload.get("error_message")
    if (
        code is not None
        or (isinstance(error_type, str) and isinstance(message, str))
        or extra["subscription_limit"]
    ):
        return {"code": code, "error_type": error_type, "message": message, **extra}
    return None


def is_subscription_limit_error(error: Any) -> bool:
    feature_code = SUBSCRIPTION_ERROR_CODES["SUBSCRIPTION_FEATURE_REQUIRES_UPGRADE"]

    def matches(code: Any) -> bool:
        if isinstance(code, bool):
            return False
        if isinstance(code, (int, float, str)):
            try:
                return float(code) == feature_code
            except ValueError:
                return False
        return False

    if matches(error):
        return True
    if isinstance(error, dict):
        return error.get("subscriptionLimit") is True or matches(error.get("code"))
    return getattr(error, "subscriptionLimit", False) is True or matches(
        getattr(error, "code", None)
    )


isSubscriptionLimitError = is_subscription_limit_error
