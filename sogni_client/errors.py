"""Public exception and error helpers."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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


# Optional fields a Sogni REST error may carry next to ``message`` and
# ``errorCode``. Mirrors sogni-client's ``lib/apiErrorFields.ts`` so both clients
# read a wait, and refuse a malformed one, the same way.
_DELTA_SECONDS = re.compile(r"^\d+$", re.ASCII)
_HTTP_DATE_TIME = re.compile(r"\d{2}:\d{2}:\d{2}", re.ASCII)
_HTTP_DATE_WORD = re.compile(r"[A-Za-z]{3}")
# JavaScript's Number.MAX_SAFE_INTEGER: the TypeScript client ignores a larger
# delta-seconds value, so this one does too.
_MAX_SAFE_INTEGER = 2**53 - 1


def normalize_retry_after_seconds(value: Any) -> int | float | None:
    """A wait in seconds, or ``None`` when the value is not a usable one."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def parse_retry_after_header(value: Any, now: datetime | None = None) -> int | None:
    """Parse an HTTP ``Retry-After`` header into whole seconds from now.

    Accepts both forms the header allows, delta-seconds (``"120"``) and an
    HTTP-date, and returns ``None`` for anything else. A date already in the
    past reads as ``0``.
    """

    if not isinstance(value, str):
        return None
    header = value.strip()
    if not header:
        return None
    if _DELTA_SECONDS.match(header):
        seconds = int(header)
        return seconds if seconds <= _MAX_SAFE_INTEGER else None
    # Only hand the date parser something shaped like an HTTP-date: a day or
    # month word and an HH:MM:SS time. "1.5", "-5" and "1e3" are not waits.
    if not _HTTP_DATE_WORD.search(header) or not _HTTP_DATE_TIME.search(header):
        return None
    try:
        at = parsedate_to_datetime(header)
    except (TypeError, ValueError, IndexError):
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0, math.ceil((at - current).total_seconds()))


def normalize_error_details(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def api_error_extras(body: Any) -> dict[str, Any]:
    """The optional fields of an error body, validated; absent fields are omitted."""

    if not isinstance(body, dict):
        return {}
    extras: dict[str, Any] = {}
    retry_after = normalize_retry_after_seconds(body.get("retryAfter"))
    if retry_after is not None:
        extras["retry_after"] = retry_after
    details = normalize_error_details(body.get("details"))
    if details is not None:
        extras["details"] = details
    return extras


class SogniError(Exception):
    """Base exception for the Python SDK."""


class ApiError(SogniError):
    """A non-successful HTTP response from a Sogni endpoint.

    ``status`` is the HTTP status and ``payload`` the error body. When the server
    says how long to wait (a ``429``, or a ``503`` during a restart),
    ``retry_after`` carries that wait in seconds, taken from the body or, failing
    that, from the ``Retry-After`` header. Wait at least that long before
    retrying: a request sent sooner is refused again. ``details`` carries any
    structured context the server attached (for example the counts behind a
    capacity refusal). Both are ``None`` when the server sent neither.

    .. code-block:: python

        try:
            await sogni.workflows.start(input=plan, idempotency_key=key)
        except ApiError as error:
            if error.retry_after is not None:
                await asyncio.sleep(error.retry_after)
                # ...then retry, reusing the same idempotency_key.
    """

    def __init__(
        self,
        status: int,
        payload: dict[str, Any] | None = None,
        retry_after_header: str | None = None,
    ) -> None:
        payload = payload or {}
        self.status = status
        self.payload = payload
        self.error_code = payload.get("errorCode", payload.get("error_code", status))
        self.errorCode = self.error_code
        extras = api_error_extras(payload)
        retry_after = extras.get("retry_after")
        if retry_after is None:
            retry_after = parse_retry_after_header(retry_after_header)
        self.retry_after: int | float | None = retry_after
        self.retryAfter = self.retry_after
        self.details: dict[str, Any] | None = extras.get("details")
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
        retry_after: int | float | None = None,
        details: dict[str, Any] | None = None,
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
        # Same contract as ApiError: the server's wait in seconds and any
        # structured context it attached. Set only for REST-originated errors.
        self.retry_after = normalize_retry_after_seconds(retry_after)
        self.retryAfter = self.retry_after
        self.details = normalize_error_details(details)
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
