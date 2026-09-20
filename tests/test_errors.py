from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from sogni_client.errors import (
    SUBSCRIPTION_ERROR_CODES,
    ApiError,
    ChatJobError,
    ProjectError,
    api_error_extras,
    extract_chat_job_error_fields,
    is_subscription_limit_error,
    parse_retry_after_header,
)


def test_api_error_preserves_http_status_payload_and_wire_code_aliases() -> None:
    error = ApiError(402, {"message": "Payment required", "errorCode": 4078})

    assert str(error) == "Payment required"
    assert error.status == 402
    assert error.payload == {"message": "Payment required", "errorCode": 4078}
    assert error.error_code == 4078
    assert error.errorCode == 4078


def test_api_error_falls_back_to_http_status() -> None:
    error = ApiError(503)

    assert str(error) == "HTTP 503"
    assert error.error_code == 503


NOW = datetime(2026, 9, 20, 22, 0, 0, tzinfo=timezone.utc)


def test_api_error_carries_the_servers_wait_and_details_from_the_body() -> None:
    payload = {
        "status": "error",
        "errorCode": 126,
        "message": "Creative workflow start rate limit exceeded.",
        "retryAfter": 1837,
        "details": {"retryAfterSeconds": 1837},
    }
    error = ApiError(429, payload)

    assert error.retry_after == error.retryAfter == 1837
    assert error.details == {"retryAfterSeconds": 1837}
    assert error.payload is payload


def test_api_error_body_wait_wins_over_the_header() -> None:
    error = ApiError(429, {"message": "slow down", "retryAfter": 30}, "120")

    assert error.retry_after == 30


def test_api_error_falls_back_to_the_retry_after_header() -> None:
    error = ApiError(503, {"message": "restarting"}, "10")

    assert error.retry_after == 10
    assert error.details is None


def test_api_error_without_a_wait_reports_none() -> None:
    error = ApiError(409, {"message": "Too many active creative workflows"})

    assert error.retry_after is None
    assert error.retryAfter is None
    assert error.details is None


def test_api_error_keeps_the_two_argument_constructor() -> None:
    assert ApiError(500).retry_after is None
    assert ApiError(500, None).details is None


def test_retry_after_header_accepts_delta_seconds() -> None:
    assert parse_retry_after_header("120") == 120
    assert parse_retry_after_header("  0 ") == 0


def test_retry_after_header_accepts_an_http_date_and_rounds_up() -> None:
    future = format_datetime(NOW + timedelta(seconds=90), usegmt=True)
    almost = NOW - timedelta(milliseconds=400)

    assert parse_retry_after_header(future, NOW) == 90
    # 90.4 seconds away: never tell a caller to come back early.
    assert parse_retry_after_header(future, almost) == 91


def test_retry_after_header_date_in_the_past_reads_as_zero() -> None:
    past = format_datetime(NOW - timedelta(hours=1), usegmt=True)

    assert parse_retry_after_header(past, NOW) == 0


@pytest.mark.parametrize(
    "value",
    ["1.5", "-5", "1e3", "", "   ", "soon", "12:00:00", "Sun", "٣٠", str(2**53), None, 30, 1.5],
)
def test_retry_after_header_ignores_anything_else(value: object) -> None:
    assert parse_retry_after_header(value, NOW) is None


@pytest.mark.parametrize("value", [-1, math.nan, math.inf, "30", True, None, [30]])
def test_body_wait_must_be_a_finite_non_negative_number(value: object) -> None:
    assert api_error_extras({"retryAfter": value}) == {}
    assert ApiError(429, {"message": "slow down", "retryAfter": value}).retry_after is None


def test_body_wait_accepts_zero_and_fractions() -> None:
    assert api_error_extras({"retryAfter": 0}) == {"retry_after": 0}
    assert ApiError(429, {"retryAfter": 2.5}).retry_after == 2.5


@pytest.mark.parametrize("value", ["capacity", 3, ["a"], None, True])
def test_details_that_are_not_an_object_are_ignored(value: object) -> None:
    assert ApiError(409, {"message": "busy", "details": value}).details is None


def test_api_error_extras_ignores_a_body_that_is_not_an_object() -> None:
    assert api_error_extras(None) == {}
    assert api_error_extras(["retryAfter"]) == {}
    assert api_error_extras("retryAfter") == {}


def test_chat_job_error_carries_the_wait_and_details() -> None:
    error = ChatJobError("slow down", status=429, retry_after=12, details={"limit": True})

    assert error.retry_after == error.retryAfter == 12
    assert error.details == {"limit": True}
    assert ChatJobError("nope").retry_after is None
    assert ChatJobError("nope", retry_after=-1, details="x").details is None  # type: ignore[arg-type]


def test_project_error_preserves_structured_failure() -> None:
    payload = {"code": 5003, "message": "Job timed out", "originalCode": "jobTimedOut"}
    error = ProjectError(payload)

    assert str(error) == "Job timed out"
    assert error.code == 5003
    assert error.error is payload


def test_chat_job_error_exposes_python_and_javascript_aliases() -> None:
    error = ChatJobError(
        "4K requires Unlimited Pro",
        code=4081,
        error_type="subscription_feature_unavailable",
        job_id="JOB-1",
        status=402,
        payload={"raw": True},
        subscription_limit=True,
        required_plans=["unlimited_pro"],
        feature="video_4k_render",
        limitation="4K video render requires Unlimited Pro",
    )

    assert error.code == "4081"
    assert error.error_code == error.errorCode == "4081"
    assert error.error_type == error.errorType == "subscription_feature_unavailable"
    assert error.job_id == error.jobID == "JOB-1"
    assert error.subscription_limit is error.subscriptionLimit is True
    assert error.required_plans == error.requiredPlans == ["unlimited_pro"]
    assert error.subscription_error_code == error.subscriptionErrorCode == 4081


@pytest.mark.parametrize("code", ["4078", "4079", "4080", "4081"])
def test_chat_job_error_recognizes_every_subscription_wire_code(code: str) -> None:
    assert ChatJobError("denied", code=code).subscription_error_code == int(code)


@pytest.mark.parametrize("code", [None, "", "not-a-number", "5000"])
def test_chat_job_error_ignores_non_subscription_codes(code: str | None) -> None:
    assert ChatJobError("failed", code=code).subscription_error_code is None


def test_extract_chat_job_error_fields_from_openai_envelope() -> None:
    assert extract_chat_job_error_fields(
        {
            "error": {
                "message": "Upgrade required",
                "type": "subscription_unavailable",
                "code": "4081",
                "subscription": {
                    "subscriptionLimit": True,
                    "requiredPlans": ["unlimited_pro", 123],
                    "feature": "video_4k_render",
                    "limitation": "4K is unavailable",
                },
            }
        }
    ) == {
        "code": "4081",
        "error_type": "subscription_unavailable",
        "message": "Upgrade required",
        "subscription_limit": True,
        "required_plans": ["unlimited_pro"],
        "feature": "video_4k_render",
        "limitation": "4K is unavailable",
    }


def test_extract_chat_job_error_fields_from_socket_shape() -> None:
    assert extract_chat_job_error_fields(
        {
            "error": "subscription_unavailable",
            "error_code": "4080",
            "error_message": "Provider is retrying renewal",
            "subscriptionLimit": False,
        }
    ) == {
        "code": "4080",
        "error_type": "subscription_unavailable",
        "message": "Provider is retrying renewal",
        "subscription_limit": False,
        "required_plans": None,
        "feature": None,
        "limitation": None,
    }


@pytest.mark.parametrize(
    "payload",
    [None, [], "error", {}, {"message": "generic"}, {"error": {"message": "generic"}}],
)
def test_extract_chat_job_error_fields_does_not_claim_generic_errors(payload: object) -> None:
    assert extract_chat_job_error_fields(payload) is None


def test_subscription_error_constants_match_javascript_contract() -> None:
    assert SUBSCRIPTION_ERROR_CODES == {
        "NOT_ENTITLED": 4078,
        "QUEUE_CAP": 4079,
        "GRACE_RETRY": 4080,
        "SUBSCRIPTION_FEATURE_REQUIRES_UPGRADE": 4081,
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (4081, True),
        ("4081", True),
        ({"code": 4081}, True),
        ({"code": "4081"}, True),
        ({"subscriptionLimit": True}, True),
        (ChatJobError("feature", code="4081"), True),
        # The JS helper is specifically a feature-limit predicate. Other
        # subscription billing denials are not plan-feature limits.
        (4078, False),
        (ChatJobError("not entitled", code="4078"), False),
        (ApiError(402, {"errorCode": 4080}), False),
        (None, False),
    ],
)
def test_is_subscription_limit_error_matches_feature_gate_semantics(
    error: object, expected: bool
) -> None:
    assert is_subscription_limit_error(error) is expected
