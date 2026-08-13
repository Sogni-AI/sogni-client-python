from __future__ import annotations

import pytest

from sogni_client.errors import (
    SUBSCRIPTION_ERROR_CODES,
    ApiError,
    ChatJobError,
    ProjectError,
    extract_chat_job_error_fields,
    is_subscription_limit_error,
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
