"""projects.get_result and projects.list_recent.

An agent or app that was offline, or stopped waiting, can still learn what
happened to its projects and fetch their media, including after the socket
stopped holding them (one hour). Ported from sogni-client's
``scripts/check-project-results.cjs``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import unquote

import pytest

from sogni_client.errors import ApiError
from sogni_client.events import EventEmitter
from sogni_client.projects import ProjectsApi

IMAGE_PATH = "/v1/image/downloadUrl"
MEDIA_PATH = "/v1/media/downloadUrl"
HISTORY_PATH = "/v1/jobs/list"
DAY_MS = 24 * 3600 * 1000


def not_found() -> ApiError:
    return ApiError(404, {"status": "error", "message": "Not Found", "errorCode": 404})


class ResultsRest:
    """Answers the live lookup, both download endpoints and the job history."""

    def __init__(
        self,
        status: dict[str, dict[str, Any]],
        history: list[dict[str, Any]],
        mint: Callable[[str, dict[str, Any]], Any] | None,
    ) -> None:
        self.status = status
        self.history = history
        self.mint = mint
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.before_answer: Callable[[str], None] | None = None

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        params = dict(params or {})
        self.calls.append((path, params))
        if self.before_answer is not None:
            self.before_answer(path)
        if path.startswith("/v2/projects/"):
            project = self.status.get(unquote(path[len("/v2/projects/") :]))
            if project is None:
                raise not_found()
            return {"status": "success", "data": {"project": project}}
        if path in {IMAGE_PATH, MEDIA_PATH}:
            if self.mint is not None:
                return self.mint(path, params)
            result_id = params.get("imageId") or params.get("id")
            url = f"https://cdn.test{path}/{params['jobId']}/{result_id}"
            return {"status": "success", "data": {"downloadUrl": url}}
        if path == HISTORY_PATH:
            return {"status": "success", "data": {"jobs": self.history, "next": None}}
        raise not_found()


class Client(EventEmitter):
    def __init__(self, rest: ResultsRest) -> None:
        super().__init__()
        self.rest = rest
        self.socket = EventEmitter()
        self.app_source = "pytest"


def make_api(
    *,
    status: dict[str, dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
    mint: Callable[[str, dict[str, Any]], Any] | None = None,
) -> tuple[ProjectsApi, ResultsRest]:
    rest = ResultsRest(status or {}, history or [], mint)
    return ProjectsApi(Client(rest)), rest


def paths(rest: ResultsRest) -> list[str]:
    return [path for path, _params in rest.calls]


def history_query(rest: ResultsRest) -> dict[str, Any]:
    return next(params for path, params in rest.calls if path == HISTORY_PATH)


def signed_in(api: ProjectsApi, address: str | None = "0xabc") -> None:
    async def resolve() -> str | None:
        return address

    api._set_account_address_resolver(resolve)


async def test_finished_video_project_signs_media_urls_and_keeps_failure_reasons() -> None:
    # Completed renders get media URLs, the withheld one says why it has none,
    # failed and cancelled renders keep their reason.
    api, rest = make_api(
        status={
            "P1": {
                "id": "P1",
                "status": "completed",
                "finished": True,
                "model": {"id": "minimax-h3-ref2va-fp8_r2v", "name": "MiniMax H3", "type": "video"},
                "workerJobs": [],
                "completedWorkerJobs": [
                    {
                        "id": "P1-0",
                        "imgID": "IMG-A",
                        "status": "jobCompleted",
                        "reason": "allJobsCompleted",
                        "seedUsed": 42,
                        "outputFormat": "mp4",
                    },
                    {
                        "id": "P1-1",
                        "imgID": "IMG-B",
                        "status": "jobCompleted",
                        "triggeredNSFWFilter": True,
                    },
                    {"id": "P1-2", "imgID": "IMG-C", "status": "jobError", "reason": "genfailure"},
                    {
                        "id": "P1-3",
                        "imgID": "IMG-D",
                        "status": "jobError",
                        "reason": "artistCanceled",
                    },
                ],
            }
        }
    )

    result = await api.get_result("P1")

    assert result["id"] == "P1"
    assert result["finished"] is True
    assert result["modelId"] == "minimax-h3-ref2va-fp8_r2v"
    assert "waitingReason" not in result
    assert result["jobs"] == [
        {
            "id": "IMG-A",
            "status": "completed",
            "seed": 42,
            "kind": "video",
            "url": "https://cdn.test/v1/media/downloadUrl/P1/IMG-A",
        },
        {"id": "IMG-B", "status": "completed", "urlUnavailable": "sensitiveContent"},
        {"id": "IMG-C", "status": "failed", "reason": "genfailure"},
        {"id": "IMG-D", "status": "canceled", "reason": "artistCanceled"},
    ]
    assert IMAGE_PATH not in paths(rest), "video never goes to the image endpoint"
    assert (MEDIA_PATH, {"jobId": "P1", "id": "IMG-A", "type": "complete"}) in rest.calls
    assert api.getResult == api.get_result


async def test_labelled_but_delivered_media_is_signed_like_a_live_result() -> None:
    api, _ = make_api(
        status={
            "P1": {
                "id": "P1",
                "status": "completed",
                "finished": True,
                "model": {"id": "z_image_turbo_bf16", "type": "image"},
                "workerJobs": [],
                "completedWorkerJobs": [
                    {
                        "id": "P1-0",
                        "imgID": "IMG-A",
                        "status": "jobCompleted",
                        "triggeredNSFWFilter": True,
                        "nsfwDetected": True,
                    }
                ],
            }
        }
    )

    result = await api.get_result("P1")

    assert result["jobs"] == [
        {
            "id": "IMG-A",
            "status": "completed",
            "kind": "image",
            "url": "https://cdn.test/v1/image/downloadUrl/P1/IMG-A",
        }
    ]


async def test_queued_project_carries_the_servers_waiting_reason() -> None:
    # No URLs; the server's reason for the wait comes through unchanged.
    waiting_reason = {
        "reason": "model_concurrency_limit",
        "message": "Your plan runs one MiniMax H3 video at a time",
        "modelFamily": "minimaxH3",
    }
    api, rest = make_api(
        status={
            "P2": {
                "id": "P2",
                "status": "queued",
                "finished": False,
                "waitingReason": waiting_reason,
                "workerJobs": [{"id": "P2-0", "imgID": "IMG-Q", "status": "queued"}],
                "completedWorkerJobs": [],
            }
        }
    )

    result = await api.get_result("P2")

    assert result["finished"] is False
    assert result["status"] == "queued"
    assert result["waitingReason"] == waiting_reason
    assert result["jobs"] == [{"id": "IMG-Q", "status": "queued"}]
    assert paths(rest) == ["/v2/projects/P2"]


async def test_a_null_waiting_reason_is_reported_as_null() -> None:
    api, _ = make_api(
        status={
            "P2": {
                "id": "P2",
                "status": "processing",
                "finished": False,
                "waitingReason": None,
                "workerJobs": [{"id": "P2-0", "status": "jobStarted"}],
                "completedWorkerJobs": [],
            }
        }
    )

    result = await api.get_result("P2")

    assert result["waitingReason"] is None
    assert result["jobs"] == [{"id": "P2-0", "status": "jobStarted"}]


UNKNOWN_MODEL_PROJECT: dict[str, Any] = {
    "id": "P3",
    "status": "completed",
    "finished": True,
    "model": {"id": "some-new-model", "name": "New"},
    "workerJobs": [],
    "completedWorkerJobs": [
        {
            "id": "P3-0",
            "imgID": "IMG-1",
            "status": "jobCompleted",
            "resultUrl": "https://vendor.test/out.mp4",
        },
        {"id": "P3-1", "imgID": "IMG-2", "status": "jobCompleted"},
    ],
}


async def test_stored_url_is_used_and_an_unknown_kind_is_never_guessed() -> None:
    api, rest = make_api(status={"P3": UNKNOWN_MODEL_PROJECT})

    unknown = await api.get_result("P3")

    assert unknown["jobs"][0] == {
        "id": "IMG-1",
        "status": "completed",
        "url": "https://vendor.test/out.mp4",
    }
    assert unknown["jobs"][1] == {
        "id": "IMG-2",
        "status": "completed",
        "urlUnavailable": "unknownMediaKind",
    }
    assert IMAGE_PATH not in paths(rest) and MEDIA_PATH not in paths(rest)


async def test_the_callers_kind_hint_covers_a_model_the_catalog_does_not_know() -> None:
    api, _ = make_api(status={"P3": UNKNOWN_MODEL_PROJECT})

    hinted = await api.get_result("P3", kind="image")

    assert hinted["jobs"][1] == {
        "id": "IMG-2",
        "status": "completed",
        "kind": "image",
        "url": "https://cdn.test/v1/image/downloadUrl/P3/IMG-2",
    }


async def test_the_kind_hint_never_overrides_what_the_model_or_result_says() -> None:
    project = {
        "id": "P4",
        "status": "completed",
        "finished": True,
        "model": {"id": "some-new-model"},
        "workerJobs": [],
        "completedWorkerJobs": [
            {
                "id": "P4-0",
                "imgID": "IMG-1",
                "status": "jobCompleted",
                "result": {"artifacts": [{"contentType": "audio/wav", "success": True}]},
            }
        ],
    }
    api, rest = make_api(status={"P4": project})

    result = await api.get_result("P4", kind="image")

    assert result["jobs"][0]["kind"] == "audio"
    assert (
        MEDIA_PATH,
        {"jobId": "P4", "id": "IMG-1", "type": "complete", "contentType": "audio/wav"},
    ) in rest.calls
    assert IMAGE_PATH not in paths(rest)


async def test_an_invalid_kind_hint_fails_before_anything_is_sent() -> None:
    api, rest = make_api(status={"P3": UNKNOWN_MODEL_PROJECT})

    with pytest.raises(ValueError, match=r'Invalid kind vid\. Must be one of "image"'):
        await api.get_result("P3", kind="vid")

    assert rest.calls == []


async def test_a_signing_failure_is_reported_on_the_render_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def refuse(_path: str, _params: dict[str, Any]) -> Any:
        raise ApiError(503, {"status": "error", "message": "Service Unavailable"})

    api, _ = make_api(status={"P3": UNKNOWN_MODEL_PROJECT}, mint=refuse)

    with caplog.at_level(logging.ERROR, logger="sogni_client"):
        failed = await api.get_result("P3", kind="image")

    assert failed["jobs"][1] == {
        "id": "IMG-2",
        "status": "completed",
        "kind": "image",
        "urlUnavailable": "downloadUrlFailed",
    }
    assert "Failed to sign a download URL for P3/IMG-2" in caplog.text


async def test_an_image_result_the_api_says_is_media_is_signed_from_the_media_endpoint() -> None:
    def respond(path: str, params: dict[str, Any]) -> Any:
        if path == IMAGE_PATH:
            raise ApiError(
                404,
                {
                    "status": "error",
                    "errorCode": 122,
                    "message": (
                        "This result is media, not an image; request it from /v1/media/downloadUrl"
                    ),
                },
            )
        return {"status": "success", "data": {"downloadUrl": f"https://cdn.test/{params['id']}"}}

    api, rest = make_api(status={"P3": UNKNOWN_MODEL_PROJECT}, mint=respond)

    result = await api.get_result("P3", kind="image")

    assert result["jobs"][1]["url"] == "https://cdn.test/IMG-2"
    assert paths(rest) == ["/v2/projects/P3", IMAGE_PATH, MEDIA_PATH]


async def test_another_accounts_or_an_unknown_project_raises_the_apis_404() -> None:
    api, _ = make_api()

    with pytest.raises(ApiError) as caught:
        await api.get_result("nope")

    assert caught.value.status == 404


async def test_an_account_change_during_the_lookup_discards_the_answer() -> None:
    api, rest = make_api(status={"P3": UNKNOWN_MODEL_PROJECT})
    rest.before_answer = lambda _path: api.client.emit("sessionChanged", None)

    with pytest.raises(RuntimeError, match="The account changed"):
        await api.get_result("P3", kind="image")

    assert paths(rest) == ["/v2/projects/P3"]


def history_record(
    job_id: str, project: dict[str, Any] | None, end_time: int, **extra: Any
) -> dict[str, Any]:
    record: dict[str, Any] = {"id": job_id, "status": "jobCompleted", "endTime": end_time, **extra}
    if project is not None:
        record["parentRequest"] = project
    return record


async def test_list_recent_reads_the_durable_history_and_groups_renders_by_project() -> None:
    # Clamps the window to 7 days and the limit to 100, like the TypeScript SDK.
    now = int(time.time() * 1000)
    history = [
        history_record(
            "A-0",
            {
                "id": "A",
                "appSource": "sogni-creative-agent-skill",
                "model": {"id": "minimax-h3-ref2va-fp8_r2v", "name": "H3"},
            },
            now - 5000,
            imgID="IMG-A0",
        ),
        history_record(
            "B-0",
            {"id": "B", "model": {"id": "flux1-schnell-fp8", "name": "Schnell"}},
            now - 1000,
            imgID="IMG-B0",
        ),
        history_record(
            "A-1",
            {"id": "A", "model": {"id": "minimax-h3-ref2va-fp8_r2v"}},
            now - 3000,
            imgID="IMG-A1",
            triggeredNSFWFilter=True,
        ),
        history_record("orphan", None, now),
    ]
    api, rest = make_api(history=history)
    signed_in(api)

    recent = await api.list_recent(
        since=now - 30 * DAY_MS, limit=500, app_source="sogni-creative-agent-skill"
    )

    query = history_query(rest)
    assert query["role"] == "artist"
    assert query["address"] == "0xabc"
    assert query["state"] == "completed"
    assert query["mediaOnly"] is True
    assert query["limit"] == 100
    assert query["appSource"] == "sogni-creative-agent-skill"
    assert now - query["since"] < 7 * DAY_MS, "never asks past the 7-day history"
    assert [project["id"] for project in recent] == ["B", "A"], "newest first, orphans dropped"
    assert recent[1] == {
        "id": "A",
        "modelId": "minimax-h3-ref2va-fp8_r2v",
        "modelName": "H3",
        "appSource": "sogni-creative-agent-skill",
        "finishedAt": now - 3000,
        "jobs": [
            {
                "id": "IMG-A0",
                "status": "completed",
                "sensitiveContentWithheld": False,
                "finishedAt": now - 5000,
            },
            {
                "id": "IMG-A1",
                "status": "completed",
                "sensitiveContentWithheld": True,
                "finishedAt": now - 3000,
            },
        ],
    }
    assert api.listRecent == api.list_recent


async def test_list_recent_defaults_to_the_last_24_hours_and_50_renders() -> None:
    api, rest = make_api()
    signed_in(api)

    assert await api.list_recent() == []

    query = history_query(rest)
    assert query["limit"] == 50
    assert abs(int(time.time() * 1000) - DAY_MS - query["since"]) < 5000
    assert "appSource" not in query
    assert paths(rest) == [HISTORY_PATH]


async def test_list_recent_takes_a_datetime_and_keeps_millisecond_numbers_exact() -> None:
    api, rest = make_api()
    signed_in(api)
    six_hours_ago = int(time.time()) - 6 * 3600
    # Sub-millisecond digits are dropped, as a JavaScript Date has none.
    since = datetime.fromtimestamp(six_hours_ago, timezone.utc) + timedelta(
        milliseconds=123, microseconds=900
    )

    await api.list_recent(since=since, limit=10.9)

    query = history_query(rest)
    assert query["since"] == six_hours_ago * 1000 + 123
    assert query["limit"] == 10

    # A naive datetime is local time, as datetime.timestamp() reads it.
    api, rest = make_api()
    signed_in(api)
    await api.list_recent(since=datetime.fromtimestamp(six_hours_ago))
    assert history_query(rest)["since"] == six_hours_ago * 1000

    api, rest = make_api()
    signed_in(api)
    recent_ms = time.time() * 1000 - 3600_000 + 0.5
    await api.list_recent(since=float(int(recent_ms)))
    # 1727000000000.0 goes out as "1727000000000", as JavaScript sends it.
    assert history_query(rest)["since"] == str(int(recent_ms))


async def test_list_recent_keeps_the_limit_at_least_one() -> None:
    api, rest = make_api()
    signed_in(api)

    await api.list_recent(limit=0)

    assert history_query(rest)["limit"] == 1


async def test_list_recent_leaves_out_what_the_history_did_not_record() -> None:
    now = int(time.time() * 1000)
    history = [
        {"id": "C-0", "status": "jobError", "reason": "genfailure", "parentRequest": {"id": "C"}},
        {"id": "C-1", "status": "jobCompleted", "endTime": 0, "parentRequest": {"id": "C"}},
        history_record("D-0", {"id": "D"}, now - 10),
    ]
    api, _ = make_api(history=history)
    signed_in(api)

    recent = await api.list_recent()

    assert recent == [
        {
            "id": "D",
            "jobs": [
                {
                    "id": "D-0",
                    "status": "completed",
                    "sensitiveContentWithheld": False,
                    "finishedAt": now - 10,
                }
            ],
            "finishedAt": now - 10,
        },
        {
            "id": "C",
            "jobs": [
                {"id": "C-0", "status": "failed", "sensitiveContentWithheld": False},
                {"id": "C-1", "status": "completed", "sensitiveContentWithheld": False},
            ],
        },
    ]


async def test_list_recent_without_a_signed_in_account_asks_nothing() -> None:
    # It says so instead of asking the API for every account's history.
    api, rest = make_api()
    signed_in(api, None)

    with pytest.raises(RuntimeError, match="signed-in account"):
        await api.list_recent()

    assert rest.calls == []

    # A ProjectsApi nothing gave an account to is not signed in either.
    api, rest = make_api()
    with pytest.raises(RuntimeError, match="list_recent needs a signed-in account"):
        await api.list_recent()
    assert rest.calls == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"since": "yesterday"}, "since must be a datetime or milliseconds"),
        ({"since": True}, "since must be a datetime or milliseconds"),
        ({"since": float("nan")}, "since must be a datetime or milliseconds"),
        ({"limit": "10"}, "limit must be a number"),
        ({"limit": float("inf")}, "limit must be a number"),
    ],
)
async def test_list_recent_refuses_values_it_cannot_send(
    options: dict[str, Any], message: str
) -> None:
    api, rest = make_api()
    signed_in(api)

    with pytest.raises(TypeError, match=message):
        await api.list_recent(**options)

    assert rest.calls == []
