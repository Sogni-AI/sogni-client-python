"""Coverage for project recovery: reconnect survival, resync, and replay."""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from sogni_client.errors import ApiError
from sogni_client.events import EventEmitter
from sogni_client.projects import Project, ProjectsApi
from sogni_client.recovery import (
    PROJECT_LOST_ORIGINAL_CODE,
    is_llm_recovered_project,
    is_project_lost_error,
    media_type_from_recovered_project,
    project_params_from_recovered_project,
)
from sogni_client.utils import b64_json_encode


class FakeRest:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append({"path": path, "params": params})
        if not self.responses:
            raise AssertionError(f"Unexpected REST call: {path}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeSocket(EventEmitter):
    def __init__(self, responses: dict[str, Any] | None = None, app_id: str = "app-self") -> None:
        super().__init__()
        self.app_id = app_id
        self.responses = responses or {}
        self.get_calls: list[tuple[str, dict[str, Any] | None]] = []
        self.sent: list[dict[str, Any]] = []

    async def send(self, message_type: str, data: Any) -> None:
        self.sent.append({"type": message_type, "data": data})

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.get_calls.append((path, params))
        if path not in self.responses:
            raise AssertionError(f"Unexpected socket GET: {path}")
        response = self.responses[path]
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient(EventEmitter):
    def __init__(
        self,
        rest_responses: list[Any] | None = None,
        socket_responses: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.rest = FakeRest(rest_responses)
        self.socket = FakeSocket(socket_responses)
        self.app_source = "pytest"


def recovered_project(**overrides: Any) -> dict[str, Any]:
    request = {
        "numberOfImages": 2,
        "network": "fast",
        "tokenType": "spark",
        "keyFrames": [
            {
                "modelID": "minimax-h3-fl2va-fp8_flf2v_turbo",
                "positivePrompt": "a paper boat",
                "seed": 7,
                "frames": 124,
                "fps": 24,
            }
        ],
    }
    base: dict[str, Any] = {
        "id": "PROJ-1",
        "appId": "app-self",
        "modelType": "video",
        "status": "active",
        "stepCount": 4,
        "width": 768,
        "height": 768,
        "clientRequestData": b64_json_encode(request),
        "workerJobs": [],
        "completedWorkerJobs": [],
    }
    base.update(overrides)
    return base


def test_recovered_params_are_rebuilt_from_the_original_request() -> None:
    params = project_params_from_recovered_project(recovered_project())

    assert params["type"] == "video"
    assert params["modelId"] == "minimax-h3-fl2va-fp8_flf2v_turbo"
    assert params["numberOfMedia"] == 2
    assert params["positivePrompt"] == "a paper boat"
    assert params["steps"] == 4
    assert params["seed"] == 7
    assert params["network"] == "fast"
    assert params["tokenType"] == "spark"
    assert params["width"] == 768 and params["height"] == 768
    assert params["frames"] == 124 and params["fps"] == 24
    # Asset inputs are not recoverable and must not be invented.
    assert "referenceImage" not in params


def test_recovered_params_survive_a_missing_or_malformed_request_blob() -> None:
    params = project_params_from_recovered_project(
        {"id": "P", "model": {"id": "flux1-schnell-fp8", "type": "image"}}
    )
    assert params == {
        "type": "image",
        "modelId": "flux1-schnell-fp8",
        "numberOfMedia": 1,
        "positivePrompt": "",
    }

    garbled = project_params_from_recovered_project(
        {"id": "P", "modelType": "audio", "clientRequestData": "not-base64-json"}
    )
    assert garbled["type"] == "audio"
    assert garbled["modelId"] == ""


def test_llm_projects_and_media_types_are_classified() -> None:
    assert is_llm_recovered_project({"jobType": "llm"}) is True
    assert is_llm_recovered_project({"model": {"type": "llm"}}) is True
    assert is_llm_recovered_project({"modelType": "video"}) is False
    assert media_type_from_recovered_project({"modelType": "music"}) == "audio"
    assert media_type_from_recovered_project({}) == "image"


def test_project_lost_error_is_recognizable() -> None:
    assert is_project_lost_error({"originalCode": PROJECT_LOST_ORIGINAL_CODE}) is True
    assert is_project_lost_error({"originalCode": "genfailure"}) is False
    assert is_project_lost_error(None) is False


async def test_a_dropped_socket_no_longer_fails_in_flight_projects() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    project = Project(
        {"type": "video", "modelId": "minimax-h3-fl2va-fp8_flf2v_turbo", "numberOfMedia": 1},
        api,
    )
    api._projects.append(project)

    client.emit("disconnected", {"code": 1006, "reason": "network blip"})

    # Generation keeps running on the Supernet; the project must stay alive.
    assert project.status == "pending"
    assert project.finished is False
    assert project.error is None
    # And its staleness timer must not fire while the transport is down.
    assert api._should_defer_project_timeouts() is True
    await project._check_for_timeout()
    assert project.finished is False

    client.emit("connected", {"network": "fast"})
    assert api._should_defer_project_timeouts() is False
    api._clear_authenticated_timer()


async def test_sync_rehydrates_untracked_projects_and_replays_completed_jobs() -> None:
    snapshot = {
        "activeProjects": [
            recovered_project(
                workerJobs=[
                    {
                        "imgID": "JOB-1",
                        "status": "jobStarted",
                        "performedSteps": 2,
                        "worker": {"name": "worker-a"},
                    }
                ]
            ),
            # LLM requests share the registry but are not media projects.
            {"id": "LLM-1", "jobType": "llm"},
        ],
        "unclaimedCompletedProjects": [
            recovered_project(
                id="PROJ-2",
                status="completed",
                workerJobs=[],
                completedWorkerJobs=[
                    {
                        "imgID": "JOB-2",
                        "status": "jobCompleted",
                        "performedSteps": 4,
                        "seedUsed": 11,
                        "resultUrl": "https://cdn.example/done.mp4",
                        "worker": {"name": "worker-b"},
                    }
                ],
            )
        ],
    }
    client = FakeClient(socket_responses={"/api/v1/artist/projects/sync": snapshot})
    api = ProjectsApi(client)

    synced: list[dict[str, Any]] = []
    active_recovered: list[Any] = []
    completed_recovered: list[Any] = []
    api.on("projectsSynced", synced.append)
    api.on("activeProjectsRecovered", active_recovered.append)
    api.on("completedProjectsRecovered", completed_recovered.append)

    result = await api.sync()

    assert client.socket.get_calls[0] == (
        "/api/v1/artist/projects/sync",
        {"appId": "app-self"},
    )
    assert [p["id"] for p in result["recoveredActive"]] == ["PROJ-1"]
    assert [p["id"] for p in result["recoveredCompleted"]] == ["PROJ-2"]
    assert result["reason"] == "manual"
    assert synced and synced[0]["reason"] == "manual"
    assert active_recovered and completed_recovered

    tracked = {p.id: p for p in api.tracked_projects}
    assert set(tracked) == {"PROJ-1", "PROJ-2"}
    assert tracked["PROJ-1"].recovered is True
    assert tracked["PROJ-1"].job("JOB-1") is not None

    finished = tracked["PROJ-2"]
    assert finished.result_urls == ["https://cdn.example/done.mp4"]
    assert result["recoveredCompleted"][0]["resultUrls"] == ["https://cdn.example/done.mp4"]

    # The sync route is read-only, so a second pass must not re-announce it.
    client.socket.responses["/api/v1/artist/projects/sync"] = {
        "activeProjects": [],
        "unclaimedCompletedProjects": [snapshot["unclaimedCompletedProjects"][0]],
    }
    again = await api.sync()
    assert again["recoveredCompleted"] == []


async def test_missing_tracked_projects_resolve_to_finished_active_or_lost() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    api._recovery_tuning["missing_project_attempts"] = 1
    api._recovery_tuning["missing_project_retry_seconds"] = 0
    api._recovery_tuning["recently_created_grace_seconds"] = 0

    lost = Project({"type": "image", "modelId": "m", "numberOfMedia": 1}, api)
    api._projects.append(lost)

    client.rest.responses.append(ApiError(404, {"message": "not found"}))
    client.socket.responses["/api/v1/artist/projects/active"] = {"projects": []}
    # The owner-scoped live lookup does not know it either.
    client.rest.responses.append(ApiError(404, {"message": "not found"}))

    errors: list[dict[str, Any]] = []
    api.on("project", lambda event: errors.append(event) if event["type"] == "error" else None)

    result = await api._reconcile(
        {"activeProjects": [], "unclaimedCompletedProjects": []}, "manual", 10**10
    )

    assert result["lost"] == [lost.id]
    assert errors and is_project_lost_error(errors[0]["error"])

    # A project the socket still lists is in flight, not lost.
    live = Project({"type": "image", "modelId": "m", "numberOfMedia": 1}, api)
    api._projects.append(live)
    client.rest.responses.append(ApiError(404, {"message": "not found"}))
    client.socket.responses["/api/v1/artist/projects/active"] = {"projects": [{"id": live.id}]}
    resolved = await api.resolve_missing([live.id])
    assert resolved[live.id] == {"state": "active"}

    # A transport error yields no verdict rather than a false "lost".
    client.rest.responses.append(ApiError(500, {"message": "boom"}))
    unknown = await api.resolve_missing(["OTHER"])
    assert unknown["OTHER"]["state"] == "unknown"


async def test_live_lookup_only_rescues_projects_the_socket_cannot_vouch_for() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    api._recovery_tuning["missing_project_attempts"] = 1
    api._recovery_tuning["missing_project_retry_seconds"] = 0
    # The socket's live list is unavailable, so only the live lookup can help.
    client.socket.responses["/api/v1/artist/projects/active"] = ApiError(503, {"message": "down"})
    jobs = {"workerJobs": [], "completedWorkerJobs": []}
    ids = ["QUEUED", "PROCESSING", "GONE", "ANON", "SETTLED", "MISMATCH"]
    client.rest.responses.extend([ApiError(404, {"message": "not found"})] * len(ids))
    client.rest.responses.extend(
        [
            {"data": {"project": {"id": "QUEUED", "status": "queued", "finished": False, **jobs}}},
            {
                "data": {
                    "project": {
                        "id": "PROCESSING",
                        "status": "processing",
                        "finished": False,
                        **jobs,
                    }
                }
            },
            ApiError(404, {"message": "not found"}),
            ApiError(401, {"message": "unauthorized"}),
            {
                "data": {
                    "project": {"id": "SETTLED", "status": "completed", "finished": True, **jobs}
                }
            },
            {
                "data": {
                    "project": {"id": "SOMEONE-ELSE", "status": "queued", "finished": False, **jobs}
                }
            },
        ]
    )

    resolved = await api.resolve_missing(ids)

    assert list(resolved) == ids, "results keep the caller's order"
    assert resolved["QUEUED"] == {"state": "active"}
    assert resolved["PROCESSING"] == {"state": "active"}
    assert resolved["GONE"] == {"state": "lost"}
    assert resolved["ANON"] == {"state": "lost"}
    assert resolved["SETTLED"]["state"] == "unknown"
    assert resolved["MISMATCH"] == {"state": "lost"}
    lookups = [call["path"] for call in client.rest.calls if call["path"].startswith("/v2/")]
    assert lookups == [f"/v2/projects/{project_id}" for project_id in ids]


async def test_a_project_the_socket_lists_is_not_looked_up_again() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    api._recovery_tuning["missing_project_attempts"] = 1
    client.rest.responses.append(ApiError(404, {"message": "not found"}))
    client.socket.responses["/api/v1/artist/projects/active"] = {"projects": [{"id": "LATE"}]}

    assert await api.resolve_missing(["LATE"]) == {"LATE": {"state": "active"}}
    assert [call["path"] for call in client.rest.calls] == ["/v1/projects/LATE"]


async def test_get_status_reads_the_live_lookup_and_get_keeps_its_v1_path() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    snapshot = {
        "id": "A/B",
        "status": "queued",
        "finished": False,
        "workerJobs": [],
        "completedWorkerJobs": [],
    }
    client.rest.responses.extend(
        [{"data": {"project": snapshot}}, ApiError(404, {"message": "not found"})]
    )

    assert await api.get_status("A/B") == snapshot
    assert api.getStatus == api.get_status
    with pytest.raises(ApiError):
        await api.get("A/B")
    assert [call["path"] for call in client.rest.calls] == [
        "/v2/projects/A%2FB",
        "/v1/projects/A/B",
    ]


async def test_list_projects_elsewhere_excludes_this_app_and_llm_requests() -> None:
    client = FakeClient(
        socket_responses={
            "/api/v1/artist/projects/sync": {
                "activeProjects": [
                    {"id": "MINE", "appId": "app-self"},
                    {"id": "OTHER", "appId": "app-other", "appSource": "sogni-web"},
                    {"id": "OTHER-LLM", "appId": "app-other", "jobType": "llm"},
                    {"id": "NO-APP-ID"},
                ]
            }
        }
    )
    api = ProjectsApi(client)

    elsewhere = await api.list_projects_elsewhere()

    assert [p["id"] for p in elsewhere] == ["OTHER"]
    assert client.socket.get_calls[0] == ("/api/v1/artist/projects/sync", None)


async def test_replay_never_downgrades_a_locally_finished_project() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    project = Project({"type": "image", "modelId": "m", "numberOfMedia": 1}, api)
    api._projects.append(project)
    project._update({"status": "completed"})

    await api._replay_raw_project(project, recovered_project(status="active"), True)

    assert project.status == "completed"


@pytest.mark.parametrize(
    ("status", "expected"),
    [("cancelled", "canceled"), ("queued", "queued"), ("active", "queued")],
)
async def test_replayed_project_status_maps_onto_local_state(status: str, expected: str) -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    project = Project({"type": "image", "modelId": "m", "numberOfMedia": 1}, api)
    api._projects.append(project)

    await api._replay_raw_project(project, recovered_project(status=status), True)

    assert project.status == expected
    # A cancelled project settles its completion future; consume it so the loop
    # does not report an unretrieved exception.
    if project._completion.done():
        with contextlib.suppress(Exception):
            project._completion.exception()


# Socket restarts (mirrors blocks 11-13 of sogni-client
# scripts/check-project-recovery.cjs).


def settle(project: Project) -> None:
    """Consume a failed project's completion so the loop reports nothing."""

    if project._completion.done():
        with contextlib.suppress(BaseException):
            project._completion.exception()


def restart_harness(
    socket_responses: dict[str, Any] | None = None,
) -> tuple[ProjectsApi, FakeClient, list[dict[str, Any]], list[dict[str, Any]]]:
    client = FakeClient(socket_responses=socket_responses)
    api = ProjectsApi(client)
    api._recovery_tuning.update(
        {
            "authenticated_grace_seconds": 0.02,
            "recently_created_grace_seconds": 0,
            "missing_project_attempts": 2,
            "missing_project_retry_seconds": 0.005,
        }
    )

    # Make the staleness watchdog's live-list lookup inert, as the JS harness does.
    async def no_live_list() -> None:
        return None

    api._list_active_project_ids = no_live_list  # type: ignore[method-assign]
    project_events: list[dict[str, Any]] = []
    api.on("project", project_events.append)
    synced: list[dict[str, Any]] = []
    api.on("projectsSynced", synced.append)
    return api, client, project_events, synced


def track(api: ProjectsApi, *, started_seconds_ago: float = 60) -> Project:
    """Track a project the way ``create()`` does once the request is sent."""

    project = Project(
        {
            "type": "image",
            "modelId": "flux1-schnell-fp8",
            "numberOfMedia": 1,
            "positivePrompt": "a lighthouse at dusk",
            "steps": 4,
        },
        api,
    )
    project._data["startedAt"] = datetime.now(timezone.utc) - timedelta(seconds=started_seconds_ago)
    api._projects.append(project)
    return project


def stop_timers(api: ProjectsApi) -> None:
    api._clear_authenticated_timer()
    if api._recheck_timer is not None:
        api._recheck_timer.cancel()
        api._recheck_timer = None
    for project in api._projects:
        if project._timeout_handle is not None:
            project._timeout_handle.cancel()


async def test_a_recoverable_drop_defers_project_timeouts_too() -> None:
    # `connecting` (the socket-deploy path) is a recoverable drop;
    # `disconnected` is only emitted for terminal closes.
    api, client, _events, _synced = restart_harness()
    track(api)

    client.emit("connecting", {"network": "fast"})
    assert api._should_defer_project_timeouts() is True, "timeouts defer while reconnecting"
    client.emit("connected", {"network": "fast"})
    assert api._should_defer_project_timeouts() is False, "timeouts resume on reconnect"
    stop_timers(api)


async def test_a_request_refused_while_the_socket_restarts_is_resubmitted_once() -> None:
    # A request refused while the socket restarts (jobError 1001, no imgID) is
    # not a failure: it is sent again, unchanged, on the next connection.
    api, client, events, _synced = restart_harness()
    project = track(api)
    request = {"jobID": project.id, "keyFrames": [{"modelID": "flux1-schnell-fp8"}]}
    api._unadmitted_requests[project.id] = request
    refusal = {
        "jobID": project.id,
        "isFromWorker": False,
        "error": "1001",
        "error_message": "Server is restarting",
    }

    client.socket.emit("jobError", refusal)

    assert project.status == "pending", "a refusal during restart does not fail the project"
    assert client.socket.sent == [], "nothing is written into the closing socket"
    # Until it is re-sent, a reconcile does not judge it missing (no lookup).
    result = await api._reconcile(
        {"activeProjects": [], "unclaimedCompletedProjects": []}, "manual", time.time()
    )
    assert result["lost"] == [] and client.rest.calls == []
    assert api._recheck_timer is None

    client.emit("connecting", {"network": "fast"})
    client.emit("connected", {"network": "fast"})
    await asyncio.sleep(0.01)

    assert client.socket.sent == [{"type": "jobRequest", "data": request}], "resubmitted once"
    assert [e for e in events if e["type"] == "error"] == [], "no error surfaced"
    assert project.id not in api._awaiting_resubmit
    assert api._unadmitted_requests[project.id] is request
    assert api._recheck_timer is not None, "the re-sent project is re-checked after its grace"

    # A second refusal is not retried again: it surfaces.
    del api._unadmitted_requests[project.id]
    client.socket.emit("jobError", refusal)
    assert project.status == "failed", "a request with nothing left to resubmit fails"
    assert project.error == {"code": 1001, "message": "Server is restarting"}
    settle(project)
    stop_timers(api)


async def test_a_project_too_new_to_judge_is_rechecked_after_its_grace() -> None:
    # A project too new to judge at the reconnect sync is re-checked once the
    # grace ends, instead of waiting minutes for the staleness watchdog.
    api, client, events, synced = restart_harness(
        socket_responses={
            "/api/v1/artist/projects/sync": {"activeProjects": [], "unclaimedCompletedProjects": []}
        }
    )
    api._recovery_tuning["recently_created_grace_seconds"] = 0.04
    project = track(api, started_seconds_ago=0)
    # The recheck's lookups: two REST attempts, then the owner-scoped live lookup.
    client.rest.responses.extend([ApiError(404, {"message": "not found"})] * 3)

    client.emit("connected", {"network": "fast"})
    client.socket.emit(
        "authenticated",
        {"clientType": "artist", "activeProjects": [], "unclaimedCompletedProjects": []},
    )
    await asyncio.sleep(0.02)
    assert len(synced) == 1
    assert synced[0]["lost"] == [], "too new to judge on the first sync"

    await asyncio.sleep(0.4)
    recheck = next((r for r in synced if r["reason"] == "recheck"), None)
    assert recheck is not None, "a recheck sync ran after the grace"
    assert recheck["lost"] == [project.id], "the recheck resolves it"
    # A lost verdict fails the project, as in the JS SDK.
    assert project.status == "failed"
    assert is_project_lost_error(project.error)
    assert [e["projectId"] for e in events if e["type"] == "error"] == [project.id]
    settle(project)
    stop_timers(api)
