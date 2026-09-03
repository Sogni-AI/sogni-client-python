"""Coverage for project recovery: reconnect survival, resync, and replay."""

from __future__ import annotations

import contextlib
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

    async def send(self, message_type: str, data: Any) -> None:  # pragma: no cover - unused
        pass

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
