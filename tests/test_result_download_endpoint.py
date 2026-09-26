"""Choosing the download endpoint of a finished job.

Images come from ``/v1/image/downloadUrl``; video, audio and 3D artifacts from
``/v1/media/downloadUrl``. The client used to read "no evidence" as "image": a
result for a project it did not track and a model whose catalog entry had no
media kind both went to the image endpoint, which answers a provable video or
audio result with 404 "This result is media, not an image; request it from
/v1/media/downloadUrl".
"""

from __future__ import annotations

from typing import Any

import pytest

from sogni_client.errors import ApiError
from sogni_client.events import EventEmitter
from sogni_client.projects import Project, ProjectsApi
from sogni_client.utils import result_media_evidence

IMAGE_PATH = "/v1/image/downloadUrl"
MEDIA_PATH = "/v1/media/downloadUrl"


def media_refusal() -> ApiError:
    return ApiError(
        404,
        {
            "status": "error",
            "errorCode": 122,
            "message": "This result is media, not an image; request it from /v1/media/downloadUrl",
        },
    )


class DownloadRest:
    """Answers both download endpoints; ``respond(path, params)`` may raise."""

    def __init__(self, respond: Any = None) -> None:
        self.respond = respond
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if path not in {IMAGE_PATH, MEDIA_PATH}:
            raise AssertionError(f"Unexpected REST call: {path}")
        params = dict(params or {})
        self.calls.append((path, params))
        if self.respond is not None:
            url = self.respond(path, params)
        else:
            url = (
                f"https://cdn.test{path}/{params['jobId']}/{params.get('imageId') or params['id']}"
            )
        return {"status": "success", "data": {"downloadUrl": url}}


class Client(EventEmitter):
    def __init__(self, respond: Any = None) -> None:
        super().__init__()
        self.rest = DownloadRest(respond)
        self.socket = EventEmitter()
        self.app_source = "pytest"


def harness(respond: Any = None, catalog: list[dict[str, Any]] | None = None):
    client = Client(respond)
    api = ProjectsApi(client)
    if catalog is not None:
        api._supported_models = catalog
    completed: list[dict[str, Any]] = []
    api.on("job", lambda event: completed.append(event) if event["type"] == "completed" else None)
    return api, client.rest.calls, completed


def track(api: ProjectsApi, **params: Any) -> Project:
    project = Project(
        {"numberOfMedia": 1, "positivePrompt": "a lighthouse at dusk", "steps": 4, **params},
        api,
    )
    api._projects.append(project)
    return project


def result(job_id: str, img_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "jobID": job_id,
        "imgID": img_id,
        "performedStepCount": 4,
        "lastSeed": "7",
        "triggeredNSFWFilter": False,
        "userCanceled": False,
        **extra,
    }


@pytest.mark.asyncio
async def test_untracked_result_without_evidence_asks_no_endpoint() -> None:
    api, calls, completed = harness()
    await api._apply_job_result(result("OTHER-1", "IMG-1"))
    assert calls == [], "no evidence must not become an image request"
    assert len(completed) == 1 and completed[0]["resultUrl"] is None

    # A failed upload is not evidence of anything either.
    api, calls, _ = harness()
    await api._apply_job_result(
        result("OTHER-2", "IMG-2", artifacts=[{"contentType": "video/mp4", "success": False}])
    )
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (
            {"artifacts": [{"file": "out.mp4", "contentType": "video/mp4", "success": True}]},
            (MEDIA_PATH, {"jobId": "P", "id": "J", "type": "complete"}),
        ),
        (
            # A still beside the video is incidental; the video is the result.
            {
                "artifacts": [
                    {"contentType": "image/png", "success": True},
                    {"contentType": "video/mp4", "success": True},
                ]
            },
            (MEDIA_PATH, {"jobId": "P", "id": "J", "type": "complete"}),
        ),
        (
            {"artifacts": [{"contentType": "audio/mpeg", "success": True}]},
            (
                MEDIA_PATH,
                {"jobId": "P", "id": "J", "type": "complete", "contentType": "audio/mpeg"},
            ),
        ),
        (
            {"artifacts": [{"contentType": "model/gltf-binary", "success": True}]},
            (
                MEDIA_PATH,
                {"jobId": "P", "id": "J", "type": "complete", "contentType": "model/gltf-binary"},
            ),
        ),
        (
            {"outputFormat": "mp4"},
            (MEDIA_PATH, {"jobId": "P", "id": "J", "type": "complete"}),
        ),
        (
            {"outputFormat": "wav"},
            (MEDIA_PATH, {"jobId": "P", "id": "J", "type": "complete", "contentType": "audio/wav"}),
        ),
        (
            # An image result keeps the image endpoint, exactly as before.
            {"artifacts": [{"contentType": "image/jpeg", "success": True}]},
            (IMAGE_PATH, {"jobId": "P", "imageId": "J", "type": "complete"}),
        ),
    ],
)
async def test_untracked_result_uses_the_frames_own_evidence(
    frame: dict[str, Any], expected: tuple[str, dict[str, Any]]
) -> None:
    api, calls, completed = harness()
    await api._apply_job_result(result("P", "J", **frame))
    assert calls == [expected]
    assert completed[0]["resultUrl"] == f"https://cdn.test{expected[0]}/P/J"


@pytest.mark.asyncio
async def test_catalog_entry_without_media_kind_is_not_an_image() -> None:
    model_id = "wan_v2.2-14b-fp8_i2v_lightx2v"
    api, calls, _ = harness(catalog=[{"id": model_id, "name": "WAN i2v", "SID": 1, "tier": "t"}])
    assert api.is_video_model_id(model_id)
    project = track(api, type="video", modelId=model_id)
    await api._apply_job_result(result(project.id, "IMG-V"))
    assert calls == [(MEDIA_PATH, {"jobId": project.id, "id": "IMG-V", "type": "complete"})]
    job = project.job("IMG-V")
    assert job is not None and job.type == "video"


@pytest.mark.asyncio
async def test_unknown_model_takes_the_project_type_for_every_url_request() -> None:
    api, calls, _ = harness(catalog=[])
    project = track(api, type="audio", modelId="future_audio_model_v1", outputFormat="flac")
    await api._apply_job_result(result(project.id, "IMG-A"))
    expected = (
        MEDIA_PATH,
        {"jobId": project.id, "id": "IMG-A", "type": "complete", "contentType": "audio/flac"},
    )
    assert calls == [expected]
    job = project.job("IMG-A")
    assert job is not None and job.type == "audio"

    # The REST resync and get_result_url() route the same way.
    job._update({"resultUrl": None})
    await job._sync_with_rest_data({"imgID": "IMG-A", "status": "jobCompleted"})
    assert calls == [expected, expected]


@pytest.mark.asyncio
async def test_image_and_pixal3d_behaviour_is_unchanged() -> None:
    api, calls, completed = harness(
        catalog=[
            {"id": "flux1-schnell-fp8", "name": "Flux", "SID": 1, "tier": "t", "media": "image"},
            {"id": "pixal3d_int8_i23d", "name": "Pixal3D", "SID": 2, "tier": "t", "media": "image"},
        ]
    )
    image = track(api, type="image", modelId="flux1-schnell-fp8", outputFormat="webp")
    await api._apply_job_result(result(image.id, "IMG-I"))
    assert calls == [
        (
            IMAGE_PATH,
            {
                "jobId": image.id,
                "imageId": "IMG-I",
                "type": "complete",
                "contentType": "image/webp",
            },
        )
    ]

    gpt = track(api, type="image", modelId="gpt-image-2")
    await api._apply_job_result(
        result(gpt.id, "IMG-G", resultUrl="https://vendor.test/gpt.png", outputFormat="png")
    )
    assert len(calls) == 1, "a frame that carries its URL mints nothing"
    assert completed[-1]["resultUrl"] == "https://vendor.test/gpt.png"

    # The Pixal3D rule still outranks a catalog that calls the model an image.
    glb = track(api, type="image", modelId="pixal3d_int8_i23d")
    await api._apply_job_result(result(glb.id, "IMG-3D"))
    assert calls[-1] == (
        MEDIA_PATH,
        {"jobId": glb.id, "id": "IMG-3D", "type": "complete", "contentType": "model/gltf-binary"},
    )


@pytest.mark.asyncio
async def test_media_refusal_switches_to_the_media_endpoint_once() -> None:
    state = {"media_fails": True}

    def respond(path: str, params: dict[str, Any]) -> str:
        if path == IMAGE_PATH:
            raise media_refusal()
        if state["media_fails"]:
            raise ApiError(500, {"status": "error", "errorCode": 1, "message": "boom"})
        return f"https://cdn.test{path}/{params['jobId']}/{params['id']}"

    api, calls, completed = harness(
        respond,
        catalog=[
            {"id": "flux1-schnell-fp8", "name": "Flux", "SID": 1, "tier": "t", "media": "image"}
        ],
    )
    project = track(api, type="image", modelId="flux1-schnell-fp8")
    await api._apply_job_result(result(project.id, "IMG-M"))
    assert [path for path, _ in calls] == [IMAGE_PATH, MEDIA_PATH]
    assert calls[1][1] == {"jobId": project.id, "id": "IMG-M", "type": "complete"}
    assert completed[0]["resultUrl"] is None

    state["media_fails"] = False
    job = project.job("IMG-M")
    assert job is not None
    assert await job.get_result_url() == f"https://cdn.test{MEDIA_PATH}/{project.id}/IMG-M"
    assert [path for path, _ in calls] == [IMAGE_PATH, MEDIA_PATH, MEDIA_PATH], (
        "a later request goes straight to the media endpoint"
    )


@pytest.mark.asyncio
async def test_other_image_failure_does_not_fall_back_to_media() -> None:
    def respond(path: str, params: dict[str, Any]) -> str:
        raise ApiError(404, {"status": "error", "errorCode": 122, "message": "Download not found"})

    api, calls, completed = harness(
        respond,
        catalog=[
            {"id": "flux1-schnell-fp8", "name": "Flux", "SID": 1, "tier": "t", "media": "image"}
        ],
    )
    project = track(api, type="image", modelId="flux1-schnell-fp8")
    await api._apply_job_result(result(project.id, "IMG-N"))
    assert [path for path, _ in calls] == [IMAGE_PATH]
    assert completed[0]["resultUrl"] is None


def test_result_media_evidence() -> None:
    assert result_media_evidence({}) is None
    assert result_media_evidence({"outputFormat": "constructor"}) is None
    assert result_media_evidence({"artifacts": "nope", "outputFormat": 7}) is None
    assert result_media_evidence({"artifacts": [{"contentType": "video"}]}) is None
    assert result_media_evidence(
        {"artifacts": [{"contentType": "Audio/MPEG; codecs=mp3", "success": True}]}
    ) == {"kind": "audio", "contentType": "Audio/MPEG; codecs=mp3"}
    assert result_media_evidence(
        {"artifacts": [{"contentType": "text/plain"}], "outputFormat": "MOV"}
    ) == {"kind": "video"}
