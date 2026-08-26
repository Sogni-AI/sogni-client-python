from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any
from unittest.mock import ANY, AsyncMock

import pytest

from sogni_client.errors import ApiError, ProjectError
from sogni_client.events import EventEmitter
from sogni_client.projects import Project, ProjectsApi, create_job_request_message
from sogni_client.projects import _now as project_now

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16
MP3 = b"ID3" + b"\0" * 16


def model_options(kind: str, **overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "type": kind,
        "sampler": {"allowed": [], "default": None},
        "scheduler": {"allowed": [], "default": None},
    }
    options.update(overrides)
    return options


class FakeRest:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def _response(self) -> Any:
        if not self.responses:
            raise AssertionError("Unexpected REST call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append({"method": "GET", "path": path, "params": params})
        return self._response()

    async def put_bytes(self, url: str, data: bytes, *, content_type: str | None = None) -> None:
        self.calls.append(
            {
                "method": "PUT",
                "url": url,
                "data": data,
                "content_type": content_type,
            }
        )

    async def get_bytes(self, url: str) -> bytes:
        self.calls.append({"method": "GET_BYTES", "url": url})
        return b"result"


class FakeSocket(EventEmitter):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[tuple[str, Any]] = []
        self.get_calls: list[tuple[str, dict[str, Any] | None]] = []

    async def send(self, message_type: str, data: Any) -> None:
        self.sent.append((message_type, data))

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.get_calls.append((path, params))
        raise AssertionError(f"Unexpected socket GET: {path}")


class FakeClient(EventEmitter):
    def __init__(
        self, responses: list[Any] | None = None, *, app_source: str | None = "pytest"
    ) -> None:
        super().__init__()
        self.rest = FakeRest(responses)
        self.socket = FakeSocket()
        self.app_source = app_source


def test_image_request_matches_required_mac_worker_template_and_undefined_wire_fields() -> None:
    message = create_job_request_message(
        "project-1",
        {
            "type": "image",
            "modelId": "flux1-schnell-fp8",
            "positivePrompt": "a glass bird",
            "numberOfMedia": 1,
        },
        model_options("image"),
    )

    keyframe = message["keyFrames"][0]
    assert keyframe["cnRotationIsEnabled"] is True
    assert keyframe["startingImageZoomPanIsOn"] is False
    assert keyframe["negativePrompt"] == ""
    assert keyframe["scheduler"] is None
    assert keyframe["timeStepSpacing"] is None
    assert "seed" not in keyframe
    assert "steps" not in keyframe
    assert "guidanceScale" not in keyframe
    assert "sizePreset" not in keyframe


def test_image_request_validates_and_normalizes_custom_sizes_and_control_net_numbers() -> None:
    options = model_options(
        "image",
        sampler={"allowed": ["euler"], "default": "euler"},
        scheduler={"allowed": ["normal"], "default": "normal"},
        vae={"allowed": ["model.vae"], "default": "model.vae"},
    )
    message = create_job_request_message(
        "project-2",
        {
            "type": "image",
            "modelId": "z_image_turbo_bf16",
            "positivePrompt": "city",
            "numberOfMedia": 1,
            "width": "512",
            "height": "768",
            "sampler": "euler",
            "scheduler": "normal",
            "vae": "model.vae",
            "controlNet": {
                "name": "canny",
                "image": True,
                "strength": "0.25",
                "guidanceStart": "0.1",
                "guidanceEnd": 0.9,
                "mode": "prompt_priority",
            },
        },
        options,
    )

    keyframe = message["keyFrames"][0]
    assert keyframe["sizePreset"] == "custom"
    assert keyframe["width"] == 512
    assert keyframe["height"] == 768
    assert keyframe["comfySampler"] == "euler"
    assert keyframe["comfyScheduler"] == "normal"
    assert keyframe["vae"] == "model.vae"
    assert keyframe["currentControlNetsJob"] == [
        {
            "name": "canny",
            "cnImageState": "original",
            "hasImage": True,
            "controlStrength": 0.25,
            "controlMode": 1,
            "controlGuidanceStart": 0.1,
            "controlGuidanceEnd": 0.9,
        }
    ]

    invalid = {
        "type": "image",
        "modelId": "z_image_turbo_bf16",
        "positivePrompt": "city",
        "numberOfMedia": 1,
        "width": 255,
        "height": 512,
    }
    with pytest.raises(ValueError, match="Width"):
        create_job_request_message("invalid-size", invalid, options)


def test_video_request_serializes_frames_mask_and_validated_numeric_fields() -> None:
    message = create_job_request_message(
        "video-1",
        {
            "type": "video",
            "modelId": "ltx23-22b-fp8_v2v_distilled",
            "positivePrompt": "watercolor motion",
            "numberOfMedia": 1,
            "duration": "5",
            "fps": 24,
            "width": "480",
            "height": "720",
            "referenceVideo": True,
            "referenceMask": True,
            "controlNet": {"name": "inpaint", "strength": "0.4"},
            "teacacheThreshold": "0.5",
            "generateAudio": False,
            "trimEndFrame": True,
        },
        model_options("video"),
    )

    keyframe = message["keyFrames"][0]
    assert keyframe["frames"] == 121
    assert keyframe["fps"] == 24
    assert keyframe["width"] == 480
    assert keyframe["height"] == 720
    assert keyframe["hasReferenceVideo"] is True
    assert keyframe["hasReferenceMask"] is True
    assert keyframe["currentControlNetsJob"] == [{"name": "inpaint", "controlStrength": 0.4}]
    assert keyframe["teacacheThreshold"] == 0.5
    assert keyframe["generateAudio"] is False
    assert keyframe["trimEndFrame"] is True

    with pytest.raises(ValueError, match="teacacheThreshold"):
        create_job_request_message(
            "invalid-cache",
            {
                "type": "video",
                "modelId": "wan_v2.2-14b-fp8_t2v",
                "positivePrompt": "clouds",
                "numberOfMedia": 1,
                "teacacheThreshold": 1.1,
            },
            model_options("video"),
        )


def test_external_video_request_uses_fixed_fps_urls_and_omits_negative_prompt() -> None:
    message = create_job_request_message(
        "external-1",
        {
            "type": "video",
            "modelId": "seedance-2-0",
            "positivePrompt": "cinematic coast",
            "negativePrompt": "text",
            "numberOfMedia": 1,
            "duration": 4,
            "referenceImageUrls": ["https://cdn.example/image.png"],
            "referenceAudioUrls": ["https://cdn.example/audio.mp3"],
        },
        model_options("video"),
    )

    keyframe = message["keyFrames"][0]
    assert keyframe["fps"] == 24
    assert keyframe["frames"] == 97
    assert keyframe["referenceImageURLs"] == ["https://cdn.example/image.png"]
    assert keyframe["referenceAudioURLs"] == ["https://cdn.example/audio.mp3"]
    assert "negativePrompt" not in keyframe

    with pytest.raises(ApiError, match="require at least one image or video"):
        create_job_request_message(
            "external-invalid",
            {
                "type": "video",
                "modelId": "seedance-2-0",
                "positivePrompt": "music",
                "numberOfMedia": 1,
                "referenceAudioUrls": ["https://cdn.example/audio.mp3"],
            },
            model_options("video"),
        )


@pytest.mark.parametrize(
    ("model_id", "params", "message"),
    [
        ("wan_v2.2-14b-fp8_i2v", {}, "requires at least one"),
        (
            "happyhorse-1.1-i2v",
            {"referenceImageUrls": ["https://cdn.example/1.png", "https://cdn.example/2.png"]},
            "exactly one",
        ),
        (
            "happyhorse-1.1-r2v",
            {"referenceVideo": True, "referenceImage": True},
            "do not support reference video",
        ),
    ],
)
def test_video_workflow_asset_requirements_are_enforced(
    model_id: str, params: dict[str, Any], message: str
) -> None:
    with pytest.raises(ApiError, match=message):
        create_job_request_message(
            "invalid-assets",
            {
                "type": "video",
                "modelId": model_id,
                "positivePrompt": "motion",
                "numberOfMedia": 1,
                **params,
            },
            model_options("video"),
        )


def test_audio_request_keeps_audio_fields_and_omits_negative_prompt() -> None:
    message = create_job_request_message(
        "audio-1",
        {
            "type": "audio",
            "modelId": "ace_step_1.5_turbo",
            "positivePrompt": "upbeat synthwave",
            "negativePrompt": "noise",
            "numberOfMedia": 2,
            "duration": 30,
            "bpm": 128,
            "lyrics": "",
            "composerMode": True,
            "promptStrength": 0.8,
        },
        model_options("audio"),
    )

    keyframe = message["keyFrames"][0]
    assert keyframe["duration"] == 30
    assert keyframe["bpm"] == 128
    assert keyframe["lyrics"] == ""
    assert keyframe["composerMode"] is True
    assert keyframe["promptStrength"] == 0.8
    assert "negativePrompt" not in keyframe
    assert message["numberOfImages"] == 2
    assert message["outputFormat"] == "mp3"


def test_minimax_h3_reference_request_uses_numbered_assets_and_frame_grid() -> None:
    message = create_job_request_message(
        "h3-reference",
        {
            "type": "video",
            "modelId": "minimax-h3-ref2va-fp8_r2v",
            "positivePrompt": "A character walks into frame and speaks.",
            "negativePrompt": "",
            "numberOfMedia": 1,
            "duration": 6,
            "referenceImage": True,
            "contextImages": [True],
            "referenceVideo": True,
            "referenceVideoDurations": [4],
            "referenceAudio": True,
            "width": 1024,
            "height": 768,
            "attribution": {
                "workloadKind": "agent_mediated",
                "operationId": "H3-OP",
            },
        },
        model_options("video"),
    )

    keyframe = message["keyFrames"][0]
    assert keyframe["fps"] == 24
    assert keyframe["frames"] == 141
    assert keyframe["hasReferenceImage"] is True
    assert keyframe["hasContextImage2"] is True
    assert keyframe["hasReferenceVideo1"] is True
    assert keyframe["referenceVideo1DurationSeconds"] == 4
    assert keyframe["hasReferenceAudio1"] is True
    assert message["workloadKind"] == "agent_mediated"
    assert message["operationId"] == "H3-OP"


def test_wan3_and_seedance25_use_current_external_video_contracts() -> None:
    wan3 = create_job_request_message(
        "wan3",
        {
            "type": "video",
            "modelId": "wan3.0-video",
            "positivePrompt": "",
            "numberOfMedia": 1,
            "smartDuration": True,
            "referenceLinkUrl": "https://example.com/reference",
            "promptExtend": False,
            "ratio": "9:16",
            "watermark": False,
            "wan3TaskType": "extend",
        },
        model_options("video"),
    )["keyFrames"][0]
    assert wan3["fps"] == 30
    assert wan3["frames"] == 901
    assert wan3["referenceLinkURL"] == "https://example.com/reference"
    assert wan3["promptExtend"] is False
    assert wan3["ratio"] == "9:16"
    assert wan3["watermark"] is False
    assert "wan3TaskType" not in wan3

    with pytest.raises(ApiError, match="promptExtend must be a boolean"):
        create_job_request_message(
            "wan3-invalid-prompt-expand",
            {
                "type": "video",
                "modelId": "wan3.0-video",
                "positivePrompt": "Keep this literal.",
                "numberOfMedia": 1,
                "promptExtend": "false",
            },
            model_options("video"),
        )

    seedance = create_job_request_message(
        "seedance25",
        {
            "type": "video",
            "modelId": "seedance-2-5",
            "positivePrompt": "Use the soundtrack as a loose timing reference.",
            "numberOfMedia": 1,
            "duration": 30,
            "seedanceTaskType": "reference",
            "referenceAudioUrls": ["https://cdn.example/audio.mp3"],
        },
        model_options("video"),
    )["keyFrames"][0]
    assert seedance["frames"] == 721
    assert seedance["seedanceTaskType"] == "reference"


@pytest.mark.asyncio
async def test_queue_eta_liveness_and_eta_confidence_are_observable() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    project = Project(
        {
            "type": "video",
            "modelId": "seedance-2-5",
            "positivePrompt": "queued",
            "numberOfMedia": 1,
        },
        api,
    )
    api._projects.append(project)

    api._handle_job_state(
        {
            "type": "queued",
            "jobID": project.id,
            "queuePosition": 2,
            "estimatedStartSeconds": 90,
            "queueStatus": "waiting",
        }
    )
    assert project.queue_status == "waiting"
    assert project.estimated_start_at is not None

    api._handle_job_state({"type": "jobStarted", "jobID": project.id, "imgID": "job-eta"})
    api._handle_job_progress(
        {
            "jobID": project.id,
            "imgID": "job-eta",
            "progress": 0.1,
            "etaMin": 40,
            "etaMax": 80,
        }
    )
    job = project.job("job-eta")
    assert job is not None
    assert job.eta_range == {"min": 40, "max": 80}
    assert project.estimated_start_at is None
    assert project.queue_status is None

    api._list_active_project_ids = AsyncMock(return_value=[project.id])
    api.get = AsyncMock(side_effect=AssertionError("REST must not be used for an active project"))
    project._last_updated = project_now() - timedelta(minutes=3)
    await project._check_for_timeout()
    assert project.status == "processing"
    api.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_waits_for_server_confirmation_and_deduplicates_requests() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    project = Project(
        {
            "type": "image",
            "modelId": "flux1-schnell-fp8",
            "positivePrompt": "cancel",
            "numberOfMedia": 1,
        },
        api,
    )
    api._projects.append(project)

    async def confirm(message_type: str, data: Any) -> None:
        client.socket.sent.append((message_type, data))
        client.socket.emit("artistCancelConfirmation", {"jobID": project.id, "didCancel": True})

    client.socket.send = AsyncMock(side_effect=confirm)
    await asyncio.gather(api.cancel(project.id), api.cancel(project.id))

    assert client.socket.send.await_count == 1
    assert project.status == "canceled"
    assert project not in api.tracked_projects


@pytest.mark.asyncio
async def test_lora_catalog_is_scoped_cached_and_exposes_constraints() -> None:
    client = FakeClient(
        [
            {
                "data": {
                    "lastUpdated": "2026-08-26T00:00:00Z",
                    "loras": [
                        {"loraId": "warm", "modelIds": ["krea2_turbo_fp8_scaled"]},
                        {"loraId": "other", "modelIds": ["z_image_bf16"]},
                    ],
                    "constraints": {"maxPerRequest": 4, "minStrength": -2, "maxStrength": 2},
                }
            },
            {
                "data": {
                    "lastUpdated": "2026-08-26T00:00:00Z",
                    "loras": [
                        {"loraId": "warm", "modelIds": ["krea2_turbo_fp8_scaled"]},
                        {"loraId": "other", "modelIds": ["z_image_bf16"]},
                    ],
                    "models": ["krea2_turbo_fp8_scaled", "z_image_bf16"],
                    "constraints": {"maxPerRequest": 4, "minStrength": -2, "maxStrength": 2},
                }
            },
        ]
    )
    api = ProjectsApi(client)

    first = await api.available_loras(model_id="krea2_turbo_fp8_scaled")
    second = await api.available_loras(model_id="krea2_turbo_fp8_scaled")

    assert [item["loraId"] for item in first["loras"]] == ["warm"]
    assert second == first
    assert (await api.get_lora("warm"))["loraId"] == "warm"
    assert await api.lora_constraints() == {
        "maxPerRequest": 4,
        "minStrength": -2,
        "maxStrength": 2,
    }


@pytest.mark.asyncio
async def test_create_normalizes_python_names_uploads_assets_and_annotates_content_types() -> None:
    client = FakeClient(
        [
            {"data": {"uploadUrl": "https://upload.example/image"}},
            {"data": {"uploadUrl": "https://upload.example/audio"}},
        ],
        app_source="python-tests",
    )
    api = ProjectsApi(client)
    api.get_model_options = AsyncMock(return_value=model_options("video"))

    project = await api.create(
        type="video",
        model_id="ltx23-22b-fp8_i2v_distilled",
        positive_prompt="portrait speaking",
        number_of_media=1,
        duration=1,
        reference_image=PNG,
        reference_audio_identity=MP3,
    )

    assert project.params["modelId"] == "ltx23-22b-fp8_i2v_distilled"
    request_type, request = client.socket.sent[-1]
    assert request_type == "jobRequest"
    assert request["appSource"] == "python-tests"
    keyframe = request["keyFrames"][0]
    assert keyframe["hasReferenceImage"] is True
    assert keyframe["hasReferenceAudioIdentity"] is True
    assert keyframe["referenceImageContentType"] == "image/png"
    assert keyframe["referenceAudioIdentityContentType"] == "audio/mpeg"
    assert keyframe["referenceAudioContentType"] == "audio/mpeg"
    assert [
        (call["path"], call["params"]) for call in client.rest.calls if call["method"] == "GET"
    ] == [
        (
            "/v1/image/uploadUrl",
            {
                "imageId": ANY,
                "jobId": project.id,
                "type": "referenceImage",
                "contentType": "image/png",
            },
        ),
        (
            "/v1/media/uploadUrl",
            {
                "jobId": project.id,
                "type": "referenceAudio",
                "contentType": "audio/mpeg",
            },
        ),
    ]
    puts = [call for call in client.rest.calls if call["method"] == "PUT"]
    assert [call["content_type"] for call in puts] == ["image/png", "audio/mpeg"]


@pytest.mark.asyncio
async def test_krea_identity_edit_uploads_context_image_and_enforces_limit() -> None:
    client = FakeClient([{"data": {"uploadUrl": "https://upload.example/context"}}])
    api = ProjectsApi(client)
    api.get_model_options = AsyncMock(return_value=model_options("image"))

    project = await api.create(
        type="image",
        model_id="krea2_identity_edit_v1_2",
        positive_prompt="Change the jacket to blue and preserve identity.",
        number_of_media=1,
        width=1024,
        height=1024,
        steps=10,
        guidance=1,
        token_type="spark",
        context_images=[PNG],
    )

    request_type, request = client.socket.sent[-1]
    assert request_type == "jobRequest"
    keyframe = request["keyFrames"][0]
    assert keyframe["modelID"] == "krea2_identity_edit_v1_2"
    assert keyframe["guidanceScale"] == 1
    assert keyframe["hasContextImage1"] is True
    assert keyframe["hasContextImage2"] is False
    assert request["tokenType"] == "spark"
    assert client.rest.calls[0] == {
        "method": "GET",
        "path": "/v1/image/uploadUrl",
        "params": {
            "imageId": ANY,
            "jobId": project.id,
            "type": "contextImage1",
            "contentType": "image/png",
        },
    }
    assert client.rest.calls[1]["method"] == "PUT"
    assert client.rest.calls[1]["content_type"] == "image/png"

    with pytest.raises(ApiError, match="Up to 2 context images"):
        await api.create(
            type="image",
            model_id="krea2_identity_edit_v1_2",
            positive_prompt="Too many references.",
            number_of_media=1,
            context_images=[PNG, PNG, PNG],
        )


@pytest.mark.asyncio
async def test_create_normalizes_nested_control_net_python_names() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    api.get_model_options = AsyncMock(return_value=model_options("image"))

    await api.create(
        type="image",
        model_id="flux1-schnell-fp8",
        positive_prompt="line art",
        number_of_media=1,
        control_net={
            "name": "canny",
            "image": True,
            "strength": 0.4,
            "guidance_start": 0.2,
            "guidance_end": 0.8,
            "mode": "balanced",
        },
    )

    control = client.socket.sent[-1][1]["keyFrames"][0]["currentControlNetsJob"][0]
    assert control["controlGuidanceStart"] == 0.2
    assert control["controlGuidanceEnd"] == 0.8


@pytest.mark.asyncio
async def test_project_state_tracks_monotonic_progress_and_completes_after_jobs_finish() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    project = Project(
        {
            "type": "image",
            "modelId": "flux1-schnell-fp8",
            "positivePrompt": "bird",
            "numberOfMedia": 1,
            "steps": 4,
        },
        api,
    )
    api._projects.append(project)
    completed: list[list[str]] = []
    project.on("completed", completed.append)

    api._handle_job_state({"jobID": project.id, "imgID": "image-1", "type": "jobStarted"})
    job = project.job("image-1")
    assert job is not None
    api._handle_job_progress({"jobID": project.id, "imgID": job.id, "step": 2, "stepCount": 4})
    api._handle_job_progress({"jobID": project.id, "imgID": job.id, "step": 1, "stepCount": 4})
    assert job.step == 2
    assert job.progress == project.progress == 50

    waiting = asyncio.create_task(project.wait_for_completion())
    await api._apply_job_result(
        {
            "jobID": project.id,
            "imgID": job.id,
            "resultUrl": "https://cdn.example/result.png",
            "performedStepCount": 4,
            "lastSeed": "123",
        }
    )
    assert not waiting.done()
    api._handle_job_state({"jobID": project.id, "type": "jobCompleted"})

    assert await waiting == ["https://cdn.example/result.png"]
    assert completed[-1] == ["https://cdn.example/result.png"]
    assert project.status == "completed"
    assert job.status == "completed"
    assert job.seed == 123
    assert project.progress == 100


@pytest.mark.asyncio
async def test_external_progress_ignores_booleans_and_eta_provides_progress_fallback() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    project = Project(
        {
            "type": "video",
            "modelId": "seedance-2-0",
            "positivePrompt": "coast",
            "numberOfMedia": 1,
        },
        api,
    )
    api._projects.append(project)
    api._handle_job_state({"jobID": project.id, "imgID": "video-1", "type": "jobStarted"})
    job = project.job("video-1")
    assert job is not None

    api._handle_job_progress({"jobID": project.id, "imgID": job.id, "progress": 0.5})
    assert job.progress == 50
    api._handle_job_progress({"jobID": project.id, "imgID": job.id, "progress": True})
    assert job.progress == 50

    job._update({"externalProgress": None, "step": 0, "stepCount": 0})
    api._handle_job_eta({"jobID": project.id, "imgID": job.id, "etaSeconds": 120})
    assert job.eta is not None
    assert job.eta_seconds is not None
    assert 1 <= job.progress <= 95


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "model_id", "output_format", "endpoint", "content_type", "id_key"),
    [
        (
            "image",
            "flux1-schnell-fp8",
            "webp",
            "/v1/image/downloadUrl",
            "image/webp",
            "imageId",
        ),
        (
            "audio",
            "ace_step_1.5_turbo",
            "wav",
            "/v1/media/downloadUrl",
            "audio/wav",
            "id",
        ),
    ],
)
async def test_result_fallback_download_uses_output_content_type(
    kind: str,
    model_id: str,
    output_format: str,
    endpoint: str,
    content_type: str,
    id_key: str,
) -> None:
    client = FakeClient([{"data": {"downloadUrl": "https://cdn.example/result"}}])
    api = ProjectsApi(client)
    project = Project(
        {
            "type": kind,
            "modelId": model_id,
            "positivePrompt": "result",
            "numberOfMedia": 1,
            "outputFormat": output_format,
        },
        api,
    )
    api._projects.append(project)
    job = project._add_job(
        {
            "id": "media-1",
            "projectId": project.id,
            "status": "processing",
            "step": 0,
            "stepCount": 1,
        }
    )

    await api._apply_job_result({"jobID": project.id, "imgID": job.id})

    assert job.result_url == "https://cdn.example/result"
    assert client.rest.calls[0] == {
        "method": "GET",
        "path": endpoint,
        "params": {
            "jobId": project.id,
            id_key: job.id,
            "type": "complete",
            "contentType": content_type,
        },
    }


@pytest.mark.asyncio
async def test_project_failure_preserves_structured_subscription_error() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    project = Project(
        {
            "type": "image",
            "modelId": "flux1-schnell-fp8",
            "positivePrompt": "bird",
            "numberOfMedia": 1,
        },
        api,
    )
    api._projects.append(project)
    waiting = asyncio.create_task(project.wait_for_completion())

    api._handle_job_error(
        {
            "jobID": project.id,
            "error": "4078",
            "error_message": "Subscription required",
            "subscriptionLimit": True,
            "requiredPlans": ["unlimited"],
            "feature": "image",
        }
    )

    with pytest.raises(ProjectError) as raised:
        await waiting
    assert raised.value.code == 4078
    assert raised.value.error["subscriptionLimit"] is True
    assert raised.value.error["requiredPlans"] == ["unlimited"]


@pytest.mark.asyncio
async def test_project_rest_sync_recovers_completed_jobs_and_direct_result_aliases() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    project = Project(
        {
            "type": "image",
            "modelId": "flux1-schnell-fp8",
            "positivePrompt": "recovered",
            "numberOfMedia": 2,
            "numberOfPreviews": 1,
            "steps": 10,
        },
        api,
    )
    api._projects.append(project)
    api.get = AsyncMock(
        return_value={
            "status": "completed",
            "imageCount": 1,
            "stepCount": 5,
            "previewCount": 0,
            "completedWorkerJobs": [
                {
                    "imgID": "rest-image",
                    "status": "jobCompleted",
                    "performedSteps": 5,
                    "worker": {"name": "worker-one"},
                    "seedUsed": 123,
                    "triggeredNSFWFilter": False,
                    "imageFile": "https://cdn.example/recovered.webp",
                }
            ],
        }
    )

    waiting = asyncio.create_task(project.wait_for_completion())
    await project._sync_to_server()

    assert await waiting == ["https://cdn.example/recovered.webp"]
    assert project.status == "completed"
    assert project.params["numberOfMedia"] == 1
    assert project.params["numberOfPreviews"] == 0
    assert project.params["steps"] == 5
    job = project.job("rest-image")
    assert job is not None
    assert job.status == "completed"
    assert job.worker_name == "worker-one"
    assert job.seed == 123
    assert job.result_url == "https://cdn.example/recovered.webp"


@pytest.mark.asyncio
async def test_project_timeout_retries_then_notifies_server_and_fails_local_state() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    project = Project(
        {
            "type": "video",
            "modelId": "seedance-2-0",
            "positivePrompt": "timeout",
            "numberOfMedia": 1,
        },
        api,
    )
    api._projects.append(project)
    job = project._add_job(
        {
            "id": "timed-job",
            "projectId": project.id,
            "status": "processing",
            "step": 0,
            "stepCount": 0,
        }
    )
    api.get = AsyncMock(
        side_effect=ApiError(
            404,
            {"status": "error", "message": "not ready", "errorCode": 404},
        )
    )
    api._list_active_project_ids = AsyncMock(return_value=[])
    waiting = asyncio.create_task(project.wait_for_completion())
    project._last_updated = project_now() - timedelta(minutes=3)

    await project._check_for_timeout()
    await project._check_for_timeout()
    assert project.status != "failed"
    await project._check_for_timeout()

    with pytest.raises(ProjectError, match="timed out"):
        await waiting
    assert job.status == "failed"
    assert project.status == "failed"
    assert client.socket.sent == [
        (
            "jobError",
            {
                "jobID": project.id,
                "error": "artistCanceled",
                "error_message": "artistCanceled",
                "isFromWorker": False,
            },
        )
    ]


@pytest.mark.asyncio
async def test_projects_api_emits_normalized_public_project_and_job_events() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    project_events: list[dict[str, Any]] = []
    job_events: list[dict[str, Any]] = []
    api.on("project", project_events.append)
    api.on("job", job_events.append)

    api._handle_job_state({"type": "queued", "jobID": "untracked", "queuePosition": 4})
    api._handle_job_state(
        {
            "type": "initiatingModel",
            "jobID": "untracked",
            "imgID": "job-1",
            "workerName": "worker",
            "positivePrompt": "prompt",
            "negativePrompt": "negative",
            "jobIndex": 0,
            "preparation": {"download": 50},
        }
    )
    api._handle_job_progress(
        {
            "jobID": "untracked",
            "imgID": "job-1",
            "step": 2,
            "stepCount": 4,
            "progress": 0.5,
        }
    )
    api._handle_job_eta({"jobID": "untracked", "imgID": "job-1", "etaSeconds": 12})
    await api._apply_job_result(
        {
            "jobID": "untracked",
            "imgID": "job-1",
            "resultUrl": "https://cdn.example/direct.mp4",
            "performedStepCount": 4,
            "lastSeed": "99",
        }
    )
    api._handle_job_error(
        {
            "jobID": "untracked-error",
            "imgID": "job-2",
            "error": "workerDisconnected",
            "error_message": "gone",
        }
    )

    assert project_events == [{"type": "queued", "projectId": "untracked", "queuePosition": 4}]
    assert [event["type"] for event in job_events] == [
        "initiating",
        "progress",
        "jobETA",
        "completed",
        "error",
    ]
    assert job_events[0]["preparation"] == {"download": 50}
    assert job_events[1] == {
        "type": "progress",
        "projectId": "untracked",
        "jobId": "job-1",
        "step": 2,
        "stepCount": 4,
        "progress": 0.5,
    }
    assert job_events[3] == {
        "type": "completed",
        "projectId": "untracked",
        "jobId": "job-1",
        "resultUrl": "https://cdn.example/direct.mp4",
        "isNSFW": False,
        "userCanceled": False,
        "steps": 4,
        "seed": 99,
    }
    assert job_events[4]["error"] == {
        "code": 5002,
        "originalCode": "workerDisconnected",
        "message": "gone",
    }


@pytest.mark.asyncio
async def test_job_enhance_and_enhanced_image_surface_use_python_and_js_aliases() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    parent = Project(
        {
            "type": "image",
            "modelId": "flux1-schnell-fp8",
            "positivePrompt": "original",
            "stylePrompt": "original style",
            "numberOfMedia": 1,
            "tokenType": "sogni",
            "sizePreset": "square",
        },
        api,
    )
    source = parent._add_job(
        {
            "id": "source",
            "projectId": parent.id,
            "status": "completed",
            "step": 5,
            "stepCount": 5,
            "seed": 42,
            "resultUrl": "https://cdn.example/source.png",
        }
    )
    enhanced = Project(
        {
            "type": "image",
            "modelId": "flux1-schnell-fp8",
            "positivePrompt": "enhanced",
            "numberOfMedia": 1,
        },
        api,
    )
    enhanced_job = enhanced._add_job(
        {
            "id": "enhanced",
            "projectId": enhanced.id,
            "status": "completed",
            "step": 5,
            "stepCount": 5,
            "resultUrl": "https://cdn.example/enhanced.png",
        }
    )
    enhanced._update({"status": "completed"})
    api.create = AsyncMock(return_value=enhanced)

    result = await source.enhance("light", positive_prompt="override")

    assert result == "https://cdn.example/enhanced.png"
    submitted = api.create.await_args.args[0]
    assert submitted["modelId"] == "flux1-schnell-fp8"
    assert submitted["positivePrompt"] == "override"
    assert submitted["stylePrompt"] == "original style"
    assert submitted["tokenType"] == "sogni"
    assert submitted["seed"] == 42
    assert submitted["startingImage"] == b"result"
    assert submitted["startingImageStrength"] == pytest.approx(0.85)
    assert submitted["sizePreset"] == "square"
    assert source.image_url == source.imageUrl == "https://cdn.example/source.png"
    assert source.enhanced_image is not None
    assert source.enhancedImage is not None
    assert source.enhanced_image["result"] == enhanced_job.result_url
    assert await source.enhanced_image["get_result_url"]() == enhanced_job.result_url
    assert await source.enhancedImage["getResultUrl"]() == enhanced_job.result_url


@pytest.mark.asyncio
async def test_estimate_enhancement_cost_delegates_to_image_estimator_defaults() -> None:
    api = ProjectsApi(FakeClient())
    expected = {"token": "1", "usd": "2", "spark": "3", "sogni": "4"}
    api.estimate_cost = AsyncMock(return_value=expected)

    assert await api.estimate_enhancement_cost("heavy", "sogni") is expected
    assert await api.estimateEnhancementCost("light") is expected
    assert api.estimate_cost.await_args_list[0].kwargs == {
        "network": "fast",
        "token_type": "sogni",
        "model": "flux1-schnell-fp8",
        "image_count": 1,
        "step_count": 5,
        "preview_count": 0,
        "cn_enabled": False,
        "starting_image_strength": 0.49,
    }
    assert api.estimate_cost.await_args_list[1].kwargs["starting_image_strength"] == 0.15
