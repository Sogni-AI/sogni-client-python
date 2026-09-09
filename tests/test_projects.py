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


def test_speech_request_keeps_voice_controls_and_clone_reference() -> None:
    message = create_job_request_message(
        "speech-1",
        {
            "type": "audio",
            "modelId": "qwen3_tts_1.7b_voice_clone_bf16",
            "positivePrompt": "The compute is borrowed. The voice is yours.",
            "numberOfMedia": 1,
            "language": "Auto",
            "creativity": 0.9,
            "speaker": "ryan",
            "instruct": "warm and unhurried",
            "referenceText": "These are the exact words in the clip.",
            "referenceAudio": True,
            "outputFormat": "wav",
        },
        model_options("audio"),
    )

    keyframe = message["keyFrames"][0]
    assert keyframe["speaker"] == "ryan"
    assert keyframe["instruct"] == "warm and unhurried"
    assert keyframe["referenceText"] == "These are the exact words in the clip."
    assert keyframe["hasReferenceAudio"] is True
    assert keyframe["language"] == "Auto"
    assert keyframe["creativity"] == 0.9
    assert keyframe["comfySampler"] is None
    assert keyframe["comfyScheduler"] is None
    assert message["outputFormat"] == "wav"


@pytest.mark.asyncio
async def test_speech_model_options_expose_voice_controls_without_music_controls() -> None:
    api = ProjectsApi(FakeClient())
    api.get_supported_models = AsyncMock(
        return_value=[
            {
                "id": "qwen3_tts_1.7b_voice_clone_bf16",
                "tier": "qwen3-tts-clone",
            }
        ]
    )
    api._get_model_tiers = AsyncMock(
        return_value={
            "qwen3-tts-clone": {
                "type": "audio",
                "steps": {"min": 1, "max": 1, "default": 1},
                "language": {"allowed": ["Auto", "English"], "default": "Auto"},
                "creativity": {"min": 0.1, "max": 2, "decimals": 1, "default": 0.9},
                "speaker": {"allowed": ["serena", "ryan"], "default": "serena"},
                "instruct": {"maxLength": 512, "required": False},
                "referenceText": {"maxLength": 1024},
                "acceptInputAudio": True,
                "requiresReferenceAudio": True,
            }
        }
    )

    options = await api.get_model_options("qwen3_tts_1.7b_voice_clone_bf16")

    assert options["type"] == "audio"
    assert options["speaker"] == {"allowed": ["serena", "ryan"], "default": "serena"}
    assert options["instruct"] == {"maxLength": 512, "required": False}
    assert options["referenceText"] == {"maxLength": 1024}
    assert options["acceptsReferenceAudio"] is True
    assert options["requiresReferenceAudio"] is True
    assert "duration" not in options
    assert "sampler" not in options
    assert "scheduler" not in options


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
            "duration": 30,
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
    # 30 seconds is the maximum smartDuration used to reach implicitly, and the
    # retired field never reaches the wire.
    assert "smartDuration" not in wan3

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

    # smartDuration is retired: it reserved the 30s maximum at admission while the
    # render usually came back far shorter. Rejected at the SDK boundary now.
    with pytest.raises(ApiError, match="smartDuration has been retired"):
        create_job_request_message(
            "wan3-smart-duration",
            {
                "type": "video",
                "modelId": "wan3.0-video",
                "positivePrompt": "Let the model pick.",
                "numberOfMedia": 1,
                "smartDuration": True,
            },
            model_options("video"),
        )

    # The retirement rejects the field's presence, not its value, so an explicit
    # false must fail the same way instead of quietly reading as "off".
    with pytest.raises(ApiError, match="smartDuration has been retired"):
        create_job_request_message(
            "wan3-smart-duration-false",
            {
                "type": "video",
                "modelId": "wan3.0-video",
                "positivePrompt": "Explicitly off.",
                "numberOfMedia": 1,
                "smartDuration": False,
            },
            model_options("video"),
        )

    # Wan 3.0 Enhanced is its own model id, with no document/link context and no
    # watermark control, and it accepts an end frame without a first frame.
    enhanced = create_job_request_message(
        "wan3-enhanced",
        {
            "type": "video",
            "modelId": "wan3.0-spicy-video",
            "positivePrompt": "Enhanced render.",
            "numberOfMedia": 1,
            "duration": 10,
            "referenceImageEnd": True,
        },
        model_options("video"),
    )["keyFrames"][0]
    assert enhanced["frames"] == 301

    for field, message in (
        ("referenceLinkUrl", "does not accept document or webpage references"),
        ("watermark", "does not expose a watermark option"),
    ):
        value = "https://example.com/ref" if field == "referenceLinkUrl" else False
        with pytest.raises(ApiError, match=message):
            create_job_request_message(
                f"wan3-enhanced-{field}",
                {
                    "type": "video",
                    "modelId": "wan3.0-spicy-video",
                    "positivePrompt": "Enhanced render.",
                    "numberOfMedia": 1,
                    "duration": 10,
                    field: value,
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
async def test_voice_clone_uploads_reference_audio_and_annotates_its_type() -> None:
    client = FakeClient([{"data": {"uploadUrl": "https://upload.example/voice"}}])
    api = ProjectsApi(client)
    api.get_model_options = AsyncMock(return_value=model_options("audio"))

    project = await api.create(
        type="audio",
        model_id="qwen3_tts_1.7b_voice_clone_bf16",
        positive_prompt="The compute is borrowed. The voice is yours.",
        number_of_media=1,
        reference_audio=MP3,
        reference_text="These are the exact words in the clip.",
        output_format="wav",
    )

    request_type, request = client.socket.sent[-1]
    assert request_type == "jobRequest"
    keyframe = request["keyFrames"][0]
    assert keyframe["hasReferenceAudio"] is True
    assert keyframe["referenceAudioContentType"] == "audio/mpeg"
    assert keyframe["referenceText"] == "These are the exact words in the clip."
    assert [
        (call["path"], call["params"]) for call in client.rest.calls if call["method"] == "GET"
    ] == [
        (
            "/v1/media/uploadUrl",
            {
                "jobId": project.id,
                "type": "referenceAudio",
                "contentType": "audio/mpeg",
            },
        )
    ]
    put = next(call for call in client.rest.calls if call["method"] == "PUT")
    assert put["data"] == MP3
    assert put["content_type"] == "audio/mpeg"


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
@pytest.mark.parametrize(
    "status,reason", [("cancelled", "artistCanceled"), ("errored", "Generation failed")]
)
async def test_project_rest_terminal_status_settles_completion(status: str, reason: str) -> None:
    api = ProjectsApi(FakeClient())
    project = Project({"type": "image", "numberOfMedia": 1, "steps": 4}, api)
    api.get = AsyncMock(
        return_value={"status": status, "reason": reason, "completedWorkerJobs": []}
    )
    waiting = asyncio.create_task(project.wait_for_completion())
    await project._sync_to_server()
    with pytest.raises(ProjectError, match=reason):
        await waiting
    with pytest.raises(ProjectError, match=reason):
        await project.wait_for_completion()
    assert project.finished
    assert project._timeout_handle is None


@pytest.mark.asyncio
async def test_live_cancellation_settles_completion_without_a_failed_event() -> None:
    project = Project({"type": "image", "numberOfMedia": 1}, ProjectsApi(FakeClient()))
    failed = []
    project.on("failed", failed.append)
    waiting = asyncio.create_task(project.wait_for_completion())
    project._update({"status": "canceled"})
    with pytest.raises(ProjectError, match="Project canceled"):
        await waiting
    assert failed == []


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
        "nsfwDetected": False,
        "nsfwSources": [],
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


def test_minimax_h3_step_counts_are_fixed_per_acceleration_tier() -> None:
    tiers = {
        "minimax-h3-fl2va-fp8_t2v": (20, ""),
        "minimax-h3-fl2va-fp8_t2v_balanced": (8, " Balanced"),
        "minimax-h3-fl2va-fp8_t2v_turbo": (4, " Turbo"),
        "minimax-h3-fastvideo-int8_t2v_turbo": (4, " Turbo"),
        "minimax-h3-ref2va-fp8_r2v_balanced": (8, " Balanced"),
    }
    for model_id, (steps, label) in tiers.items():
        params: dict[str, Any] = {
            "type": "video",
            "modelId": model_id,
            "positivePrompt": "a kite",
            "numberOfMedia": 1,
            "steps": steps,
        }
        if model_id.endswith("_r2v_balanced"):
            params["referenceImage"] = True
        # The matching step count is accepted.
        create_job_request_message("h3-steps", params, model_options("video"))

        with pytest.raises(ApiError, match=f"MiniMax H3{label} steps are fixed at {steps}"):
            create_job_request_message(
                "h3-steps-bad", {**params, "steps": steps + 1}, model_options("video")
            )


def test_minimax_h3_reference_video_durations_are_optional_preflight_hints() -> None:
    params: dict[str, Any] = {
        "type": "video",
        "modelId": "minimax-h3-ref2va-fp8_r2v",
        "positivePrompt": "a subject walks",
        "numberOfMedia": 1,
        "referenceImage": True,
        "referenceVideo": True,
    }
    # Socket probes the uploaded media, so omitting the hints is valid.
    create_job_request_message("h3-r2v", params, model_options("video"))

    # When supplied they are still validated early.
    with pytest.raises(ApiError, match="must contain one entry for each uploaded reference video"):
        create_job_request_message(
            "h3-r2v-bad",
            {**params, "referenceVideoDurations": [3.0, 4.0]},
            model_options("video"),
        )
    with pytest.raises(ApiError, match=r"referenceVideoDurations\[0\] must be between 2 and 15"):
        create_job_request_message(
            "h3-r2v-range",
            {**params, "referenceVideoDurations": [0.5]},
            model_options("video"),
        )


def test_cost_estimates_carry_live_benchmark_seconds_only_when_the_server_has_samples() -> None:
    quote = {
        "quote": {
            "project": {
                "costInToken": "1",
                "costInUSD": "2",
                "costInSpark": "3",
                "costInSogni": "4",
            }
        }
    }
    assert ProjectsApi._cost(quote) == {
        "token": "1",
        "usd": "2",
        "spark": "3",
        "sogni": "4",
    }

    benchmarked = ProjectsApi._cost(
        {
            **quote,
            "benchmark": {
                "estimatedRenderTimeSec": 91.5,
                "medianRenderTimeSec": 88,
                "sampleCount": 42,
                "confidence": 0.8,
                "estimatedWaitTimeSec": 30,
                "estimatedTotalTimeSec": 121.5,
            },
        }
    )
    assert benchmarked["estimatedRenderSeconds"] == 91.5
    assert benchmarked["estimatedTotalSeconds"] == 121.5

    # A render-time-only benchmark omits the total rather than guessing one.
    render_only = ProjectsApi._cost({**quote, "benchmark": {"estimatedRenderTimeSec": 10}})
    assert render_only["estimatedRenderSeconds"] == 10
    assert "estimatedTotalSeconds" not in render_only


async def test_labelled_sensitive_media_is_delivered_while_withheld_media_is_not() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    project = Project(
        {"type": "image", "modelId": "flux1-schnell-fp8", "numberOfMedia": 2},
        api,
    )
    api._projects.append(project)

    # Filter OFF: the signal fired but the media was delivered and merely labelled.
    await api._apply_job_result(
        {
            "jobID": project.id,
            "imgID": "JOB-LABELLED",
            "resultUrl": "https://cdn.example/labelled.png",
            "nsfwDetected": True,
            "nsfwSources": ["image"],
        }
    )
    labelled = project.job("JOB-LABELLED")
    assert labelled is not None
    # isNSFW keeps its historical meaning so upgrading changes nothing.
    assert labelled.is_nsfw is False
    assert labelled.nsfw_detected is True
    assert labelled.nsfw_sources == ["image"]
    assert labelled.is_withheld is False
    assert labelled.has_result_media is True
    assert labelled.result_url == "https://cdn.example/labelled.png"

    # Filter ON: the server withheld the media and there is nothing to fetch.
    await api._apply_job_result(
        {
            "jobID": project.id,
            "imgID": "JOB-WITHHELD",
            "triggeredNSFWFilter": True,
        }
    )
    withheld = project.job("JOB-WITHHELD")
    assert withheld is not None
    assert withheld.is_nsfw is True
    assert withheld.nsfw_detected is False
    assert withheld.is_withheld is True
    assert withheld.has_result_media is False
    assert withheld.result_url is None
    with pytest.raises(RuntimeError, match="did not pass NSFW filter"):
        await withheld.enhance("light")


def sam3_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "type": "image",
        "modelId": "sam3_image_segment_bf16",
        "positivePrompt": "",
        "numberOfMedia": 4,
        "numberOfPreviews": 5,
        "outputFormat": "jpg",
        "startingImage": True,
        "sam3Prompt": {"points": [{"x": 0.42, "y": 0.61, "label": "positive"}]},
    }
    params.update(overrides)
    return params


def pixal3d_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "type": "image",
        "modelId": "pixal3d_int8_i23d",
        "positivePrompt": "the red ceramic teapot",
        "numberOfMedia": 1,
        "startingImage": True,
    }
    params.update(overrides)
    return params


def test_sam3_segmentation_pins_its_single_mask_request_shape() -> None:
    message = create_job_request_message("sam3-wire", sam3_params(), model_options("image"))

    # One source image produces exactly one lossless mask PNG, whatever the
    # caller asked for.
    assert message["numberOfImages"] == 1
    assert message["previews"] == 0
    assert message["outputFormat"] == "png"
    keyframe = message["keyFrames"][0]
    assert keyframe["hasStartingImage"] is True
    assert keyframe["sam3Prompt"] == {
        "points": [{"x": 0.42, "y": 0.61, "label": "positive"}],
        "boxes": [],
        "threshold": 0.5,
        "multimask": True,
        # An omitted applyMask must serialize as an explicit false, not absence.
        "applyMask": False,
    }
    # An omitted cap stays off the wire entirely rather than being defaulted.
    assert "maxInstances" not in keyframe["sam3Prompt"]


def test_sam3_prompt_accepts_apply_mask_and_max_instances() -> None:
    """The JS SDK's root-level key gate rejected both names in 5.32.0 and 5.33.0.

    They were validated and serialized a hundred lines further down, so the
    whole feature was unreachable. This pins the fixed end state.
    """

    cutout = create_job_request_message(
        "sam3-apply-mask",
        sam3_params(sam3Prompt={"text": "the teapot", "applyMask": True}),
        model_options("image"),
    )
    assert cutout["keyFrames"][0]["sam3Prompt"] == {
        "points": [],
        "boxes": [],
        "text": "the teapot",
        "threshold": 0.5,
        "applyMask": True,
    }

    capped = create_job_request_message(
        "sam3-max-instances",
        sam3_params(sam3Prompt={"text": "the teapots", "maxInstances": 4}),
        model_options("image"),
    )
    assert capped["keyFrames"][0]["sam3Prompt"]["maxInstances"] == 4
    assert capped["keyFrames"][0]["sam3Prompt"]["applyMask"] is False

    for max_instances in (0, 17):
        with pytest.raises(
            ValueError, match="sam3Prompt.maxInstances must be an integer from 1 to 16"
        ):
            create_job_request_message(
                "sam3-max-instances-range",
                sam3_params(sam3Prompt={"text": "the teapots", "maxInstances": max_instances}),
                model_options("image"),
            )
    # A boolean is not an integer on the wire even though bool subclasses int.
    with pytest.raises(ValueError, match="sam3Prompt.maxInstances must be an integer from 1 to 16"):
        create_job_request_message(
            "sam3-max-instances-bool",
            sam3_params(sam3Prompt={"text": "the teapots", "maxInstances": True}),
            model_options("image"),
        )
    with pytest.raises(ValueError, match="sam3Prompt.applyMask must be a boolean"):
        create_job_request_message(
            "sam3-apply-mask-type",
            sam3_params(sam3Prompt={"text": "the teapot", "applyMask": "yes"}),
            model_options("image"),
        )


def test_sam3_prompt_still_rejects_a_name_the_contract_has_no_field_for() -> None:
    with pytest.raises(ValueError, match="sam3Prompt contains unsupported fields: bogus"):
        create_job_request_message(
            "sam3-unknown-root-key",
            sam3_params(sam3Prompt={"text": "the teapot", "bogus": 1}),
            model_options("image"),
        )


def test_sam3_negative_box_excludes_one_instance_of_a_text_concept() -> None:
    excluded = create_job_request_message(
        "sam3-negative-box",
        sam3_params(
            sam3Prompt={
                "text": "the teapots",
                "boxes": [{"x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.4, "label": "negative"}],
            }
        ),
        model_options("image"),
    )
    assert excluded["keyFrames"][0]["sam3Prompt"]["boxes"] == [
        {"x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.4, "label": "negative"}
    ]

    # An absent label leaves every existing caller on its current behavior.
    positive = create_job_request_message(
        "sam3-default-box-label",
        sam3_params(
            sam3Prompt={
                "text": "the teapots",
                "boxes": [{"x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.4}],
            }
        ),
        model_options("image"),
    )
    assert positive["keyFrames"][0]["sam3Prompt"]["boxes"][0]["label"] == "positive"

    with pytest.raises(ValueError, match="sam3Prompt negative boxes require a text prompt"):
        create_job_request_message(
            "sam3-negative-box-without-text",
            sam3_params(
                sam3Prompt={
                    "points": [{"x": 0.4, "y": 0.4, "label": "positive"}],
                    "boxes": [{"x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.4, "label": "negative"}],
                }
            ),
            model_options("image"),
        )


def test_sam3_multimask_is_a_point_only_control() -> None:
    """multimask picks among SAM's whole/part/subpart candidates for one click.

    It never meant anything for a text prompt. The socket drops it silently
    because older SDKs sent it unconditionally, so the SDK boundary is the only
    place that can tell a caller they asked for something meaningless.
    """

    with pytest.raises(ValueError, match="sam3Prompt.multimask requires point prompts"):
        create_job_request_message(
            "sam3-multimask-with-text",
            sam3_params(sam3Prompt={"text": "the teapot", "multimask": True}),
            model_options("image"),
        )
    with pytest.raises(ValueError, match="sam3Prompt.multimask must be a boolean"):
        create_job_request_message(
            "sam3-multimask-type",
            sam3_params(
                sam3Prompt={"points": [{"x": 0.4, "y": 0.4, "label": "positive"}], "multimask": 1}
            ),
            model_options("image"),
        )
    explicit = create_job_request_message(
        "sam3-multimask-off",
        sam3_params(
            sam3Prompt={
                "points": [{"x": 0.4, "y": 0.4, "label": "positive"}],
                "multimask": False,
            }
        ),
        model_options("image"),
    )
    assert explicit["keyFrames"][0]["sam3Prompt"]["multimask"] is False

    declined_for_text = create_job_request_message(
        "sam3-text-with-multimask-off",
        sam3_params(sam3Prompt={"text": "the teapot", "multimask": False}),
        model_options("image"),
    )
    assert "multimask" not in declined_for_text["keyFrames"][0]["sam3Prompt"]


def test_sam3_prompt_validates_points_boxes_text_and_threshold() -> None:
    cases = [
        (
            {"points": [{"x": 0.4, "y": 0.4, "label": "maybe"}]},
            'sam3Prompt.points\\[0\\].label must be "positive" or "negative"',
        ),
        (
            {"points": [{"x": 1.4, "y": 0.4, "label": "positive"}]},
            "sam3Prompt.points\\[0\\].x must be a finite normalized coordinate from 0 to 1",
        ),
        (
            {"points": [{"x": 0.4, "y": 0.4, "label": "positive", "z": 1}]},
            "sam3Prompt.points\\[0\\] contains unsupported fields",
        ),
        ({"points": ["nope"]}, "sam3Prompt.points\\[0\\] must be an object"),
        (
            {"points": [{"x": 0.4, "y": 0.4, "label": "positive"}] * 33},
            "sam3Prompt.points must contain at most 32 entries",
        ),
        (
            {"boxes": [{"x0": 0.5, "y0": 0.2, "x1": 0.3, "y1": 0.4}]},
            "sam3Prompt.boxes\\[0\\] must have x0 < x1 and y0 < y1",
        ),
        (
            {"boxes": [{"x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.4, "nope": 1}]},
            "sam3Prompt.boxes\\[0\\] contains unsupported fields",
        ),
        (
            {"boxes": [{"x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.4, "label": "sideways"}]},
            'sam3Prompt.boxes\\[0\\].label must be "positive" or "negative"',
        ),
        (
            {"boxes": [{"x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.4}] * 17},
            "sam3Prompt.boxes must contain at most 16 entries",
        ),
        ({"text": 7}, "sam3Prompt.text must be a string"),
        ({"text": "   "}, "sam3Prompt.text must contain 1 to 240 characters"),
        ({"text": "a" * 241}, "sam3Prompt.text must contain 1 to 240 characters"),
        ({}, "sam3Prompt requires at least one point, box, or text prompt"),
        (
            {"text": "the teapot", "points": [{"x": 0.4, "y": 0.4, "label": "positive"}]},
            "sam3Prompt cannot combine text and point prompts",
        ),
        (
            {
                "points": [{"x": 0.4, "y": 0.4, "label": "positive"}],
                "boxes": [
                    {"x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.4},
                    {"x0": 0.5, "y0": 0.6, "x1": 0.7, "y1": 0.8},
                ],
            },
            "sam3Prompt supports at most one box when point prompts are present",
        ),
        (
            {"text": "the teapot", "threshold": 1.5},
            "sam3Prompt.threshold must be a finite number from 0 to 1",
        ),
    ]
    for prompt, message in cases:
        with pytest.raises(ValueError, match=message):
            create_job_request_message(
                "sam3-invalid", sam3_params(sam3Prompt=prompt), model_options("image")
            )

    with pytest.raises(ValueError, match="sam3Prompt must be an object"):
        create_job_request_message(
            "sam3-not-an-object", sam3_params(sam3Prompt=["nope"]), model_options("image")
        )
    with pytest.raises(ValueError, match="SAM3 image segmentation requires startingImage"):
        create_job_request_message(
            "sam3-missing-source", sam3_params(startingImage=None), model_options("image")
        )
    with pytest.raises(ValueError, match="SAM3 image segmentation requires sam3Prompt"):
        create_job_request_message(
            "sam3-missing-prompt", sam3_params(sam3Prompt=None), model_options("image")
        )
    with pytest.raises(ValueError, match="sam3Prompt is only supported by sam3_image_segment_bf16"):
        create_job_request_message(
            "sam3-wrong-model",
            sam3_params(modelId="krea2_turbo_fp8_scaled"),
            model_options("image"),
        )


def test_pixal3d_pins_the_glb_output_and_reduce_only_options() -> None:
    message = create_job_request_message("pixal3d-wire", pixal3d_params(), model_options("image"))
    assert message["outputFormat"] == "glb"
    keyframe = message["keyFrames"][0]
    assert keyframe["hasStartingImage"] is True
    assert keyframe["positivePrompt"] == "the red ceramic teapot"

    # Every option may only reduce work: each maximum is the shipped default,
    # so the flat price stays an upper bound.
    reduced = create_job_request_message(
        "pixal3d-options",
        pixal3d_params(
            textureSize=2048,
            meshTargetFaces=60000,
            normalMapSize=1024,
            ambientOcclusionSize=512,
            shapeResolution=1024,
        ),
        model_options("image"),
    )["keyFrames"][0]
    assert reduced["textureSize"] == 2048
    assert reduced["meshTargetFaces"] == 60000
    assert reduced["normalMapSize"] == 1024
    assert reduced["ambientOcclusionSize"] == 512
    assert reduced["shapeResolution"] == 1024

    with pytest.raises(ValueError, match="Pixal3D reconstruction requires startingImage"):
        create_job_request_message(
            "pixal3d-missing-source",
            pixal3d_params(startingImage=None),
            model_options("image"),
        )
    with pytest.raises(ValueError, match="meshTargetFaces must be an integer from 5000 to 700000"):
        create_job_request_message(
            "pixal3d-out-of-range",
            pixal3d_params(meshTargetFaces=700001),
            model_options("image"),
        )
    with pytest.raises(ValueError, match="textureSize is only supported by pixal3d_int8_i23d"):
        create_job_request_message(
            "pixal3d-wrong-model",
            pixal3d_params(modelId="flux1-schnell-fp8", textureSize=2048),
            model_options("image"),
        )


def birefnet_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "type": "image",
        "modelId": "birefnet_image_background_removal_fp16",
        # BiRefNet takes no prompt: it finds the salient foreground on its own.
        "positivePrompt": "",
        "numberOfMedia": 4,
        "numberOfPreviews": 5,
        "outputFormat": "jpg",
        "startingImage": True,
    }
    params.update(overrides)
    return params


def test_birefnet_pins_its_single_matte_request_shape() -> None:
    message = create_job_request_message("birefnet-wire", birefnet_params(), model_options("image"))

    # The graph is deterministic and takes no seed, so four copies of one source
    # are four identical mattes at four times the price.
    assert message["numberOfImages"] == 1
    # There is no diffusion to preview.
    assert message["previews"] == 0
    # A soft matte, and a cutout whose alpha IS that matte: jpg would quantize
    # the first and flatten the second away.
    assert message["outputFormat"] == "png"
    keyframe = message["keyFrames"][0]
    assert keyframe["hasStartingImage"] is True
    # Omitting applyMask serializes as an explicit false, not absence: which
    # artifact the job returns is decided by the request, not by whichever
    # worker build picked it up.
    assert keyframe["applyMask"] is False


def test_birefnet_apply_mask_selects_the_cutout_branch() -> None:
    cutout = create_job_request_message(
        "birefnet-apply-mask", birefnet_params(applyMask=True), model_options("image")
    )
    assert cutout["keyFrames"][0]["applyMask"] is True

    explicit = create_job_request_message(
        "birefnet-explicit-false", birefnet_params(applyMask=False), model_options("image")
    )
    assert explicit["keyFrames"][0]["applyMask"] is False

    with pytest.raises(ValueError, match="applyMask must be a boolean"):
        create_job_request_message(
            "birefnet-non-boolean", birefnet_params(applyMask="yes"), model_options("image")
        )
    with pytest.raises(ValueError, match="BiRefNet background removal requires startingImage"):
        create_job_request_message(
            "birefnet-missing-source",
            birefnet_params(startingImage=None),
            model_options("image"),
        )


def test_birefnet_apply_mask_belongs_to_exactly_one_model() -> None:
    """SAM 3 has its own applyMask, nested inside sam3Prompt.

    They are different fields on different models with different graphs, and the
    top-level one must not silently do nothing anywhere else. Each case below is
    otherwise a valid request for its model, so the rejection is the applyMask
    gate rather than some earlier requirement.
    """

    wrong_model_cases = [
        {"modelId": "flux1-schnell-fp8", "positivePrompt": "a teapot"},
        {"modelId": "sam3_image_segment_bf16", "sam3Prompt": {"text": "the teapot"}},
        {"modelId": "pixal3d_int8_i23d", "positivePrompt": "the red ceramic teapot"},
    ]
    for overrides in wrong_model_cases:
        with pytest.raises(
            ValueError,
            match="applyMask is only supported by birefnet_image_background_removal_fp16",
        ):
            create_job_request_message(
                "birefnet-wrong-model",
                birefnet_params(applyMask=True, **overrides),
                model_options("image"),
            )

    # And a SAM 3 request still carries its own nested applyMask untouched.
    sam3 = create_job_request_message(
        "sam3-nested-apply-mask",
        sam3_params(sam3Prompt={"text": "the teapot", "applyMask": True}),
        model_options("image"),
    )
    assert sam3["keyFrames"][0]["sam3Prompt"]["applyMask"] is True
    assert "applyMask" not in sam3["keyFrames"][0]


def test_pixal3d_template_variant_selects_the_reconstruction_graph() -> None:
    """ComfyUI registers one prompt-free BiRefNet Pixal3D graph."""

    unnamed = create_job_request_message(
        "pixal3d-default-variant", pixal3d_params(), model_options("image")
    )
    # An unnamed request must let each worker run its own default graph: a
    # worker resolves only the variants its own manifest declares.
    assert "templateVariant" not in unnamed["keyFrames"][0]

    named = create_job_request_message(
        "pixal3d-variant-i23d-birefnet",
        pixal3d_params(templateVariant="i23d-birefnet"),
        model_options("image"),
    )
    assert named["keyFrames"][0]["templateVariant"] == "i23d-birefnet"

    # The prompt-free graph does not need one.
    birefnet = create_job_request_message(
        "pixal3d-birefnet-no-prompt",
        pixal3d_params(templateVariant="i23d-birefnet", positivePrompt=""),
        model_options("image"),
    )
    assert birefnet["keyFrames"][0]["templateVariant"] == "i23d-birefnet"

    # The retired prompted graph is no longer accepted.
    with pytest.raises(ValueError, match="templateVariant must be one of: i23d-birefnet"):
        create_job_request_message(
            "pixal3d-retired-variant",
            pixal3d_params(templateVariant="i23d", positivePrompt="the subject"),
            model_options("image"),
        )
    # A closed list, not a passthrough.
    with pytest.raises(ValueError, match="templateVariant must be one of: i23d-birefnet"):
        create_job_request_message(
            "pixal3d-unknown-variant",
            pixal3d_params(templateVariant="i23d-experimental"),
            model_options("image"),
        )
    with pytest.raises(ValueError, match="templateVariant is only supported by pixal3d_int8_i23d"):
        create_job_request_message(
            "pixal3d-variant-wrong-model",
            pixal3d_params(modelId="flux1-schnell-fp8", templateVariant="i23d-birefnet"),
            model_options("image"),
        )


def test_world_generation_receipt_binds_its_stage_model_and_hashes() -> None:
    source = "a" * 64
    selection = "b" * 64
    target = create_job_request_message(
        "world-target-still",
        {
            "type": "image",
            "modelId": "krea2_identity_edit_sogni_v0_3_alpha",
            "positivePrompt": "swap the sky",
            "numberOfMedia": 1,
            "appSource": "sogni-world",
            "worldGenerationReceipt": {
                "stage": "target_still",
                "sourceImageSha256": source.upper(),
                "selectionHash": selection,
            },
        },
        model_options("image"),
    )
    assert target["keyFrames"][0]["worldGenerationReceipt"] == {
        "stage": "target_still",
        "sourceImageSha256": source,
        "selectionHash": selection,
    }

    transition = create_job_request_message(
        "world-transition",
        {
            "type": "video",
            "modelId": "minimax-h3-fastvideo-int8_flf2v_turbo",
            "positivePrompt": "walk through the doorway",
            "numberOfMedia": 1,
            "appSource": "sogni-world",
            "referenceImage": True,
            "referenceImageEnd": True,
            "worldGenerationReceipt": {
                "stage": "transition",
                "firstFrameSha256": source,
                "lastFrameSha256": selection,
            },
        },
        model_options("video"),
    )
    assert transition["keyFrames"][0]["worldGenerationReceipt"] == {
        "stage": "transition",
        "firstFrameSha256": source,
        "lastFrameSha256": selection,
    }

    with pytest.raises(ApiError, match='worldGenerationReceipt requires appSource "sogni-world".'):
        create_job_request_message(
            "world-wrong-app-source",
            {
                "type": "image",
                "modelId": "krea2_identity_edit_sogni_v0_3_alpha",
                "positivePrompt": "swap the sky",
                "numberOfMedia": 1,
                "worldGenerationReceipt": {
                    "stage": "target_still",
                    "sourceImageSha256": source,
                    "selectionHash": selection,
                },
            },
            model_options("image"),
        )
    with pytest.raises(
        ApiError,
        match="The target_still receipt requires krea2_identity_edit_sogni_v0_3_alpha.",
    ):
        create_job_request_message(
            "world-wrong-model",
            {
                "type": "image",
                "modelId": "flux1-schnell-fp8",
                "positivePrompt": "swap the sky",
                "numberOfMedia": 1,
                "appSource": "sogni-world",
                "worldGenerationReceipt": {
                    "stage": "target_still",
                    "sourceImageSha256": source,
                    "selectionHash": selection,
                },
            },
            model_options("image"),
        )
    with pytest.raises(
        ApiError, match="worldGenerationReceipt.selectionHash must be a SHA-256 hex digest."
    ):
        create_job_request_message(
            "world-bad-hash",
            {
                "type": "image",
                "modelId": "krea2_identity_edit_sogni_v0_3_alpha",
                "positivePrompt": "swap the sky",
                "numberOfMedia": 1,
                "appSource": "sogni-world",
                "worldGenerationReceipt": {
                    "stage": "target_still",
                    "sourceImageSha256": source,
                    "selectionHash": "nope",
                },
            },
            model_options("image"),
        )
    with pytest.raises(
        ApiError, match="worldGenerationReceipt.stage must be target_still or transition."
    ):
        create_job_request_message(
            "world-bad-stage",
            {
                "type": "image",
                "modelId": "krea2_identity_edit_sogni_v0_3_alpha",
                "positivePrompt": "swap the sky",
                "numberOfMedia": 1,
                "appSource": "sogni-world",
                "worldGenerationReceipt": {"stage": "teleport"},
            },
            model_options("image"),
        )


@pytest.mark.asyncio
async def test_sam3_create_normalizes_the_project_and_surfaces_worker_provenance() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    api.get_model_options = AsyncMock(return_value=model_options("image"))

    project = await api.create(
        type="image",
        model_id="sam3_image_segment_bf16",
        positive_prompt="",
        number_of_media=4,
        number_of_previews=5,
        output_format="jpg",
        starting_image=True,
        sam3_prompt={"text": "the teapot", "apply_mask": True, "max_instances": 2},
    )

    assert project.params["numberOfMedia"] == 1
    assert project.params["numberOfPreviews"] == 0
    assert project.params["outputFormat"] == "png"
    request_type, request = client.socket.sent[-1]
    assert request_type == "jobRequest"
    assert request["numberOfImages"] == 1
    assert request["outputFormat"] == "png"
    # Python names normalize to the camelCase wire spellings the socket expects.
    assert request["keyFrames"][0]["sam3Prompt"] == {
        "points": [],
        "boxes": [],
        "text": "the teapot",
        "threshold": 0.5,
        "applyMask": True,
        "maxInstances": 2,
    }

    api._handle_job_state(
        {
            "type": "jobStarted",
            "jobID": project.id,
            "imgID": "mask-result-1",
            "workerName": "receipt-test-worker",
        }
    )
    await api._apply_job_result(
        {
            "jobID": project.id,
            "imgID": "mask-result-1",
            "resultUrl": "https://cdn.example/mask.png",
            "performedStepCount": 1,
            "lastSeed": "42",
            "triggeredNSFWFilter": False,
            "userCanceled": False,
            "sha256": "c" * 64,
            "sourceImageSha256": "A" * 64,
            "samPromptSha256": "d" * 64,
            "maskRleSha256": "b" * 64,
            "maskWidth": 1024,
            "maskHeight": 576,
            "maskBox": [0.1, 0.2, 0.6, 0.7],
            "maskCoverage": 0.25,
            "maskDetectedCount": 3,
            "maskReturnedCount": 1,
            "maskSelections": [
                {"score": 0.9, "box": [0.1, 0.2, 0.6, 0.7], "coverage": 0.25, "included": True},
                {"score": None, "box": None, "coverage": 0.05, "included": False},
                # Malformed entries are dropped rather than surfaced half-valid.
                {"score": 0.4, "coverage": "lots", "included": True},
                "not-a-selection",
            ],
            # Rejected by their own format checks, so they never reach a caller.
            "selectionHash": "too-short",
            "samVersion": "not a version!",
        }
    )

    job = project.job("mask-result-1")
    assert job is not None
    assert job.provenance == {
        "sha256": "c" * 64,
        "sourceImageSha256": "a" * 64,
        "samPromptSha256": "d" * 64,
        "maskRleSha256": "b" * 64,
        "maskWidth": 1024,
        "maskHeight": 576,
        "maskBox": [0.1, 0.2, 0.6, 0.7],
        "maskCoverage": 0.25,
        "maskDetectedCount": 3,
        "maskReturnedCount": 1,
        "maskSelections": [
            {"score": 0.9, "box": [0.1, 0.2, 0.6, 0.7], "coverage": 0.25, "included": True},
            {"score": None, "box": None, "coverage": 0.05, "included": False},
        ],
    }


@pytest.mark.asyncio
async def test_pixal3d_job_downloads_a_gltf_artifact_and_refuses_enhancement() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    api.get_model_options = AsyncMock(return_value=model_options("image"))
    api.media_download_url = AsyncMock(return_value="https://cdn.example/object.glb")
    api.download_url = AsyncMock(side_effect=AssertionError("Pixal3D must not use /v1/image"))

    project = await api.create(
        type="image",
        model_id="pixal3d_int8_i23d",
        positive_prompt="the red ceramic teapot",
        number_of_media=1,
        starting_image=True,
        # Python names normalize to the camelCase spellings the worker reads.
        texture_size=2048,
        mesh_target_faces=60000,
    )
    request_type, request = client.socket.sent[-1]
    assert request_type == "jobRequest"
    assert request["outputFormat"] == "glb"
    assert request["keyFrames"][0]["textureSize"] == 2048
    assert request["keyFrames"][0]["meshTargetFaces"] == 60000

    api._handle_job_state({"type": "jobStarted", "jobID": project.id, "imgID": "model-result-1"})
    await api._apply_job_result(
        {
            "jobID": project.id,
            "imgID": "model-result-1",
            "performedStepCount": 56,
            "lastSeed": "42",
            "triggeredNSFWFilter": False,
            "userCanceled": False,
        }
    )

    job = project.job("model-result-1")
    assert job is not None
    assert job.type == "model"
    assert job.result_url == "https://cdn.example/object.glb"
    api.media_download_url.assert_awaited_once_with(
        {
            "jobId": project.id,
            "id": "model-result-1",
            "type": "complete",
            "contentType": "model/gltf-binary",
        }
    )
    # A Pixal3D project's params say type 'image', so the params check alone
    # would have let a GLB job be sent for image enhancement.
    with pytest.raises(RuntimeError, match="Enhancement is only available for images"):
        await job.enhance("subtle")


@pytest.mark.asyncio
async def test_model_artifact_predicate_survives_a_catalog_that_says_image() -> None:
    api = ProjectsApi(FakeClient())
    assert api.is_model_artifact_model_id("pixal3d_int8_i23d") is True
    assert api.is_model_artifact_model_id("flux1-schnell-fp8") is False
    # The live /models/list route regressed to media 'image' for Pixal3D, which
    # routed every GLB to the image download endpoint. The pixal3d_ prefix is a
    # positive override precisely so that cannot happen: the prefix is
    # structural, and the image endpoint cannot serve a binary glTF.
    api._supported_models = [
        {"id": "future_i23d", "media": "model"},
        {"id": "pixal3d_int8_i23d", "media": "image"},
    ]
    assert api.is_model_artifact_model_id("pixal3d_int8_i23d") is True
    # The catalog stays authoritative for ids the SDK has no prefix knowledge of.
    assert api.is_model_artifact_model_id("future_i23d") is True
    assert api.is_model_artifact_model_id("flux1-schnell-fp8") is False


def _pixal3d_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "type": "image",
        "modelId": "pixal3d_int8_i23d",
        "positivePrompt": "the red ceramic teapot",
        "numberOfMedia": 1,
        "startingImage": True,
        "steps": 56,
    }
    params.update(overrides)
    return params


def test_pixal3d_never_requests_image_previews() -> None:
    # A 3D reconstruction has no intermediate images to preview.
    message = create_job_request_message(
        "pixal3d-previews", _pixal3d_params(numberOfPreviews=6), model_options("image")
    )
    assert message["previews"] == 0


@pytest.mark.asyncio
async def test_pixal3d_create_pins_previews_in_params_and_on_the_wire() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    api.get_model_options = AsyncMock(return_value=model_options("image"))

    project = await api.create(_pixal3d_params(numberOfPreviews=6))

    assert project.params["numberOfPreviews"] == 0
    request_type, request = client.socket.sent[-1]
    assert request_type == "jobRequest"
    assert request["previews"] == 0


@pytest.mark.asyncio
async def test_pixal3d_job_found_only_by_rest_sync_still_gets_a_result_url() -> None:
    # Parity with the TS SDK: a job learned about for the first time through the
    # REST snapshot must have its download URL minted. A GLB carries none of the
    # legacy *Url aliases the raw record can inline, so nothing else would.
    client = FakeClient()
    api = ProjectsApi(client)
    api.media_download_url = AsyncMock(return_value="https://cdn.example/rest-synced.glb")
    api.download_url = AsyncMock(
        side_effect=AssertionError("Pixal3D must not use the image endpoint")
    )
    project = Project(_pixal3d_params(), api)
    api._projects.append(project)
    api.get = AsyncMock(
        return_value={
            "status": "completed",
            "imageCount": 1,
            "stepCount": 56,
            "previewCount": 0,
            "completedWorkerJobs": [
                {
                    "imgID": "rest-only-1",
                    "status": "jobCompleted",
                    "performedSteps": 56,
                    "worker": {"name": "pixal3d-test-worker"},
                    "seedUsed": 42,
                    "triggeredNSFWFilter": False,
                }
            ],
        }
    )

    waiting = asyncio.create_task(project.wait_for_completion())
    await project._sync_to_server()

    assert await waiting == ["https://cdn.example/rest-synced.glb"]
    job = project.job("rest-only-1")
    assert job is not None
    assert job.type == "model"
    assert job.result_url == "https://cdn.example/rest-synced.glb"
    api.media_download_url.assert_awaited_once_with(
        {
            "jobId": project.id,
            "id": "rest-only-1",
            "type": "complete",
            "contentType": "model/gltf-binary",
        }
    )


def _tracked_job(api: ProjectsApi, **overrides: Any) -> tuple[Project, Any]:
    project = Project(
        {
            "type": "image",
            "modelId": "flux1-schnell-fp8",
            "positivePrompt": "a glass bird",
            "numberOfMedia": 1,
            "steps": 4,
        },
        api,
    )
    data: dict[str, Any] = {
        "id": "job-1",
        "projectId": project.id,
        "status": "processing",
        "step": 0,
        "stepCount": 4,
    }
    data.update(overrides)
    return project, project._add_job(data)


_REST_COMPLETION = {
    "imgID": "job-1",
    "status": "jobCompleted",
    "performedSteps": 4,
    "worker": {"name": "worker-one"},
    "seedUsed": 7,
    "triggeredNSFWFilter": False,
}


@pytest.mark.asyncio
async def test_rest_sync_emits_completed_once_carrying_the_minted_url() -> None:
    # The URL must be minted into the same delta as the status change. Minting
    # after `_update` emitted `completed` with None, and the follow-up update
    # carried no `status` key so it never emitted again.
    api = ProjectsApi(FakeClient())
    api.download_url = AsyncMock(return_value="https://cdn.example/minted.png")
    _project, job = _tracked_job(api)
    seen: list[Any] = []
    job.on("completed", seen.append)

    await job._sync_with_rest_data(dict(_REST_COMPLETION))

    assert seen == ["https://cdn.example/minted.png"]
    assert job.result_url == "https://cdn.example/minted.png"


@pytest.mark.asyncio
async def test_rest_sync_logs_a_failed_result_url_mint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = ProjectsApi(FakeClient())
    api.download_url = AsyncMock(side_effect=RuntimeError("signing service down"))
    _project, job = _tracked_job(api)

    with caplog.at_level("ERROR", logger="sogni_client"):
        await job._sync_with_rest_data(dict(_REST_COMPLETION))

    # The job still settles; the failure is reported rather than swallowed.
    assert job.status == "completed"
    assert job.result_url is None
    assert any("Failed to mint result URL" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_sam3_mask_is_not_enhanceable_and_spends_nothing() -> None:
    # A segmentation job reports type 'image' on an image project, so the media
    # guard alone lets it through: enhance() would download the mask PNG and
    # submit it as the starting image of a paid Flux render.
    client = FakeClient()
    api = ProjectsApi(client)
    api.create = AsyncMock(
        side_effect=AssertionError("a rejected enhancement must not create a project")
    )
    project = Project(
        {
            "type": "image",
            "modelId": "sam3_image_segment_bf16",
            "positivePrompt": "",
            "numberOfMedia": 1,
            "steps": 1,
        },
        api,
    )
    job = project._add_job(
        {
            "id": "mask-result-1",
            "projectId": project.id,
            "status": "completed",
            "step": 1,
            "stepCount": 1,
            "resultUrl": "https://cdn.example/mask.png",
        }
    )
    assert job.type == "image"

    with pytest.raises(RuntimeError, match="Enhancement is not available for segmentation masks"):
        await job.enhance("medium")

    api.create.assert_not_awaited()
    assert client.socket.sent == []


@pytest.mark.asyncio
async def test_birefnet_create_normalizes_the_project_and_refuses_enhancement() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    api.get_model_options = AsyncMock(return_value=model_options("image"))

    project = await api.create(birefnet_params(applyMask=True))

    # The Project's own params must agree with the request that was sent.
    assert project.params["numberOfMedia"] == 1
    assert project.params["numberOfPreviews"] == 0
    assert project.params["outputFormat"] == "png"
    request_type, request = client.socket.sent[-1]
    assert request_type == "jobRequest"
    assert request["numberOfImages"] == 1
    assert request["previews"] == 0
    assert request["outputFormat"] == "png"
    assert request["keyFrames"][0]["applyMask"] is True

    api.create = AsyncMock(
        side_effect=AssertionError("a rejected enhancement must not create a project")
    )
    job = project._add_job(
        {
            "id": "birefnet-result-1",
            "projectId": project.id,
            "status": "completed",
            "step": 1,
            "stepCount": 1,
            "resultUrl": "https://cdn.example/cutout.png",
        }
    )
    assert job.type == "image"
    # A matte reports type 'image' on an image project, so the media guard alone
    # lets it through: enhance() would download it and submit it as the starting
    # image of a paid render.
    sent_before = len(client.socket.sent)
    with pytest.raises(RuntimeError, match="Enhancement is not available for segmentation masks"):
        await job.enhance("medium")
    api.create.assert_not_awaited()
    assert len(client.socket.sent) == sent_before


@pytest.mark.asyncio
async def test_create_accepts_an_explicit_control_net_of_none() -> None:
    # The `{}` default only applies when the key is absent, so an explicit None
    # used to raise AttributeError out of create().
    client = FakeClient()
    api = ProjectsApi(client)
    api.get_model_options = AsyncMock(return_value=model_options("image"))

    project = await api.create(
        {
            "type": "image",
            "modelId": "flux1-schnell-fp8",
            "positivePrompt": "a glass bird",
            "numberOfMedia": 1,
            "controlNet": None,
        }
    )

    assert project.params["controlNet"] is None
    request_type, request = client.socket.sent[-1]
    assert request_type == "jobRequest"
    assert "cnImage" not in request["keyFrames"][0]


def test_video_request_accepts_an_explicit_control_net_of_none() -> None:
    message = create_job_request_message(
        "video-control-net-none",
        {
            "type": "video",
            "modelId": "ltx23-22b-fp8_v2v_distilled",
            "positivePrompt": "a kite over the sea",
            "numberOfMedia": 1,
            "controlNet": None,
            "referenceVideo": True,
            "referenceMask": True,
        },
        model_options("video"),
    )
    assert "hasReferenceMask" not in message["keyFrames"][0]
