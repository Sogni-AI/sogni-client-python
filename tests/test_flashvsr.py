"""FlashVSR v1.1 video upscaling, mirroring sogni-client 5.37.1.

The TypeScript behavioral checks live in `scripts/check-flashvsr-video.cjs`;
the messages below are copied from `createJobRequestMessage.ts` byte for byte.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock

import pytest

import sogni_client
from sogni_client import (
    FLASHVSR_VIDEO_UPSCALE_MODEL_ID,
    ApiError,
    calculate_video_frames,
    create_job_request_message,
    is_video_model,
    is_video_upscale_model,
)
from sogni_client.events import EventEmitter
from sogni_client.projects import ProjectsApi, _validate_video_upscale_params
from sogni_client.utils import get_video_workflow_type

MODEL_ID = FLASHVSR_VIDEO_UPSCALE_MODEL_ID
RESOLUTION = "Choose 1080p or 1440p for video upscaling."
REFERENCE_VIDEO = "FlashVSR requires an uploaded referenceVideo."
TIMING = (
    "Omit the source timing, or supply the source video’s exact frame count "
    "and frame rate (up to 362 frames and 15 seconds)."
)
PROMPTLESS = "FlashVSR is promptless."
CONTROLS = (
    "Video upscaling preserves the complete source video and its audio; "
    "generation controls are unsupported."
)
ONE_SOURCE = "Upscale one source video per project."
EXTERNAL_URLS = (
    "External reference URLs are supported only by Seedance, HappyHorse, and Wan 3 models."
)

# The server-advertised FlashVSR tier, as delivered in the model catalog.
TIER = {
    "type": "video",
    "task": "video-upscale",
    "requiresReferenceVideo": True,
    "preservesSourceTiming": True,
    "outputResolutions": [1080, 1440],
    "maxPixels": 3686400,
    "width": {"min": 2, "max": 2560, "step": 2, "default": 2520},
    "height": {"min": 2, "max": 2560, "step": 2, "default": 1440},
    "frames": {"min": 1, "max": 362, "step": 1, "default": 158},
    "fps": {"min": 1, "max": 60, "default": 24},
    "steps": {"min": 1, "max": 1, "default": 1},
    "guidance": {"min": 1, "max": 1, "default": 1},
    "comfySampler": {"allowed": ["euler"], "default": "euler"},
    "comfyScheduler": {"allowed": ["simple"], "default": "simple"},
}
OPTIONS = {
    "type": "video",
    "sampler": {"allowed": ["euler"], "default": "euler"},
    "scheduler": {"allowed": ["simple"], "default": "simple"},
}
_OMIT = object()


def message(**changes: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "type": "video",
        "modelId": MODEL_ID,
        "positivePrompt": "",
        "numberOfMedia": 1,
        "referenceVideo": b"video",
        "width": 2520,
        "height": 1440,
        "frames": 158,
        "fps": 24,
        "steps": 1,
    }
    for key, value in changes.items():
        if value is _OMIT:
            params.pop(key, None)
        else:
            params[key] = value
    return create_job_request_message("upscale-project", params, OPTIONS)


def request(**changes: Any) -> dict[str, Any]:
    return message(**changes)["keyFrames"][0]


def exactly(text: str) -> str:
    return f"^{re.escape(text)}$"


def test_flashvsr_is_a_distinct_upscale_workflow_exported_from_the_package_root() -> None:
    assert sogni_client.FLASHVSR_VIDEO_UPSCALE_MODEL_ID == "flashvsr_v1.1_tiny_long_bf16"
    assert is_video_model(MODEL_ID)
    assert is_video_upscale_model(MODEL_ID)
    assert sogni_client.isVideoUpscaleModel(MODEL_ID)
    assert not is_video_upscale_model("ltx23-22b-fp8_t2v_distilled")
    assert get_video_workflow_type(MODEL_ID) == "upscale"
    assert calculate_video_frames(MODEL_ID, 158 / 24, 24) == 158


@pytest.mark.asyncio
async def test_video_asset_config_requires_only_the_source_video() -> None:
    api = ProjectsApi(FakeClient())
    config = await api.get_video_asset_config(MODEL_ID)
    assert config["workflowType"] == "upscale"
    assert config["assets"]["referenceVideo"] == "required"
    assert {asset for asset, rule in config["assets"].items() if rule != "forbidden"} == {
        "referenceVideo"
    }


def test_request_preserves_source_timing_and_pins_generation_fields() -> None:
    key = request(duration=5, seed=987, steps=20, generateAudio=True)
    assert key["frames"] == 158, "explicit source count must win over duration"
    assert key["fps"] == 24
    assert key["hasReferenceVideo"] is True
    assert key["generateAudio"] is True
    assert key["interpolation"] == "none"
    assert key["upscaleResolution"] == 1440
    assert key["steps"] == 1
    assert key["seed"] == 0
    assert key["comfySampler"] is None
    assert request(width=1890, height=1080)["upscaleResolution"] == 1080
    assert request(fps=24000 / 1001)["fps"] == 24000 / 1001
    assert message()["numberOfImages"] == 1


def test_upscale_resolution_param_overrides_the_shorter_edge() -> None:
    key = request(upscaleResolution=1080.0, width=_OMIT, height=_OMIT)
    assert key["upscaleResolution"] == 1080
    assert type(key["upscaleResolution"]) is int
    assert "width" not in key


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 3780, "height": 2160},
        {"upscaleResolution": 0},
        {"upscaleResolution": 720},
        {"upscaleResolution": "1440"},
        {"width": None},
        {"height": _OMIT},
        {"width": _OMIT, "height": _OMIT},
    ],
)
def test_rejects_other_delivery_resolutions(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=exactly(RESOLUTION)):
        request(**changes)


def test_source_video_is_required_and_other_assets_are_refused() -> None:
    with pytest.raises(
        ApiError,
        match=exactly("upscale workflow requires referenceVideo. Please provide this asset."),
    ):
        request(referenceVideo=None)
    with pytest.raises(
        ApiError,
        match=exactly(
            "upscale workflow does not support referenceImage. Please remove this asset."
        ),
    ):
        request(referenceImage=b"image")
    # The asset check above answers first in both clients; the upscale check
    # keeps its own message for callers that reach it directly.
    with pytest.raises(ValueError, match=exactly(REFERENCE_VIDEO)):
        _validate_video_upscale_params({"upscaleResolution": 1440, "numberOfMedia": 1})


@pytest.mark.parametrize(
    "field", ["referenceVideoUrls", "referenceImageUrls", "referenceAudioUrls"]
)
@pytest.mark.parametrize("value", [[], ["https://cdn.example/source.mp4"]])
def test_external_reference_url_arrays_are_refused_even_when_empty(field: str, value: list) -> None:
    with pytest.raises(ApiError, match=exactly(EXTERNAL_URLS)):
        request(**{field: value})


def test_empty_reference_url_arrays_are_refused_for_every_non_external_model() -> None:
    # `[]` is truthy in JavaScript, so the TypeScript check refuses any array.
    with pytest.raises(ApiError, match=exactly(EXTERNAL_URLS)):
        create_job_request_message(
            "ltx-project",
            {
                "type": "video",
                "modelId": "ltx23-22b-fp8_t2v_distilled",
                "positivePrompt": "a lighthouse at dusk",
                "numberOfMedia": 1,
                "referenceImageUrls": [],
            },
            OPTIONS,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"frames": 0},
        {"frames": 363},
        {"frames": 158.5},
        {"frames": "158"},
        {"frames": True},
        {"frames": float("nan")},
        {"fps": 0},
        {"fps": 61},
        {"fps": "24"},
        {"fps": True},
        {"fps": float("inf")},
        {"fps": 120},
        {"frames": 362, "fps": 23},
        {"frames": _OMIT, "duration": 0.02},
        {"frames": _OMIT, "duration": "soon"},
        {"frames": _OMIT, "duration": 16},
        # A duration identifies the source's frames only together with its exact rate.
        {"frames": _OMIT, "fps": _OMIT, "duration": 5},
        {"frames": None, "fps": None, "duration": 5},
    ],
)
def test_rejects_source_timing_that_is_sent_but_invalid(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=exactly(TIMING)):
        request(**changes)


def test_minimal_request_leaves_the_source_timing_and_size_to_the_server() -> None:
    # The server probes the uploaded source and adopts its exact frames, rate and size.
    key = request(width=_OMIT, height=_OMIT, frames=_OMIT, fps=_OMIT, upscaleResolution=1080)
    for field in ("frames", "fps", "width", "height"):
        assert field not in key, f"a minimal upscale must leave {field} to the verified source"
    assert key["upscaleResolution"] == 1080
    assert key["hasReferenceVideo"] is True
    assert key["steps"] == 1
    assert key["seed"] == 0
    # None counts as omitted, as everywhere else in the Python client.
    none_key = request(width=None, height=None, frames=None, fps=None, upscaleResolution=1440)
    assert {"frames", "fps", "width", "height"}.isdisjoint(none_key)


def test_rate_only_and_frames_only_requests_send_just_what_was_given() -> None:
    rate_only = request(
        width=_OMIT, height=_OMIT, frames=_OMIT, upscaleResolution=1440, fps=30000 / 1001
    )
    assert rate_only["fps"] == 30000 / 1001
    assert "frames" not in rate_only
    frames_only = request(fps=_OMIT)
    assert frames_only["frames"] == 158
    assert "fps" not in frames_only


def test_integral_float_frame_count_is_accepted_and_sent_as_an_int() -> None:
    key = request(frames=158.0)
    assert key["frames"] == 158
    assert type(key["frames"]) is int


def test_duration_only_request_derives_the_source_frame_count() -> None:
    key = request(frames=_OMIT, duration=158 / 24)
    assert key["frames"] == 158
    assert request(frames=None, duration="5")["frames"] == 120
    assert request(frames=_OMIT, duration=1 / 24)["frames"] == 1
    assert request(frames=_OMIT, duration=362 / 24)["frames"] == 362


def test_duration_only_request_applies_the_one_frame_minimum_and_15_second_maximum() -> None:
    # Rounds to one frame, so it passes the frame check, then fails the 1/fps minimum.
    with pytest.raises(
        ValueError,
        match=exactly("Video duration must greater or equal 0.041666666666666664, got 0.03"),
    ):
        request(frames=_OMIT, duration=0.03)
    # Rounds to 362 frames within the 1ms tolerance, then fails the 362/24 maximum.
    with pytest.raises(
        ValueError,
        match=exactly("Video duration must be less or equal 15.083333333333334, got 15.1"),
    ):
        request(frames=_OMIT, duration=15.1)


def test_explicit_frame_count_skips_duration_validation() -> None:
    assert request(frames=158, duration=99)["frames"] == 158


@pytest.mark.parametrize(
    "changes",
    [
        {"positivePrompt": "replace the face"},
        {"negativePrompt": "blurry"},
        {"positivePrompt": None, "negativePrompt": " keep the grain "},
    ],
)
def test_rejects_prompts(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=exactly(PROMPTLESS)):
        request(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"positivePrompt": None, "negativePrompt": None},
        {"positivePrompt": " \n\t", "negativePrompt": ""},
        # String.prototype.trim strips U+FEFF; Python's str.strip() would not.
        {"positivePrompt": "\ufeff\u3000"},
    ],
)
def test_accepts_absent_and_blank_prompts(changes: dict[str, Any]) -> None:
    assert request(**changes)["upscaleResolution"] == 1440


@pytest.mark.parametrize(
    "changes",
    [
        {"teacacheThreshold": 0},
        {"trimEndFrame": True},
        {"controlNet": {"name": "canny"}},
        {"controlNet": {}},
        {"videoStart": 0},
        {"generateAudio": False},
    ],
)
def test_rejects_generation_controls(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=exactly(CONTROLS)):
        request(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"numberOfMedia": 2},
        {"numberOfMedia": _OMIT},
        {"numberOfMedia": None},
        {"numberOfMedia": True},
        {"numberOfMedia": "1"},
    ],
)
def test_rejects_anything_but_one_source_video(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=exactly(ONE_SOURCE)):
        request(**changes)


def test_accepts_an_integral_float_media_count() -> None:
    assert message(numberOfMedia=1.0)["numberOfImages"] == 1


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"width": 3780, "height": 2160, "frames": 0, "positivePrompt": "x"}, RESOLUTION),
        ({"frames": 0, "positivePrompt": "x", "generateAudio": False}, TIMING),
        ({"positivePrompt": "x", "generateAudio": False, "numberOfMedia": 2}, PROMPTLESS),
        ({"generateAudio": False, "numberOfMedia": 2}, CONTROLS),
        # Upscale checks run before the duration range check, as in TypeScript.
        ({"frames": _OMIT, "duration": 0.03, "numberOfMedia": 2}, ONE_SOURCE),
    ],
)
def test_checks_run_in_the_typescript_order(changes: dict[str, Any], expected: str) -> None:
    with pytest.raises(ValueError, match=exactly(expected)):
        request(**changes)


class FakeSocket(EventEmitter):
    def __init__(self, responses: list[Any] | None = None) -> None:
        super().__init__()
        self.responses = list(responses or [])
        self.get_calls: list[tuple[str, dict[str, Any] | None]] = []

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.get_calls.append((path, params))
        return self.responses.pop(0)


class FakeClient(EventEmitter):
    def __init__(self, responses: list[Any] | None = None) -> None:
        super().__init__()
        self.socket = FakeSocket(responses)


QUOTE = {
    "quote": {"project": {"costInToken": 1, "costInUSD": 0.01, "costInSpark": 1, "costInSogni": 1}}
}


@pytest.mark.asyncio
async def test_model_options_carry_the_catalog_fields_and_the_fps_range() -> None:
    api = ProjectsApi(FakeClient())
    api.get_supported_models = AsyncMock(
        return_value=[
            {"id": MODEL_ID, "tier": "flashvsr"},
            {"id": "wan_v2.2-14b-fp8_t2v", "tier": "wan"},
        ]
    )
    api._get_model_tiers = AsyncMock(
        return_value={
            "flashvsr": TIER,
            "wan": {
                "type": "video",
                "width": {"min": 480, "max": 1280, "step": 16, "default": 640},
                "height": {"min": 480, "max": 1280, "step": 16, "default": 640},
                "fps": {"allowed": [16, 32], "default": 16},
            },
        }
    )

    options = await api.get_model_options(MODEL_ID)

    assert options["task"] == "video-upscale"
    assert options["outputResolutions"] == [1080, 1440]
    assert options["outputResolutions"] is not TIER["outputResolutions"]
    assert options["preservesSourceTiming"] is True
    assert options["requiresReferenceVideo"] is True
    assert options["fps"] == {"min": 1, "max": 60, "default": 24}
    assert options["maxPixels"] == 3686400
    wan = await api.get_model_options("wan_v2.2-14b-fp8_t2v")
    assert wan["fps"] == {"allowed": [16, 32], "default": 16}
    assert "task" not in wan


@pytest.mark.asyncio
async def test_estimate_sends_source_geometry_only_as_a_pair() -> None:
    client = FakeClient([QUOTE, QUOTE, QUOTE])
    api = ProjectsApi(client)
    base = {
        "tokenType": "spark",
        "model": MODEL_ID,
        "width": 2520,
        "height": 1440,
        "frames": 158,
        "fps": 24,
        "referenceVideo": b"video",
    }

    await api.estimate_video_cost({**base, "sourceWidth": 1260.0, "sourceHeight": 720})
    await api.estimate_video_cost({**base, "sourceWidth": 1260})
    await api.estimate_video_cost({**base, "source_height": 720})

    path, query = client.socket.get_calls[0]
    assert path == f"/api/v1/job-video/estimate/spark/{MODEL_ID}/2520/1440/158/24"
    sent = {key: value for key, value in (query or {}).items() if value is not None}
    assert list(sent) == ["sourceWidth", "sourceHeight", "hasVideoInput"]
    assert sent["sourceWidth"] == "1260"
    assert sent["sourceHeight"] == 720
    for _path, lone in client.socket.get_calls[1:]:
        assert (lone or {}).get("sourceWidth") is None
        assert (lone or {}).get("sourceHeight") is None
