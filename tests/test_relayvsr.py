"""RelayVSR video upscaling, mirroring sogni-client.

The TypeScript behavioral checks live in `scripts/check-relayvsr-video.cjs`;
the messages below are copied from `createJobRequestMessage.ts` byte for byte.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

import sogni_client
from sogni_client import (
    FLASHVSR_VIDEO_UPSCALE_MODEL_ID,
    RELAYVSR_X2_VIDEO_UPSCALE_MODEL_ID,
    RELAYVSR_X4_VIDEO_UPSCALE_MODEL_ID,
    SogniTools,
    calculate_video_frames,
    create_job_request_message,
    is_relay_vsr_model,
    is_video_model,
    is_video_upscale_model,
)
from sogni_client.utils import get_video_workflow_type

MODELS = (RELAYVSR_X2_VIDEO_UPSCALE_MODEL_ID, RELAYVSR_X4_VIDEO_UPSCALE_MODEL_ID)
OPTIONS = {
    "type": "video",
    "sampler": {"allowed": ["euler"], "default": "euler"},
    "scheduler": {"allowed": ["simple"], "default": "simple"},
}


def message(model_id: str, **changes: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "type": "video",
        "modelId": model_id,
        "positivePrompt": "",
        "numberOfMedia": 1,
        "referenceVideo": b"video",
        "upscaleResolution": 2160,
        **changes,
    }
    return create_job_request_message("upscale-project", params, OPTIONS)


def request(model_id: str, **changes: Any) -> dict[str, Any]:
    return message(model_id, **changes)["keyFrames"][0]


def exactly(text: str) -> str:
    return f"^{re.escape(text)}$"


def test_ids_are_exported_from_the_package_root() -> None:
    assert sogni_client.RELAYVSR_X2_VIDEO_UPSCALE_MODEL_ID == "relayvsr_x2_lightx2v_int8_convrot"
    assert sogni_client.RELAYVSR_X4_VIDEO_UPSCALE_MODEL_ID == "relayvsr_x4_int8_convrot"
    assert sogni_client.isRelayVsrModel(RELAYVSR_X2_VIDEO_UPSCALE_MODEL_ID)
    assert not is_relay_vsr_model(FLASHVSR_VIDEO_UPSCALE_MODEL_ID)


@pytest.mark.parametrize("model_id", MODELS)
def test_relayvsr_is_an_upscale_workflow(model_id: str) -> None:
    assert is_relay_vsr_model(model_id)
    assert is_video_model(model_id)
    assert is_video_upscale_model(model_id)
    assert get_video_workflow_type(model_id) == "upscale"
    assert calculate_video_frames(model_id, 158 / 24, 24) == 158


@pytest.mark.parametrize("model_id", MODELS)
def test_minimal_call_leaves_the_source_to_the_server_and_sends_no_flashvsr_options(
    model_id: str,
) -> None:
    key = request(model_id)
    assert key["upscaleResolution"] == 2160
    assert key["steps"] == 1
    assert key["seed"] == 0
    assert key["generateAudio"] is True
    assert key["interpolation"] == "none"
    for field in ("frames", "fps", "width", "height", "detailPreference", "processingSpeed"):
        assert field not in key


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("resolution", [1080, 1440, 2160])
def test_targets(model_id: str, resolution: int) -> None:
    assert request(model_id, upscaleResolution=resolution)["upscaleResolution"] == resolution


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("resolution", [720, 2880])
def test_other_targets_are_refused(model_id: str, resolution: int) -> None:
    with pytest.raises(
        ValueError, match=exactly("Choose 1080p, 1440p or 2160p for RelayVSR video upscaling.")
    ):
        request(model_id, upscaleResolution=resolution)


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("detailPreference", "stable"),
        ("detailPreference", "sharper"),
        ("processingSpeed", "stable"),
        ("processingSpeed", "faster"),
        ("seed", 0),
        ("seed", 42),
        ("seed", -1),
    ],
)
def test_flashvsr_options_are_refused(model_id: str, field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=exactly(f"RelayVSR has no {field}.")):
        request(model_id, **{field: value})


@pytest.mark.parametrize("model_id", MODELS)
def test_promptless_and_generation_controls(model_id: str) -> None:
    with pytest.raises(ValueError, match=exactly("RelayVSR is promptless.")):
        request(model_id, positivePrompt="sharpen the face")
    with pytest.raises(ValueError, match="generation controls are unsupported"):
        request(model_id, generateAudio=False)
    with pytest.raises(ValueError, match=exactly("Upscale one source video per project.")):
        request(model_id, numberOfMedia=2)


@pytest.mark.parametrize("model_id", MODELS)
def test_no_client_side_length_cap(model_id: str) -> None:
    assert request(model_id, frames=1800, fps=30)["frames"] == 1800


def test_flashvsr_keeps_its_limits_options_and_wire_shape() -> None:
    def flash(**changes: Any) -> dict[str, Any]:
        return request(FLASHVSR_VIDEO_UPSCALE_MODEL_ID, upscaleResolution=1440, **changes)

    with pytest.raises(ValueError, match=exactly("Choose 1080p or 1440p for video upscaling.")):
        request(FLASHVSR_VIDEO_UPSCALE_MODEL_ID, upscaleResolution=2160)
    key = flash()
    assert key["detailPreference"] == "stable"
    assert key["processingSpeed"] == "stable"
    assert key["seed"] == 0
    tuned = flash(seed=42, detailPreference="sharper", processingSpeed="faster")
    assert (tuned["seed"], tuned["detailPreference"], tuned["processingSpeed"]) == (
        42,
        "sharper",
        "faster",
    )


def test_hosted_tool_offers_relayvsr_as_an_optional_selector() -> None:
    parameters = SogniTools.upscaleVideo["function"]["parameters"]
    assert parameters["properties"]["model"]["enum"] == ["flashvsr", "relayvsr"]
    assert parameters["properties"]["targetResolution"]["enum"] == [1080, 1440, 2160]
    assert parameters["required"] == []
