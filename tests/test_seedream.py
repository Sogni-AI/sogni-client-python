"""Seedream 5.0 Pro Uncensored, in parity with sogni-client's
scripts/check-seedream-5-pro-uncensored.cjs."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

import sogni_client
from sogni_client.chat import ChatToolsApi
from sogni_client.errors import ApiError, ProjectError
from sogni_client.projects import (
    Project,
    ProjectsApi,
    _custom_image_size_bounds,
    create_job_request_message,
    is_gpt_image_model,
)
from sogni_client.utils import is_seedream_model, is_video_model
from tests.test_assets import PNG, AssetsRest
from tests.test_assets import FakeClient as AssetsClient
from tests.test_chat import FakeProjects, tool_call
from tests.test_projects import FakeClient

SEEDREAM = "seedream-5-0-pro-uncensored"
IMAGE_OPTIONS = {
    "type": "image",
    "sampler": {"allowed": [], "default": None},
    "scheduler": {"allowed": [], "default": None},
}


def seedream_request(**params: Any) -> dict[str, Any]:
    message = create_job_request_message(
        "seedream-wire",
        {
            "type": "image",
            "modelId": SEEDREAM,
            "positivePrompt": "A portrait in window light",
            "numberOfMedia": 1,
            **params,
        },
        IMAGE_OPTIONS,
    )
    return message["keyFrames"][0]


def test_seedream_identity_is_exported_from_the_package_root() -> None:
    assert sogni_client.SEEDREAM_5_PRO_UNCENSORED_MODEL_ID == SEEDREAM
    assert sogni_client.is_seedream_model is is_seedream_model
    assert sogni_client.isSeedreamModel is is_seedream_model
    assert is_seedream_model(SEEDREAM)
    for other in (
        "seedream-5-0-pro",
        "seedream-5-0-flash-uncensored",
        "seedream-5-0-pro-uncensored-v2",
        "seedream",
        "seedance-2-5-uncensored",
        "gpt-image-2",
        None,
    ):
        assert not is_seedream_model(other), other
    assert not is_gpt_image_model(SEEDREAM)
    assert not is_video_model(SEEDREAM)


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (1024, 1024),
        (1152, 864),
        (864, 1152),
        (1280, 720),
        (720, 1280),
        (1248, 832),
        (832, 1248),
        (1512, 648),
        (1536, 1536),
        (2048, 2048),
        (2304, 1728),
        (1728, 2304),
        (2560, 1440),
        (1440, 2560),
        (2496, 1664),
        (1664, 2496),
        (3024, 1296),
        # The extremes the pixel and aspect-ratio limits allow.
        (3840, 240),
        (8192, 512),
    ],
)
def test_seedream_sizes_inside_the_pixel_and_aspect_limits_pass(width: int, height: int) -> None:
    keyframe = seedream_request(sizePreset="custom", width=width, height=height)
    assert (keyframe["width"], keyframe["height"]) == (width, height)


def test_seedream_sizes_outside_the_limits_are_refused() -> None:
    assert _custom_image_size_bounds(SEEDREAM) == (240, 8192)
    with pytest.raises(ValueError, match="Width"):
        seedream_request(width=8193, height=512)
    for width, height in ((1024, 768), (2048, 2049), (4096, 240)):
        with pytest.raises(
            ApiError, match="921,600 to 4,194,304 pixels at an aspect ratio from 1:16 to 16:1"
        ):
            seedream_request(width=width, height=height)
    # A named size preset is resolved by the server, not checked here.
    assert seedream_request(sizePreset="square_hd", width=512, height=512)["sizePreset"] == (
        "square_hd"
    )


def test_seedream_takes_up_to_ten_references_and_no_negative_prompt() -> None:
    keyframe = seedream_request(contextImages=[PNG] * 10, negativePrompt="blurry")
    assert keyframe["hasContextImage10"] is True
    assert "negativePrompt" not in keyframe
    for contexts in ([PNG] * 11, [PNG, None]):
        with pytest.raises(ApiError, match="up to 10 non-empty reference images in source order"):
            seedream_request(contextImages=contexts)
    # GPT Image keeps the negative prompt it has always sent.
    gpt = create_job_request_message(
        "gpt-wire",
        {
            "type": "image",
            "modelId": "gpt-image-2",
            "positivePrompt": "a mug",
            "negativePrompt": "blurry",
        },
        IMAGE_OPTIONS,
    )["keyFrames"][0]
    assert gpt["negativePrompt"] == "blurry"


async def test_seedream_uploads_ten_references_in_source_order() -> None:
    rest = AssetsRest(enabled=False)
    api = ProjectsApi(AssetsClient(rest))
    api.get_model_options = AsyncMock(return_value=IMAGE_OPTIONS)
    await api.create(
        type="image",
        model_id=SEEDREAM,
        positive_prompt="Combine these",
        number_of_media=1,
        context_images=[PNG] * 10,
    )
    uploads = [
        params["type"] for method, path, params in rest.calls if path == "/v1/image/uploadUrl"
    ]
    assert uploads == [f"contextImage{index}" for index in range(1, 11)]


@pytest.mark.asyncio
async def test_seedream_image_jobs_surface_the_shared_consent_refusal() -> None:
    consent = {"key": "seedance-2-5-uncensored", "version": 2, "modelId": SEEDREAM}
    message = (
        "Seedream 5.0 Pro Uncensored requires a one-time likeness and consent agreement. "
        "Review and accept it in the Sogni app, then try again."
    )
    api = ProjectsApi(FakeClient())
    project = Project(
        {"type": "image", "modelId": SEEDREAM, "positivePrompt": "A portrait", "numberOfMedia": 1},
        api,
    )
    api._projects.append(project)
    events: list[dict[str, Any]] = []
    api.on("project", events.append)
    waiting = asyncio.create_task(project.wait_for_completion())

    api._handle_job_error(
        {
            "jobID": project.id,
            "isFromWorker": False,
            "error": "4103",
            "error_message": message,
            "consentRequired": consent,
        }
    )

    with pytest.raises(ProjectError) as raised:
        await waiting
    assert str(raised.value) == message
    assert raised.value.code == sogni_client.MODEL_CONSENT_REQUIRED_ERROR_CODE == 4103
    assert raised.value.consent_required == consent
    assert sogni_client.is_model_consent_required_error(raised.value)
    assert events[0]["error"] == {"code": 4103, "message": message, "consentRequired": consent}


def test_seedream_selectors_route_generate_and_edit() -> None:
    for tool_name in ("generate_image", "edit_image"):
        for selector in (
            SEEDREAM,
            "Seedream 5.0 Pro Uncensored",
            "Seedream 5 Pro Uncensored",
            "seedream_uncensored",
            "Seedream 5 Pro Spicy",
            "Seedream 5.0 Pro Spicy",
        ):
            assert ChatToolsApi._resolve_model(tool_name, {"model": selector}) == SEEDREAM
        # A bare "seedream" is not claimed for the uncensored model.
        assert ChatToolsApi._resolve_model(tool_name, {"model": "seedream"}) == "seedream"


@pytest.mark.asyncio
async def test_chat_tools_choose_seedream_only_by_name_without_sampling_args() -> None:
    class ImageProjects(FakeProjects):
        def __init__(self, models: list[dict[str, Any]]) -> None:
            super().__init__()
            self.models = models

        async def wait_for_models(self, _timeout: float) -> list[dict[str, Any]]:
            return self.models

    # External API models report nine workers; the worker-count choice must
    # never land a call that names no model on the agreement-gated one.
    projects = ImageProjects(
        [
            {"id": SEEDREAM, "media": "image", "workerCount": 50},
            {"id": "gpt-image-2", "media": "image", "workerCount": 9},
            {"id": "z_image_turbo_bf16", "media": "image", "workerCount": 3},
        ]
    )
    api = ChatToolsApi(projects)  # type: ignore[arg-type]
    for name in ("generate_image", "edit_image"):
        result = await api.execute(tool_call(name, '{"prompt":"A portrait"}'))
        assert result["success"] is True
        assert projects.created[-1]["modelId"] == "gpt-image-2"

    only_seedream = ImageProjects([{"id": SEEDREAM, "media": "image", "workerCount": 9}])
    refused = await ChatToolsApi(only_seedream).execute(  # type: ignore[arg-type]
        tool_call("generate_image", '{"prompt":"A portrait"}')
    )
    assert refused["success"] is False
    assert only_seedream.created == []

    # Named, it is chosen, and takes no negative prompt or seed.
    named = {
        "prompt": "A portrait",
        "model": "Seedream 5.0 Pro Uncensored",
        "negative_prompt": "blurry",
        "seed": 7,
        "width": 2048,
        "height": 2048,
    }
    await api.execute(tool_call("generate_image", json.dumps(named)))
    created = projects.created[-1]
    assert created["modelId"] == SEEDREAM
    assert (created["width"], created["height"]) == (2048, 2048)
    assert "negativePrompt" not in created
    assert "seed" not in created

    # Other models still receive both.
    gpt = named | {"model": "gpt-image-2"}
    await api.execute(tool_call("generate_image", json.dumps(gpt)))
    assert projects.created[-1]["negativePrompt"] == "blurry"
    assert projects.created[-1]["seed"] == 7
