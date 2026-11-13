"""Seedream 5.0 Pro Uncensored and Seedream 5.5 Uncensored, in parity with
sogni-client's scripts/check-seedream-5-pro-uncensored.cjs and
scripts/check-seedream-5-5-uncensored.cjs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
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
    _image_layers_from_result,
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


# Seedream 5.5 Uncensored: an additive model with Seedream 5.0 Pro Uncensored's
# contract, in parity with sogni-client's scripts/check-seedream-5-5-uncensored.cjs.
SEEDREAM_55 = "seedream-5-5-uncensored"


def test_seedream_55_is_a_seedream_model_and_5_0_pro_keeps_its_selectors() -> None:
    assert sogni_client.SEEDREAM_5_5_UNCENSORED_MODEL_ID == SEEDREAM_55
    assert is_seedream_model(SEEDREAM_55)
    assert is_seedream_model(SEEDREAM)
    for other in ("seedream-5-5", "seedream-5-5-uncensored-v2"):
        assert not is_seedream_model(other), other
    # 4,624,220 pixels at 16:1 allow edges up to 8595 px (8595x538).
    assert _custom_image_size_bounds(SEEDREAM_55) == (240, 8595)
    assert _custom_image_size_bounds(SEEDREAM) == (240, 8192)
    for tool_name in ("generate_image", "edit_image"):
        for selector in (
            SEEDREAM_55,
            "Seedream 5.5 Uncensored",
            "Seedream 5.5 Spicy",
            "seedream-5-5-spicy",
        ):
            assert ChatToolsApi._resolve_model(tool_name, {"model": selector}) == SEEDREAM_55, (
                selector
            )
        for selector in (
            SEEDREAM,
            "Seedream 5.0 Pro Uncensored",
            "seedream_uncensored",
            "Seedream 5 Pro Spicy",
        ):
            assert ChatToolsApi._resolve_model(tool_name, {"model": selector}) == SEEDREAM, selector


def test_seedream_55_takes_the_vendor_size_ten_references_and_no_negative_prompt() -> None:
    def request(**params: Any) -> dict[str, Any]:
        return create_job_request_message(
            "seedream-55-wire",
            {
                "type": "image",
                "modelId": SEEDREAM_55,
                "positivePrompt": "A white cat in a bar",
                "numberOfMedia": 1,
                **params,
            },
            IMAGE_OPTIONS,
        )["keyFrames"][0]

    keyframe = request(
        sizePreset="custom",
        width=2512,
        height=1664,
        contextImages=[PNG] * 10,
        negativePrompt="blurry",
    )
    assert (keyframe["width"], keyframe["height"]) == (2512, 1664)
    assert keyframe["hasContextImage10"] is True
    assert "negativePrompt" not in keyframe
    with pytest.raises(ApiError, match="up to 10 non-empty reference images in source order"):
        request(contextImages=[PNG] * 11)
    with pytest.raises(ApiError, match="921,600 to 4,624,220 pixels"):
        request(width=1024, height=768)
    # No new option was set, so none is written.
    assert [key for key in keyframe if key.startswith("seedream")] == []


@pytest.mark.asyncio
async def test_chat_tools_never_choose_seedream_55_unless_named() -> None:
    class ImageProjects(FakeProjects):
        def __init__(self, models: list[dict[str, Any]]) -> None:
            super().__init__()
            self.models = models

        async def wait_for_models(self, _timeout: float) -> list[dict[str, Any]]:
            return self.models

    projects = ImageProjects(
        [
            {"id": SEEDREAM_55, "media": "image", "workerCount": 50},
            {"id": SEEDREAM, "media": "image", "workerCount": 50},
            {"id": "gpt-image-2", "media": "image", "workerCount": 9},
        ]
    )
    api = ChatToolsApi(projects)  # type: ignore[arg-type]
    for name in ("generate_image", "edit_image"):
        result = await api.execute(tool_call(name, '{"prompt":"A portrait"}'))
        assert result["success"] is True
        assert projects.created[-1]["modelId"] == "gpt-image-2"
    await api.execute(
        tool_call(
            "generate_image",
            json.dumps({"prompt": "A portrait", "model": "Seedream 5.5 Uncensored", "seed": 7}),
        )
    )
    assert projects.created[-1]["modelId"] == SEEDREAM_55
    assert "seed" not in projects.created[-1]


# Byte-for-byte compatibility: requests and quotes without any seedream* option
# serialize exactly as sogni-client (Python) 5.80.0 sent them.
BASELINE = json.loads(
    (Path(__file__).parent / "fixtures" / "seedream-request-baseline-5.80.0.json").read_text()
)


@pytest.mark.parametrize("name", sorted(BASELINE["requests"]))
def test_requests_without_seedream_options_are_byte_identical_to_5_80_0(name: str) -> None:
    case = BASELINE["requests"][name]
    current = create_job_request_message(
        f"baseline-{name}", dict(case["params"]), BASELINE["options"]
    )
    assert json.dumps(current) == json.dumps(case["message"])
    assert not [key for key in current["keyFrames"][0] if key.startswith("seedream")]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(BASELINE["estimates"]))
async def test_quotes_without_seedream_options_are_identical_to_5_80_0(name: str) -> None:
    case = BASELINE["estimates"][name]
    get, api = _estimate_api()
    await api.estimate_cost(dict(case["request"]))
    path, query = get.await_args.args
    assert (path, json.dumps(query)) == (case["path"], json.dumps(case["query"]))


def _estimate_api() -> tuple[AsyncMock, ProjectsApi]:
    api = object.__new__(ProjectsApi)
    api.get_model_options = AsyncMock(return_value=IMAGE_OPTIONS)
    quote = {
        "quote": {"project": {"costInToken": 1, "costInUSD": 1, "costInSpark": 1, "costInSogni": 1}}
    }
    get = AsyncMock(return_value=quote)
    api.client = SimpleNamespace(socket=SimpleNamespace(get=get))
    return get, api


def seedream_55_request(**params: Any) -> dict[str, Any]:
    return seedream_request(modelId=SEEDREAM_55, **params)


def seedream_keys(keyframe: dict[str, Any]) -> list[str]:
    return [key for key in keyframe if key.startswith("seedream")]


@pytest.mark.parametrize(
    ("width", "height"),
    [
        *[(1024, 1024), (1152, 864), (864, 1152), (1424, 800), (800, 1424), (1248, 832)],
        *[(832, 1248), (1568, 672), (1536, 1536), (1792, 1344), (1344, 1792), (2048, 1152)],
        *[(1152, 2048), (1872, 1248), (1248, 1872), (2352, 1008), (2048, 2048), (2368, 1776)],
        *[(1776, 2368), (2816, 1584), (1584, 2816), (2496, 1664), (1664, 2496), (3136, 1344)],
        # The boundaries: exactly 4,624,220 pixels, and the extreme edges.
        *[(2830, 1634), (1634, 2830), (3840, 240), (8595, 538), (538, 8595)],
    ],
)
def test_seedream_55_native_and_boundary_sizes_pass(width: int, height: int) -> None:
    keyframe = seedream_55_request(sizePreset="custom", width=width, height=height)
    assert (keyframe["width"], keyframe["height"]) == (width, height)


def test_seedream_55_refuses_sizes_beyond_4_624_220_pixels_and_5_0_pro_keeps_its_limit() -> None:
    with pytest.raises(ValueError, match="Width"):
        seedream_55_request(width=8596, height=538)
    for width, height in ((1024, 768), (2830, 1635), (2151, 2150), (4096, 240)):
        with pytest.raises(
            ApiError,
            match=(
                "seedream-5-5-uncensored renders 921,600 to 4,624,220 pixels at an aspect "
                "ratio from 1:16 to 16:1"
            ),
        ):
            seedream_55_request(width=width, height=height)
    for width, height in ((2830, 1634), (2816, 1584)):
        with pytest.raises(ApiError, match="921,600 to 4,194,304 pixels"):
            seedream_request(width=width, height=height)
    with pytest.raises(ValueError, match="Width"):
        seedream_request(width=8595, height=538)


@pytest.mark.parametrize("background", ["opaque", "transparent", "auto"])
@pytest.mark.parametrize("output_format", [None, "png", "webp"])
@pytest.mark.parametrize("references", [0, 1, 10])
def test_seedream_55_backgrounds_are_written_only_when_set(
    background: str, output_format: str | None, references: int
) -> None:
    params: dict[str, Any] = {"seedreamBackground": background}
    if output_format:
        params["outputFormat"] = output_format
    if references:
        params["contextImages"] = [PNG] * references
    keyframe = seedream_55_request(**params)
    assert keyframe["seedreamBackground"] == background
    assert seedream_keys(keyframe) == ["seedreamBackground"]


def test_seedream_55_background_rules() -> None:
    assert seedream_55_request(seedreamBackground="opaque", outputFormat="jpg")[
        "seedreamBackground"
    ] == ("opaque")
    for background in ("transparent", "auto"):
        with pytest.raises(
            ApiError, match="Transparent or auto Seedream background requires PNG or WebP output"
        ):
            seedream_55_request(seedreamBackground=background, outputFormat="jpg")
    for background in ("Transparent", "none", "", True):
        with pytest.raises(
            ApiError, match="seedreamBackground must be opaque, transparent or auto"
        ):
            seedream_55_request(seedreamBackground=background)
    for background in ("opaque", "transparent", "auto"):
        with pytest.raises(
            ApiError, match="seedreamBackground cannot combine with seedreamLayerDecomposition"
        ):
            seedream_55_request(
                seedreamBackground=background,
                seedreamLayerDecomposition=True,
                contextImages=[PNG],
            )
    # An explicit non-decomposition still takes a background.
    assert (
        seedream_55_request(seedreamBackground="transparent", seedreamLayerDecomposition=False)[
            "seedreamBackground"
        ]
        == "transparent"
    )


@pytest.mark.parametrize("model_id", [SEEDREAM, SEEDREAM_55])
@pytest.mark.parametrize("layer_size", ["1K", "1.5K", "2K"])
def test_layer_decomposition_runs_on_both_seedream_models(model_id: str, layer_size: str) -> None:
    keyframe = seedream_request(
        modelId=model_id,
        contextImages=[PNG],
        seedreamLayerDecomposition=True,
        seedreamLayerSize=layer_size,
    )
    assert seedream_keys(keyframe) == ["seedreamLayerDecomposition", "seedreamLayerSize"]
    assert keyframe["seedreamLayerDecomposition"] is True
    assert keyframe["seedreamLayerSize"] == layer_size


@pytest.mark.parametrize("model_id", [SEEDREAM, SEEDREAM_55])
def test_layer_decomposition_rules_shared_by_both_models(model_id: str) -> None:
    assert seedream_keys(
        seedream_request(modelId=model_id, contextImages=[PNG], seedreamLayerDecomposition=True)
    ) == ["seedreamLayerDecomposition"]
    for contexts in (None, [PNG, PNG]):
        with pytest.raises(ApiError, match="exactly one reference image"):
            seedream_request(
                modelId=model_id, contextImages=contexts, seedreamLayerDecomposition=True
            )
    with pytest.raises(ApiError, match="set numberOfMedia to 1"):
        seedream_request(
            modelId=model_id,
            contextImages=[PNG],
            seedreamLayerDecomposition=True,
            numberOfMedia=2,
        )
    with pytest.raises(ApiError, match="seedreamLayerSize must be 1K, 1.5K or 2K"):
        seedream_request(
            modelId=model_id,
            contextImages=[PNG],
            seedreamLayerDecomposition=True,
            seedreamLayerSize="auto",
        )
    with pytest.raises(ApiError, match="seedreamLayerSize requires seedreamLayerDecomposition"):
        seedream_request(modelId=model_id, contextImages=[PNG], seedreamLayerSize="2K")
    with pytest.raises(ApiError, match="seedreamLayerDecomposition must be a boolean"):
        seedream_request(modelId=model_id, contextImages=[PNG], seedreamLayerDecomposition="yes")


@pytest.mark.parametrize("max_images", [2, 8, 17, 100])
@pytest.mark.parametrize("text_recognition", [True, False])
def test_seedream_55_layer_options_are_written_with_their_wire_names(
    max_images: int, text_recognition: bool
) -> None:
    keyframe = seedream_55_request(
        contextImages=[PNG],
        seedreamLayerDecomposition=True,
        seedreamLayerSize="1.5K",
        seedreamLayerMaxImages=max_images,
        seedreamLayerTextRecognition=text_recognition,
    )
    assert keyframe["seedreamLayerMaxImages"] == max_images
    assert keyframe["seedreamLayerTextRecognition"] is text_recognition
    assert "seedreamBackground" not in keyframe


def test_seedream_55_layer_option_rules() -> None:
    def layers(**params: Any) -> dict[str, Any]:
        return seedream_55_request(contextImages=[PNG], seedreamLayerDecomposition=True, **params)

    for max_images in (0, 1, 101, 2.5, 17.0, "17", True):
        with pytest.raises(
            ApiError, match="seedreamLayerMaxImages must be an integer from 2 to 100"
        ):
            layers(seedreamLayerMaxImages=max_images)
    for text_recognition in ("true", 1):
        with pytest.raises(ApiError, match="seedreamLayerTextRecognition must be a boolean"):
            layers(seedreamLayerTextRecognition=text_recognition)
    for decomposition in (None, False):
        with pytest.raises(
            ApiError, match="seedreamLayerMaxImages requires seedreamLayerDecomposition"
        ):
            seedream_55_request(
                contextImages=[PNG],
                seedreamLayerDecomposition=decomposition,
                seedreamLayerMaxImages=17,
            )
        with pytest.raises(
            ApiError, match="seedreamLayerTextRecognition requires seedreamLayerDecomposition"
        ):
            seedream_55_request(
                contextImages=[PNG],
                seedreamLayerDecomposition=decomposition,
                seedreamLayerTextRecognition=False,
            )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("seedreamBackground", "opaque"),
        ("seedreamBackground", "transparent"),
        ("seedreamBackground", "auto"),
        ("seedreamLayerMaxImages", 17),
        ("seedreamLayerTextRecognition", True),
        ("seedreamLayerTextRecognition", False),
    ],
)
@pytest.mark.parametrize("with_layers", [False, True])
def test_seedream_5_0_pro_refuses_the_5_5_options(
    option: str, value: Any, with_layers: bool
) -> None:
    extra: dict[str, Any] = (
        {"contextImages": [PNG], "seedreamLayerDecomposition": True} if with_layers else {}
    )
    with pytest.raises(ApiError, match=f"{option} requires seedream-5-5-uncensored"):
        seedream_request(**extra, **{option: value})


@pytest.mark.parametrize("model_id", ["gpt-image-2.5-sunburst", "z_image_turbo_bf16"])
def test_other_models_refuse_every_seedream_option(model_id: str) -> None:
    for option, value in (
        ("seedreamLayerDecomposition", True),
        ("seedreamLayerSize", "1K"),
        ("seedreamLayerMaxImages", 17),
        ("seedreamLayerTextRecognition", False),
    ):
        with pytest.raises(ApiError, match="Seedream layer options require a Seedream model"):
            seedream_request(modelId=model_id, contextImages=[PNG], **{option: value})
    with pytest.raises(ApiError, match="seedreamBackground requires seedream-5-5-uncensored"):
        seedream_request(modelId=model_id, seedreamBackground="opaque")


def _box(left: int, top: int, right: int, bottom: int) -> dict[str, list[int]]:
    return {"absolute": [left, top, right, bottom], "normalized": [0, 0, 500, 500]}


def _style(**extra: Any) -> dict[str, Any]:
    return {
        "similarFontFamily": "Helvetica Neue",
        "foregroundColor": "#FFCC00",
        "bold": True,
        "italic": False,
        "underline": False,
        "strikethrough": False,
        "shadow": True,
        "translucent": False,
        **extra,
    }


def _recognition() -> dict[str, Any]:
    return {
        "text": "GRAND OPENING\nFriday 8 PM",
        "lines": [
            {
                "boundingBox": _box(100, 80, 900, 160),
                "runs": [{"text": "GRAND OPENING", "style": _style()}],
            },
            {
                "boundingBox": _box(100, 180, 700, 230),
                "runs": [
                    {"text": "Friday ", "style": _style(bold=False, shadow=False)},
                    {"text": "8 PM", "style": _style(foregroundColor="#FFFFFF")},
                ],
            },
        ],
    }


def _layer(z_index: int, **extra: Any) -> dict[str, Any]:
    return {
        "url": f"https://cdn.sogni.ai/layer-{z_index}.png",
        "key": f"vendor/bytedance/seedream-5-5-uncensored/2026-11-13/job.layer-{z_index}.png",
        "width": 1429,
        "height": 1652,
        "zIndex": z_index,
        "boundingBox": _box(10, 20, 719, 840),
        "name": f"Layer {z_index}",
        "description": f"Layer {z_index} description",
        **extra,
    }


def _public(layer: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in layer.items() if key != "key"}


def test_layer_results_are_ordered_by_z_index_1_to_99_with_text_recognition() -> None:
    text = _layer(3, textRecognition=_recognition())
    parsed = _image_layers_from_result([_layer(99), text, _layer(1)])
    assert parsed == [_public(_layer(1)), _public(text), _public(_layer(99))]
    assert parsed is not None and parsed[1]["textRecognition"] == _recognition()
    # Layers without text recognition (the usual case) carry no such field.
    assert "textRecognition" not in parsed[0]
    # zIndex 0 is the base and 100 is beyond the 99 layers Seedream returns.
    assert _image_layers_from_result([_layer(0), _layer(100), _layer(2)]) == [_public(_layer(2))]
    for empty in (None, [], [_layer(100)], [_layer(1, url="")], "layers"):
        assert _image_layers_from_result(empty) is None
    # An empty recognition is still well formed.
    empty_text = _image_layers_from_result([_layer(5, textRecognition={"text": "", "lines": []})])
    assert empty_text is not None and empty_text[0]["textRecognition"] == {"text": "", "lines": []}


@pytest.mark.parametrize(
    "recognition",
    [
        None,
        "GRAND OPENING",
        {"text": "GRAND OPENING"},
        {"lines": []},
        {**_recognition(), "text": 7},
        {**_recognition(), "lines": [{"boundingBox": _box(0, 0, 1, 1)}]},
        {**_recognition(), "lines": [{"runs": [{"text": "x", "style": _style()}]}]},
        {**_recognition(), "lines": [{"boundingBox": {"absolute": [0, 0, 1]}, "runs": []}]},
        {**_recognition(), "lines": [{"boundingBox": _box(0, 0, 1, 1), "runs": [{"text": "x"}]}]},
        {
            **_recognition(),
            "lines": [{"boundingBox": _box(0, 0, 1, 1), "runs": [{"style": _style()}]}],
        },
        {
            **_recognition(),
            "lines": [
                {
                    "boundingBox": _box(0, 0, 1, 1),
                    "runs": [{"text": "x", "style": _style(bold="yes")}],
                }
            ],
        },
        {
            **_recognition(),
            "lines": [
                *_recognition()["lines"],
                {
                    "boundingBox": _box(0, 0, 1, 1),
                    "runs": [{"text": "x", "style": {"similarFontFamily": "unknown"}}],
                },
            ],
        },
        # The vendor's snake_case shape is the API's to camelCase, not the SDK's.
        {"text": "x", "lines": [{"bounding_box": _box(0, 0, 1, 1), "runs": []}]},
    ],
)
def test_malformed_text_recognition_is_left_out_whole(recognition: Any) -> None:
    parsed = _image_layers_from_result([_layer(4, textRecognition=recognition)])
    assert parsed is not None
    assert parsed[0]["zIndex"] == 4
    assert "textRecognition" not in parsed[0]


@pytest.mark.asyncio
async def test_seedream_55_layers_reach_the_job_event_refresh_and_rest_sync() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    api.get_model_options = AsyncMock(return_value=IMAGE_OPTIONS)
    project = Project(
        {
            "type": "image",
            "modelId": SEEDREAM_55,
            "positivePrompt": "",
            "numberOfMedia": 1,
            "seedreamLayerDecomposition": True,
            "seedreamLayerMaxImages": 100,
            "seedreamLayerTextRecognition": True,
        },
        api,
    )
    api._projects.append(project)
    completed: list[dict[str, Any]] = []
    api.on("job", lambda event: completed.append(event) if event["type"] == "completed" else None)
    api._handle_job_state(
        {"type": "jobStarted", "jobID": project.id, "imgID": "layers-55", "workerName": ""}
    )
    text = _layer(2, textRecognition=_recognition())
    await api._apply_job_result(
        {
            "jobID": project.id,
            "imgID": "layers-55",
            "resultUrl": "https://cdn.sogni.ai/base.png",
            "performedStepCount": 1,
            "lastSeed": "-1",
            "triggeredNSFWFilter": False,
            "userCanceled": False,
            "layers": [_layer(99), text, _layer(1)],
        }
    )
    job = project.job("layers-55")
    assert job is not None
    expected = [_public(_layer(1)), _public(text), _public(_layer(99))]
    assert job.result_url == "https://cdn.sogni.ai/base.png"
    assert job.layers == expected
    assert completed[-1]["layers"] == expected

    api.download_url = AsyncMock(return_value="https://cdn.sogni.ai/fresh-layer.png")
    assert await job.get_layer_url(99) == "https://cdn.sogni.ai/fresh-layer.png"
    assert await job.getLayerUrl(1) == "https://cdn.sogni.ai/fresh-layer.png"
    assert [call.args[0] for call in api.download_url.await_args_list] == [
        {"jobId": project.id, "imageId": "layers-55", "type": "complete", "layerIndex": 99},
        {"jobId": project.id, "imageId": "layers-55", "type": "complete", "layerIndex": 1},
    ]
    for missing in (0, 3, 100):
        with pytest.raises(RuntimeError, match="Layer is not available"):
            await job.get_layer_url(missing)
    assert api.download_url.await_count == 2

    # A REST sync of the finished job reads the layers from its stored result.
    job._update({"layers": None})
    await job._sync_with_rest_data(
        {"imgID": "layers-55", "status": "jobCompleted", "result": {"layers": [text, _layer(1)]}}
    )
    assert job.layers == [_public(_layer(1)), _public(text)]


@pytest.mark.asyncio
async def test_ordinary_seedream_results_carry_no_layers() -> None:
    api = ProjectsApi(FakeClient())
    project = Project(
        {"type": "image", "modelId": SEEDREAM, "positivePrompt": "A portrait", "numberOfMedia": 1},
        api,
    )
    api._projects.append(project)
    completed: list[dict[str, Any]] = []
    api.on("job", lambda event: completed.append(event) if event["type"] == "completed" else None)
    api._handle_job_state(
        {"type": "jobStarted", "jobID": project.id, "imgID": "plain", "workerName": ""}
    )
    await api._apply_job_result(
        {"jobID": project.id, "imgID": "plain", "resultUrl": "https://cdn.sogni.ai/plain.png"}
    )
    job = project.job("plain")
    assert job is not None and job.layers is None
    assert "layers" not in completed[-1]
    assert "layers" not in job.to_dict()


@pytest.mark.asyncio
async def test_recovered_projects_replay_seedream_layers() -> None:
    api = ProjectsApi(FakeClient())
    project = Project(
        {"type": "image", "modelId": SEEDREAM_55, "positivePrompt": "", "numberOfMedia": 1}, api
    )
    api._projects.append(project)
    text = _layer(99, textRecognition=_recognition())
    await api._replay_recovered_project(
        project,
        {
            "status": "completed",
            "reason": "allJobsCompleted",
            "completedWorkerJobs": [
                {
                    "imgID": "recovered-layers",
                    "status": "jobCompleted",
                    "resultUrl": "https://cdn.sogni.ai/base.png",
                    "result": {"layers": [text]},
                }
            ],
        },
    )
    job = project.job("recovered-layers")
    assert job is not None and job.layers == [_public(text)]


@pytest.mark.asyncio
async def test_create_sends_python_named_seedream_options_with_wire_names() -> None:
    client = FakeClient()
    api = ProjectsApi(client)
    api.get_model_options = AsyncMock(return_value=IMAGE_OPTIONS)
    await api.create(
        type="image",
        model_id=SEEDREAM_55,
        positive_prompt="A glossy cherry sticker",
        number_of_media=1,
        size_preset="custom",
        width=1536,
        height=1536,
        seedream_background="transparent",
    )
    request_type, request = client.socket.sent[-1]
    assert request_type == "jobRequest"
    assert request["keyFrames"][0]["seedreamBackground"] == "transparent"
    assert seedream_keys(request["keyFrames"][0]) == ["seedreamBackground"]


@pytest.mark.asyncio
async def test_layer_quotes_send_the_ceiling_and_not_the_free_options() -> None:
    get, api = _estimate_api()
    base = {
        "model": SEEDREAM_55,
        "image_count": 1,
        "step_count": 1,
        "preview_count": 0,
        "width": 1536,
        "height": 1536,
    }
    await api.estimate_cost(**base, seedream_background="transparent")
    assert get.await_args.args == (
        "/api/v2/job/estimate/spark/fast/seedream-5-5-uncensored/1/1/0/0/0/1536/1536",
        {},
    )
    await api.estimate_cost(**base, context_images=1, seedream_layer_decomposition=True)
    assert get.await_args.args[1] == {"seedreamLayerDecomposition": "true"}
    await api.estimate_cost(
        **base,
        context_images=1,
        seedream_layer_decomposition=True,
        seedream_layer_size="2K",
        seedream_layer_max_images=100,
        seedream_layer_text_recognition=True,
        billing_mode="subscription",
    )
    path, query = get.await_args.args
    assert (
        path == "/api/v3/job/estimate/spark/fast/seedream-5-5-uncensored/1/1/0/0/0/1536/1536/0/_/1"
    )
    assert json.dumps(query) == json.dumps(
        {
            "seedreamLayerDecomposition": "true",
            "seedreamLayerSize": "2K",
            "seedreamLayerMaxImages": "100",
            "billingMode": "subscription",
        }
    )
