from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sogni_client.projects import ProjectsApi, create_job_request_message

OPTIONS = {
    "type": "image",
    "sampler": {"allowed": [], "default": None},
    "scheduler": {"allowed": [], "default": None},
}


@pytest.mark.parametrize("variant", ["sunburst", "flare"])
def test_native_gif_wire_and_defaults(variant: str) -> None:
    params = {
        "type": "image",
        "modelId": f"gpt-image-2.6-{variant}",
        "positivePrompt": "A gently swaying lantern, seamless loop",
        "numberOfMedia": 1,
        "outputFormat": "gif",
        "gptImageFrameCount": 9,
        "gptImageFrameDurationMs": 200,
        "gptImageComplexity": "low",
    }
    request = create_job_request_message("gif", params, OPTIONS)
    assert request["outputFormat"] == "gif"
    frame = request["keyFrames"][0]
    assert (frame["width"], frame["height"]) == (896, 896)
    assert frame["gptImageFrameCount"] == 9
    assert frame["gptImageFrameDurationMs"] == 200
    assert frame["gptImageComplexity"] == "low"
    small = create_job_request_message("small", {**params, "width": 320, "height": 320}, OPTIONS)
    assert small["keyFrames"][0]["width"] == 320
    with pytest.raises(Exception, match="pixel budget"):
        create_job_request_message("large", {**params, "width": 1024, "height": 1024}, OPTIONS)
    with pytest.raises(Exception, match="text-to-image"):
        create_job_request_message("edit", {**params, "contextImages": [b"ref"]}, OPTIONS)


@pytest.mark.parametrize("model", ["gpt-image-2", "gpt-image-2.5-flare", "z_image_turbo_bf16"])
def test_gif_requires_explicit_26_model(model: str) -> None:
    with pytest.raises(Exception, match="require GPT Image 2.6"):
        create_job_request_message(
            "legacy",
            {
                "type": "image",
                "modelId": model,
                "positivePrompt": "A loop",
                "numberOfMedia": 1,
                "outputFormat": "gif",
            },
            OPTIONS,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [{}, {"size_preset": "custom"}, {"size_preset": "1024x1024", "width": 320, "height": 320}],
)
async def test_estimate_uses_same_gif_dimensions_as_create(extra):
    api = object.__new__(ProjectsApi)
    api.get_model_options = AsyncMock(return_value=OPTIONS)
    api.get_size_presets = AsyncMock(
        side_effect=AssertionError("GIF custom size replaced by preset")
    )
    get = AsyncMock(return_value={"quote": {"project": {"costInToken": 1, "costInUSD": 0.01, "costInSpark": 2, "costInSogni": 3}}})
    api.client = SimpleNamespace(socket=SimpleNamespace(get=get))
    await api.estimate_cost(
        model="gpt-image-2.6-flare",
        image_count=1,
        step_count=1,
        preview_count=0,
        output_format="gif",
        gpt_image_frame_count=9,
        gpt_image_frame_duration_ms=200,
        gpt_image_complexity="low",
        **extra,
    )
    path, query = get.await_args.args
    side = extra.get("width", 896)
    assert path.endswith(f"/0/0/{side}/{side}")
    assert query == {
        "outputFormat": "gif",
        "gptImageFrameCount": 9,
        "gptImageFrameDurationMs": 200,
        "gptImageComplexity": "low",
    }
