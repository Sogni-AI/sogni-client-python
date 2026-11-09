import copy
import json
from typing import Any

import httpx
import pytest

from sogni_client import (
    SAM3_VIDEO_SEGMENT_MODEL_ID,
    Sam3PromptMask,
    Sam3VideoFramePrompt,
    Sam3VideoPrompt,
    Sam3VideoSelection,
    is_video_segmentation_model,
)
from sogni_client.auth import ApiKeyAuthManager
from sogni_client.errors import ApiError
from sogni_client.events import EventEmitter
from sogni_client.projects import (
    ProjectsApi,
    _job_provenance_from_result,
    create_job_request_message,
)
from sogni_client.transport import RestClient
from sogni_client.utils import (
    b64_json_decode,
    b64_json_encode,
    calculate_video_frames,
    is_video_model,
)

OPTIONS = {"type": "video"}
PARAMS = {
    "type": "video",
    "modelId": SAM3_VIDEO_SEGMENT_MODEL_ID,
    "positivePrompt": "",
    "numberOfMedia": 1,
    "referenceVideo": True,
    "sam3VideoPrompt": {"text": "unicorn horn", "frameIndex": 126},
    "width": 1920,
    "height": 1080,
    "frames": 255,
    "fps": 24,
}


def test_native_video_selection_wire_and_fractional_timing():
    assert is_video_segmentation_model(PARAMS["modelId"])
    assert is_video_model(PARAMS["modelId"])
    wire = create_job_request_message("selection-wire", PARAMS, OPTIONS)
    assert wire["outputFormat"] == "mp4"
    frame = wire["keyFrames"][0]
    assert frame["hasReferenceVideo"] is True
    assert frame["sam3VideoPrompt"]["frameIndex"] == 126
    assert frame["frames"] == 255
    assert frame["interpolation"] == "none"
    fractional = create_job_request_message("fractional", {**PARAMS, "fps": 30000 / 1001}, OPTIONS)
    assert fractional["keyFrames"][0]["fps"] == 30000 / 1001
    assert calculate_video_frames(PARAMS["modelId"], 300 / (30000 / 1001), 30000 / 1001) == 300


@pytest.mark.parametrize(
    "overrides",
    [
        {"numberOfMedia": 2},
        {"referenceVideo": None},
        {"sam3VideoPrompt": {"text": "horn", "frameIndex": 255}},
        {"sam3VideoPrompt": {"text": "horn", "applyMask": True}},
        {"sam3VideoPrompt": {"text": "horn", "maxInstances": 2}},
        {"positivePrompt": "change the video"},
        {"trimEndFrame": True},
        {
            "sam3VideoPrompt": {
                "points": [{"x": 0.5, "y": 0.5, "label": "positive"}],
                "objectIndex": 1,
            }
        },
        {"sam3VideoPrompt": {"boxes": [{"x0": 0, "y0": 0, "x1": 1, "y1": 1}], "objectIndex": 1}},
    ],
)
def test_rejects_incompatible_video_requests(overrides):
    with pytest.raises((ValueError, ApiError)):
        create_job_request_message("invalid", {**PARAMS, **overrides}, OPTIONS)


def test_video_mask_receipt_keeps_source_hash_and_exact_timing():
    receipt = {
        "sha256": "a" * 64,
        "sourceVideoSha256": "b" * 64,
        "maskWidth": 1920,
        "maskHeight": 1080,
        "maskFrames": 255,
        "maskFps": 30000 / 1001,
        "samVersion": "sam3.1@203579918+fp16@9ba99c92",
    }
    assert _job_provenance_from_result(receipt) == receipt


def test_omitted_metadata_leaves_source_dimensions_for_server_probe():
    params = {
        key: value
        for key, value in PARAMS.items()
        if key not in ("width", "height", "frames", "fps")
    }
    frame = create_job_request_message("server-probed", params, OPTIONS)["keyFrames"][0]
    assert all(name not in frame for name in ("width", "height", "frames", "fps"))


# SAM 3 video selection opt-in options (mask, objects + combine, frames, memory),
# mirroring sogni-client `scripts/check-sam3-video-segment.cjs` and
# `scripts/check-video-estimates.cjs`.

# A prompt without the opt-in fields is sent byte for byte as before. These key
# frames were captured from 5.79.0 (20692d2) before the opt-in fields existed,
# serialized exactly as the socket sends them.
OLD_KEY_FRAMES = (
    '{"stepsIsEnabled":true,"siRotation":0,"siDragOffsetIsEnabled":true,"strength":0.5,"siZoomScaleIsEnabled":true,"isEnabled":true,"processing":"CPU, GPU","useLastImageAsGuideImageInAnimation":true,"guidanceScaleIsEnabled":true,"siImageBackgroundColor":"black","cnDragOffset":[0,0],"scheduler":null,"timeStepSpacing":null,"cnRotation":0,"siZoomScale":1,"modelID":"sam3_video_segment_fp16","cnRotationIsEnabled":true,"startingImageZoomPanIsOn":false,"siRotationIsEnabled":true,"cnImageBackgroundColor":"clear","strengthIsEnabled":true,"siDragOffset":[0,0],"useLastImageAsCNImageInAnimation":false,"positivePrompt":"","controlNetZoomPanIsOn":false,"cnZoomScaleIsEnabled":true,"currentControlNets":null,"stylePrompt":"","cnDragOffsetIsEnabled":true,"frameIndex":0,"startingImage":null,"cnZoomScale":1,"hasReferenceVideo":true,"sam3VideoPrompt":{"points":[],"boxes":[],"text":"unicorn horn","threshold":0.5,"applyMask":false,"maxInstances":1,"frameIndex":126,"objectIndex":0,"output":"mask","edges":"hard"},"steps":1,"seed":0,"interpolation":"none","generateAudio":false,"width":1920,"height":1080,"frames":255,"fps":24}',
    '{"stepsIsEnabled":true,"siRotation":0,"siDragOffsetIsEnabled":true,"strength":0.5,"siZoomScaleIsEnabled":true,"isEnabled":true,"processing":"CPU, GPU","useLastImageAsGuideImageInAnimation":true,"guidanceScaleIsEnabled":true,"siImageBackgroundColor":"black","cnDragOffset":[0,0],"scheduler":null,"timeStepSpacing":null,"cnRotation":0,"siZoomScale":1,"modelID":"sam3_video_segment_fp16","cnRotationIsEnabled":true,"startingImageZoomPanIsOn":false,"siRotationIsEnabled":true,"cnImageBackgroundColor":"clear","strengthIsEnabled":true,"siDragOffset":[0,0],"useLastImageAsCNImageInAnimation":false,"positivePrompt":"","controlNetZoomPanIsOn":false,"cnZoomScaleIsEnabled":true,"currentControlNets":null,"stylePrompt":"","cnDragOffsetIsEnabled":true,"frameIndex":0,"startingImage":null,"cnZoomScale":1,"hasReferenceVideo":true,"sam3VideoPrompt":{"points":[{"x":0.25,"y":0.75,"label":"positive"}],"boxes":[{"x0":0.1,"y0":0.2,"x1":0.6,"y1":0.9,"label":"positive"}],"threshold":0.5,"multimask":false,"applyMask":false,"maxInstances":1,"frameIndex":0,"objectIndex":0,"output":"mask","edges":"hard"},"steps":1,"seed":0,"interpolation":"none","generateAudio":false}',
    '{"stepsIsEnabled":true,"siRotation":0,"siDragOffsetIsEnabled":true,"strength":0.5,"siZoomScaleIsEnabled":true,"isEnabled":true,"processing":"CPU, GPU","useLastImageAsGuideImageInAnimation":true,"guidanceScaleIsEnabled":true,"siImageBackgroundColor":"black","cnDragOffset":[0,0],"scheduler":null,"timeStepSpacing":null,"cnRotation":0,"siZoomScale":1,"modelID":"sam3_video_segment_fp16","cnRotationIsEnabled":true,"startingImageZoomPanIsOn":false,"siRotationIsEnabled":true,"cnImageBackgroundColor":"clear","strengthIsEnabled":true,"siDragOffset":[0,0],"useLastImageAsCNImageInAnimation":false,"positivePrompt":"","controlNetZoomPanIsOn":false,"cnZoomScaleIsEnabled":true,"currentControlNets":null,"stylePrompt":"","cnDragOffsetIsEnabled":true,"frameIndex":0,"startingImage":null,"cnZoomScale":1,"hasReferenceVideo":true,"sam3VideoPrompt":{"points":[],"boxes":[{"x0":0,"y0":0,"x1":0.5,"y1":0.5,"label":"positive"},{"x0":0.5,"y0":0.5,"x1":1,"y1":1,"label":"positive"}],"threshold":0.3,"applyMask":false,"maxInstances":1,"frameIndex":0,"objectIndex":1,"output":"mask","edges":"hard"},"steps":1,"seed":0,"interpolation":"none","generateAudio":false,"fps":29.97002997002997}',
    '{"stepsIsEnabled":true,"siRotation":0,"siDragOffsetIsEnabled":true,"strength":0.5,"siZoomScaleIsEnabled":true,"isEnabled":true,"processing":"CPU, GPU","useLastImageAsGuideImageInAnimation":true,"guidanceScaleIsEnabled":true,"siImageBackgroundColor":"black","cnDragOffset":[0,0],"scheduler":null,"timeStepSpacing":null,"cnRotation":0,"siZoomScale":1,"modelID":"sam3_video_segment_fp16","cnRotationIsEnabled":true,"startingImageZoomPanIsOn":false,"siRotationIsEnabled":true,"cnImageBackgroundColor":"clear","strengthIsEnabled":true,"siDragOffset":[0,0],"useLastImageAsCNImageInAnimation":false,"positivePrompt":"","controlNetZoomPanIsOn":false,"cnZoomScaleIsEnabled":true,"currentControlNets":null,"stylePrompt":"","cnDragOffsetIsEnabled":true,"frameIndex":0,"startingImage":null,"cnZoomScale":1,"hasReferenceVideo":true,"sam3VideoPrompt":{"points":[],"boxes":[{"x0":0.1,"y0":0.2,"x1":0.4,"y1":0.8,"label":"negative"}],"text":"horn","threshold":0.5,"applyMask":false,"maxInstances":1,"frameIndex":7,"objectIndex":0,"output":"mask","edges":"hard"},"steps":1,"seed":0,"interpolation":"none","generateAudio":false}',
)
OLD_PROMPTS = (
    (
        {"text": "unicorn horn", "frameIndex": 126},
        {"width": 1920, "height": 1080, "frames": 255, "fps": 24},
    ),
    (
        {
            "points": [{"x": 0.25, "y": 0.75, "label": "positive"}],
            "boxes": [{"x0": 0.1, "y0": 0.2, "x1": 0.6, "y1": 0.9}],
            "multimask": False,
        },
        {},
    ),
    (
        {
            "boxes": [
                {"x0": 0, "y0": 0, "x1": 0.5, "y1": 0.5},
                {"x0": 0.5, "y0": 0.5, "x1": 1, "y1": 1},
            ],
            "objectIndex": 1,
            "threshold": 0.3,
        },
        {"fps": 30000 / 1001},
    ),
    (
        {
            "text": "horn",
            "boxes": [{"x0": 0.1, "y0": 0.2, "x1": 0.4, "y1": 0.8, "label": "negative"}],
            "frameIndex": 7,
        },
        {},
    ),
)
_SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
OPT_IN_PROMPTS = (
    {"mask": {"polygons": [_SQUARE, [[0.4, 0.4], [0.6, 0.4], [0.5, 0.6]]]}, "frameIndex": 4},
    {
        "objects": [
            {"text": "monitor"},
            {"points": [{"x": 0.5, "y": 0.5, "label": "positive"}], "multimask": False},
            {"mask": {"polygons": [_SQUARE]}, "objectIndex": 0},
        ],
        "combine": "union",
        "frameIndex": 12,
    },
    {
        "frames": [
            {"frameIndex": 0, "text": "horn"},
            {
                "frameIndex": 254,
                "boxes": [{"x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.5}],
                "objectIndex": 0,
            },
        ]
    },
    {"text": "horn", "frameIndex": 9, "memory": "anchor"},
    {"objects": [{"text": "a"}, {"text": "b"}], "combine": "union", "memory": "chunk"},
)


def _wire(value: Any) -> str:
    # Exactly the bytes `b64_json_encode` puts on the socket.
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


@pytest.mark.parametrize(
    ("prompt", "extra", "expected"),
    [(*pair, OLD_KEY_FRAMES[i]) for i, pair in enumerate(OLD_PROMPTS)],
)
def test_prompts_without_opt_in_fields_send_the_5_79_0_key_frame(prompt, extra, expected):
    base = {k: v for k, v in PARAMS.items() if k not in ("width", "height", "frames", "fps")}
    message = create_job_request_message(
        "selection-wire", {**base, **extra, "sam3VideoPrompt": prompt}, OPTIONS
    )
    assert _wire(message["keyFrames"][0]) == expected


@pytest.mark.parametrize("prompt", OPT_IN_PROMPTS)
def test_opt_in_fields_reach_the_server_unchanged(prompt):
    sent = copy.deepcopy(prompt)
    frame = create_job_request_message("opt-in", {**PARAMS, "sam3VideoPrompt": prompt}, OPTIONS)[
        "keyFrames"
    ][0]
    assert frame["sam3VideoPrompt"] == sent
    assert _wire(frame["sam3VideoPrompt"]) == _wire(sent)
    assert prompt == sent, "the caller prompt is not modified"
    assert frame["frames"] == 255
    assert frame["hasReferenceVideo"] is True
    assert frame["steps"] == 1
    assert frame["interpolation"] == "none"
    for overrides in (
        {"numberOfMedia": 2},
        {"referenceVideo": None},
        {"positivePrompt": "change the video"},
        {"frames": 0},
    ):
        with pytest.raises((ValueError, ApiError)):
            create_job_request_message(
                "invalid-opt-in", {**PARAMS, "sam3VideoPrompt": prompt, **overrides}, OPTIONS
            )


def test_held_opt_in_request_does_not_follow_later_edits_to_the_caller_prompt():
    # The request is held and resent after a reconnect, so it must not share
    # nested lists with the caller's dict.
    prompt = copy.deepcopy(OPT_IN_PROMPTS[1])
    frame = create_job_request_message("held", {**PARAMS, "sam3VideoPrompt": prompt}, OPTIONS)[
        "keyFrames"
    ][0]
    prompt["objects"].append({"text": "desk"})
    prompt["objects"][2]["mask"]["polygons"][0][0][0] = 0.5
    assert frame["sam3VideoPrompt"] == OPT_IN_PROMPTS[1]


@pytest.mark.parametrize("key", ["mask", "objects", "combine", "frames", "memory"])
def test_each_opt_in_field_is_left_for_the_server_to_validate(key):
    # Present with any value, the prompt is sent as given (null included, as
    # the JS SDK sends it); the server applies the limits and combination rules.
    prompt = {"text": "horn", key: None}
    frame = create_job_request_message(
        "opt-in-key", {**PARAMS, "sam3VideoPrompt": prompt}, OPTIONS
    )["keyFrames"][0]
    assert frame["sam3VideoPrompt"] == prompt


def test_unknown_fields_are_still_rejected_before_the_wire():
    with pytest.raises(ValueError, match="unsupported fields: memories"):
        create_job_request_message(
            "unknown",
            {**PARAMS, "sam3VideoPrompt": {"text": "horn", "memories": "anchor"}},
            OPTIONS,
        )


def test_typed_opt_in_prompts_round_trip_through_the_socket_frame():
    region: Sam3PromptMask = {"polygons": [[(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]]}
    monitor: Sam3VideoSelection = {"text": "monitor", "threshold": 0.4}
    webcam: Sam3VideoSelection = {"mask": region, "objectIndex": 0}
    first: Sam3VideoFramePrompt = {"frameIndex": 0, "text": "horn"}
    last: Sam3VideoFramePrompt = {"frameIndex": 254, "mask": region}
    prompts: list[Sam3VideoPrompt] = [
        {"objects": [monitor, webcam], "combine": "union", "frameIndex": 3, "memory": "anchor"},
        {"frames": [first, last], "memory": "chunk"},
        {"mask": region, "frameIndex": 4, "output": "mask", "edges": "hard"},
    ]
    for prompt in prompts:
        message = create_job_request_message(
            "typed", {**PARAMS, "sam3VideoPrompt": prompt}, OPTIONS
        )
        received = b64_json_decode(b64_json_encode(message))["keyFrames"][0]["sam3VideoPrompt"]
        # Tuples arrive as JSON arrays; nothing else changes.
        assert received == json.loads(json.dumps(prompt))


_QUOTE = {
    "quote": {
        "project": {"costInToken": "1", "costInUSD": "2", "costInSpark": "3", "costInSogni": "4"}
    }
}
_SAM_ESTIMATE = {
    "tokenType": "spark",
    "model": SAM3_VIDEO_SEGMENT_MODEL_ID,
    "width": 1920,
    "height": 1080,
    "frames": 121,
    "fps": 24,
}
_SAM_URL = "https://socket.sogni.ai/api/v1/job-video/estimate/spark/sam3_video_segment_fp16/1920/1080/121/24"


class _QuoteSocket(EventEmitter):
    """Sends quote requests through the real REST client, recording each query and URL."""

    def __init__(self, rest: RestClient) -> None:
        super().__init__()
        self.rest = rest
        self.queries: list[dict[str, Any] | None] = []

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.queries.append(copy.deepcopy(params))
        return await self.rest.get(path, params)


class _QuoteClient(EventEmitter):
    app_source = "pytest"
    rest = None


@pytest.fixture
async def quotes():
    urls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json=_QUOTE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = _QuoteClient()
        client.socket = _QuoteSocket(
            RestClient("https://socket.sogni.ai", ApiKeyAuthManager(), http_client=http)
        )
        yield ProjectsApi(client), client.socket, urls


async def test_sam3_video_quote_without_objects_or_memory_is_unchanged(quotes):
    api, socket, urls = quotes
    legacy = (
        None,
        {"text": "horn"},
        {
            "points": [{"x": 0.5, "y": 0.5, "label": "positive"}],
            "boxes": [{"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.9}],
            "frameIndex": 7,
        },
        {"mask": {"polygons": [[[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]]}},
        {"frames": [{"frameIndex": 0, "text": "horn"}, {"frameIndex": 120, "text": "horn"}]},
        {"text": "horn", "memory": None},
    )
    for prompt in legacy:
        await api.estimate_video_cost(
            _SAM_ESTIMATE if prompt is None else {**_SAM_ESTIMATE, "sam3VideoPrompt": prompt}
        )
    assert urls == [_SAM_URL] * len(legacy)
    # The query handed to the transport has 5.79.0's keys in 5.79.0's order.
    assert {json.dumps(list(query.items())) for query in socket.queries} == {
        json.dumps(
            [
                ["sourceWidth", None],
                ["sourceHeight", None],
                ["hasVideoInput", None],
                ["referenceImageCount", None],
                ["referenceVideoCount", None],
                ["referenceVideoDurationSeconds", None],
                ["keyframeCount", None],
                ["network", None],
                ["billingMode", None],
            ]
        )
    }
    # Other quote parameters keep their 5.79.0 URLs (captured from 20692d2).
    await api.estimate_video_cost(
        {**_SAM_ESTIMATE, "network": "fast", "sam3VideoPrompt": {"text": "horn"}}
    )
    await api.estimate_video_cost(
        {
            **_SAM_ESTIMATE,
            "network": "relaxed",
            "billingMode": "token",
            "referenceVideo": True,
            "sam3VideoPrompt": {"text": "horn"},
        }
    )
    assert urls[-2:] == [
        f"{_SAM_URL}?network=fast",
        f"{_SAM_URL}?hasVideoInput=1&network=relaxed&billingMode=token",
    ]


async def test_sam3_video_quote_sends_object_count_and_memory(quotes):
    api, _socket, urls = quotes
    two_objects: Sam3VideoPrompt = {
        "objects": [{"text": "monitor"}, {"points": [{"x": 0.4, "y": 0.3, "label": "positive"}]}],
        "combine": "union",
    }
    sent = copy.deepcopy(two_objects)
    await api.estimate_video_cost({**_SAM_ESTIMATE, "sam3VideoPrompt": two_objects})
    assert urls[-1] == f"{_SAM_URL}?sam3ObjectCount=2"
    assert two_objects == sent, "the caller prompt is not modified"
    await api.estimate_video_cost(
        {**_SAM_ESTIMATE, "sam3VideoPrompt": {"text": "horn", "memory": "anchor"}}
    )
    assert urls[-1] == f"{_SAM_URL}?sam3Memory=anchor"
    await api.estimate_video_cost(
        {**_SAM_ESTIMATE, "sam3VideoPrompt": {"text": "horn", "memory": "chunk"}}
    )
    assert urls[-1] == f"{_SAM_URL}?sam3Memory=chunk"
    await api.estimate_video_cost(
        {
            **_SAM_ESTIMATE,
            "sam3VideoPrompt": {
                **two_objects,
                "objects": [*two_objects["objects"], {"text": "desk"}],
                "memory": "anchor",
            },
        }
    )
    assert urls[-1] == f"{_SAM_URL}?sam3ObjectCount=3&sam3Memory=anchor"
    # An invalid count is sent as it is, so the quote refuses it as admission refuses the job.
    await api.estimate_video_cost(
        {**_SAM_ESTIMATE, "sam3VideoPrompt": {"objects": [{"text": "monitor"}], "combine": "union"}}
    )
    assert urls[-1] == f"{_SAM_URL}?sam3ObjectCount=1"
    # The Python key style reaches the same quote.
    await api.estimate_video_cost(
        {**_SAM_ESTIMATE, "sam3_video_prompt": {**two_objects, "memory": "anchor"}}
    )
    assert urls[-1] == f"{_SAM_URL}?sam3ObjectCount=2&sam3Memory=anchor"
    await api.estimate_video_cost(
        {**_SAM_ESTIMATE, "network": "fast", "sam3VideoPrompt": {**two_objects, "memory": "anchor"}}
    )
    assert urls[-1] == f"{_SAM_URL}?sam3ObjectCount=2&sam3Memory=anchor&network=fast"
