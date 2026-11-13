"""MiniMax H3 Control: wire parity with the JS SDK, uploads, refusals, and LTX v2v
requests that stay byte-identical to 5.81.0."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_projects import FakeClient, model_options

from sogni_client.errors import ApiError
from sogni_client.minimax_h3_control import MINIMAX_H3_CONTROL_MODEL_ID, MINIMAX_H3_CONTROL_TYPES
from sogni_client.projects import ProjectsApi, create_job_request_message
from sogni_client.utils import (
    camel_to_snake,
    get_video_workflow_type,
    is_minimax_h3_keyframe_model,
    is_minimax_h3_model,
)

FIXTURES = Path(__file__).parent / "fixtures"
PARITY = json.loads((FIXTURES / "minimax-h3-control-parity.json").read_text())
BASELINE = json.loads((FIXTURES / "ltx-v2v-control-baseline-5.81.0.json").read_text())
OPTIONS = model_options(
    "video",
    sampler={"allowed": ["euler"], "default": "euler"},
    scheduler={"allowed": ["simple"], "default": "simple"},
)
PARAMS = PARITY["cases"][0]["params"]


def test_control_model_is_an_h3_v2v_model():
    assert is_minimax_h3_model(MINIMAX_H3_CONTROL_MODEL_ID)
    assert get_video_workflow_type(MINIMAX_H3_CONTROL_MODEL_ID) == "v2v"
    assert not is_minimax_h3_keyframe_model(MINIMAX_H3_CONTROL_MODEL_ID)
    assert MINIMAX_H3_CONTROL_TYPES == ("canny", "depth", "pose", "hed", "mlsd", "inpaint")


@pytest.mark.parametrize("case", PARITY["cases"], ids=lambda case: case["name"])
def test_javascript_wire_parity(case):
    frame = create_job_request_message("minimax-control-parity", case["params"], OPTIONS)[
        "keyFrames"
    ][0]
    for name, expected in case["expected"].items():
        actual = bool(frame.get(name)) if name.startswith("hasReference") else frame.get(name)
        assert actual == expected, name


@pytest.mark.parametrize("strength", [0, 1, 2.5, 10])
def test_control_strength_range_reaches_wire(strength):
    frame = create_job_request_message(
        "control-strength",
        {**PARAMS, "controlNet": {"name": "depth", "strength": strength}},
        OPTIONS,
    )["keyFrames"][0]
    assert frame["currentControlNetsJob"] == [{"name": "depth", "controlStrength": strength}]


def test_a_mask_can_accompany_a_structural_guide():
    frame = create_job_request_message(
        "pose-mask", {**PARAMS, "controlNet": {"name": "pose"}, "referenceMask": True}, OPTIONS
    )["keyFrames"][0]
    assert frame["hasReferenceMask"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"referenceVideo": None},
        {"controlNet": {"name": "inpaint"}},
        {"controlNet": {"name": "detailer"}},
        {"controlNet": {"name": "outpaint"}},
        {"referenceMask": True, "referenceMaskVideo": True},
        {"referenceVideo": None, "controlVideo": True, "referenceMask": True},
        {"referenceVideo": None, "controlVideo": True, "referenceMaskVideo": True},
        {"controlVideo": True, "controlNet": {"name": "pose", "preprocess": True}},
        {"referenceMask": True, "controlNet": {"name": "inpaint", "preprocess": True}},
        {"controlVideo": True, "controlNet": {"name": "scribble", "preprocess": False}},
        {"controlVideo": True, "controlNet": {"name": "layout", "preprocess": False}},
        {"controlVideo": True, "controlNet": {"name": "gray", "preprocess": False}},
        {"controlNet": {"name": "depth", "preprocess": "yes"}},
        {"controlNet": {"name": "depth", "guidanceStart": 0.9, "guidanceEnd": 0.1}},
        {"controlNet": {"name": "depth", "guidanceStart": "0.1"}},
        {"controlNet": {"name": "depth", "guidanceEnd": 2}},
        {"controlNet": {"name": "depth", "strength": float("nan")}},
        {"controlNet": {"name": "depth", "strength": 10.01}},
        {"controlNet": {"name": "depth", "strength": -0.1}},
        {"referenceVideos": [True]},
        {"referenceAudios": [True]},
        {"referenceAudio": True},
        {"referenceAudioIdentity": True},
        {"referenceImageEnd": True},
        {"contextImages": [True]},
        {"referenceVideoUrls": ["https://example.com/a.mp4"]},
        {"keyframes": [{"frameIndex": 40, "image": True}]},
        {"steps": 20},
        {"guidance": 2},
        {"negativePrompt": "blur"},
        {"frames": 244},
        {"fps": 30},
    ],
)
def test_invalid_control_requests_are_rejected(changes):
    with pytest.raises((ValueError, ApiError)):
        create_job_request_message("invalid", {**PARAMS, **changes}, OPTIONS)


def _with_extras(params):
    extras = BASELINE["extras"]
    result = {
        **params,
        "referenceMaskVideo": extras["referenceMaskVideo"],
        "controlVideo": extras["controlVideo"],
    }
    if params.get("controlNet"):
        result["controlNet"] = {**params["controlNet"], **extras["controlNet"]}
    return result


@pytest.mark.parametrize("name", sorted(BASELINE["requests"]))
def test_ltx_v2v_requests_are_byte_identical_to_5_81_0(name):
    case = BASELINE["requests"][name]
    expected = json.dumps(case["message"])
    plain = create_job_request_message(
        f"baseline-{name}", dict(case["params"]), BASELINE["options"]
    )
    assert json.dumps(plain) == expected
    extra = create_job_request_message(
        f"baseline-{name}", _with_extras(case["params"]), BASELINE["options"]
    )
    assert json.dumps(extra) == expected


def test_ltx_strength_stays_capped_at_one():
    params = BASELINE["requests"]["ltx25-depth"]["params"]
    with pytest.raises(ValueError, match="strength"):
        create_job_request_message(
            "ltx-strength", {**params, "controlNet": {"name": "depth", "strength": 2.5}}, OPTIONS
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("snake_case", [False, True])
async def test_project_create_uploads_numbered_roles(moving, snake_case):
    api = ProjectsApi(FakeClient())
    api.get_model_options = AsyncMock(return_value=OPTIONS)
    api._upload_asset = AsyncMock(return_value="video/mp4")
    video, mask, control = b"source-video", b"white-mask", b"prepared-control"
    data = {
        **PARAMS,
        "referenceVideo": video,
        "controlVideo": control,
        "controlNet": {"name": "depth", "guidanceStart": 0.1, "guidanceEnd": 0.9},
        ("referenceMaskVideo" if moving else "referenceMask"): mask,
    }
    if snake_case:
        data["controlNet"] = {
            camel_to_snake(key): value for key, value in data["controlNet"].items()
        }
        project = await api.create(**{camel_to_snake(key): value for key, value in data.items()})
    else:
        project = await api.create(data)
    try:
        calls = api._upload_asset.call_args_list
        assert [(call.args[1], call.args[2]) for call in calls] == [
            ("referenceVideo", video),
            *(([("referenceVideo2", mask)]) if moving else [("referenceMask", mask)]),
            ("referenceVideo3", control),
        ]
        sent = api.client.socket.sent[-1][1]["keyFrames"][0]
        assert sent["hasReferenceVideo3"] is True
        assert sent["hasReferenceVideo2" if moving else "hasReferenceMask"] is True
        assert sent["currentControlNetsJob"][0]["controlGuidanceStart"] == 0.1
        assert sent["currentControlNetsJob"][0]["controlGuidanceEnd"] == 0.9
        # The JS SDK sends no content type for these slots, and the Socket reads none.
        assert "referenceVideo2ContentType" not in sent
        assert "referenceVideo3ContentType" not in sent
    finally:
        project._dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("inpaint", [False, True])
async def test_ltx_project_create_uploads_only_what_it_did_before(inpaint):
    api = ProjectsApi(FakeClient())
    api.get_model_options = AsyncMock(return_value=OPTIONS)
    api._upload_asset = AsyncMock(return_value="video/mp4")
    video, mask = b"source-video", b"white-mask"
    data = {
        **_with_extras(BASELINE["requests"]["ltx25-depth"]["params"]),
        "referenceVideo": video,
        "referenceMaskVideo": b"moving-mask",
        "controlVideo": b"prepared-control",
    }
    if inpaint:
        data["controlNet"] = {"name": "inpaint"}
        data["referenceMask"] = mask
    project = await api.create(data)
    try:
        calls = [(call.args[1], call.args[2]) for call in api._upload_asset.call_args_list]
        assert calls == [("referenceVideo", video), *([("referenceMask", mask)] if inpaint else [])]
        sent = api.client.socket.sent[-1][1]["keyFrames"][0]
        assert "hasReferenceVideo2" not in sent and "hasReferenceVideo3" not in sent
    finally:
        project._dispose()
