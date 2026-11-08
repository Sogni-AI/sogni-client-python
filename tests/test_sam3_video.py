import pytest

from sogni_client import SAM3_VIDEO_SEGMENT_MODEL_ID, is_video_segmentation_model
from sogni_client.errors import ApiError
from sogni_client.projects import _job_provenance_from_result, create_job_request_message
from sogni_client.utils import calculate_video_frames, is_video_model

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
        {"sam3VideoPrompt": {"points": [{"x": 0.5, "y": 0.5, "label": "positive"}], "objectIndex": 1}},
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
