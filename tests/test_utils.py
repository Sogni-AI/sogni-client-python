from __future__ import annotations

import json
from io import BytesIO

import pytest

from sogni_client.utils import (
    b64_json_decode,
    b64_json_encode,
    calculate_video_frames,
    camel_to_snake,
    detect_content_type,
    drop_none,
    get_video_workflow_type,
    is_audio_model,
    is_external_video_model,
    is_happyhorse_model,
    is_ltx_model,
    is_seedance_model,
    is_video_model,
    is_wan_model,
    new_id,
    normalize_params,
    parse_sse_chunk,
    read_media,
    snake_to_camel,
)


def test_base64_json_codec_round_trips_unicode_and_compact_data() -> None:
    payload = {
        "jobID": "ABC",
        "prompt": "a café on 🌙",
        "nested": [1, True, None, {"value": "λ"}],
    }

    encoded = b64_json_encode(payload)

    assert "\n" not in encoded
    assert b64_json_decode(encoded) == payload


@pytest.mark.parametrize("encoded", ["not base64", "e30", "////"])
def test_base64_json_decode_rejects_malformed_payloads(encoded: str) -> None:
    with pytest.raises((ValueError, UnicodeDecodeError, json.JSONDecodeError)):
        b64_json_decode(encoded)


def test_case_conversion_and_param_normalization_accepts_both_styles() -> None:
    assert snake_to_camel("number_of_media") == "numberOfMedia"
    assert camel_to_snake("numberOfMedia") == "number_of_media"
    assert normalize_params(
        {"modelId": "wan-model", "number_of_media": 2},
        positive_prompt="hello",
        number_of_media=3,
    ) == {
        "modelId": "wan-model",
        "numberOfMedia": 3,
        "positivePrompt": "hello",
    }


def test_drop_none_is_recursive_but_preserves_false_zero_and_list_positions() -> None:
    assert drop_none(
        {
            "missing": None,
            "false": False,
            "zero": 0,
            "nested": {"drop": None, "keep": ""},
            "items": [None, {"drop": None, "keep": 1}],
        }
    ) == {
        "false": False,
        "zero": 0,
        "nested": {"keep": ""},
        "items": [None, {"keep": 1}],
    }


def test_new_id_returns_uppercase_uuid_strings() -> None:
    first = new_id()
    second = new_id()

    assert first == first.upper()
    assert len(first) == 36
    assert first != second


@pytest.mark.parametrize(
    ("model", "duration", "fps", "expected"),
    [
        # WAN always generates at 16 fps; the requested output fps only controls
        # interpolation after generation.
        ("wan_v2.2-14b-fp8_t2v", 5, 16, 81),
        ("wan_v2.2-14b-fp8_t2v", 5, 32, 81),
        # LTX generates at the requested fps and snaps to the 1 + n*8 lattice.
        ("ltx23-22b-dev_t2v", 5, 24, 121),
        ("ltx2-19b-dev_t2v", 4, 24, 97),
        ("ltx23-22b-dev_t2v", 1, 25, 25),
        # Partner/external models use the ordinary duration * fps + 1 rule.
        ("seedance-2-0-fast", 5, 24, 121),
        ("happyhorse-1.1-t2v", 3, 24, 73),
        # Python's built-in round() would produce 2 here; JS Math.round() is 3.
        ("seedance-2-0-fast", 0.125, 20, 4),
    ],
)
def test_calculate_video_frames_matches_javascript_sdk(
    model: str, duration: float, fps: float, expected: int
) -> None:
    assert calculate_video_frames(model, duration, fps) == expected


def test_calculate_video_frames_applies_explicit_bounds_after_model_rules() -> None:
    assert calculate_video_frames("ltx23-22b-dev_t2v", 1, 8, min_frames=17) == 17
    assert calculate_video_frames("wan_v2.2-14b-fp8_t2v", 20, 32, max_frames=161) == 161


def test_video_workflow_detection_rejects_family_invalid_suffixes() -> None:
    assert get_video_workflow_type("wan_v2.2-14b-fp8_v2v") is None
    assert get_video_workflow_type("seedance-2-0_a2v") is None


@pytest.mark.parametrize(
    ("model", "wan", "ltx", "seedance", "happyhorse", "video", "external", "audio"),
    [
        ("wan_v2.2-14b-fp8_i2v", True, False, False, False, True, False, False),
        ("ltx23-22b-dev_t2v", False, True, False, False, True, False, False),
        ("seedance-2-0-mini", False, False, True, False, True, True, False),
        ("happyhorse-1.1-r2v", False, False, False, True, True, True, False),
        ("ace_step_1.5_xl_turbo", False, False, False, False, False, False, True),
        ("flux1-schnell-fp8", False, False, False, False, False, False, False),
    ],
)
def test_model_family_predicates(
    model: str,
    wan: bool,
    ltx: bool,
    seedance: bool,
    happyhorse: bool,
    video: bool,
    external: bool,
    audio: bool,
) -> None:
    assert is_wan_model(model) is wan
    assert is_ltx_model(model) is ltx
    assert is_seedance_model(model) is seedance
    assert is_happyhorse_model(model) is happyhorse
    assert is_video_model(model) is video
    assert is_external_video_model(model) is external
    assert is_audio_model(model) is audio


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("wan_v2.2-14b-fp8_i2v", "i2v"),
        ("wan_v2.2-14b-fp8_animate-move", "animate-move"),
        ("ltx23-22b-dev_v2v", "v2v"),
        ("happyhorse-1.1-r2v", "r2v"),
        ("happyhorse-1.1-t2v", "t2v"),
        ("flux1-schnell-fp8", None),
    ],
)
def test_get_video_workflow_type(model: str, expected: str | None) -> None:
    assert get_video_workflow_type(model) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"\xff\xd8\xffmore", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"GIF89a", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBP", "image/webp"),
        (b"RIFF\x00\x00\x00\x00WAVE", "audio/wav"),
        (b"ID3music", "audio/mpeg"),
        (b"\x00\x00\x00\x18ftypM4A ", "audio/mp4"),
        (b"\x00\x00\x00\x18ftypqt  ", "video/quicktime"),
        (b"\x00\x00\x00\x18ftypisom", "video/mp4"),
    ],
)
def test_detect_content_type_by_signature(raw: bytes, expected: str) -> None:
    assert detect_content_type(raw) == expected


def test_detect_content_type_and_read_media_support_paths_and_binary_files(tmp_path) -> None:
    image = tmp_path / "input.webp"
    image.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")

    assert detect_content_type(image) == "image/webp"
    assert read_media(image) == image.read_bytes()
    assert read_media(BytesIO(b"file object")) == b"file object"


def test_read_media_rejects_unsupported_values() -> None:
    with pytest.raises(TypeError, match="bytes, a path, or a binary file object"):
        read_media(object())


def test_parse_sse_chunk_handles_crlf_comments_multiline_json_and_text() -> None:
    chunk = (
        ": keep-alive\r\n"
        "id: 41\r\n"
        "event: workflow_event\r\n"
        'data: {"status":"running",\r\n'
        'data: "step":"image"}\r\n'
        "\r\n"
        "event:\n"
        "data: not-json\n"
        "\n"
    )

    assert parse_sse_chunk(chunk) == [
        {
            "id": "41",
            "event": "workflow_event",
            "data": {"status": "running", "step": "image"},
            "raw": (
                ": keep-alive\r\nid: 41\r\nevent: workflow_event\r\n"
                'data: {"status":"running",\r\ndata: "step":"image"}'
            ),
        },
        {"event": "message", "data": "not-json", "raw": "event:\ndata: not-json"},
    ]


def test_parse_sse_chunk_preserves_empty_data_and_ignores_unknown_fields() -> None:
    assert parse_sse_chunk("retry: 1000\ndata:\n\n") == [
        {"event": "message", "data": "", "raw": "retry: 1000\ndata:"}
    ]
