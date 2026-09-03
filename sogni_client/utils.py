"""Wire-format, media, model, and SSE utilities."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any

_LTX_WORKFLOWS = ("t2v", "i2v", "a2v", "ia2v", "v2v")
_LTX_VIDEO_MODEL_IDS = {
    *{
        f"{prefix}_{workflow}{suffix}"
        for workflow in _LTX_WORKFLOWS
        for prefix, suffix in (
            ("ltx2-19b-fp8", ""),
            ("ltx2-19b-fp8", "_distilled"),
            ("ltx23-22b-fp8", "_distilled"),
            ("ltx23-22b-fp8", "_dev"),
            ("ltx25-22b-int8", "_distilled"),
            ("ltx25-22b-int8", "_dev"),
        )
    },
    "ltx23-22b-10eros-v1.4-fp8mixed_i2v",
}
_WAN_VIDEO_MODEL_IDS = {
    "wan_v2.2-14b-fp8_t2v",
    "wan_v2.2-14b-fp8_i2v",
    "wan_v2.2-14b-fp8_t2v_lightx2v",
    "wan_v2.2-14b-fp8_i2v_lightx2v",
    "wan_v2.2-14b-fp8_s2v_lightx2v",
    "wan_v2.2-14b-fp8_animate-move_lightx2v",
    "wan_v2.2-14b-fp8_animate-replace_lightx2v",
}
_SEEDANCE_VIDEO_MODEL_IDS = {
    "seedance-2-0",
    "seedance-2-0-mini",
    "seedance-2-0-fast",
    "seedance-2-5",
}
_HAPPYHORSE_VIDEO_MODEL_IDS = {
    "happyhorse-1.1-t2v",
    "happyhorse-1.1-i2v",
    "happyhorse-1.1-r2v",
}
_WAN3_VIDEO_MODEL_IDS = {"wan3.0-video", "wan3.0-spicy-video"}
_MINIMAX_H3_VIDEO_MODEL_IDS = {
    "minimax-h3-fl2va-fp8_t2v",
    "minimax-h3-fl2va-fp8_i2v",
    "minimax-h3-fl2va-fp8_flf2v",
    "minimax-h3-ref2va-fp8_r2v",
    "minimax-h3-fl2va-fp8_t2v_turbo",
    "minimax-h3-fl2va-fp8_i2v_turbo",
    "minimax-h3-fl2va-fp8_flf2v_turbo",
    "minimax-h3-fastvideo-int8_t2v_turbo",
    "minimax-h3-fastvideo-int8_i2v_turbo",
    "minimax-h3-fastvideo-int8_flf2v_turbo",
    "minimax-h3-ref2va-fp8_r2v_turbo",
    "minimax-h3-fl2va-fp8_t2v_balanced",
    "minimax-h3-fl2va-fp8_i2v_balanced",
    "minimax-h3-fl2va-fp8_flf2v_balanced",
    "minimax-h3-ref2va-fp8_r2v_balanced",
}
_MINIMAX_H3_TURBO_PATTERN = re.compile(
    r"^minimax-h3-(?:fl2va-fp8|fastvideo-int8)_(?:t2v|i2v|flf2v)_turbo$"
)
_MINIMAX_H3_BALANCED_PATTERN = re.compile(r"^minimax-h3-fl2va-fp8_(?:t2v|i2v|flf2v)_balanced$")

LTX2_FRAME_STEP = 8
MINIMAX_H3_FPS = 24
MINIMAX_H3_FRAME_STEP = 17
MINIMAX_H3_BASE_FRAMES = 124
MINIMAX_H3_MIN_FRAMES = 124
MINIMAX_H3_MAX_FRAMES = 362
MINIMAX_H3_DIMENSION_STEP = 32
MINIMAX_H3_MAX_DIMENSION = 1344
MINIMAX_H3_MAX_PIXELS = 1_032_192
MINIMAX_H3_MIN_DURATION = MINIMAX_H3_MIN_FRAMES / MINIMAX_H3_FPS
MINIMAX_H3_MAX_DURATION = MINIMAX_H3_MAX_FRAMES / MINIMAX_H3_FPS


def new_id() -> str:
    return str(uuid.uuid4()).upper()


def snake_to_camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


def camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def normalize_params(params: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Merge a mapping and keywords, accepting both Python and JS key styles."""

    merged = dict(params or {})
    merged.update(kwargs)
    return {snake_to_camel(key) if "_" in key else key: value for key, value in merged.items()}


def drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_none(item) for item in value]
    return value


def b64_json_encode(data: Any) -> str:
    raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode()
    return base64.b64encode(raw).decode()


def b64_json_decode(data: str) -> Any:
    return json.loads(base64.b64decode(data).decode())


def is_wan_model(model_id: str) -> bool:
    return model_id in _WAN_VIDEO_MODEL_IDS


def is_wan_animate_model(model_id: str) -> bool:
    return model_id in {
        "wan_v2.2-14b-fp8_animate-move_lightx2v",
        "wan_v2.2-14b-fp8_animate-replace_lightx2v",
    }


def is_ltx_model(model_id: str) -> bool:
    return model_id in _LTX_VIDEO_MODEL_IDS


def is_seedance_model(model_id: str) -> bool:
    return model_id in _SEEDANCE_VIDEO_MODEL_IDS


def is_seedance25_model(model_id: str) -> bool:
    return model_id == "seedance-2-5"


def is_happyhorse_model(model_id: str) -> bool:
    return model_id in _HAPPYHORSE_VIDEO_MODEL_IDS


def is_wan3_model(model_id: str) -> bool:
    return model_id in _WAN3_VIDEO_MODEL_IDS


def is_wan3_enhanced_model(model_id: str) -> bool:
    """Check for the Wan 3.0 Enhanced model specifically."""

    return model_id == "wan3.0-spicy-video"


def is_minimax_h3_model(model_id: str) -> bool:
    return model_id in _MINIMAX_H3_VIDEO_MODEL_IDS


def is_minimax_h3_turbo_model(model_id: str) -> bool:
    """One of the 4-step MiniMax H3 Turbo workflows.

    FL2VA and FastH3 both cover t2v/i2v/flf2v; Ref2VA uses its dedicated r2v
    Turbo LoRA. FastH3 has no r2v mode.
    """

    return bool(_MINIMAX_H3_TURBO_PATTERN.match(model_id)) or (
        model_id == "minimax-h3-ref2va-fp8_r2v_turbo"
    )


def is_minimax_h3_balanced_model(model_id: str) -> bool:
    """One of the 8-step MiniMax H3 Balanced workflows.

    FL2VA covers t2v/i2v/flf2v; Ref2VA uses its matching Larry v4 adapter for r2v.
    """

    return bool(_MINIMAX_H3_BALANCED_PATTERN.match(model_id)) or (
        model_id == "minimax-h3-ref2va-fp8_r2v_balanced"
    )


def is_minimax_h3_reference_model(model_id: str) -> bool:
    return is_minimax_h3_model(model_id) and get_video_workflow_type(model_id) == "r2v"


def is_external_video_model(model_id: str) -> bool:
    return is_seedance_model(model_id) or is_happyhorse_model(model_id) or is_wan3_model(model_id)


def is_video_model(model_id: str) -> bool:
    return any(
        predicate(model_id)
        for predicate in (
            is_wan_model,
            is_ltx_model,
            is_seedance_model,
            is_happyhorse_model,
            is_wan3_model,
            is_minimax_h3_model,
        )
    )


def is_audio_model(model_id: str) -> bool:
    return model_id.startswith("ace_step") or model_id == "minimax_music3"


def calculate_video_frames(
    model_id: str,
    duration: float,
    fps: float,
    min_frames: int | None = None,
    max_frames: int | None = None,
) -> int:
    """Match the JS SDK's WAN and LTX frame-count behavior."""

    # Python uses bankers' rounding while JavaScript's Math.round chooses the
    # next integer for positive half values. Durations and frame rates are
    # non-negative, so this is the exact wire-compatible operation here.
    def js_round(value: float) -> int:
        return math.floor(value + 0.5)

    if is_wan_model(model_id):
        frames = js_round(duration * 16) + 1
    elif is_minimax_h3_model(model_id):
        requested_frames = js_round(duration * MINIMAX_H3_FPS)
        minimum = max(MINIMAX_H3_MIN_FRAMES, min_frames or MINIMAX_H3_MIN_FRAMES)
        maximum = min(MINIMAX_H3_MAX_FRAMES, max_frames or MINIMAX_H3_MAX_FRAMES)
        minimum_step = math.ceil((minimum - MINIMAX_H3_BASE_FRAMES) / MINIMAX_H3_FRAME_STEP)
        maximum_step = math.floor((maximum - MINIMAX_H3_BASE_FRAMES) / MINIMAX_H3_FRAME_STEP)
        if minimum_step > maximum_step:
            raise ValueError(
                f"No valid MiniMax H3 frame count exists between {minimum} and {maximum}"
            )
        requested_step = js_round(
            (requested_frames - MINIMAX_H3_BASE_FRAMES) / MINIMAX_H3_FRAME_STEP
        )
        steps = min(maximum_step, max(minimum_step, requested_step))
        return MINIMAX_H3_BASE_FRAMES + steps * MINIMAX_H3_FRAME_STEP
    else:
        frames = js_round(duration * fps) + 1
        if is_ltx_model(model_id):
            frames = js_round((frames - 1) / LTX2_FRAME_STEP) * LTX2_FRAME_STEP + 1
    if min_frames is not None:
        frames = max(min_frames, frames)
    if max_frames is not None:
        frames = min(max_frames, frames)
    return frames


def get_video_workflow_type(model_id: str) -> str | None:
    if is_wan3_model(model_id):
        return "t2v"
    if is_happyhorse_model(model_id):
        for kind in ("r2v", "i2v", "t2v"):
            if f"-{kind}" in model_id:
                return kind
        return None
    if is_minimax_h3_model(model_id):
        for kind in ("r2v", "flf2v", "i2v", "t2v"):
            if f"_{kind}" in model_id:
                return kind
        return None
    if not (is_wan_model(model_id) or is_ltx_model(model_id) or is_seedance_model(model_id)):
        return None
    # Keep this ordering aligned with the TypeScript helper. ``_ia2v`` also
    # contains ``_a2v`` conceptually, and several workflows are family-bound.
    if "_i2v" in model_id:
        return "i2v"
    if "_t2v" in model_id:
        return "t2v"
    if (is_ltx_model(model_id) or is_seedance_model(model_id)) and "_v2v" in model_id:
        return "v2v"
    if (is_ltx_model(model_id) or is_seedance_model(model_id)) and "_ia2v" in model_id:
        return "ia2v"
    if is_ltx_model(model_id) and "_a2v" in model_id:
        return "a2v"
    if is_wan_model(model_id):
        if "_s2v" in model_id:
            return "s2v"
        if "_animate-move" in model_id:
            return "animate-move"
        if "_animate-replace" in model_id:
            return "animate-replace"
    return None


def detect_content_type(value: Any) -> str | None:
    if isinstance(value, (str, Path)):
        return mimetypes.guess_type(str(value))[0]
    raw = bytes(value) if isinstance(value, (bytes, bytearray, memoryview)) else b""
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"\x89PNG"):
        return "image/png"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "audio/wav"
    if raw.startswith(b"ID3") or (len(raw) >= 2 and raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0):
        return "audio/mpeg"
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        brand = raw[8:12].lower()
        if b"m4a" in brand or b"m4b" in brand:
            return "audio/mp4"
        if b"qt" in brand:
            return "video/quicktime"
        return "video/mp4"
    return None


def read_media(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, (str, Path)):
        return Path(value).expanduser().read_bytes()
    read = getattr(value, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, str):
            return data.encode()
        return bytes(data)
    raise TypeError("Media must be bytes, a path, or a binary file object")


def parse_sse_chunk(chunk: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for block in re.split(r"\r?\n\r?\n", chunk):
        block = block.strip()
        if not block:
            continue
        frame: dict[str, Any] = {"event": "message", "data": None, "raw": block}
        data_lines: list[str] = []
        for line in re.split(r"\r?\n", block):
            if not line or line.startswith(":"):
                continue
            field, _, raw_value = line.partition(":")
            value = raw_value[1:] if raw_value.startswith(" ") else raw_value
            if field == "id":
                frame["id"] = value
            elif field == "event":
                frame["event"] = value or "message"
            elif field == "data":
                data_lines.append(value)
        if data_lines:
            data = "\n".join(data_lines)
            try:
                frame["data"] = json.loads(data)
            except json.JSONDecodeError:
                frame["data"] = data
        frames.append(frame)
    return frames


parse_creative_workflow_sse_chunk = parse_sse_chunk
parseCreativeWorkflowSseChunk = parse_sse_chunk
