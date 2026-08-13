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
    return model_id.startswith("wan_")


def is_ltx_model(model_id: str) -> bool:
    return model_id.startswith(("ltx2-", "ltx23-"))


def is_seedance_model(model_id: str) -> bool:
    return model_id.startswith("seedance-2-0")


def is_happyhorse_model(model_id: str) -> bool:
    return model_id.startswith("happyhorse-1.1")


def is_external_video_model(model_id: str) -> bool:
    return is_seedance_model(model_id) or is_happyhorse_model(model_id)


def is_video_model(model_id: str) -> bool:
    return model_id.startswith(("wan_", "ltx2-", "ltx23-", "seedance-2-0", "happyhorse-1.1"))


def is_audio_model(model_id: str) -> bool:
    return model_id.startswith("ace_step")


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
    else:
        frames = js_round(duration * fps) + 1
        if is_ltx_model(model_id):
            frames = js_round((frames - 1) / 8) * 8 + 1
    if min_frames is not None:
        frames = max(min_frames, frames)
    if max_frames is not None:
        frames = min(max_frames, frames)
    return frames


def get_video_workflow_type(model_id: str) -> str | None:
    if is_happyhorse_model(model_id):
        for kind in ("r2v", "i2v", "t2v"):
            if f"-{kind}" in model_id:
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
