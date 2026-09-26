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
    "wan_v2.2-14b-fp8_s2v",
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
    "minimax-h3-fastvideo-int8_t2v_turbo_2stage",
    "minimax-h3-fastvideo-int8_i2v_turbo_2stage",
    "minimax-h3-fastvideo-int8_flf2v_turbo_2stage",
    "minimax-h3-fastvideo-int8_ia2v_turbo",
    "minimax-h3-fastvideo-int8_flfa2v_turbo",
    "minimax-h3-fastvideo-int8_a2v_turbo",
    "minimax-h3-fastvideo-int8_ia2v_turbo_2stage",
    "minimax-h3-fastvideo-int8_flfa2v_turbo_2stage",
    "minimax-h3-fastvideo-int8_a2v_turbo_2stage",
    "minimax-h3-ref2va-fp8_r2v_turbo",
    "minimax-h3-fl2va-fp8_t2v_balanced",
    "minimax-h3-fl2va-fp8_i2v_balanced",
    "minimax-h3-fl2va-fp8_flf2v_balanced",
    "minimax-h3-ref2va-fp8_r2v_balanced",
    "minimax-h3-ref2va-fp8_r2v_2stage",
    "minimax-h3-ref2va-fp8_r2v_balanced_2stage",
}
_MINIMAX_H3_TURBO_PATTERN = re.compile(
    r"^minimax-h3-(?:fl2va-fp8_(?:t2v|i2v|flf2v)_turbo"
    r"|fastvideo-int8_(?:t2v|i2v|flf2v)_turbo(?:_2stage)?"
    r"|fastvideo-int8_(?:ia2v|flfa2v|a2v)_turbo(?:_2stage)?)$"
)
_MINIMAX_H3_BALANCED_PATTERN = re.compile(r"^minimax-h3-fl2va-fp8_(?:t2v|i2v|flf2v)_balanced$")
_MINIMAX_H3_AUDIO_GUIDE_PATTERN = re.compile(
    r"^minimax-h3-fastvideo-int8_(?:ia2v|flfa2v|a2v)_turbo(?:_2stage)?$"
)
# MiniMax H3 FastH3 audio guide base ids (each also has a ``_2stage`` id).
MINIMAX_H3_FASTH3_IA2V_MODEL_ID = "minimax-h3-fastvideo-int8_ia2v_turbo"
MINIMAX_H3_FASTH3_FLFA2V_MODEL_ID = "minimax-h3-fastvideo-int8_flfa2v_turbo"
MINIMAX_H3_FASTH3_A2V_MODEL_ID = "minimax-h3-fastvideo-int8_a2v_turbo"

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

    FL2VA and FastH3 both cover t2v/i2v/flf2v, and FastH3 also covers the
    ia2v/flfa2v/a2v audio guide; Ref2VA uses its dedicated r2v Turbo LoRA.
    FastH3 has no r2v mode. The FastH3 Two-Stage ids
    (``..._turbo_2stage``) share FastH3's 4-step sampling and request; they
    deliver the clip at twice the canvas width and height, and one ``_2stage``
    id per workflow serves every canvas class (672x384 for 720p, 960x544 for
    1080p, 1344x768 for 2K). Only a ``_2stage`` id renders two-stage: a base
    FastH3 id always runs one-stage at the canvas it sends. The short-lived
    ``..._turbo_2stage_720p`` spellings were retired by the socket on
    2026-09-14 and are not MiniMax H3 ids. The Ref2VA two-stage ids
    (``..._r2v_2stage``, ``..._r2v_balanced_2stage``) keep their Standard or
    Balanced tier and are never Turbo.
    """

    return bool(_MINIMAX_H3_TURBO_PATTERN.match(model_id)) or (
        model_id == "minimax-h3-ref2va-fp8_r2v_turbo"
    )


def is_minimax_h3_audio_guide_model(model_id: str) -> bool:
    """A MiniMax H3 FastH3 audio-guide workflow, standard or two-stage.

    ``ia2v`` takes ``referenceImage`` + ``referenceAudio``, ``flfa2v`` takes
    ``referenceImage`` + ``referenceImageEnd`` + ``referenceAudio``, and ``a2v``
    takes ``referenceAudio`` only. The uploaded audio drives the video from frame
    0 and is trimmed to the video length (``frames / 24`` seconds from the
    optional ``audioStart``); the output always carries it, so
    ``generateAudio=False`` and ``audioDuration`` are refused, as are LoRAs.
    """

    return bool(_MINIMAX_H3_AUDIO_GUIDE_PATTERN.match(model_id))


def is_minimax_h3_balanced_model(model_id: str) -> bool:
    """One of the 8-step MiniMax H3 Balanced workflows.

    FL2VA covers t2v/i2v/flf2v; Ref2VA uses its matching Larry v4 adapter for
    r2v, on its one-stage and two-stage (``..._r2v_balanced_2stage``) ids alike.
    """

    return bool(_MINIMAX_H3_BALANCED_PATTERN.match(model_id)) or (
        model_id == "minimax-h3-ref2va-fp8_r2v_balanced"
        or model_id == "minimax-h3-ref2va-fp8_r2v_balanced_2stage"
    )


def is_minimax_h3_reference_model(model_id: str) -> bool:
    return is_minimax_h3_model(model_id) and get_video_workflow_type(model_id) == "r2v"


def is_external_video_model(model_id: str) -> bool:
    return is_seedance_model(model_id) or is_happyhorse_model(model_id) or is_wan3_model(model_id)


FLASHVSR_VIDEO_UPSCALE_MODEL_ID = "flashvsr_v1.1_tiny_long_bf16"


def is_video_upscale_model(model_id: str) -> bool:
    return model_id == FLASHVSR_VIDEO_UPSCALE_MODEL_ID


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
            is_video_upscale_model,
        )
    )


def is_audio_model(model_id: str) -> bool:
    return (
        model_id.startswith("ace_step")
        or model_id.startswith("qwen3_tts_")
        or model_id == "minimax_music3"
    )


#: Canonical id of the prompt-free single-image (front view only) image-to-3D workflow.
PIXAL3D_IMAGE_TO_3D_MODEL_ID = "pixal3d_int8_i23d"

#: Canonical id of the multi-view image-to-3D workflow: a required front view
#: (``startingImage``) plus any of the optional ``leftViewImage``,
#: ``backViewImage`` and ``rightViewImage`` orbit views.
PIXAL3D_MULTIVIEW_IMAGE_TO_3D_MODEL_ID = "pixal3d_multiview_int8_i23d"

#: Pixal3D multi-view orbit views and the ``contextImage<slot>`` upload each one
#: travels in. The slots are the worker's asset keys, so the order is fixed.
#: Views are named from the subject's own point of view: ``leftViewImage`` is
#: the subject turned so its own left side faces the camera (it faces
#: screen-left), ``rightViewImage`` its own right side (it faces screen-right),
#: ``backViewImage`` the subject seen from behind.
PIXAL3D_ORBIT_VIEW_SLOTS: dict[str, int] = {
    "leftViewImage": 1,
    "backViewImage": 2,
    "rightViewImage": 3,
}

#: Canonical id of the SAM 3 interactive image-segmentation workflow.
SAM3_IMAGE_SEGMENT_MODEL_ID = "sam3_image_segment_bf16"

#: Canonical id of the standalone BiRefNet background-removal workflow.
BIREFNET_BACKGROUND_REMOVAL_MODEL_ID = "birefnet_image_background_removal_fp16"


def is_model_artifact_model(model_id: str) -> bool:
    """Check if a model returns a downloadable 3D model artifact."""

    return model_id.startswith("pixal3d_")


#: What a finished job's result is, which decides the download endpoint: an
#: ``image`` comes from ``/v1/image/downloadUrl``, everything else from
#: ``/v1/media/downloadUrl``.
RESULT_MEDIA_KINDS = frozenset({"image", "video", "audio", "model"})


def as_result_media_kind(value: Any) -> str | None:
    """Narrow a declared kind (a catalog ``media`` value, a project ``type``).

    Anything else, including a missing value, is no evidence and returns
    ``None``; it never reads as ``image``.
    """

    return value if isinstance(value, str) and value in RESULT_MEDIA_KINDS else None


def _result_media_kind_from_content_type(content_type: str) -> str | None:
    top, slash, _ = content_type.split(";")[0].strip().lower().partition("/")
    return top if slash and top in RESULT_MEDIA_KINDS else None


_OUTPUT_FORMAT_EVIDENCE: dict[str, dict[str, str]] = {
    "mp4": {"kind": "video"},
    "mov": {"kind": "video"},
    "mp3": {"kind": "audio", "contentType": "audio/mpeg"},
    "wav": {"kind": "audio", "contentType": "audio/wav"},
    "flac": {"kind": "audio", "contentType": "audio/flac"},
    "glb": {"kind": "model", "contentType": "model/gltf-binary"},
    "png": {"kind": "image"},
    "jpg": {"kind": "image"},
    "jpeg": {"kind": "image"},
    "webp": {"kind": "image"},
}

# A media artifact beside a still is the result; the still is incidental.
_ARTIFACT_KIND_PRECEDENCE = ("model", "video", "audio", "image")


def result_media_evidence(data: Any) -> dict[str, str] | None:
    """What a ``jobResult`` frame says the job produced, or ``None`` when it says nothing.

    ComfyUI workers list each uploaded artifact with its content type, and
    partner-model results name an output format. A frame with neither (a Mac
    worker's result, for one) is no evidence, and must not be read as an image.
    Returns ``{"kind": ..., "contentType": ...}``; ``contentType`` only when known.
    """

    if not isinstance(data, dict):
        return None
    by_kind: dict[str, str] = {}
    artifacts = data.get("artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            content_type = artifact.get("contentType")
            if artifact.get("success") is False or not isinstance(content_type, str):
                continue
            kind = _result_media_kind_from_content_type(content_type)
            if kind and kind not in by_kind:
                by_kind[kind] = content_type.strip()
    for kind in _ARTIFACT_KIND_PRECEDENCE:
        if kind in by_kind:
            return {"kind": kind, "contentType": by_kind[kind]}
    output_format = data.get("outputFormat")
    if isinstance(output_format, str):
        evidence = _OUTPUT_FORMAT_EVIDENCE.get(output_format.strip().lower())
        if evidence:
            return dict(evidence)
    return None


def is_segmentation_model(model_id: str) -> bool:
    """Check if a model performs image segmentation rather than generation.

    Segmentation returns a lossless mask PNG the same size as the source, not a
    new image, so callers must not treat it as a generated result: it has no
    meaningful prompt-to-pixels relationship and is not enhanceable.

    BiRefNet counts. It reaches the same artifact with no prompt at all, and its
    cutout branch is that mask carried as an alpha channel, so every consumer
    that hides a mask from a gallery, refuses to enhance one, or requires a
    source image has to treat it exactly as it treats SAM 3.
    """

    return model_id in (SAM3_IMAGE_SEGMENT_MODEL_ID, BIREFNET_BACKGROUND_REMOVAL_MODEL_ID)


def is_pixal3d_model(model_id: str) -> bool:
    """Check if a model ID is one of the Pixal3D image-to-3D workflows."""

    return model_id in (PIXAL3D_IMAGE_TO_3D_MODEL_ID, PIXAL3D_MULTIVIEW_IMAGE_TO_3D_MODEL_ID)


def is_pixal3d_multiview_model(model_id: str) -> bool:
    """Check if a model ID is the Pixal3D workflow that accepts orbit views."""

    return model_id == PIXAL3D_MULTIVIEW_IMAGE_TO_3D_MODEL_ID


def get_pixal3d_orbit_view_slots(params: dict[str, Any]) -> list[tuple[str, int, Any]]:
    """The orbit views supplied on a request as ``(view, slot, media)``.

    Accepts camelCase wire keys. Unset (``None``) or empty views are omitted, so
    any subset keeps its own ``contextImage<slot>`` rather than being renumbered.
    """

    return [
        (view, slot, params[view])
        for view, slot in PIXAL3D_ORBIT_VIEW_SLOTS.items()
        if params.get(view)
    ]


def requires_starting_image(model_id: str) -> bool:
    """Models that need a starting image because they transform one rather than
    generating from a prompt alone.
    """

    return is_segmentation_model(model_id) or is_model_artifact_model(model_id)


isVideoModel = is_video_model
isVideoUpscaleModel = is_video_upscale_model
isAudioModel = is_audio_model
isModelArtifactModel = is_model_artifact_model
isSegmentationModel = is_segmentation_model
requiresStartingImage = requires_starting_image
isPixal3dModel = is_pixal3d_model
isPixal3dMultiViewModel = is_pixal3d_multiview_model
getPixal3dOrbitViewSlots = get_pixal3d_orbit_view_slots


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

    if is_video_upscale_model(model_id):
        frames = js_round(duration * fps)
    elif is_wan_model(model_id):
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


def get_minimax_h3_frames_for_audio_duration(audio_duration_seconds: float) -> int:
    """Smallest valid MiniMax H3 frame count that covers an audio clip.

    Returns the first ``124 + n*17`` value at or above ``seconds * 24``, clamped
    to 124-362, to size a FastH3 audio-guide request to its uploaded audio.
    Mirrors the TypeScript ``getMinimaxH3FramesForAudioDuration``.
    """

    if (
        isinstance(audio_duration_seconds, bool)
        or not isinstance(audio_duration_seconds, (int, float))
        or not math.isfinite(audio_duration_seconds)
        or audio_duration_seconds <= 0
    ):
        raise ValueError("Audio duration must be a finite number of seconds greater than 0.")
    # The epsilon keeps exact grid durations (e.g. 141/24 s) from rounding up a step.
    needed_frames = math.ceil(audio_duration_seconds * MINIMAX_H3_FPS - 1e-6)
    steps = max(0, math.ceil((needed_frames - MINIMAX_H3_BASE_FRAMES) / MINIMAX_H3_FRAME_STEP))
    return min(MINIMAX_H3_MAX_FRAMES, MINIMAX_H3_BASE_FRAMES + steps * MINIMAX_H3_FRAME_STEP)


getMinimaxH3FramesForAudioDuration = get_minimax_h3_frames_for_audio_duration
isMinimaxH3AudioGuideModel = is_minimax_h3_audio_guide_model


def get_video_workflow_type(model_id: str) -> str | None:
    if is_video_upscale_model(model_id):
        return "upscale"
    if is_wan3_model(model_id):
        return "t2v"
    if is_happyhorse_model(model_id):
        for kind in ("r2v", "i2v", "t2v"):
            if f"-{kind}" in model_id:
                return kind
        return None
    if is_minimax_h3_model(model_id):
        # ``_flfa2v`` contains neither ``_flf2v`` nor ``_ia2v``, and ``_ia2v``
        # does not contain ``_a2v``, so each audio-guide suffix is its own test.
        for kind in ("r2v", "flfa2v", "ia2v", "a2v", "flf2v", "i2v", "t2v"):
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
