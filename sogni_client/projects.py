"""Generation projects and jobs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

from .attribution import workload_attribution_to_wire_fields
from .errors import ApiError, ProjectError
from .events import DataEntity, EventEmitter
from .transport import ApiClient
from .utils import (
    MINIMAX_H3_BASE_FRAMES,
    MINIMAX_H3_DIMENSION_STEP,
    MINIMAX_H3_FRAME_STEP,
    MINIMAX_H3_MAX_DIMENSION,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MAX_FRAMES,
    MINIMAX_H3_MAX_PIXELS,
    MINIMAX_H3_MIN_DURATION,
    MINIMAX_H3_MIN_FRAMES,
    calculate_video_frames,
    detect_content_type,
    get_video_workflow_type,
    is_audio_model,
    is_external_video_model,
    is_happyhorse_model,
    is_ltx_model,
    is_minimax_h3_model,
    is_minimax_h3_reference_model,
    is_minimax_h3_turbo_model,
    is_seedance25_model,
    is_seedance_model,
    is_video_model,
    is_wan3_model,
    new_id,
    normalize_params,
    read_media,
)

_LOGGER = logging.getLogger("sogni_client")

VIDEO_WORKFLOW_ASSETS: dict[str, dict[str, str]] = {
    "t2v": {
        "referenceImage": "forbidden",
        "referenceImageEnd": "forbidden",
        "referenceAudio": "forbidden",
        "referenceAudioIdentity": "optional",
        "referenceVideo": "forbidden",
        "referenceMask": "forbidden",
    },
    "i2v": {
        "referenceImage": "optional",
        "referenceImageEnd": "optional",
        "referenceAudio": "forbidden",
        "referenceAudioIdentity": "optional",
        "referenceVideo": "forbidden",
        "referenceMask": "forbidden",
    },
    "flf2v": {
        "referenceImage": "required",
        "referenceImageEnd": "required",
        "referenceAudio": "forbidden",
        "referenceAudioIdentity": "forbidden",
        "referenceVideo": "forbidden",
        "referenceMask": "forbidden",
    },
    "s2v": {
        "referenceImage": "required",
        "referenceImageEnd": "forbidden",
        "referenceAudio": "required",
        "referenceAudioIdentity": "forbidden",
        "referenceVideo": "forbidden",
        "referenceMask": "forbidden",
    },
    "ia2v": {
        "referenceImage": "required",
        "referenceImageEnd": "forbidden",
        "referenceAudio": "required",
        "referenceAudioIdentity": "forbidden",
        "referenceVideo": "forbidden",
        "referenceMask": "forbidden",
    },
    "a2v": {
        "referenceImage": "forbidden",
        "referenceImageEnd": "forbidden",
        "referenceAudio": "required",
        "referenceAudioIdentity": "forbidden",
        "referenceVideo": "forbidden",
        "referenceMask": "forbidden",
    },
    "animate-move": {
        "referenceImage": "required",
        "referenceImageEnd": "forbidden",
        "referenceAudio": "forbidden",
        "referenceAudioIdentity": "forbidden",
        "referenceVideo": "required",
        "referenceMask": "forbidden",
    },
    "animate-replace": {
        "referenceImage": "required",
        "referenceImageEnd": "forbidden",
        "referenceAudio": "forbidden",
        "referenceAudioIdentity": "forbidden",
        "referenceVideo": "required",
        "referenceMask": "forbidden",
    },
    "v2v": {
        "referenceImage": "optional",
        "referenceImageEnd": "forbidden",
        "referenceAudio": "forbidden",
        "referenceAudioIdentity": "optional",
        "referenceVideo": "required",
        "referenceMask": "optional",
    },
    "r2v": {
        "referenceImage": "optional",
        "referenceImageEnd": "forbidden",
        "referenceAudio": "forbidden",
        "referenceAudioIdentity": "forbidden",
        "referenceVideo": "forbidden",
        "referenceMask": "forbidden",
    },
}

_MINIMAX_H3_R2V_ASSETS = {
    "referenceImage": "optional",
    "referenceImageEnd": "forbidden",
    "referenceAudio": "optional",
    "referenceAudioIdentity": "forbidden",
    "referenceVideo": "optional",
    "referenceMask": "forbidden",
}
_MINIMAX_H3_I2V_ASSETS = {
    "referenceImage": "optional",
    "referenceImageEnd": "optional",
    "referenceAudio": "forbidden",
    "referenceAudioIdentity": "forbidden",
    "referenceVideo": "forbidden",
    "referenceMask": "forbidden",
}
_MINIMAX_H3_MAX_REFERENCE_IMAGES = 9
_MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3
_MINIMAX_H3_MAX_REFERENCE_AUDIOS = 3
_MINIMAX_H3_MAX_REFERENCE_FILES = 12
_SEEDANCE_REFERENCE_LIMITS = {
    "seedance-2-0": (9, 3, 3, 12),
    "seedance-2-0-mini": (9, 3, 3, 12),
    "seedance-2-0-fast": (9, 3, 3, 12),
    "seedance-2-5": (30, 10, 10, 50),
}

_SAMPLER_ALIASES = {
    "Euler": "euler",
    "Euler a": "euler_a",
    "Euler Ancestral": "euler_ancestral",
    "Heun": "heun",
    "DPM++ 2M": "dpmpp_2m",
    "DPM++ 2M SDE": "dpmpp_2m_sde",
    "DPM++ SDE": "dpmpp_sde",
    "DPM++ 3M SDE": "dpmpp_3m_sde",
    "UniPC": "uni_pc",
    "LCM (Latent Consistency Model)": "lcm",
}
_SCHEDULER_ALIASES = {
    "Simple": "simple",
    "Normal": "normal",
    "Karras": "karras",
    "Exponential": "exponential",
    "SGM Uniform": "sgm_uniform",
    "DDIM Uniform": "ddim_uniform",
    "Beta": "beta",
    "Linear Quadratic": "linear_quadratic",
    "KL Optimal": "kl_optimal",
    "DDIM": "ddim",
    "Leading": "leading",
    "Linear": "linear",
}

_EXTENDED_IMAGE_SIZE_MODEL_IDS = {
    "z_image_bf16",
    "z_image_turbo_bf16",
    "krea2_turbo_fp8_scaled",
    "qwen_image_edit_2511_fp8",
    "qwen_image_edit_2511_fp8_lightning",
    "qwen_image_2512_fp8",
    "qwen_image_2512_fp8_lightning",
}
_KREA_IDENTITY_EDIT_MODEL_IDS = {
    "krea2_identity_edit_v1_2",
    "dark_beast_krea2_identity_edit_v1_2",
}
_PROJECT_TIMEOUT_SECONDS = 2 * 60
_MAX_FAILED_SYNC_ATTEMPTS = 3
_CANCELLATION_CONFIRMATION_TIMEOUT_SECONDS = 120
_RUNTIME_LIMIT_FLOOR_SECONDS = {
    "fast": {"image": 30 * 60, "audio": 30 * 60, "video": 90 * 60},
    "relaxed": {"image": 2 * 60 * 60, "audio": 2 * 60 * 60, "video": 8 * 60 * 60},
}
_RUNTIME_LIMIT_ETA_MULTIPLIER = 6
_RUNTIME_LIMIT_MAX_SECONDS = 12 * 60 * 60
_DEFAULT_LORA_CONSTRAINTS = {
    "maxPerRequest": 8,
    "minStrength": -100,
    "maxStrength": 100,
}
_PROJECT_STATUS_MAP = {
    "pending": "pending",
    "active": "queued",
    "assigned": "processing",
    "progress": "processing",
    "completed": "completed",
    "errored": "failed",
    "cancelled": "canceled",
}
_JOB_STATUS_MAP = {
    "created": "pending",
    "queued": "pending",
    "assigned": "initiating",
    "initiatingModel": "initiating",
    "jobStarted": "processing",
    "jobProgress": "processing",
    "jobCompleted": "completed",
    "jobError": "failed",
}
_ENHANCEMENT_DEFAULTS: dict[str, Any] = {
    "network": "fast",
    "modelId": "flux1-schnell-fp8",
    "positivePrompt": "",
    "negativePrompt": "",
    "stylePrompt": "",
    "startingImageStrength": 0.5,
    "steps": 5,
    "guidance": 1,
    "numberOfMedia": 1,
    "numberOfPreviews": 0,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _format_runtime_limit(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


def _api_error(message: str) -> ApiError:
    return ApiError(400, {"status": "error", "message": message, "errorCode": 0})


def _validate_option(value: str | None, options: dict[str, Any], key: str) -> str | None:
    allowed = options.get(key, {}).get("allowed", [])
    if not value or not allowed:
        return None
    if value not in allowed:
        raise _api_error(f'Invalid {key} {value}. Must be one of "' + '", "'.join(allowed) + '".')
    return value


def _validate_number(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    property_name: str = "Value",
) -> int | float:
    """Match the JS SDK's finite numeric coercion and range checks."""

    if isinstance(value, bool):
        number = int(value)
    else:
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError) as error:
            raise ValueError(f"{property_name} must be a number, got {value}") from error
    if not math.isfinite(number):
        raise ValueError(f"{property_name} must be a number, got {value}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{property_name} must greater or equal {minimum:g}, got {number:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{property_name} must be less or equal {maximum:g}, got {number:g}")
    return int(number) if number.is_integer() else number


def _custom_image_size_bounds(model_id: str) -> tuple[int, int]:
    if model_id == "rtx_vsr_pro":
        return 512, 15360
    if model_id in _KREA_IDENTITY_EDIT_MODEL_IDS:
        return 512, 2048
    if model_id == "gpt-image-2":
        return 256, 3840
    if model_id in _EXTENDED_IMAGE_SIZE_MODEL_IDS:
        return 256, 2560
    return 256, 2048


def _enhancement_strength(value: str) -> float:
    return {"light": 0.15, "heavy": 0.49}.get(value, 0.35)


def _raw_result_url(data: dict[str, Any]) -> str | None:
    for key in ("resultUrl", "imageUrl", "imageFile", "videoUrl", "videoFile"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _nested_params(value: Any) -> Any:
    """Normalize Python names in nested public parameter records."""

    if isinstance(value, dict):
        normalized = normalize_params(value)
        return {key: _nested_params(item) for key, item in normalized.items()}
    if isinstance(value, list):
        return [_nested_params(item) for item in value]
    return value


def _template() -> dict[str, Any]:
    return {
        "selectedUpscalingModel": "OFF",
        "cnVideoFramesSketch": [],
        "cnVideoFramesSegmentedSubject": [],
        "cnVideoFramesFace": [],
        "doCanvasBlending": False,
        "animationIsOn": False,
        "cnVideoFramesBoth": [],
        "cnVideoFramesDepth": [],
        "keyFrames": [
            {
                "stepsIsEnabled": True,
                "siRotation": 0,
                "siDragOffsetIsEnabled": True,
                "strength": 0.5,
                "siZoomScaleIsEnabled": True,
                "isEnabled": True,
                "processing": "CPU, GPU",
                "useLastImageAsGuideImageInAnimation": True,
                "guidanceScaleIsEnabled": True,
                "siImageBackgroundColor": "black",
                "cnDragOffset": [0, 0],
                "scheduler": None,
                "timeStepSpacing": None,
                "steps": 20,
                "cnRotation": 0,
                "guidanceScale": 7.5,
                "siZoomScale": 1,
                "modelID": "",
                "cnRotationIsEnabled": True,
                "negativePrompt": "",
                "startingImageZoomPanIsOn": False,
                "seed": None,
                "siRotationIsEnabled": True,
                "cnImageBackgroundColor": "clear",
                "strengthIsEnabled": True,
                "siDragOffset": [0, 0],
                "useLastImageAsCNImageInAnimation": False,
                "positivePrompt": "",
                "controlNetZoomPanIsOn": False,
                "cnZoomScaleIsEnabled": True,
                "currentControlNets": None,
                "stylePrompt": "",
                "cnDragOffsetIsEnabled": True,
                "frameIndex": 0,
                "startingImage": None,
                "cnZoomScale": 1,
            }
        ],
        "previews": 5,
        "frameRate": 24,
        "generatedVideoSeconds": 10,
        "canvasIsOn": False,
        "cnVideoFrames": [],
        "disableSafety": False,
        "cnVideoFramesSegmentedBackground": [],
        "cnVideoFramesSegmented": [],
        "numberOfImages": 1,
        "cnVideoFramesPose": [],
        "jobID": "",
        "siVideoFrames": [],
    }


def _validate_reference_array(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise _api_error(f"{name} must contain only non-empty URL strings.")
    return [item for item in value if item.strip()]


def _video_asset_requirements(model_id: str) -> dict[str, str] | None:
    workflow = get_video_workflow_type(model_id)
    if not workflow:
        return None
    if workflow == "r2v" and is_minimax_h3_model(model_id):
        return _MINIMAX_H3_R2V_ASSETS
    if workflow == "i2v" and is_minimax_h3_model(model_id):
        return _MINIMAX_H3_I2V_ASSETS
    return VIDEO_WORKFLOW_ASSETS[workflow]


def _media_slots(single: Any, multiple: Any) -> list[tuple[int, Any]]:
    values = [single, *(multiple or [])]
    return [(index, value) for index, value in enumerate(values, 1) if value]


def _video_context_slots(params: dict[str, Any]) -> list[tuple[int, Any]]:
    contexts = params.get("contextImages") or []
    offset = 1 if params.get("referenceImage") else 0
    return [(offset + index, value) for index, value in enumerate(contexts, 1)]


def _validate_h3_params(params: dict[str, Any]) -> None:
    if not is_minimax_h3_model(params["modelId"]):
        return
    if params.get("fps") is not None and params["fps"] != 24:
        raise _api_error("MiniMax H3 fps is fixed at 24. Omit fps or set it to 24.")
    expected_steps = 4 if is_minimax_h3_turbo_model(params["modelId"]) else 20
    if params.get("steps") is not None and params["steps"] != expected_steps:
        suffix = " Turbo" if expected_steps == 4 else ""
        raise _api_error(f"MiniMax H3{suffix} steps are fixed at {expected_steps}.")
    if params.get("guidance") is not None and params["guidance"] != 1:
        raise _api_error("MiniMax H3 guidance is fixed at 1.")
    if str(params.get("negativePrompt") or "").strip():
        raise _api_error(
            "MiniMax H3 has no negative-prompt input. Put requested exclusions in positivePrompt."
        )
    if params.get("frames") is not None:
        frames = params["frames"]
        if (
            isinstance(frames, bool)
            or not isinstance(frames, int)
            or not MINIMAX_H3_MIN_FRAMES <= frames <= MINIMAX_H3_MAX_FRAMES
            or (frames - MINIMAX_H3_BASE_FRAMES) % MINIMAX_H3_FRAME_STEP
        ):
            raise _api_error("MiniMax H3 frames must be 124 + n*17 in the inclusive range 124-362.")
    if (params.get("width") is None) != (params.get("height") is None):
        raise _api_error("MiniMax H3 width and height must be provided together.")
    if params.get("width") is not None:
        width, height = params["width"], params["height"]
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width < MINIMAX_H3_DIMENSION_STEP
            or height < MINIMAX_H3_DIMENSION_STEP
            or width > MINIMAX_H3_MAX_DIMENSION
            or height > MINIMAX_H3_MAX_DIMENSION
            or width % MINIMAX_H3_DIMENSION_STEP
            or height % MINIMAX_H3_DIMENSION_STEP
            or width * height > MINIMAX_H3_MAX_PIXELS
        ):
            raise _api_error(
                "MiniMax H3 dimensions must use a 32px grid, stay at or below 1344px per axis, and fit within 1,032,192 pixels."
            )


def _validate_h3_references(params: dict[str, Any]) -> None:
    for field in ("referenceImageUrls", "referenceVideoUrls", "referenceAudioUrls"):
        if params.get(field) is not None:
            raise _api_error(
                f"MiniMax H3 r2v does not accept {field}; pass files through the Sogni asset upload fields instead."
            )
    images = bool(params.get("referenceImage")) + len(params.get("contextImages") or [])
    videos = len(_media_slots(params.get("referenceVideo"), params.get("referenceVideos")))
    audios = len(_media_slots(params.get("referenceAudio"), params.get("referenceAudios")))
    durations = params.get("referenceVideoDurations")
    if videos and durations is None:
        raise _api_error(
            f"MiniMax H3 r2v referenceVideoDurations must contain one entry for each uploaded reference video (expected {videos})."
        )
    if durations is not None:
        if not isinstance(durations, list) or len(durations) != videos:
            raise _api_error(
                f"MiniMax H3 r2v referenceVideoDurations must contain one entry for each uploaded reference video (expected {videos})."
            )
        total = 0.0
        for index, duration in enumerate(durations):
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < 1.95
                or duration > 15.05
            ):
                raise _api_error(
                    f"MiniMax H3 r2v referenceVideoDurations[{index}] must be between 2 and 15 seconds."
                )
            total += duration
        if total > 15.05:
            raise _api_error(
                f"MiniMax H3 r2v reference videos may total at most 15 seconds (got {total:g})."
            )
    for count, maximum, label in (
        (images, _MINIMAX_H3_MAX_REFERENCE_IMAGES, "reference images"),
        (videos, _MINIMAX_H3_MAX_REFERENCE_VIDEOS, "reference videos"),
        (audios, _MINIMAX_H3_MAX_REFERENCE_AUDIOS, "reference audios"),
    ):
        if count > maximum:
            raise _api_error(
                f"MiniMax H3 r2v supports at most {maximum} uploaded {label} (got {count})."
            )
    if images + videos + audios > _MINIMAX_H3_MAX_REFERENCE_FILES:
        raise _api_error(
            f"MiniMax H3 r2v supports at most 12 reference files in total (got {images + videos + audios}: {images} image, {videos} video, {audios} audio)."
        )
    if images + videos < 1:
        raise _api_error(
            "MiniMax H3 r2v needs at least one uploaded visual reference. Attach an image through referenceImage/contextImages or a video through referenceVideo/referenceVideos."
        )


def _validate_seedance_task(params: dict[str, Any]) -> None:
    task = params.get("seedanceTaskType")
    is_25 = is_seedance25_model(params["modelId"])
    if task is not None and task not in {"reference", "edit", "extend"}:
        raise _api_error("seedanceTaskType must be reference, edit, or extend.")
    if task is not None and not is_25:
        raise _api_error("seedanceTaskType is supported only by Seedance 2.5.")
    if not is_25:
        return
    image_urls = _validate_reference_array(params.get("referenceImageUrls"), "referenceImageUrls")
    video_urls = _validate_reference_array(params.get("referenceVideoUrls"), "referenceVideoUrls")
    audio_urls = _validate_reference_array(params.get("referenceAudioUrls"), "referenceAudioUrls")
    has_frames = bool(params.get("referenceImage") or params.get("referenceImageEnd"))
    has_video = bool(params.get("referenceVideo") or video_urls)
    has_loose = bool(image_urls or has_video or params.get("referenceAudio") or audio_urls)
    if task is None and has_loose:
        raise _api_error("Seedance 2.5 loose-reference requests require seedanceTaskType.")
    if task is not None and has_frames:
        raise _api_error(
            "seedanceTaskType is for Seedance 2.5 loose-reference, edit, or extend requests; omit it for first/last-frame generation."
        )
    if task in {"edit", "extend"} and not has_video:
        raise _api_error(f"Seedance 2.5 {task} requires at least one reference video.")
    if task == "reference" and not has_loose:
        raise _api_error(
            "Seedance 2.5 reference requires at least one loose image, video, or audio reference."
        )


def _validate_wan3_references(params: dict[str, Any]) -> None:
    images = _validate_reference_array(params.get("referenceImageUrls"), "referenceImageUrls")
    videos = _validate_reference_array(params.get("referenceVideoUrls"), "referenceVideoUrls")
    audios = _validate_reference_array(params.get("referenceAudioUrls"), "referenceAudioUrls")
    for field in ("referenceFileUrl", "referenceLinkUrl"):
        value = params.get(field)
        if value is not None and (
            not isinstance(value, str) or not value.strip().startswith("https://")
        ):
            raise _api_error(f"{field} must be a valid public HTTPS URL.")
    if params.get("referenceFileUrl") and params.get("referenceLinkUrl"):
        raise _api_error("Wan 3 accepts either one reference file or one reference link, not both.")
    for field in ("promptExtend", "watermark", "smartDuration"):
        if params.get(field) is not None and not isinstance(params[field], bool):
            raise _api_error(f"Wan 3 {field} must be a boolean.")
    if params.get("smartDuration") and params.get("duration") is not None:
        raise _api_error("Wan 3 smartDuration and duration are mutually exclusive.")
    if params.get("fps") is not None and params["fps"] != 30:
        raise _api_error("Wan 3 output is fixed at 30 fps.")
    if params.get("ratio") is not None and params["ratio"] not in {
        "adaptive",
        "16:9",
        "4:3",
        "1:1",
        "3:4",
        "9:16",
    }:
        raise _api_error("Wan 3 ratio must be adaptive, 16:9, 4:3, 1:1, 3:4, or 9:16.")
    if params.get("referenceAudioIdentity") or params.get("referenceMask"):
        raise _api_error("Wan 3 does not support audio-identity or mask inputs.")
    if params.get("seed") is not None and (
        isinstance(params["seed"], bool)
        or not isinstance(params["seed"], int)
        or not 0 <= params["seed"] <= 2_147_483_647
    ):
        raise _api_error("Wan 3 seed must be an integer from 0 through 2147483647.")
    video_count = bool(params.get("referenceVideo")) + len(videos)
    audio_count = bool(params.get("referenceAudio")) + len(audios)
    has_frames = bool(params.get("referenceImage") or params.get("referenceImageEnd"))
    has_document = bool(params.get("referenceFileUrl") or params.get("referenceLinkUrl"))
    has_loose = bool(images or video_count or audio_count or has_document)
    if params.get("referenceImageEnd") and not params.get("referenceImage"):
        raise _api_error("Wan 3 last-frame generation requires a first-frame referenceImage.")
    if has_frames and has_loose:
        raise _api_error(
            "Wan 3 first/last-frame anchors cannot be combined with loose media, file, or link references."
        )
    if len(images) > 10:
        raise _api_error("Wan 3 supports at most 10 reference images.")
    if video_count > 5:
        raise _api_error("Wan 3 supports at most 5 reference videos.")
    if audio_count > 5:
        raise _api_error("Wan 3 supports at most 5 reference audio clips.")
    if not str(params.get("positivePrompt") or "").strip() and not has_frames and not has_loose:
        raise _api_error("Wan 3 requires a prompt or at least one media, file, or link input.")


def _validate_video_assets(params: dict[str, Any]) -> None:
    model_id = params["modelId"]
    for field in ("contextImages", "referenceVideos", "referenceAudios"):
        value = params.get(field)
        if value is not None and (not isinstance(value, list) or any(not item for item in value)):
            raise _api_error(f"{field} must be an array without empty entries.")
        if value is not None and not is_minimax_h3_reference_model(model_id):
            raise _api_error(f"{field} is supported only by MiniMax H3 r2v models.")
    if params.get("referenceVideoDurations") is not None and not is_minimax_h3_reference_model(
        model_id
    ):
        raise _api_error("referenceVideoDurations is supported only by MiniMax H3 r2v models.")
    images = _validate_reference_array(params.get("referenceImageUrls"), "referenceImageUrls")
    videos = _validate_reference_array(params.get("referenceVideoUrls"), "referenceVideoUrls")
    audios = _validate_reference_array(params.get("referenceAudioUrls"), "referenceAudioUrls")
    if is_happyhorse_model(model_id):
        if params.get("referenceVideo") or videos:
            raise _api_error("HappyHorse models do not support reference video assets.")
        if params.get("referenceAudio") or params.get("referenceAudioIdentity") or audios:
            raise _api_error("HappyHorse models do not support reference audio assets.")
        if params.get("referenceImageEnd"):
            raise _api_error(
                "HappyHorse models do not support a separate end-frame image (referenceImageEnd)."
            )
        workflow = get_video_workflow_type(model_id)
        count = bool(params.get("referenceImage")) + len(images)
        if workflow == "i2v" and count != 1:
            raise _api_error("HappyHorse i2v requires exactly one first-frame reference image.")
        if workflow == "r2v" and not 1 <= count <= 9:
            raise _api_error("HappyHorse r2v requires between 1 and 9 reference images.")
        if workflow == "t2v" and count:
            raise _api_error("HappyHorse t2v does not support reference images.")
        return
    if is_wan3_model(model_id):
        _validate_wan3_references(params)
        return
    if is_seedance_model(model_id):
        _validate_seedance_task(params)
        image_count = (
            bool(params.get("referenceImage")) + bool(params.get("referenceImageEnd")) + len(images)
        )
        video_count = bool(params.get("referenceVideo")) + len(videos)
        audio_count = bool(
            params.get("referenceAudio") or params.get("referenceAudioIdentity")
        ) + len(audios)
        try:
            image_max, video_max, audio_max, total_max = _SEEDANCE_REFERENCE_LIMITS[model_id]
        except KeyError as error:
            raise _api_error(
                f'Unknown Seedance model "{model_id}"; no reference-asset limits are defined for it.'
            ) from error
        if image_count > image_max:
            raise _api_error(f"{model_id} supports at most {image_max} image assets.")
        if video_count > video_max:
            raise _api_error(f"{model_id} supports at most {video_max} video assets.")
        if audio_count > audio_max:
            raise _api_error(f"{model_id} supports at most {audio_max} audio assets.")
        if image_count + video_count + audio_count > total_max:
            raise _api_error(f"{model_id} supports at most {total_max} total asset files.")
        if (
            not is_seedance25_model(model_id)
            and audio_count
            and not image_count
            and not video_count
        ):
            raise _api_error(
                "Seedance audio references require at least one image or video reference."
            )
        return
    if is_minimax_h3_reference_model(model_id):
        _validate_h3_references(params)
    elif (
        images
        or videos
        or audios
        or params.get("referenceFileUrl")
        or params.get("referenceLinkUrl")
    ):
        raise _api_error(
            "External reference URLs are supported only by Seedance, HappyHorse, and Wan 3 models."
        )
    workflow = get_video_workflow_type(model_id)
    if not workflow:
        return
    if workflow == "i2v" and not (params.get("referenceImage") or params.get("referenceImageEnd")):
        raise _api_error(
            "i2v workflow requires at least one of referenceImage or referenceImageEnd. Please provide this asset."
        )
    if params.get("sam2Coordinates") and workflow != "animate-replace":
        raise _api_error("sam2Coordinates is only supported for animate-replace workflows.")
    requirements = _video_asset_requirements(model_id)
    if not requirements:
        return
    for asset, requirement in requirements.items():
        present = bool(params.get(asset))
        if requirement == "required" and not present:
            raise _api_error(f"{workflow} workflow requires {asset}. Please provide this asset.")
        if requirement == "forbidden" and present:
            raise _api_error(
                f"{workflow} workflow does not support {asset}. Please remove this asset."
            )


def create_job_request_message(
    project_id: str, params: dict[str, Any], options: dict[str, Any]
) -> dict[str, Any]:
    """Build the legacy Supernet `jobRequest` envelope used by the JS SDK."""

    project_type = params.get("type")
    if project_type not in {"image", "video", "audio"}:
        raise _api_error('Invalid project type. Must be "image", "video", or "audio".')
    if options.get("type") != project_type:
        raise _api_error(
            f"Invalid model type. Model does not support {project_type} generation. Please use a different model."
        )
    template = _template()
    keyframe = dict(template["keyFrames"][0])
    keyframe.update(
        {
            "steps": params.get("steps"),
            "guidanceScale": params.get("guidance"),
            "modelID": params["modelId"],
            "seed": params.get("seed"),
            "positivePrompt": params.get("positivePrompt", ""),
        }
    )
    for undefined_key in ("steps", "guidanceScale", "seed"):
        if keyframe[undefined_key] is None:
            keyframe.pop(undefined_key)
    if params.get("negativePrompt") and not (
        project_type == "audio"
        or (
            project_type == "video"
            and (
                is_external_video_model(params["modelId"]) or is_minimax_h3_model(params["modelId"])
            )
        )
    ):
        keyframe["negativePrompt"] = params["negativePrompt"]
    elif project_type in {"audio", "video"}:
        keyframe.pop("negativePrompt", None)
    if params.get("stylePrompt"):
        keyframe["stylePrompt"] = params["stylePrompt"]
    if params.get("loras"):
        keyframe["loras"] = params["loras"]
    if params.get("loraStrengths"):
        keyframe["loraStrengths"] = params["loraStrengths"]

    if project_type == "image":
        keyframe["sizePreset"] = params.get("sizePreset")
        contexts = params.get("contextImages") or []
        for index in range(1, 17):
            keyframe[f"hasContextImage{index}"] = (
                bool(contexts[index - 1]) if index <= len(contexts) else False
            )
        comfy = params["modelId"].startswith(
            (
                "z_image_",
                "dark_beast_z_image_",
                "krea2_",
                "dark_beast_krea2_",
                "qwen_image_",
                "wan_",
                "ace_step",
                "rtx_vsr_",
                "minimax_music3",
            )
        )
        if comfy:
            keyframe["comfySampler"] = _validate_option(params.get("sampler"), options, "sampler")
            keyframe["comfyScheduler"] = _validate_option(
                params.get("scheduler"), options, "scheduler"
            )
            keyframe["vae"] = (
                _validate_option(params.get("vae"), options, "vae") if "vae" in options else None
            )
        else:
            keyframe["scheduler"] = _validate_option(params.get("sampler"), options, "sampler")
            keyframe["timeStepSpacing"] = _validate_option(
                params.get("scheduler"), options, "scheduler"
            )
        if params.get("startingImage"):
            keyframe.update(
                {
                    "hasStartingImage": True,
                    "strengthIsEnabled": True,
                    "strength": 1 - (float(params.get("startingImageStrength") or 0.5)),
                }
            )
        control = params.get("controlNet")
        if control:
            raw: dict[str, Any] = {
                "name": control["name"],
                "cnImageState": "original",
                "hasImage": bool(control.get("image")),
            }
            if control.get("strength") is not None:
                raw["controlStrength"] = _validate_number(
                    control["strength"],
                    minimum=0,
                    maximum=1,
                    property_name="strength",
                )
            if control.get("mode"):
                raw["controlMode"] = {"balanced": 0, "prompt_priority": 1, "cn_priority": 2}[
                    control["mode"]
                ]
            if control.get("guidanceStart") is not None:
                raw["controlGuidanceStart"] = _validate_number(
                    control["guidanceStart"],
                    minimum=0,
                    maximum=1,
                    property_name="guidanceStart",
                )
            if control.get("guidanceEnd") is not None:
                raw["controlGuidanceEnd"] = _validate_number(
                    control["guidanceEnd"],
                    minimum=0,
                    maximum=1,
                    property_name="guidanceEnd",
                )
            keyframe["currentControlNetsJob"] = [raw]
        size_preset = params.get("sizePreset")
        if params.get("width") and params.get("height") and not size_preset:
            size_preset = "custom"
        keyframe["sizePreset"] = size_preset
        if size_preset == "custom" and params.get("width") and params.get("height"):
            minimum, maximum = _custom_image_size_bounds(params["modelId"])
            keyframe["width"] = _validate_number(
                params["width"],
                minimum=minimum,
                maximum=maximum,
                property_name="Width",
            )
            keyframe["height"] = _validate_number(
                params["height"],
                minimum=minimum,
                maximum=maximum,
                property_name="Height",
            )
        elif size_preset is None:
            keyframe.pop("sizePreset", None)
        if params.get("gptImageQuality") is not None:
            keyframe["gptImageQuality"] = params["gptImageQuality"]
        if params.get("gptImageBackground") is not None:
            keyframe["gptImageBackground"] = params["gptImageBackground"]

    elif project_type == "video":
        if not is_video_model(params["modelId"]):
            raise _api_error("Video generation is only supported for video models.")
        _validate_video_assets(params)
        _validate_h3_params(params)
        if params.get("referenceImage"):
            keyframe["hasReferenceImage"] = True
        for slot, _value in _video_context_slots(params):
            keyframe[f"hasContextImage{slot}"] = True
        if params.get("referenceImageEnd"):
            keyframe["hasReferenceImageEnd"] = True
        if is_minimax_h3_reference_model(params["modelId"]):
            for slot, _value in _media_slots(
                params.get("referenceAudio"), params.get("referenceAudios")
            ):
                keyframe[f"hasReferenceAudio{slot}"] = True
            for slot, _value in _media_slots(
                params.get("referenceVideo"), params.get("referenceVideos")
            ):
                keyframe[f"hasReferenceVideo{slot}"] = True
                duration = (params.get("referenceVideoDurations") or [])[slot - 1]
                keyframe[f"referenceVideo{slot}DurationSeconds"] = duration
        else:
            if params.get("referenceAudio"):
                keyframe["hasReferenceAudio"] = True
            if params.get("referenceVideo"):
                keyframe["hasReferenceVideo"] = True
        if params.get("referenceAudioIdentity"):
            keyframe["hasReferenceAudioIdentity"] = True
        if params.get("referenceMask") and params.get("controlNet", {}).get("name") == "inpaint":
            keyframe["hasReferenceMask"] = True
        for source, target in (
            ("referenceImageUrls", "referenceImageURLs"),
            ("referenceAudioUrls", "referenceAudioURLs"),
            ("referenceVideoUrls", "referenceVideoURLs"),
        ):
            if params.get(source):
                keyframe[target] = params[source]
        for source, target in (
            ("referenceFileUrl", "referenceFileURL"),
            ("referenceLinkUrl", "referenceLinkURL"),
            ("promptExtend", "promptExtend"),
            ("watermark", "watermark"),
            ("ratio", "ratio"),
            ("seedanceTaskType", "seedanceTaskType"),
        ):
            if params.get(source) is not None:
                keyframe[target] = params[source]
        if params.get("smartDuration"):
            keyframe["smartDuration"] = True
            keyframe["frames"] = calculate_video_frames(params["modelId"], 30, 30)
        field_map = {
            "generateAudio": "generateAudio",
            "audioIdentityStrength": "identityGuidanceScale",
            "frames": "frames",
            "shift": "shift",
            "teacacheThreshold": "teacacheThreshold",
            "audioStart": "audioStart",
            "audioDuration": "audioDuration",
            "videoStart": "videoStart",
            "firstFrameStrength": "firstFrameStrength",
            "lastFrameStrength": "lastFrameStrength",
            "detailerStrength": "detailerStrength",
            "outpaintPosition": "outpaintPosition",
        }
        for source, target in field_map.items():
            if params.get(source) is not None:
                value = params[source]
                if source == "teacacheThreshold":
                    value = _validate_number(
                        value,
                        minimum=0,
                        maximum=1,
                        property_name="teacacheThreshold",
                    )
                keyframe[target] = value
        if params.get("fps") is not None:
            keyframe["fps"] = params["fps"]
        elif is_wan3_model(params["modelId"]):
            keyframe["fps"] = 30
        elif is_external_video_model(params["modelId"]) or is_minimax_h3_model(params["modelId"]):
            keyframe["fps"] = 24
        if params.get("duration") is not None:
            duration = float(params["duration"])
            minimum = (
                MINIMAX_H3_MIN_DURATION
                if is_minimax_h3_model(params["modelId"])
                else 2
                if is_wan3_model(params["modelId"])
                else 3
                if is_happyhorse_model(params["modelId"])
                else 4
                if is_seedance_model(params["modelId"])
                else 1
            )
            maximum = (
                MINIMAX_H3_MAX_DURATION
                if is_minimax_h3_model(params["modelId"])
                else 30
                if is_seedance25_model(params["modelId"]) or is_wan3_model(params["modelId"])
                else 15
                if is_external_video_model(params["modelId"])
                else 20
                if is_ltx_model(params["modelId"]) or "_animate-" in params["modelId"]
                else 10
            )
            if not minimum <= duration <= maximum:
                raise ValueError(f"Video duration must be between {minimum} and {maximum}")
            keyframe["frames"] = calculate_video_frames(
                params["modelId"],
                duration,
                params.get("fps", 30 if is_wan3_model(params["modelId"]) else 24),
            )
        if params.get("sam2Coordinates") is not None:
            keyframe["sam2Coordinates"] = json.dumps(
                params["sam2Coordinates"], separators=(",", ":")
            )
        if params.get("trimEndFrame"):
            keyframe["trimEndFrame"] = True
        if params.get("controlNet"):
            control = params["controlNet"]
            raw = {"name": control["name"]}
            if control.get("strength") is not None:
                raw["controlStrength"] = _validate_number(
                    control["strength"],
                    minimum=0,
                    maximum=1,
                    property_name="strength",
                )
            keyframe["currentControlNetsJob"] = [raw]
        if params.get("width") and params.get("height"):
            if is_minimax_h3_model(params["modelId"]):
                keyframe["width"] = params["width"]
                keyframe["height"] = params["height"]
            else:
                keyframe["width"] = _validate_number(
                    params["width"], minimum=480, property_name="Video width"
                )
                keyframe["height"] = _validate_number(
                    params["height"], minimum=480, property_name="Video height"
                )
        keyframe["comfySampler"] = _validate_option(params.get("sampler"), options, "sampler")
        keyframe["comfyScheduler"] = _validate_option(params.get("scheduler"), options, "scheduler")

    else:
        for field in (
            "duration",
            "bpm",
            "timesignature",
            "language",
            "lyrics",
            "keyscale",
            "composerMode",
            "promptStrength",
            "creativity",
            "shift",
        ):
            if params.get(field) is not None:
                keyframe[field] = params[field]
        keyframe["comfySampler"] = _validate_option(params.get("sampler"), options, "sampler")
        keyframe["comfyScheduler"] = _validate_option(params.get("scheduler"), options, "scheduler")

    template["keyFrames"] = [keyframe]
    template.update(
        {
            "previews": params.get("numberOfPreviews", 0) if project_type == "image" else 0,
            "numberOfImages": params.get("numberOfMedia") or 1,
            "jobID": project_id,
            "disableSafety": bool(params.get("disableNSFWFilter")),
            "tokenType": params.get("tokenType"),
            "billingMode": params.get("billingMode"),
            "outputFormat": params.get("outputFormat")
            or ("mp3" if project_type == "audio" else "mp4" if project_type == "video" else "png"),
            **workload_attribution_to_wire_fields(params.get("attribution")),
        }
    )
    if params.get("network"):
        template["network"] = params["network"]
    if params.get("appSource"):
        template["appSource"] = params["appSource"]
    # JSON.stringify in the JS client omits undefined keys.
    return {key: value for key, value in template.items() if value is not None}


class Job(DataEntity):
    def __init__(self, data: dict[str, Any], project: Project, api: ProjectsApi) -> None:
        super().__init__(data)
        self._project = project
        self._api = api
        self._enhancement_project: Project | None = None
        self._enhancement_listener: Any = None
        self._runtime_timeout: asyncio.TimerHandle | None = None
        self.on("updated", self._handle_updated)
        if self.status == "processing":
            self._start_runtime_timeout()

    @property
    def id(self) -> str:
        return self._data["id"]

    @property
    def project_id(self) -> str:
        return self._data["projectId"]

    projectId = project_id

    @property
    def status(self) -> str:
        return self._data["status"]

    @property
    def finished(self) -> bool:
        return self.status in {"completed", "failed", "canceled"}

    @property
    def progress(self) -> int:
        if self.status == "completed":
            return 100
        external = self._data.get("externalProgress")
        if (
            isinstance(external, (int, float))
            and not isinstance(external, bool)
            and math.isfinite(external)
        ):
            return max(0, min(100, round(external * 100 if 0 <= external <= 1 else external)))
        count = self._data.get("stepCount", 0)
        if count:
            return max(0, min(100, round(self._data.get("step", 0) / count * 100)))
        started = self._data.get("etaStartedAt")
        eta = self.eta
        if isinstance(started, datetime) and isinstance(eta, datetime):
            total = (eta - started).total_seconds()
            if math.isfinite(total) and total > 0:
                elapsed = (_now() - started).total_seconds()
                if not math.isfinite(elapsed) or elapsed <= 0:
                    return 1
                return max(1, min(95, round(elapsed / total * 100)))
        return 0

    @property
    def step(self) -> int:
        return self._data.get("step", 0)

    @property
    def step_count(self) -> int:
        return self._data.get("stepCount", 0)

    stepCount = step_count

    @property
    def seed(self) -> int | None:
        return self._data.get("seed")

    @property
    def result_url(self) -> str | None:
        return self._data.get("resultUrl")

    resultUrl = result_url

    @property
    def image_url(self) -> str | None:
        return self.result_url or self.preview_url

    imageUrl = image_url

    @property
    def preview_url(self) -> str | None:
        return self._data.get("previewUrl")

    previewUrl = preview_url

    @property
    def error(self) -> dict[str, Any] | None:
        return self._data.get("error")

    @property
    def is_nsfw(self) -> bool:
        return bool(self._data.get("isNSFW"))

    isNSFW = is_nsfw

    @property
    def has_result_media(self) -> bool:
        return self.status == "completed" and not self.is_nsfw

    hasResultMedia = has_result_media

    @property
    def type(self) -> str:
        return self._project.type

    @property
    def worker_name(self) -> str | None:
        return self._data.get("workerName")

    workerName = worker_name

    @property
    def eta(self) -> datetime | None:
        return self._data.get("eta")

    @property
    def eta_seconds(self) -> int | float | None:
        return self._data.get("etaSeconds")

    etaSeconds = eta_seconds

    @property
    def eta_range(self) -> dict[str, float] | None:
        return self._data.get("etaRange")

    etaRange = eta_range

    @property
    def enhanced_image(self) -> dict[str, Any] | None:
        project = self._enhancement_project
        if project is None:
            return None
        job = project.jobs[0] if project.jobs else None

        async def result_url() -> str | None:
            return await job.get_result_url() if job is not None else None

        return {
            "status": project.status,
            "progress": project.progress,
            "result": job.result_url if job is not None else None,
            "error": project.error,
            "get_result_url": result_url,
            "getResultUrl": result_url,
        }

    enhancedImage = enhanced_image

    @property
    def _audio_content_type(self) -> str:
        return {"flac": "audio/flac", "wav": "audio/wav"}.get(
            self._project.params.get("outputFormat"), "audio/mpeg"
        )

    @property
    def _image_content_type(self) -> str | None:
        return {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "png": "image/png",
        }.get(self._project.params.get("outputFormat"))

    async def get_result_url(self) -> str:
        if self.result_url:
            return self.result_url
        if self.status != "completed":
            raise RuntimeError("Job is not completed yet")
        if self.type in {"video", "audio"}:
            params: dict[str, Any] = {"jobId": self.project_id, "id": self.id, "type": "complete"}
            if self.type == "audio":
                params["contentType"] = self._audio_content_type
            url = await self._api.media_download_url(params)
        else:
            params = {"jobId": self.project_id, "imageId": self.id, "type": "complete"}
            if self._image_content_type:
                params["contentType"] = self._image_content_type
            url = await self._api.download_url(params)
        self._update({"resultUrl": url})
        return url

    getResultUrl = get_result_url

    async def get_result_data(self) -> bytes:
        if not self.has_result_media:
            raise RuntimeError("No result media available")
        return await self._api.client.rest.get_bytes(await self.get_result_url())

    getResultData = get_result_data

    async def enhance(
        self,
        strength: str,
        overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | None:
        if self._project.params.get("type") != "image":
            raise RuntimeError("Enhancement is only available for images")
        if self.status != "completed":
            raise RuntimeError("Job is not completed yet")
        if self.is_nsfw:
            raise RuntimeError("Job did not pass NSFW filter")

        values = _nested_params(normalize_params(overrides, **kwargs))
        if self._enhancement_project is not None and self._enhancement_listener is not None:
            self._enhancement_project.off("updated", self._enhancement_listener)
        self._enhancement_project = None
        self._enhancement_listener = None

        params = dict(_ENHANCEMENT_DEFAULTS)
        params.update(
            {
                "type": "image",
                "positivePrompt": values.get("positivePrompt")
                or self._project.params.get("positivePrompt", ""),
                "stylePrompt": values.get("stylePrompt")
                or self._project.params.get("stylePrompt", ""),
                "tokenType": values.get("tokenType") or self._project.params.get("tokenType"),
                "seed": self.seed or self._project.params.get("seed"),
                "startingImage": await self.get_result_data(),
                "startingImageStrength": 1 - _enhancement_strength(strength),
                "sizePreset": self._project.params.get("sizePreset"),
            }
        )
        project = await self._api.create(params)
        self._enhancement_project = project

        def updated(_keys: Any) -> None:
            self.emit("updated", ["enhancedImage"])

        self._enhancement_listener = updated
        project.on("updated", updated)
        results = await project.wait_for_completion()
        return results[0] if results else None

    async def _sync_with_rest_data(self, data: dict[str, Any]) -> None:
        delta: dict[str, Any] = {
            "step": data.get("performedSteps", self.step),
            "workerName": (
                data.get("worker", {}).get("name")
                if isinstance(data.get("worker"), dict)
                else self.worker_name
            ),
            "seed": data.get("seedUsed", self.seed),
            "isNSFW": bool(data.get("triggeredNSFWFilter")),
        }
        status = _JOB_STATUS_MAP.get(data.get("status"))
        if status:
            delta["status"] = status
        direct_url = _raw_result_url(data)
        if not self.result_url and direct_url:
            delta["resultUrl"] = direct_url
        self._update(delta)
        if (
            not self.result_url
            and status == "completed"
            and not bool(data.get("triggeredNSFWFilter"))
        ):
            with contextlib.suppress(Exception):
                await self.get_result_url()

    def _update(self, delta: dict[str, Any]) -> None:
        if "eta" in delta and isinstance(delta.get("eta"), datetime):
            delta["etaSeconds"] = round((delta["eta"] - _now()).total_seconds())
            if not isinstance(self._data.get("etaStartedAt"), datetime):
                delta.setdefault("etaStartedAt", _now())
        super()._update(delta)
        if self.status == "processing":
            self._start_runtime_timeout()
        elif self.finished:
            self._stop_runtime_timeout()

    def _runtime_limit_seconds(self) -> float:
        network = self._project.params.get("network")
        if network not in {"fast", "relaxed"}:
            network = self._api._current_network()
        if network not in {"fast", "relaxed"}:
            network = "relaxed"
        media = self._project.type if self._project.type in {"video", "audio"} else "image"
        floor = _RUNTIME_LIMIT_FLOOR_SECONDS[network][media]
        eta = self._data.get("etaSeconds")
        eta_budget = (
            float(eta) * _RUNTIME_LIMIT_ETA_MULTIPLIER
            if isinstance(eta, (int, float)) and not isinstance(eta, bool) and eta > 0
            else 0
        )
        return min(_RUNTIME_LIMIT_MAX_SECONDS, max(floor, eta_budget))

    def _start_runtime_timeout(self) -> None:
        if self._runtime_timeout is not None or self.finished:
            return
        limit = self._runtime_limit_seconds()
        self._runtime_timeout = asyncio.get_running_loop().call_later(
            limit, self._runtime_timeout_elapsed, limit
        )

    def _runtime_timeout_elapsed(self, limit: float) -> None:
        self._runtime_timeout = None
        if self.status == "processing" and not self._project.finished:
            self._project._handle_job_runtime_timeout(self, limit)

    def _stop_runtime_timeout(self) -> None:
        if self._runtime_timeout is not None:
            self._runtime_timeout.cancel()
            self._runtime_timeout = None

    def _handle_updated(self, keys: list[str]) -> None:
        if any(key in keys for key in ("step", "stepCount", "externalProgress", "eta")):
            self.emit("progress", self.progress)
        if "status" in keys and self.status == "completed":
            self.emit("completed", self.result_url)
        if "status" in keys and self.status == "failed":
            self.emit("failed", self.error)


class Project(DataEntity):
    def __init__(self, params: dict[str, Any], api: ProjectsApi) -> None:
        super().__init__(
            {
                "id": new_id(),
                "startedAt": _now(),
                "params": params,
                "queuePosition": -1,
                "status": "pending",
            }
        )
        self._api = api
        self._jobs: list[Job] = []
        self._completion = asyncio.get_running_loop().create_future()
        self._last_emitted_progress = -1
        self._last_updated = _now()
        self._failed_sync_attempts = 0
        self._timeout_handle: asyncio.TimerHandle | None = None
        self.on("updated", self._handle_updated)
        self._arm_timeout()

    @property
    def id(self) -> str:
        return self._data["id"]

    @property
    def params(self) -> dict[str, Any]:
        return self._data["params"]

    @property
    def type(self) -> str:
        return self.params["type"]

    @property
    def status(self) -> str:
        return self._data["status"]

    @property
    def finished(self) -> bool:
        return self.status in {"completed", "failed", "canceled"}

    @property
    def error(self) -> dict[str, Any] | None:
        return self._data.get("error")

    @property
    def eta(self) -> datetime | None:
        return self._data.get("eta")

    @property
    def queue_position(self) -> int:
        return self._data["queuePosition"]

    queuePosition = queue_position

    @property
    def estimated_start_at(self) -> datetime | None:
        return self._data.get("estimatedStartAt")

    estimatedStartAt = estimated_start_at

    @property
    def queue_status(self) -> str | None:
        return self._data.get("queueStatus")

    queueStatus = queue_status

    @property
    def jobs(self) -> list[Job]:
        return list(self._jobs)

    @property
    def result_urls(self) -> list[str]:
        return [job.result_url for job in self._jobs if job.result_url]

    resultUrls = result_urls

    @property
    def progress(self) -> int:
        if self.status == "completed":
            return 100
        count = max(1, int(self.params.get("numberOfMedia", 1)))
        return max(0, min(100, round(sum(job.progress for job in self._jobs) / count)))

    async def wait_for_completion(self, timeout: float | None = None) -> list[str]:
        if self.status == "completed":
            return self.result_urls
        if self.status == "failed":
            if self._completion.done():
                return await self._completion
            raise ProjectError(self.error or {})
        if timeout is None:
            return await self._completion
        return await asyncio.wait_for(asyncio.shield(self._completion), timeout)

    waitForCompletion = wait_for_completion

    async def cancel(self) -> None:
        await self._api.cancel(self.id)

    def job(self, job_id: str) -> Job | None:
        return next((job for job in self._jobs if job.id == job_id), None)

    def _add_job(self, data: dict[str, Any]) -> Job:
        job = Job(data, self, self._api)
        self._jobs.append(job)

        def job_updated(_keys: Any) -> None:
            self._keep_alive()
            self.emit("updated", ["jobs"])

        job.on("updated", job_updated)
        job.on("completed", lambda _url: self.emit("jobCompleted", job))
        job.on("failed", lambda _error: self.emit("jobFailed", job))
        self.emit("jobStarted", job)
        self.emit("updated", ["jobs"])
        return job

    def _handle_updated(self, keys: list[str]) -> None:
        progress = self.progress
        if progress != self._last_emitted_progress:
            self._last_emitted_progress = progress
            self.emit("progress", progress)
        if self.finished and self._timeout_handle is not None:
            self._timeout_handle.cancel()
            self._timeout_handle = None
        if self.finished:
            for job in self._jobs:
                job._stop_runtime_timeout()
        if ("status" in keys or "jobs" in keys) and self.status == "completed":
            all_started = len(self._jobs) >= int(self.params.get("numberOfMedia", 1))
            if all_started and all(job.finished for job in self._jobs):
                if not self._completion.done():
                    self._completion.set_result(self.result_urls)
                self.emit("completed", self.result_urls)
        if "status" in keys and self.status == "failed":
            error = self.error or {"code": 0, "message": "Project failed"}
            if not self._completion.done():
                self._completion.set_exception(ProjectError(error))
            self.emit("failed", error)

    def _update(self, delta: dict[str, Any]) -> None:
        self._keep_alive()
        super()._update(delta)

    def _keep_alive(self) -> None:
        self._last_updated = _now()
        self._arm_timeout()

    def _arm_timeout(self) -> None:
        if self.finished:
            if self._timeout_handle is not None:
                self._timeout_handle.cancel()
                self._timeout_handle = None
            return
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
        self._timeout_handle = asyncio.get_running_loop().call_later(
            _PROJECT_TIMEOUT_SECONDS, self._timeout_elapsed
        )

    def _timeout_elapsed(self) -> None:
        self._timeout_handle = None
        if not self.finished:
            asyncio.create_task(self._check_for_timeout())

    def _handle_job_runtime_timeout(self, job: Job, limit_seconds: float) -> None:
        if self.finished or job.finished or job not in self._jobs:
            return
        limit = _format_runtime_limit(limit_seconds)
        job_error = {"code": 0, "message": f"Job exceeded the maximum runtime of {limit}"}
        asyncio.create_task(self._notify_runtime_timeout(job))
        for project_job in self._jobs:
            if not project_job.finished:
                project_job._update({"status": "failed", "error": job_error})
        self._update(
            {
                "status": "failed",
                "error": {
                    "code": 0,
                    "message": f"Job {job.id} exceeded the maximum runtime of {limit}; project canceled",
                },
            }
        )

    async def _notify_runtime_timeout(self, job: Job) -> None:
        try:
            await self._api._notify_project_timed_out(self.id)
        except Exception:
            _LOGGER.exception("Failed to cancel project %s after job %s timed out", self.id, job.id)

    async def _check_for_timeout(self) -> None:
        if self.finished:
            return
        idle_seconds = (_now() - self._last_updated).total_seconds()
        if idle_seconds < _PROJECT_TIMEOUT_SECONDS:
            self._arm_timeout()
            return
        live_project_ids = await self._api._list_active_project_ids()
        if live_project_ids is not None and self.id in live_project_ids:
            self._failed_sync_attempts = 0
            self._keep_alive()
            return
        socket_confirms_gone = live_project_ids is not None
        try:
            await self._sync_to_server()
        except Exception as error:
            if isinstance(error, ApiError) and error.status == 404 and not socket_confirms_gone:
                self._arm_timeout()
                return
            self._failed_sync_attempts += 1
            if self._failed_sync_attempts < _MAX_FAILED_SYNC_ATTEMPTS:
                self._arm_timeout()
                return
            with contextlib.suppress(Exception):
                await self._api._notify_project_timed_out(self.id)
            for job in self.jobs:
                if not job.finished:
                    job._update(
                        {
                            "status": "failed",
                            "error": {"code": 0, "message": "Job timed out"},
                        }
                    )
            self._update(
                {
                    "status": "failed",
                    "error": {
                        "code": 0,
                        "message": "Project timed out. Please try again or contact support.",
                    },
                }
            )
            return
        self._failed_sync_attempts = 0
        if not self.finished:
            self._keep_alive()

    async def _sync_to_server(self) -> None:
        data = await self._api.get(self.id)
        raw_jobs = data.get("completedWorkerJobs")
        if not isinstance(raw_jobs, list):
            raw_jobs = []
        jobs_by_id = {
            str(raw.get("imgID")): raw
            for raw in raw_jobs
            if isinstance(raw, dict) and raw.get("imgID")
        }
        for job in self._jobs:
            raw = jobs_by_id.pop(job.id, None)
            if raw is not None:
                await job._sync_with_rest_data(raw)
        for raw in jobs_by_id.values():
            raw_status = _JOB_STATUS_MAP.get(raw.get("status"), "pending")
            worker = raw.get("worker") if isinstance(raw.get("worker"), dict) else {}
            job = self._add_job(
                {
                    "id": raw.get("imgID") or new_id(),
                    "projectId": self.id,
                    "status": raw_status,
                    "step": raw.get("performedSteps", 0),
                    "stepCount": data.get("stepCount", self.params.get("steps", 0)),
                    "workerName": worker.get("name"),
                    "seed": raw.get("seedUsed"),
                    "isNSFW": bool(raw.get("triggeredNSFWFilter")),
                    "resultUrl": _raw_result_url(raw),
                }
            )
            await job._sync_with_rest_data(raw)

        params = dict(self.params)
        if data.get("imageCount") is not None:
            params["numberOfMedia"] = data["imageCount"]
        if data.get("stepCount") is not None:
            params["steps"] = data["stepCount"]
        if self.type == "image" and data.get("previewCount") is not None:
            params["numberOfPreviews"] = data["previewCount"]
        delta: dict[str, Any] = {"params": params}
        status = _PROJECT_STATUS_MAP.get(data.get("status"))
        if status:
            delta["status"] = status
        self._update(delta)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["jobs"] = [job.to_dict() for job in self._jobs]
        return data


class ProjectsApi(EventEmitter):
    def __init__(self, client: ApiClient) -> None:
        super().__init__()
        self.client = client
        self._available_models: list[dict[str, Any]] = []
        self._projects: list[Project] = []
        self._supported_models: list[dict[str, Any]] | None = None
        self._supported_models_at = 0.0
        self._model_tiers: dict[str, Any] | None = None
        self._model_tiers_at = 0.0
        self._current_network_type: str | None = None
        self._cancellation_requests: dict[str, asyncio.Task[None]] = {}
        self._lora_catalog_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        socket = client.socket
        socket.on("changeNetwork", self._handle_change_network)
        socket.on("swarmModels", self._handle_swarm_models)
        socket.on("jobState", self._handle_job_state)
        socket.on("jobProgress", self._handle_job_progress)
        socket.on("jobETA", self._handle_job_eta)
        socket.on("jobResult", self._handle_job_result)
        socket.on("jobError", self._handle_job_error)
        client.on("disconnected", self._handle_disconnect)

    @property
    def available_models(self) -> list[dict[str, Any]]:
        return list(self._available_models)

    availableModels = available_models

    @property
    def tracked_projects(self) -> list[Project]:
        return list(self._projects)

    trackedProjects = tracked_projects

    def _handle_change_network(self, data: Any) -> None:
        network = data.get("network") if isinstance(data, dict) else data
        if network in {"fast", "relaxed"}:
            self._current_network_type = network
        self._set_available_models([])

    def _current_network(self) -> str | None:
        return self._current_network_type

    def is_video_model_id(self, model_id: str) -> bool:
        model = next(
            (item for item in self._supported_models or [] if item.get("id") == model_id), None
        )
        return model.get("media") == "video" if model else is_video_model(model_id)

    isVideoModelId = is_video_model_id

    def is_audio_model_id(self, model_id: str) -> bool:
        model = next(
            (item for item in self._supported_models or [] if item.get("id") == model_id), None
        )
        return model.get("media") == "audio" if model else is_audio_model(model_id)

    isAudioModelId = is_audio_model_id

    def _set_available_models(self, models: list[dict[str, Any]]) -> None:
        self._available_models = models
        self.emit("availableModels", self.available_models)

    async def _resolve_swarm_models(self, data: dict[str, int]) -> None:
        with contextlib.suppress(Exception):
            supported = await self.get_supported_models()
            index = {model["id"]: model for model in supported}
            self._set_available_models(
                [
                    {
                        "id": model_id,
                        "name": index.get(model_id, {}).get("name", model_id.replace("-", " ")),
                        "workerCount": count,
                        "media": index.get(model_id, {}).get("media", "image"),
                    }
                    for model_id, count in data.items()
                ]
            )

    def _handle_swarm_models(self, data: Any) -> None:
        if isinstance(data, dict):
            asyncio.create_task(self._resolve_swarm_models(data))

    async def wait_for_models(self, timeout: float = 10.0) -> list[dict[str, Any]]:
        if self._available_models:
            return self.available_models
        future: asyncio.Future[list[dict[str, Any]]] = asyncio.get_running_loop().create_future()

        def ready(models: list[dict[str, Any]]) -> None:
            if models and not future.done():
                future.set_result(list(models))

        remove = self.on("availableModels", ready)
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError as error:
            raise TimeoutError("Timeout waiting for models") from error
        finally:
            remove()

    waitForModels = wait_for_models

    async def create(self, params: dict[str, Any] | None = None, **kwargs: Any) -> Project:
        data = _nested_params(normalize_params(params, **kwargs))
        for required in ("type", "modelId", "positivePrompt", "numberOfMedia"):
            if required not in data:
                raise ValueError(f"{required} is required")
        project = Project(data, self)
        options = await self.get_model_options(data["modelId"])
        request_params = dict(data)
        request_params["appSource"] = data.get("appSource") or self.client.app_source
        resolver = getattr(self.client, "resolve_workload_attribution", None)
        request_params["attribution"] = (
            resolver(data.get("attribution"), project.id) if callable(resolver) else None
        )
        request = create_job_request_message(project.id, request_params, options)
        await self._process_assets(project, data, request)
        await self.client.socket.send("jobRequest", request)
        self._projects.append(project)
        return project

    async def _process_assets(
        self, project: Project, data: dict[str, Any], request: dict[str, Any]
    ) -> None:
        if data["type"] == "image":
            assets: list[tuple[str, Any, bool]] = [
                ("startingImage", data.get("startingImage"), False),
                ("cnImage", data.get("controlNet", {}).get("image"), False),
            ]
            contexts = data.get("contextImages") or []
            max_context = (
                16
                if data["modelId"] == "gpt-image-2"
                else 2
                if "kontext" in data["modelId"] or "identity_edit" in data["modelId"]
                else 3
            )
            if len(contexts) > max_context:
                raise _api_error(f"Up to {max_context} context images are supported for this model")
            assets.extend(
                (f"contextImage{index}", value, False) for index, value in enumerate(contexts, 1)
            )
            for role, value, media in assets:
                if value and value is not True:
                    await self._upload_asset(project.id, role, value, media=media)
        elif data["type"] == "video":
            h3_reference = is_minimax_h3_reference_model(data["modelId"])
            if data.get("referenceImage") and data["referenceImage"] is not True:
                content_type = await self._upload_asset(
                    project.id, "referenceImage", data["referenceImage"], media=False
                )
                request["keyFrames"][0]["referenceImageContentType"] = content_type
            for slot, value in _video_context_slots(data):
                if value is not True:
                    await self._upload_asset(project.id, f"contextImage{slot}", value, media=False)
            if data.get("referenceImageEnd") and data["referenceImageEnd"] is not True:
                content_type = await self._upload_asset(
                    project.id, "referenceImageEnd", data["referenceImageEnd"], media=False
                )
                request["keyFrames"][0]["referenceImageEndContentType"] = content_type
            if h3_reference:
                for media_name, single_name, plural_name in (
                    ("referenceAudio", "referenceAudio", "referenceAudios"),
                    ("referenceVideo", "referenceVideo", "referenceVideos"),
                ):
                    for slot, value in _media_slots(data.get(single_name), data.get(plural_name)):
                        if value is True:
                            continue
                        role = f"{media_name}{slot}"
                        content_type = await self._upload_asset(project.id, role, value, media=True)
                        request["keyFrames"][0][f"{role}ContentType"] = content_type
            for role, media in (
                ("referenceAudio", True),
                ("referenceAudioIdentity", True),
                ("referenceVideo", True),
                ("referenceMask", False),
            ):
                if h3_reference and role in {"referenceAudio", "referenceVideo"}:
                    continue
                value = data.get(role)
                if role == "referenceMask" and data.get("controlNet", {}).get("name") != "inpaint":
                    continue
                if value and value is not True:
                    content_type = await self._upload_asset(
                        project.id,
                        "referenceAudio" if role == "referenceAudioIdentity" else role,
                        value,
                        media=media,
                    )
                    if request.get("keyFrames"):
                        request["keyFrames"][0][f"{role}ContentType"] = content_type
                        if role == "referenceAudioIdentity":
                            request["keyFrames"][0].setdefault(
                                "referenceAudioContentType", content_type
                            )

    async def _upload_asset(self, job_id: str, role: str, value: Any, *, media: bool) -> str | None:
        content_type = detect_content_type(value)
        body = read_media(value)
        if media:
            url = await self.media_upload_url(
                {"jobId": job_id, "type": role, "contentType": content_type}
            )
        else:
            url = await self.upload_url(
                {
                    "imageId": new_id(),
                    "jobId": job_id,
                    "type": role,
                    "contentType": content_type,
                }
            )
        await self.client.rest.put_bytes(url, body, content_type=content_type)
        return content_type

    async def get(self, project_id: str) -> dict[str, Any]:
        response = await self.client.rest.get(f"/v1/projects/{quote(project_id)}")
        return response["data"]["project"]

    async def _list_active_project_ids(self) -> list[str] | None:
        try:
            response = await self.client.socket.get("/api/v1/artist/projects/active")
        except Exception:
            return None
        projects = response.get("projects") if isinstance(response, dict) else None
        if not isinstance(projects, list):
            return None
        return [
            item["id"]
            for item in projects
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]

    async def cancel(self, project_id: str) -> None:
        existing = self._cancellation_requests.get(project_id)
        if existing is not None:
            await existing
            return
        task = asyncio.create_task(self._cancel_once(project_id))
        self._cancellation_requests[project_id] = task
        try:
            await task
        finally:
            if self._cancellation_requests.get(project_id) is task:
                self._cancellation_requests.pop(project_id, None)

    async def _cancel_once(self, project_id: str) -> None:
        project = next((item for item in self._projects if item.id == project_id), None)
        if project is not None and project.finished:
            return
        confirmation: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def confirmed(data: Any) -> None:
            if not isinstance(data, dict) or data.get("jobID") != project_id:
                return
            if confirmation.done():
                return
            if data.get("didCancel"):
                confirmation.set_result(None)
            else:
                confirmation.set_exception(
                    RuntimeError(
                        data.get("error_message")
                        or "Cancellation could not be confirmed. The job is still running; please try again."
                    )
                )

        remove = self.client.socket.on("artistCancelConfirmation", confirmed)
        try:
            await asyncio.gather(
                self.client.socket.send(
                    "jobError",
                    {
                        "jobID": project_id,
                        "error": "artistCanceled",
                        "error_message": "artistCanceled",
                        "isFromWorker": False,
                    },
                ),
                asyncio.wait_for(confirmation, timeout=_CANCELLATION_CONFIRMATION_TIMEOUT_SECONDS),
            )
        except asyncio.TimeoutError as error:
            raise RuntimeError(
                "Cancellation could not be confirmed. The job is still running; please try again."
            ) from error
        finally:
            remove()
        if project:
            self._projects.remove(project)
            for job in project.jobs:
                if not job.finished:
                    job._update({"status": "canceled"})
            if not project.finished:
                project._update({"status": "canceled"})

    async def _notify_project_timed_out(self, project_id: str) -> None:
        await self.client.socket.send(
            "jobError",
            {
                "jobID": project_id,
                "error": "artistCanceled",
                "error_message": "artistCanceled",
                "isFromWorker": False,
            },
        )

    async def upload_url(self, params: dict[str, Any] | None = None, **kwargs: Any) -> str:
        response = await self.client.rest.get(
            "/v1/image/uploadUrl", normalize_params(params, **kwargs)
        )
        return response["data"]["uploadUrl"]

    uploadUrl = upload_url

    async def download_url(self, params: dict[str, Any] | None = None, **kwargs: Any) -> str:
        response = await self.client.rest.get(
            "/v1/image/downloadUrl", normalize_params(params, **kwargs)
        )
        try:
            return response["data"]["downloadUrl"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(f"API returned no downloadUrl: {response!r}") from error

    downloadUrl = download_url

    async def media_upload_url(self, params: dict[str, Any] | None = None, **kwargs: Any) -> str:
        response = await self.client.rest.get(
            "/v1/media/uploadUrl", normalize_params(params, **kwargs)
        )
        return response["data"]["uploadUrl"]

    mediaUploadUrl = media_upload_url

    async def media_download_url(self, params: dict[str, Any] | None = None, **kwargs: Any) -> str:
        response = await self.client.rest.get(
            "/v1/media/downloadUrl", normalize_params(params, **kwargs)
        )
        try:
            return response["data"]["downloadUrl"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(f"API returned no downloadUrl: {response!r}") from error

    mediaDownloadUrl = media_download_url

    async def get_supported_models(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        if (
            self._supported_models is not None
            and not force_refresh
            and time.time() - self._supported_models_at < 86400
        ):
            return list(self._supported_models)
        self._supported_models = await self.client.socket.get("/api/v1/models/list")
        self._supported_models_at = time.time()
        return list(self._supported_models)

    getSupportedModels = get_supported_models

    async def _get_model_tiers(self, force_refresh: bool = False) -> dict[str, Any]:
        if (
            self._model_tiers is not None
            and not force_refresh
            and time.time() - self._model_tiers_at < 86400
        ):
            return dict(self._model_tiers)
        self._model_tiers = await self.client.socket.get("/api/v2/models/tiers")
        self._model_tiers_at = time.time()
        return dict(self._model_tiers)

    @staticmethod
    def _map_options(
        data: dict[str, Any] | None, aliases: dict[str, str] | None = None
    ) -> dict[str, Any]:
        if not data:
            return {"allowed": [], "default": None}
        aliases = aliases or {}
        return {
            "allowed": [aliases.get(item, item) for item in data.get("allowed", [])],
            "default": aliases.get(data.get("default"), data.get("default")),
        }

    @staticmethod
    def _map_range(data: dict[str, Any]) -> dict[str, Any]:
        return {
            "min": data["min"],
            "max": data["max"],
            "step": 10 ** -data["decimals"] if data.get("decimals") else data.get("step", 1),
            "default": data["default"],
        }

    async def get_model_options(self, model_id: str) -> dict[str, Any]:
        models, tiers = await asyncio.gather(self.get_supported_models(), self._get_model_tiers())
        model = next((item for item in models if item.get("id") == model_id), None)
        if not model:
            raise ValueError(f"Model {model_id} not supported")
        tier = tiers.get(model.get("tier"))
        if not tier:
            raise ValueError(
                f'Unable to find model tier "{model.get("tier")}" please contact support'
            )
        kind = tier.get("type") or "image"
        options: dict[str, Any] = {
            "type": kind,
            "sampler": self._map_options(
                tier.get("comfySampler") or tier.get("sampler"), _SAMPLER_ALIASES
            ),
            "scheduler": self._map_options(
                tier.get("comfyScheduler") or tier.get("scheduler"), _SCHEDULER_ALIASES
            ),
        }
        for field in (
            "steps",
            "guidance",
            "width",
            "height",
            "duration",
            "bpm",
            "promptStrength",
            "creativity",
            "shift",
        ):
            if tier.get(field):
                options[field] = self._map_range(tier[field])
        for field in ("fps", "timesignature", "language", "keyscale", "vae"):
            if tier.get(field):
                options[field] = self._map_options(tier[field])
        if tier.get("composerMode"):
            options["composerMode"] = {"default": tier["composerMode"]["default"]}
        if tier.get("maxPixels") is not None:
            options["maxPixels"] = tier["maxPixels"]
        return options

    getModelOptions = get_model_options

    async def get_size_presets(
        self, network: str, model_id: str, force_refresh: bool = False
    ) -> list[dict[str, Any]]:
        # Unlike JS's global 10m cache, rely on the HTTP layer; callers can cache safely.
        return await self.client.socket.get(
            f"/api/v1/size-presets/network/{quote(network)}/model/{quote(model_id)}"
        )

    getSizePresets = get_size_presets

    async def get_available_models(self, network: str) -> list[dict[str, Any]]:
        workers, models = await asyncio.gather(
            self.client.socket.get(f"/api/v1/status/network/{quote(network)}/models"),
            self.get_supported_models(),
        )
        result = []
        for sid, count in workers.items():
            model = next((item for item in models if item.get("SID") == int(sid)), {})
            result.append(
                {
                    "id": model.get("id", sid),
                    "name": model.get("name", sid.replace("-", " ")),
                    "workerCount": count,
                    "media": model.get("media", "image"),
                }
            )
        return result

    getAvailableModels = get_available_models

    async def get_video_asset_config(self, model_id: str) -> dict[str, Any]:
        if not self.is_video_model_id(model_id):
            raise _api_error(f"Model {model_id} is not a video model")
        workflow = get_video_workflow_type(model_id)
        return {
            "workflowType": workflow,
            **({"assets": _video_asset_requirements(model_id)} if workflow else {}),
        }

    getVideoAssetConfig = get_video_asset_config

    @staticmethod
    def _cost(response: dict[str, Any]) -> dict[str, Any]:
        quote_data = response["quote"]["project"]
        return {
            "token": quote_data["costInToken"],
            "usd": quote_data["costInUSD"],
            "spark": quote_data["costInSpark"],
            "sogni": quote_data["costInSogni"],
        }

    async def estimate_cost(
        self, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        data = normalize_params(params, **kwargs)
        network = data.get("network", "fast")
        model = data["model"]
        await self.get_model_options(model)
        path: list[Any] = [
            data.get("tokenType", "spark"),
            network,
            model,
            data["imageCount"],
            data["stepCount"],
            data["previewCount"],
            1 if data.get("cnEnabled") else 0,
            1 - data["startingImageStrength"] if data.get("startingImageStrength") else 0,
        ]
        if data.get("sizePreset"):
            presets = await self.get_size_presets(network, model)
            preset = next((item for item in presets if item["id"] == data["sizePreset"]), None)
            if not preset:
                raise ValueError("Invalid size preset")
            path.extend([preset["width"], preset["height"]])
        else:
            path.extend([data.get("width", 0), data.get("height", 0)])
        version = 2
        if data.get("sampler") or "contextImages" in data:
            version = 3
            path.extend(
                [data.get("guidance", 0), data.get("sampler", "_"), data.get("contextImages", 0)]
            )
        query = {key: data[key] for key in ("gptImageQuality", "outputFormat") if data.get(key)}
        response = await self.client.socket.get(
            f"/api/v{version}/job/estimate/" + "/".join(map(str, path)), query
        )
        return self._cost(response)

    estimateCost = estimate_cost

    async def estimate_enhancement_cost(
        self, strength: str, token_type: str = "spark"
    ) -> dict[str, Any]:
        return await self.estimate_cost(
            network=_ENHANCEMENT_DEFAULTS["network"],
            token_type=token_type,
            model=_ENHANCEMENT_DEFAULTS["modelId"],
            image_count=1,
            step_count=_ENHANCEMENT_DEFAULTS["steps"],
            preview_count=0,
            cn_enabled=False,
            starting_image_strength=_enhancement_strength(strength),
        )

    estimateEnhancementCost = estimate_enhancement_cost

    async def estimate_video_cost(
        self, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        data = normalize_params(params, **kwargs)
        frames = data.get("frames") or calculate_video_frames(
            data["model"], data["duration"], data["fps"]
        )
        path: list[Any] = [
            data["tokenType"],
            data["model"],
            data["width"],
            data["height"],
            frames,
            data["fps"],
        ]
        count = data.get("numberOfMedia", 1)
        if data.get("steps") is not None:
            path.extend([data["steps"], count])
        elif count != 1:
            path.extend([0, count])
        has_video = (
            data.get("hasVideoInput")
            or data.get("referenceVideo")
            or data.get("referenceVideoUrls")
        )
        response = await self.client.socket.get(
            "/api/v1/job-video/estimate/" + "/".join(quote(str(item)) for item in path),
            {
                "hasVideoInput": 1 if has_video else None,
                "referenceImageCount": (
                    math.floor(data["referenceImageCount"])
                    if isinstance(data.get("referenceImageCount"), (int, float))
                    and not isinstance(data.get("referenceImageCount"), bool)
                    and math.isfinite(data["referenceImageCount"])
                    and data["referenceImageCount"] >= 0
                    else None
                ),
                "referenceVideoCount": (
                    math.floor(data["referenceVideoCount"])
                    if isinstance(data.get("referenceVideoCount"), (int, float))
                    and not isinstance(data.get("referenceVideoCount"), bool)
                    and math.isfinite(data["referenceVideoCount"])
                    and data["referenceVideoCount"] >= 0
                    else None
                ),
                "referenceVideoDurationSeconds": (
                    data["referenceVideoDurationSeconds"]
                    if isinstance(data.get("referenceVideoDurationSeconds"), (int, float))
                    and not isinstance(data.get("referenceVideoDurationSeconds"), bool)
                    and math.isfinite(data["referenceVideoDurationSeconds"])
                    and data["referenceVideoDurationSeconds"] >= 0
                    else None
                ),
            },
        )
        return self._cost(response)

    estimateVideoCost = estimate_video_cost

    async def estimate_audio_cost(
        self, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        data = normalize_params(params, **kwargs)
        path = [
            data["tokenType"],
            data["model"],
            data["duration"],
            data["steps"],
            data["numberOfMedia"],
        ]
        response = await self.client.socket.get(
            "/api/v1/job-audio/estimate/" + "/".join(quote(str(item)) for item in path)
        )
        return self._cost(response)

    estimateAudioCost = estimate_audio_cost

    async def available_loras(
        self, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        data = normalize_params(params, **kwargs)
        model_id = data.get("modelId")
        cache_key = model_id or ""
        cached = self._lora_catalog_cache.get(cache_key)
        if cached and not data.get("forceRefresh") and time.time() - cached[0] < 300:
            return dict(cached[1])
        response = await self.client.rest.get(
            "/v1/loras/comfy", {"modelId": model_id} if model_id else {}
        )
        payload = response.get("data", {}) if isinstance(response, dict) else {}
        all_loras = payload.get("loras") if isinstance(payload.get("loras"), list) else []
        scoped = (
            [item for item in all_loras if model_id in (item.get("modelIds") or [])]
            if model_id
            else all_loras
        )
        models = payload.get("models")
        if not isinstance(models, list):
            models = sorted(
                {
                    candidate
                    for item in all_loras
                    if isinstance(item, dict)
                    for candidate in item.get("modelIds") or []
                    if isinstance(candidate, str)
                }
            )
        catalog = {
            "lastUpdated": payload.get("lastUpdated"),
            "loras": scoped,
            "models": models,
            "constraints": payload.get("constraints") or dict(_DEFAULT_LORA_CONSTRAINTS),
        }
        self._lora_catalog_cache[cache_key] = (time.time(), catalog)
        return dict(catalog)

    availableLoras = available_loras

    async def get_lora(self, lora_id: str) -> dict[str, Any] | None:
        catalog = await self.available_loras()
        return next((item for item in catalog["loras"] if item.get("loraId") == lora_id), None)

    getLora = get_lora

    async def supports_loras(self, model_id: str) -> bool:
        return model_id in (await self.available_loras())["models"]

    supportsLoras = supports_loras

    async def lora_constraints(self) -> dict[str, Any]:
        return dict((await self.available_loras())["constraints"])

    loraConstraints = lora_constraints

    def _project(self, project_id: str) -> Project | None:
        return next((item for item in self._projects if item.id == project_id), None)

    def _handle_job_state(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        kind = data.get("type")
        if kind == "queued":
            seconds = data.get("estimatedStartSeconds")
            estimated = (
                seconds
                if isinstance(seconds, (int, float))
                and not isinstance(seconds, bool)
                and math.isfinite(seconds)
                and seconds >= 0
                else None
            )
            queue_status = (
                data.get("queueStatus")
                if data.get("queueStatus") in {"waiting", "no-workers"}
                else None
            )
            event = {
                "type": "queued",
                "projectId": data.get("jobID"),
                "queuePosition": data.get("queuePosition"),
            }
            if estimated is not None:
                event["estimatedStartSeconds"] = estimated
            if queue_status is not None:
                event["queueStatus"] = queue_status
            self.emit("project", event)
        elif kind == "jobCompleted":
            self.emit(
                "project",
                {"type": "completed", "projectId": data.get("jobID")},
            )
        elif kind in {"initiatingModel", "jobStarted"}:
            event = {
                "type": "initiating" if kind == "initiatingModel" else "started",
                "projectId": data.get("jobID"),
                "jobId": data.get("imgID"),
                "workerName": data.get("workerName"),
                "positivePrompt": data.get("positivePrompt"),
                "negativePrompt": data.get("negativePrompt"),
                "jobIndex": data.get("jobIndex"),
            }
            if kind == "initiatingModel":
                event["preparation"] = data.get("preparation")
            self.emit("job", event)

        project = self._project(data.get("jobID", ""))
        if not project:
            return
        if kind == "queued":
            project._update(
                {
                    "status": "queued",
                    "queuePosition": data.get("queuePosition", -1),
                    "estimatedStartAt": (
                        _now() + timedelta(seconds=estimated) if estimated is not None else None
                    ),
                    "queueStatus": queue_status,
                }
            )
        elif kind == "jobCompleted":
            project._update({"status": "completed"})
            self._schedule_gc()
        elif kind in {"initiatingModel", "jobStarted"}:
            if project.estimated_start_at is not None or project.queue_status is not None:
                project._update({"estimatedStartAt": None, "queueStatus": None})
            job = project.job(data.get("imgID", "")) or project._add_job(
                {
                    "id": data.get("imgID"),
                    "projectId": project.id,
                    "status": "pending",
                    "step": 0,
                    "stepCount": project.params.get("steps", 0),
                }
            )
            job._update(
                {
                    "status": "initiating" if kind == "initiatingModel" else "processing",
                    "workerName": data.get("workerName"),
                    "positivePrompt": data.get("positivePrompt"),
                    "negativePrompt": data.get("negativePrompt"),
                    "jobIndex": data.get("jobIndex"),
                }
            )

    def _ensure_job(self, data: dict[str, Any]) -> tuple[Project | None, Job | None]:
        project = self._project(data.get("jobID", ""))
        if not project:
            return None, None
        job_id = data.get("imgID") or ""
        job = project.job(job_id) or project._add_job(
            {
                "id": job_id,
                "projectId": project.id,
                "status": "pending",
                "step": 0,
                "stepCount": project.params.get("steps", 0),
            }
        )
        return project, job

    def _handle_job_progress(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        event: dict[str, Any] = {
            "type": "progress",
            "projectId": data.get("jobID"),
            "jobId": data.get("imgID"),
        }
        for source, target in (
            ("step", "step"),
            ("stepCount", "stepCount"),
            ("progress", "progress"),
            ("etaSeconds", "etaSeconds"),
            ("etaMin", "etaMin"),
            ("etaMax", "etaMax"),
        ):
            value = data.get(source)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                event[target] = value
        self.emit("job", event)
        project, job = self._ensure_job(data)
        if not project or not job:
            return
        delta: dict[str, Any] = {"status": "processing"}
        if isinstance(data.get("step"), (int, float)):
            delta["step"] = max(job.step, data["step"])
        if isinstance(data.get("stepCount"), (int, float)):
            delta["stepCount"] = data["stepCount"]
        if isinstance(data.get("progress"), (int, float)) and not isinstance(
            data.get("progress"), bool
        ):
            delta["externalProgress"] = data["progress"]
        if (
            isinstance(data.get("etaMin"), (int, float))
            and not isinstance(data.get("etaMin"), bool)
            and isinstance(data.get("etaMax"), (int, float))
            and not isinstance(data.get("etaMax"), bool)
            and data["etaMax"] > 0
        ):
            delta["etaRange"] = {"min": data["etaMin"], "max": data["etaMax"]}
        job._update(delta)
        if project.status != "processing":
            project._update({"status": "processing"})
        if data.get("hasImage"):
            asyncio.create_task(self._load_preview(project, job))

    async def _load_preview(self, project: Project, job: Job) -> None:
        with contextlib.suppress(Exception):
            url = await self.download_url(job_id=project.id, image_id=job.id, type="preview")
            job._update({"previewUrl": url})
            self.emit(
                "job",
                {
                    "type": "preview",
                    "projectId": project.id,
                    "jobId": job.id,
                    "url": url,
                },
            )

    def _handle_job_eta(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        self.emit(
            "job",
            {
                "type": "jobETA",
                "projectId": data.get("jobID"),
                "jobId": data.get("imgID") or "",
                "etaSeconds": data.get("etaSeconds"),
            },
        )
        project, job = self._ensure_job(data)
        if not project or not job:
            return
        now = _now()
        eta = now + timedelta(seconds=float(data.get("etaSeconds", 0)))
        delta: dict[str, Any] = {"eta": eta, "etaSeconds": data.get("etaSeconds")}
        if not isinstance(job._data.get("etaStartedAt"), datetime):
            delta["etaStartedAt"] = now
        job._update(delta)
        project._keep_alive()
        project._update({"eta": max((item.eta for item in project.jobs if item.eta), default=None)})

    def _handle_job_result(self, data: Any) -> None:
        if isinstance(data, dict):
            asyncio.create_task(self._apply_job_result(data))

    async def _apply_job_result(self, data: dict[str, Any]) -> None:
        project = self._project(data.get("jobID", ""))
        _, job = self._ensure_job(data) if project is not None else (None, None)
        url = _raw_result_url(data)
        nsfw = bool(data.get("triggeredNSFWFilter"))
        canceled = bool(data.get("userCanceled"))
        pass_nsfw = not nsfw or project is None or project.params.get("disableNSFWFilter")
        if not url and pass_nsfw and not canceled:
            with contextlib.suppress(Exception):
                if project is not None and project.type in {"video", "audio"}:
                    download: dict[str, Any] = {
                        "jobId": project.id,
                        "id": data.get("imgID"),
                        "type": "complete",
                    }
                    if project.type == "audio":
                        download["contentType"] = (
                            job._audio_content_type if job is not None else "audio/mpeg"
                        )
                    url = await self.media_download_url(download)
                else:
                    download = {
                        "jobId": data.get("jobID"),
                        "imageId": data.get("imgID"),
                        "type": "complete",
                    }
                    if job is not None and job._image_content_type:
                        download["contentType"] = job._image_content_type
                    url = await self.download_url(download)
        steps = data.get("performedStepCount")
        if not isinstance(steps, (int, float)):
            steps = (job.step_count or job.step) if job is not None else None
        seed: int | None = None
        with contextlib.suppress(KeyError, TypeError, ValueError):
            seed = int(data["lastSeed"])
        if job is not None:
            delta: dict[str, Any] = {
                "status": "canceled" if canceled else "completed",
                "seed": seed if seed is not None else job.seed,
                "resultUrl": url,
                "isNSFW": nsfw,
                "userCanceled": canceled,
            }
            if isinstance(steps, (int, float)):
                delta["step"] = steps
            job._update(delta)
        event: dict[str, Any] = {
            "type": "completed",
            "projectId": data.get("jobID"),
            "jobId": data.get("imgID"),
            "resultUrl": url,
            "isNSFW": nsfw,
            "userCanceled": canceled,
        }
        if isinstance(steps, (int, float)):
            event["steps"] = steps
        if seed is not None:
            event["seed"] = seed
        self.emit("job", event)

    def _handle_job_error(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        symbolic = {
            "serverRestarting": 5001,
            "workerDisconnected": 5002,
            "jobTimedOut": 5003,
            "artistCanceled": 5004,
            "workerCancelled": 5005,
        }
        try:
            code = int(data.get("error"))
            error = {"code": code, "message": data.get("error_message")}
        except (TypeError, ValueError):
            original = str(data.get("error"))
            error = {
                "code": symbolic.get(original, 5000),
                "originalCode": original,
                "message": data.get("error_message"),
            }
        for key in ("subscriptionLimit", "requiredPlans", "feature", "limitation"):
            if data.get(key) is not None:
                error[key] = data[key]
        if not data.get("imgID"):
            self.emit(
                "project",
                {"type": "error", "projectId": data.get("jobID"), "error": error},
            )
        else:
            self.emit(
                "job",
                {
                    "type": "error",
                    "projectId": data.get("jobID"),
                    "jobId": data.get("imgID"),
                    "error": error,
                },
            )
        project = self._project(data.get("jobID", ""))
        if not project:
            return
        if not data.get("imgID"):
            project._update({"status": "failed", "error": error})
            return
        _, job = self._ensure_job(data)
        if not job:
            return
        job._update({"status": "failed", "error": error})
        all_started = len(project.jobs) >= project.params.get("numberOfMedia", 1)
        if project.params.get("numberOfMedia", 1) == 1 or (
            all_started and all(item.status == "failed" for item in project.jobs)
        ):
            project._update({"status": "failed", "error": error})

    def _handle_disconnect(self, _data: Any) -> None:
        self._set_available_models([])
        for project in self._projects:
            if not project.finished:
                project._update(
                    {"status": "failed", "error": {"code": 0, "message": "Server disconnected"}}
                )

    def _schedule_gc(self) -> None:
        async def collect() -> None:
            await asyncio.sleep(30)
            self._projects = [project for project in self._projects if not project.finished]

        asyncio.create_task(collect())
