"""Helpers for rebuilding local project state from the server's recovery payloads.

The server hands back two views of what this app instance owns:
``authenticated.activeProjects`` / ``unclaimedCompletedProjects`` on the socket
handshake, and ``GET /api/v1/artist/projects/sync`` over REST. See
``docs/artist-project-recovery.md`` in sogni-socket for the wire contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .utils import b64_json_decode

#: ``originalCode`` of the error the SDK assigns when a remembered project is
#: gone from the server and the REST API has no record of it.
PROJECT_LOST_ORIGINAL_CODE = "projectLost"

PROJECT_LOST_ERROR: dict[str, Any] = {
    "code": 0,
    "originalCode": PROJECT_LOST_ORIGINAL_CODE,
    "message": (
        "The server has no record of this generation. It may have been "
        "interrupted by a restart - please try again."
    ),
}

_RAW_JOB_FINISHED = frozenset({"jobCompleted", "jobError"})


def is_project_lost_error(error: Mapping[str, Any] | None) -> bool:
    """True for the failure the SDK assigns when a project was not found on the server."""

    if not isinstance(error, Mapping):
        return False
    return error.get("originalCode") == PROJECT_LOST_ORIGINAL_CODE


isProjectLostError = is_project_lost_error


def decode_client_request_data(encoded: str | None) -> dict[str, Any] | None:
    """Decode the base64 JSON ``clientRequestData`` blob; ``None`` when absent or malformed."""

    if not encoded or not isinstance(encoded, str):
        return None
    try:
        parsed = b64_json_decode(encoded)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


decodeClientRequestData = decode_client_request_data


def is_llm_recovered_project(project: Mapping[str, Any]) -> bool:
    """LLM requests share the socket's project registry but are not media projects."""

    if not isinstance(project, Mapping):
        return False
    model = project.get("model")
    model_type = model.get("type") if isinstance(model, Mapping) else None
    return project.get("jobType") == "llm" or model_type == "llm"


isLLMRecoveredProject = is_llm_recovered_project


def media_type_from_recovered_project(project: Mapping[str, Any]) -> str:
    model = project.get("model") if isinstance(project, Mapping) else None
    kind = project.get("modelType") or (model.get("type") if isinstance(model, Mapping) else None)
    if kind == "video":
        return "video"
    if kind in {"audio", "music"}:
        return "audio"
    return "image"


mediaTypeFromRecoveredProject = media_type_from_recovered_project


def is_recovered_job_finished(status: str) -> bool:
    return status in _RAW_JOB_FINISHED


isRecoveredJobFinished = is_recovered_job_finished


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def project_params_from_recovered_project(project: Mapping[str, Any]) -> dict[str, Any]:
    """Best-effort reconstruction of the params a project was created with.

    Rebuilt from the server's copy of the original request. Enough for the SDK to
    route events, mint result URLs, gate NSFW handling and report the prompt;
    asset inputs (starting images, reference media) are not recoverable and are
    left out.
    """

    request = decode_client_request_data(project.get("clientRequestData")) or {}
    key_frames = request.get("keyFrames")
    key_frame = (
        key_frames[0]
        if isinstance(key_frames, list) and key_frames and isinstance(key_frames[0], dict)
        else {}
    )
    media_type = media_type_from_recovered_project(project)
    model = project.get("model") if isinstance(project.get("model"), Mapping) else {}

    image_count = project.get("imageCount")
    if _is_number(image_count) and image_count > 0:
        number_of_media = image_count
    else:
        try:
            number_of_media = int(request.get("numberOfImages") or 0)
        except (TypeError, ValueError):
            number_of_media = 0
    number_of_media = number_of_media or 1

    positive_prompt = key_frame.get("positivePrompt")
    base: dict[str, Any] = {
        "type": media_type,
        "modelId": key_frame.get("modelID") or model.get("id") or "",
        "numberOfMedia": number_of_media,
        "positivePrompt": positive_prompt if isinstance(positive_prompt, str) else "",
    }

    for source, target in (("negativePrompt", "negativePrompt"), ("stylePrompt", "stylePrompt")):
        value = key_frame.get(source)
        if isinstance(value, str) and value:
            base[target] = value

    steps = (
        project.get("stepCount") if _is_number(project.get("stepCount")) else key_frame.get("steps")
    )
    if _is_number(steps) and steps > 0:
        base["steps"] = steps
    if _is_number(key_frame.get("guidanceScale")):
        base["guidance"] = key_frame["guidanceScale"]
    if _is_number(key_frame.get("seed")):
        base["seed"] = key_frame["seed"]
    if isinstance(key_frame.get("loras"), list) and key_frame["loras"]:
        base["loras"] = key_frame["loras"]
    if isinstance(key_frame.get("loraStrengths"), list) and key_frame["loraStrengths"]:
        base["loraStrengths"] = key_frame["loraStrengths"]

    network = project.get("network") or request.get("network")
    if network in {"fast", "relaxed"}:
        base["network"] = network
    token_type = project.get("tokenType") or request.get("tokenType")
    if token_type:
        base["tokenType"] = token_type
    billing_mode = project.get("billingMode") or request.get("billingMode")
    if billing_mode:
        base["billingMode"] = billing_mode
    if request.get("disableSafety") is True:
        base["disableNSFWFilter"] = True
    output_format = request.get("outputFormat")
    if isinstance(output_format, str) and output_format:
        base["outputFormat"] = output_format
    app_source = project.get("appSource") or request.get("appSource")
    if app_source:
        base["appSource"] = app_source

    if media_type == "image":
        previews = (
            project.get("previewCount")
            if _is_number(project.get("previewCount"))
            else request.get("previews")
        )
        if _is_number(previews) and previews > 0:
            base["numberOfPreviews"] = previews
        size_preset = project.get("sizePreset")
        if size_preset and size_preset != "custom":
            base["sizePreset"] = size_preset
        for field in ("width", "height"):
            if _is_number(project.get(field)):
                base[field] = project[field]
    elif media_type == "video":
        for field in ("width", "height"):
            if _is_number(project.get(field)):
                base[field] = project[field]
        for field in ("frames", "fps", "duration"):
            if _is_number(key_frame.get(field)):
                base[field] = key_frame[field]
    elif _is_number(key_frame.get("duration")):
        base["duration"] = key_frame["duration"]

    return base


projectParamsFromRecoveredProject = project_params_from_recovered_project


__all__ = [
    "PROJECT_LOST_ERROR",
    "PROJECT_LOST_ORIGINAL_CODE",
    "decode_client_request_data",
    "decodeClientRequestData",
    "is_llm_recovered_project",
    "is_project_lost_error",
    "is_recovered_job_finished",
    "isLLMRecoveredProject",
    "isProjectLostError",
    "isRecoveredJobFinished",
    "media_type_from_recovered_project",
    "mediaTypeFromRecoveredProject",
    "project_params_from_recovered_project",
    "projectParamsFromRecoveredProject",
]
