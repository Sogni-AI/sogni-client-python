"""MiniMax H3 Control request validation and numbered video upload roles."""

from __future__ import annotations

import math
from typing import Any

MINIMAX_H3_CONTROL_MODEL_ID = "minimax-h3-ref2va-fp8_v2v_controlnet"
MINIMAX_H3_CONTROL_TYPES = (
    "canny",
    "depth",
    "pose",
    "hed",
    "mlsd",
    "scribble",
    "layout",
    "gray",
    "inpaint",
)
# Control types the worker can extract from the source video itself.
_AUTOMATIC_CONTROL_TYPES = ("canny", "depth", "pose", "hed", "mlsd")


def is_minimax_h3_control_model(model_id: str) -> bool:
    return model_id == MINIMAX_H3_CONTROL_MODEL_ID


def control_uploads(params: dict[str, Any]) -> list[tuple[str, Any]]:
    """The moving mask and prepared control video, with the numbered private
    video slots the server reads for this model."""

    return [
        (role, params[field])
        for role, field in (
            ("referenceVideo2", "referenceMaskVideo"),
            ("referenceVideo3", "controlVideo"),
        )
        if params.get(field)
    ]


def validate_minimax_h3_control(params: dict[str, Any]) -> None:
    """Refuse a MiniMax H3 Control request the server and worker would refuse,
    before anything is uploaded.

    Other models never read ``controlVideo``, ``referenceMaskVideo`` or the
    H3-only ``controlNet`` fields, so this returns at once for them and their
    requests are built exactly as before. Extra references, end frames, uploaded
    audio and keyframes are already refused for this model by
    ``_validate_video_assets``, the 40 steps by ``_validate_h3_params`` and
    ``controlNet.strength`` by the request builder. Mirrors
    ``validateMinimaxH3Control`` in the JS SDK.
    """

    if not is_minimax_h3_control_model(params["modelId"]):
        return
    control = params.get("controlNet") or {}
    name = control.get("name")
    if name is None:
        name = "depth"
    if name not in MINIMAX_H3_CONTROL_TYPES:
        raise ValueError("Choose a supported MiniMax H3 Control type.")
    if not params.get("referenceVideo") and not params.get("controlVideo"):
        raise ValueError("MiniMax H3 Control requires referenceVideo or a prepared controlVideo.")
    if params.get("referenceMask") and params.get("referenceMaskVideo"):
        raise ValueError("Choose one referenceMask image or referenceMaskVideo.")
    masked = bool(params.get("referenceMask") or params.get("referenceMaskVideo"))
    if masked and not params.get("referenceVideo"):
        raise ValueError("MiniMax H3 masked editing requires the source referenceVideo.")
    if name == "inpaint" and not masked:
        raise ValueError("MiniMax H3 inpaint requires referenceMask or referenceMaskVideo.")
    start = control.get("guidanceStart")
    end = control.get("guidanceEnd")
    for key, value in (("guidanceStart", start), ("guidanceEnd", end)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"controlNet.{key} must be from 0 to 1.")
    if (0 if start is None else start) > (1 if end is None else end):
        raise ValueError("controlNet.guidanceStart must not exceed guidanceEnd.")
    preprocess = control.get("preprocess")
    if preprocess is not None and not isinstance(preprocess, bool):
        raise ValueError("controlNet.preprocess must be a boolean.")
    if params.get("controlVideo") and preprocess is True:
        raise ValueError("controlVideo is a prepared map; omit preprocess or set it to false.")
    if preprocess and name not in _AUTOMATIC_CONTROL_TYPES:
        raise ValueError(f"{name} requires prepared control maps.")
