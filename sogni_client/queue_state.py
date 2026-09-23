"""Normalize current server-provided queue descriptions."""

from __future__ import annotations

from typing import Any

_REASONS = frozenset(
    {"concurrency_limit", "model_concurrency_limit", "payment_pending", "no_workers", "queued"}
)
_OPTIONAL_VALUES = {
    "mediaType": {"video", "media"},
    "paymentModel": {"subscription", "paid_spark", "free_spark", "sogni"},
    "subscriptionTier": {"unlimited", "unlimited_pro"},
    "modelFamily": {"minimax_h3"},
}


def normalize_waiting_reason(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    reason, message = raw.get("reason"), raw.get("message")
    if (
        not isinstance(reason, str)
        or reason not in _REASONS
        or not isinstance(message, str)
        or not message.strip()
        or len(message) > 600
    ):
        return None
    result = {"reason": reason, "message": message}
    for key, allowed in _OPTIONAL_VALUES.items():
        value = raw.get(key)
        if isinstance(value, str) and value in allowed:
            result[key] = value
    return result


def normalize_job_waiting_reasons(raw: Any, number_of_media: int) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for entry in raw[:number_of_media]:
        if not isinstance(entry, dict):
            continue
        index = entry.get("jobIndex")
        if (
            not isinstance(index, (int, float))
            or isinstance(index, bool)
            or not 0 <= index < number_of_media
            or int(index) != index
            or index in seen
        ):
            continue
        reason = normalize_waiting_reason(entry.get("waitingReason"))
        if reason is None:
            continue
        index = int(index)
        seen.add(index)
        item = {"jobIndex": index, "waitingReason": reason}
        img_id = entry.get("imgID")
        if isinstance(img_id, str) and 0 < len(img_id) <= 128:
            item["imgID"] = img_id
        result.append(item)
    return result
