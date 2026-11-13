"""Normalize the pay-as-you-go offer and the Premium Spark switch reply.

An Unlimited plan project that has waited a while, at a moment when a card
outside the plan's worker pool could start it now, can be run on Premium Spark
instead, at its normal price. The server describes that in the optional
``payAsYouGo`` field of the ``projectQueue`` event; nothing is charged until the
owner calls :meth:`ProjectsApi.switch_to_premium_spark`.

A switch result is a dict with ``projectId`` and ``status``:

- ``"switched"``: the project runs on Premium Spark and its price was charged
  once (``paymentModel`` is ``"paid_spark"``).
- ``"refused"``: this request costs nothing; ``reason`` says why. A debit the
  server had already started (after ``"payment_failed"``, ``"no_idle_worker"`` or
  ``"started"``) is cancelled, and the balance can take a moment to show it, so
  tell the member they won't be charged rather than that they weren't. The server's
  reasons are in :data:`PAY_AS_YOU_GO_REFUSAL_REASONS`. ``"not_sent"`` is set by
  this client, never by the server: the request never left the client, so it
  says nothing about an earlier request.
- ``"unknown"``: the server may have switched and charged the project. Treat the
  payment as possibly made. ``reason`` is ``"timeout"``, ``"not_found"``,
  ``"session_changed"`` or ``"unrecognized_reply"``.

Both may carry ``message`` (plain text) and ``costInSpark`` (a decimal string).
"""

from __future__ import annotations

import re
from typing import Any

#: Reasons the server gives for refusing a switch. A refusal charges nothing.
#: ``not_found`` is not here: it can follow a request that did switch.
PAY_AS_YOU_GO_REFUSAL_REASONS = frozenset(
    {
        "not_eligible",
        "started",
        "no_idle_worker",
        "concurrency_limit",
        "insufficient_premium_spark",
        "insufficient_funds",
        "payment_failed",
        "staff_only",
        "unavailable",
    }
)

#: Refusals that end the offer for good, so the client clears it at once. Every
#: other outcome leaves the offer to the server's next ``projectQueue`` update.
PAY_AS_YOU_GO_REFUSALS_CLEARING_OFFER = frozenset({"started", "not_eligible"})

_DECIMAL = re.compile(r"\d+(\.\d+)?")


def _is_decimal_string(value: Any) -> bool:
    return isinstance(value, str) and len(value) <= 64 and _DECIMAL.fullmatch(value) is not None


def normalize_pay_as_you_go_offer(raw: Any) -> dict[str, Any] | None:
    """The offer as ``{"available": True, "costInSpark", "coveredByPremiumSpark"}``, or None."""
    if not isinstance(raw, dict):
        return None
    if raw.get("available") is not True or not _is_decimal_string(raw.get("costInSpark")):
        return None
    return {
        "available": True,
        "costInSpark": raw["costInSpark"],
        "coveredByPremiumSpark": raw.get("coveredByPremiumSpark") is True,
    }


def normalize_pay_as_you_go_switch_result(project_id: str, raw: Any) -> dict[str, Any]:
    """Map one ``projectPaymentSwitch`` reply to a switch result."""
    value = raw if isinstance(raw, dict) else {}
    extra: dict[str, Any] = {}
    message = value.get("message")
    if isinstance(message, str) and message.strip() and len(message) <= 600:
        extra["message"] = message
    if _is_decimal_string(value.get("costInSpark")):
        extra["costInSpark"] = value["costInSpark"]
    status, reason = value.get("status"), value.get("reason")
    if status == "switched":
        return {
            "projectId": project_id,
            "status": "switched",
            "paymentModel": "paid_spark",
            **extra,
        }
    if status == "refused" and reason == "not_found":
        # Its reply may have gone to a dropped socket before the project finished,
        # so not_found never proves nothing was charged.
        return {"projectId": project_id, "status": "unknown", "reason": "not_found", **extra}
    if status == "refused":
        return {
            "projectId": project_id,
            "status": "refused",
            "reason": reason if reason in PAY_AS_YOU_GO_REFUSAL_REASONS else "not_eligible",
            **extra,
        }
    return {"projectId": project_id, "status": "unknown", "reason": "unrecognized_reply", **extra}
