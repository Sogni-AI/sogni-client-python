"""Hosted world builds (``/v1/world-builds``).

sogni-api builds the paths, moments and collectibles a person chose for one
scene of their Sogni World in the background, with the account's own key. The
build starts from a signed-in session (an API key cannot author a world),
pauses for the quote before anything paid, and pauses again when finished takes
await review; publication stays in the World studio.

This mirrors ``src/Worlds/index.ts`` in the TypeScript SDK: the same REST
transport, the same bodies, and the same SSE iterator that skips
``run_status`` frames (the run snapshot, which ``get`` also returns).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import quote, urlencode

from .transport import ApiClient
from .utils import new_id, parse_sse_chunk

#: What a ``waiting_for_user`` build needs, on its ``waiting.reason`` field.
WORLD_BUILD_WAITING_REASONS = frozenset(
    {
        "cost_approval_required",
        "review_required",
        "insufficient_credit",
        "safety_review_required",
    }
)

_TERMINAL_BUILD_STATUSES = frozenset({"completed", "partial_failure", "failed", "cancelled"})


def _attribution_headers(
    client: Any, app_source: str | None, override: Any, operation_id: str
) -> dict[str, str]:
    builder = getattr(client, "attribution_headers", None)
    return builder(app_source, override, operation_id) if callable(builder) else {}


def _run_data(response: Any, key: str) -> Any:
    if not isinstance(response, Mapping) or not isinstance(response.get("data"), Mapping):
        raise ValueError("World build response did not include data")
    data = response["data"]
    if key not in data:
        raise ValueError(f"World build response did not include data.{key}")
    return data[key]


class WorldBuildsApi:
    """``client.worlds.builds``: start, follow, answer and review a hosted world build."""

    def __init__(self, client: ApiClient) -> None:
        self.client = client

    async def start(
        self,
        params: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Start a build. ``hotspots`` name each object as it appears, its kind
        (``path``, ``moment`` or ``collectible``), the action, and for a path
        where it leads. ``idempotency_key`` replays return the same run.
        """
        values = dict(params or {})
        values.update(kwargs)
        hotspots = values.get("hotspots")
        if not isinstance(hotspots, list):
            raise ValueError("World build start requires a list of hotspots")
        body: dict[str, Any] = {
            "worldId": values.get("world_id", values.get("worldId")),
            "nodeId": values.get("node_id", values.get("nodeId")),
            "hotspots": hotspots,
        }
        for key, wire in (
            ("look", "look"),
            ("audience", "audience"),
            ("token_type", "tokenType"),
            ("billing_mode", "billingMode"),
        ):
            value = values.get(key, values.get(wire))
            if value is not None:
                body[wire] = value
        if not body["worldId"] or not body["nodeId"]:
            raise ValueError("World build start requires world_id and node_id")
        app_source = getattr(self.client, "app_source", None)
        headers = _attribution_headers(self.client, app_source, values.get("attribution"), new_id())
        idempotency_key = values.get("idempotency_key", values.get("idempotencyKey"))
        if idempotency_key:
            headers["Idempotency-Key"] = str(idempotency_key)
        response = await self.client.rest.request(
            "POST", "/v1/world-builds", json_body=body, headers=headers
        )
        return _run_data(response, "run")

    async def list(
        self, world_id: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """The account's builds, newest first, optionally for one world (events left out)."""
        query: dict[str, str] = {}
        if world_id:
            query["worldId"] = world_id
        if limit is not None:
            query["limit"] = str(limit)
        suffix = f"?{urlencode(query)}" if query else ""
        response = await self.client.rest.request("GET", f"/v1/world-builds{suffix}")
        return list(_run_data(response, "runs"))

    async def get(self, run_id: str) -> dict[str, Any]:
        response = await self.client.rest.request(
            "GET", f"/v1/world-builds/{quote(run_id, safe='')}"
        )
        return _run_data(response, "run")

    async def events(self, run_id: str, after: int | None = None) -> list[dict[str, Any]]:
        suffix = f"?after={after}" if after is not None else ""
        response = await self.client.rest.request(
            "GET", f"/v1/world-builds/{quote(run_id, safe='')}/events{suffix}"
        )
        return list(_run_data(response, "events"))

    async def confirm_cost(self, run_id: str, decision: str) -> dict[str, Any]:
        """Answer the quote: ``confirm`` starts the paid work, ``cancel`` ends the
        run, ``requote`` asks for a fresh price (a quote is good for ten minutes).
        """
        if decision not in ("confirm", "cancel", "requote"):
            raise ValueError('decision must be "confirm", "cancel" or "requote"')
        response = await self.client.rest.request(
            "POST",
            f"/v1/world-builds/{quote(run_id, safe='')}/confirm-cost",
            json_body={"decision": decision},
        )
        return _run_data(response, "run")

    async def review(
        self, run_id: str, task_id: str, decision: str, note: str | None = None
    ) -> dict[str, Any]:
        """A verdict on a take in ``review``. A rejection with a note sends it
        back for one rewrite; a second rejection ends it.
        """
        if decision not in ("approved", "rejected"):
            raise ValueError('decision must be "approved" or "rejected"')
        body: dict[str, Any] = {"taskId": task_id, "decision": decision}
        if note:
            body["note"] = note
        response = await self.client.rest.request(
            "POST", f"/v1/world-builds/{quote(run_id, safe='')}/review", json_body=body
        )
        return _run_data(response, "run")

    async def cancel(self, run_id: str, reason: str | None = None) -> dict[str, Any]:
        response = await self.client.rest.request(
            "POST",
            f"/v1/world-builds/{quote(run_id, safe='')}/cancel",
            json_body={"reason": reason} if reason else {},
        )
        return _run_data(response, "run")

    async def stream_events(
        self, run_id: str, last_event_id: int | str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """SSE events, replayed after ``last_event_id`` and then live until the
        run ends. ``run_status`` frames are the run snapshot and are skipped.
        """
        headers = {"Accept": "text/event-stream"}
        if last_event_id is not None:
            headers["Last-Event-ID"] = str(last_event_id)
        buffer: list[str] = []

        def frames() -> list[dict[str, Any]]:
            parsed = parse_sse_chunk("\n".join(buffer))
            buffer.clear()
            return parsed

        async for line in self.client.rest.stream_lines(
            f"/v1/world-builds/{quote(run_id, safe='')}/events/stream",
            headers=headers,
            timeout=None,
        ):
            if line:
                buffer.append(line)
                continue
            for frame in frames():
                if frame["event"] == "run_status" or not isinstance(frame["data"], dict):
                    continue
                yield frame["data"]
        if buffer:
            for frame in frames():
                if frame["event"] == "run_status" or not isinstance(frame["data"], dict):
                    continue
                yield frame["data"]


class WorldsApi:
    """``client.worlds``: Sogni Worlds as the platform hosts them."""

    def __init__(self, client: ApiClient) -> None:
        self.client = client
        self.builds = WorldBuildsApi(client)


def is_world_build_terminal(status: Any) -> bool:
    return isinstance(status, str) and status in _TERMINAL_BUILD_STATUSES
