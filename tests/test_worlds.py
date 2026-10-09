"""``client.worlds.builds`` sends what sogni-api reads and parses what it streams,
the same way the TypeScript SDK's ``scripts/check-world-builds.cjs`` proves."""

from __future__ import annotations

from typing import Any

import pytest

from sogni_client.worlds import WORLD_BUILD_WAITING_REASONS, WorldsApi, is_world_build_terminal


class FakeRest:
    def __init__(
        self, responses: list[Any] | None = None, *, stream_lines: list[str] | None = None
    ) -> None:
        self.responses = list(responses or [])
        self.lines = list(stream_lines or [])
        self.calls: list[dict[str, Any]] = []

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        **_: Any,
    ) -> Any:
        self.calls.append({"method": method, "path": path, "body": json_body, "headers": headers})
        if not self.responses:
            raise AssertionError("Unexpected REST call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def stream_lines(self, path: str, *, headers: dict[str, str] | None = None, **_: Any):
        self.calls.append({"method": "STREAM", "path": path, "headers": headers})
        for line in self.lines:
            yield line


class FakeClient:
    app_source = "sogni-world"

    def __init__(self, rest: FakeRest) -> None:
        self.rest = rest

    def attribution_headers(
        self, app_source: str | None, override: Any, operation_id: str
    ) -> dict[str, str]:
        return {"X-App-Source": app_source or "", "X-Operation-Id": operation_id}


def make_api(
    responses: list[Any] | None = None, *, lines: list[str] | None = None
) -> tuple[WorldsApi, FakeRest]:
    rest = FakeRest(responses, stream_lines=lines)
    return WorldsApi(FakeClient(rest)), rest  # type: ignore[arg-type]


RUN = {"status": "success", "data": {"run": {"runId": "wbuild_1", "status": "queued"}}}


@pytest.mark.asyncio
async def test_start_sends_the_scene_hotspots_billing_and_idempotency_key() -> None:
    api, rest = make_api([RUN])
    hotspot = {
        "kind": "path",
        "object": "the door",
        "action": {"verb": "Open", "subject": "the door", "intent": "Step through"},
        "leadsTo": "Garden",
    }
    run = await api.builds.start(
        world_id="w",
        node_id="n",
        hotspots=[hotspot],
        look="painted",
        token_type="spark",
        billing_mode="subscription",
        idempotency_key="once",
    )
    assert run["runId"] == "wbuild_1"
    call = rest.calls[0]
    assert (call["method"], call["path"]) == ("POST", "/v1/world-builds")
    assert call["body"] == {
        "worldId": "w",
        "nodeId": "n",
        "hotspots": [hotspot],
        "look": "painted",
        "tokenType": "spark",
        "billingMode": "subscription",
    }
    assert call["headers"]["Idempotency-Key"] == "once"
    assert call["headers"]["X-App-Source"] == "sogni-world"
    with pytest.raises(ValueError):
        await api.builds.start(world_id="w", node_id="n", hotspots="door")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await api.builds.start(hotspots=[hotspot])


@pytest.mark.asyncio
async def test_list_get_and_events_address_the_run_by_its_encoded_id() -> None:
    api, rest = make_api(
        [
            {"status": "success", "data": {"runs": [{"runId": "a"}]}},
            {"status": "success", "data": {"run": {"runId": "wbuild / 1"}}},
            {"status": "success", "data": {"events": [{"sequence": 3, "type": "task_step"}]}},
        ]
    )
    assert await api.builds.list(world_id="w", limit=5) == [{"runId": "a"}]
    assert rest.calls[0]["path"] == "/v1/world-builds?worldId=w&limit=5"
    await api.builds.get("wbuild / 1")
    assert rest.calls[1]["path"] == "/v1/world-builds/wbuild%20%2F%201"
    assert await api.builds.events("wbuild_1", after=2) == [{"sequence": 3, "type": "task_step"}]
    assert rest.calls[2]["path"] == "/v1/world-builds/wbuild_1/events?after=2"


@pytest.mark.asyncio
async def test_decisions_are_forwarded_unchanged_and_validated() -> None:
    api, rest = make_api([RUN, RUN, RUN, RUN, RUN])
    await api.builds.confirm_cost("wbuild_1", "requote")
    assert rest.calls[0]["path"].endswith("/confirm-cost") and rest.calls[0]["body"] == {
        "decision": "requote"
    }
    await api.builds.review("wbuild_1", "door", "rejected", note="too dark")
    assert rest.calls[1]["path"].endswith("/review") and rest.calls[1]["body"] == {
        "taskId": "door",
        "decision": "rejected",
        "note": "too dark",
    }
    await api.builds.review("wbuild_1", "door", "approved")
    assert rest.calls[2]["body"] == {"taskId": "door", "decision": "approved"}
    await api.builds.cancel("wbuild_1", "changed my mind")
    assert rest.calls[3]["body"] == {"reason": "changed my mind"}
    await api.builds.cancel("wbuild_1")
    assert rest.calls[4]["body"] == {}
    with pytest.raises(ValueError):
        await api.builds.confirm_cost("wbuild_1", "maybe")
    with pytest.raises(ValueError):
        await api.builds.review("wbuild_1", "door", "meh")


@pytest.mark.asyncio
async def test_stream_events_replays_from_last_event_id_and_skips_run_status() -> None:
    api, rest = make_api(
        lines=[
            "id: 1",
            "event: run_started",
            'data: {"sequence": 1, "type": "run_started"}',
            "",
            ": keep-alive",
            "",
            "event: run_status",
            'data: {"runId": "wbuild_1", "status": "running"}',
            "",
            "id: 2",
            "event: task_ready",
            'data: {"sequence": 2, "type": "task_ready", "payload": {"taskId": "door"}}',
            "",
        ]
    )
    received = [event async for event in api.builds.stream_events("wbuild_1", last_event_id=0)]
    assert [(event["sequence"], event["type"]) for event in received] == [
        (1, "run_started"),
        (2, "task_ready"),
    ]
    assert rest.calls[0]["path"] == "/v1/world-builds/wbuild_1/events/stream"
    assert rest.calls[0]["headers"] == {"Accept": "text/event-stream", "Last-Event-ID": "0"}


def test_waiting_reasons_and_terminal_statuses_match_the_api() -> None:
    assert WORLD_BUILD_WAITING_REASONS == {
        "cost_approval_required",
        "review_required",
        "insufficient_credit",
        "safety_review_required",
    }
    assert is_world_build_terminal("partial_failure") and not is_world_build_terminal(
        "waiting_for_user"
    )
