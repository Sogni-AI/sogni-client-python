from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sogni_client.errors import ApiError
from sogni_client.replay import ReplayApi
from sogni_client.stats import StatsApi
from sogni_client.workflows import CreativeWorkflowsApi


class FakeRest:
    def __init__(
        self, responses: list[Any] | None = None, stream_lines: list[str] | None = None
    ) -> None:
        self.responses = list(responses or [])
        self.lines = list(stream_lines or [])
        self.calls: list[dict[str, Any]] = []

    def _response(self) -> Any:
        if not self.responses:
            raise AssertionError("Unexpected REST call")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append({"method": "GET", "path": path, "params": params})
        return self._response()

    async def post(
        self,
        path: str,
        body: Any = None,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        self.calls.append(
            {
                "method": "POST",
                "path": path,
                "body": body,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self._response()

    async def patch(self, path: str, body: Any = None) -> Any:
        self.calls.append({"method": "PATCH", "path": path, "body": body})
        return self._response()

    async def delete(self, path: str) -> Any:
        self.calls.append({"method": "DELETE", "path": path})
        return self._response()

    async def stream_lines(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ):
        self.calls.append(
            {
                "method": "STREAM",
                "path": path,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        for line in self.lines:
            yield line


def fake_client(rest: FakeRest, *, app_source: str | None = "pytest-client") -> Any:
    return SimpleNamespace(rest=rest, app_source=app_source)


def attributed_client(rest: FakeRest) -> Any:
    return SimpleNamespace(
        rest=rest,
        app_source="pytest-client",
        attribution_headers=lambda app_source, override, operation_id: {
            "X-App-Source": app_source,
            "X-Sogni-Workload-Kind": override["workloadKind"],
            "X-Sogni-Operation-Id": operation_id,
        },
    )


@pytest.mark.asyncio
async def test_workflow_start_serializes_python_names_to_canonical_wire_shape() -> None:
    workflow = {"workflowId": "wf-1", "status": "queued"}
    rest = FakeRest([{"status": "success", "data": {"workflow": workflow}}])
    api = CreativeWorkflowsApi(fake_client(rest))
    media = [{"id": "input-1", "url": "https://cdn.example/image.png"}]

    result = await api.start(
        input={"title": "test", "steps": []},
        token_type="spark",
        billing_mode="subscription",
        max_estimated_capacity_units=42,
        confirm_cost=False,
        media_references=media,
        idempotency_key="request-123",
    )

    assert result is workflow
    assert rest.calls == [
        {
            "method": "POST",
            "path": "/v1/creative-agent/workflows",
            "body": {
                "input": {"title": "test", "steps": []},
                "token_type": "spark",
                "billing_mode": "subscription",
                "app_source": "pytest-client",
                "max_estimated_capacity_units": 42,
                "confirm_cost": False,
                "media_references": media,
            },
            "headers": {"Idempotency-Key": "request-123"},
            "timeout": None,
        }
    ]


@pytest.mark.asyncio
async def test_workflow_start_accepts_javascript_aliases_and_template_inputs() -> None:
    workflow = {"workflowId": "wf-from-template"}
    rest = FakeRest([{"data": {"workflow": workflow}}])
    api = CreativeWorkflowsApi(fake_client(rest, app_source="default-source"))

    result = await api.start(
        {
            "workflowId": "template-1",
            "inputs": {"prompt": "a glass robot"},
            "tokenType": "sogni",
            "billingMode": "tokens",
            "appSource": "explicit-source",
            "maxEstimatedCapacityUnits": 10,
            "confirmCost": True,
            "idempotencyKey": "camel-key",
        }
    )

    assert result is workflow
    assert rest.calls[0]["body"] == {
        "workflow_id": "template-1",
        "inputs": {"prompt": "a glass robot"},
        "token_type": "sogni",
        "billing_mode": "tokens",
        "app_source": "explicit-source",
        "max_estimated_capacity_units": 10,
        "confirm_cost": True,
    }
    assert rest.calls[0]["headers"] == {"Idempotency-Key": "camel-key"}


@pytest.mark.asyncio
async def test_workflow_start_combines_idempotency_and_attribution_headers() -> None:
    rest = FakeRest([{"data": {"workflow": {"workflowId": "wf-attributed"}}}])
    api = CreativeWorkflowsApi(attributed_client(rest))

    await api.start(
        input={"steps": []},
        idempotency_key="workflow-operation",
        attribution={"workloadKind": "agent_mediated"},
    )

    assert rest.calls[0]["headers"] == {
        "X-App-Source": "pytest-client",
        "X-Sogni-Workload-Kind": "agent_mediated",
        "X-Sogni-Operation-Id": rest.calls[0]["headers"]["X-Sogni-Operation-Id"],
        "Idempotency-Key": "workflow-operation",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({}, "requires either"),
        ({"input": {"steps": []}, "workflow_id": "template-1"}, "not both"),
    ],
)
async def test_workflow_start_requires_exactly_one_plan_source(
    params: dict[str, Any], message: str
) -> None:
    rest = FakeRest()
    api = CreativeWorkflowsApi(fake_client(rest))

    with pytest.raises(ValueError, match=message):
        await api.start(params)

    assert rest.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "media_references",
    [
        [{"url": "data:image/png;base64,AAAA"}],
        [{"url": "/local/file.png"}],
        [{"url": "file:///tmp/file.png"}],
        [{"url": "https://cdn.example/ok.png", "dataUri": "data:image/png;base64,AAAA"}],
        [{"data_uri": "data:image/png;base64,AAAA"}],
    ],
)
async def test_workflow_start_rejects_non_retrievable_durable_media(
    media_references: list[dict[str, str]],
) -> None:
    rest = FakeRest()
    api = CreativeWorkflowsApi(fake_client(rest))

    with pytest.raises(ValueError, match=r"HTTP\(S\) URLs"):
        await api.start(input={"steps": []}, media_references=media_references)

    assert rest.calls == []


@pytest.mark.asyncio
async def test_workflow_resume_and_reseed_use_encoded_ids_and_snake_wire_fields() -> None:
    resumed_workflow = {"workflowId": "wf / one", "status": "running"}
    reseeded_workflow = {"workflowId": "wf-2", "status": "queued"}
    rest = FakeRest(
        [
            {"data": {"workflow": resumed_workflow, "resumed": True}},
            {
                "data": {
                    "workflow": reseeded_workflow,
                    "reseed": {
                        "cloned_from_run_id": "wf / one",
                        "steps": [{"id": "image", "seed": 123}],
                    },
                }
            },
        ]
    )
    api = CreativeWorkflowsApi(fake_client(rest))

    resumed = await api.resume("wf / one", tokenType="spark", billingMode="auto")
    reseeded = await api.reseed("wf / one", seedOverrides={"image": 123}, appSource="reseed-test")

    assert resumed == {"workflow": resumed_workflow, "resumed": True}
    assert reseeded == {
        "workflow": reseeded_workflow,
        "reseed": {
            "cloned_from_run_id": "wf / one",
            "steps": [{"id": "image", "seed": 123}],
        },
    }
    assert rest.calls[0]["path"].endswith("/wf%20%2F%20one/resume")
    assert rest.calls[0]["body"] == {
        "token_type": "spark",
        "billing_mode": "auto",
        "app_source": "pytest-client",
    }
    assert rest.calls[1]["path"].endswith("/wf%20%2F%20one/reseed")
    assert rest.calls[1]["body"] == {
        "seed_overrides": {"image": 123},
        "app_source": "reseed-test",
    }


@pytest.mark.asyncio
async def test_workflow_list_get_events_and_cancel_unwrap_envelopes() -> None:
    workflows = [{"workflowId": "wf-1"}]
    workflow = workflows[0]
    events = [{"sequence": 1, "type": "workflow_started"}]
    rest = FakeRest(
        [
            {"data": {"workflows": workflows}},
            {"data": {"workflow": workflow}},
            {"data": {"events": events}},
            {"data": {"workflow": {**workflow, "status": "cancelled"}}},
        ]
    )
    api = CreativeWorkflowsApi(fake_client(rest))

    assert await api.list(limit=20, offset=5) is workflows
    assert await api.get("wf-1") is workflow
    assert await api.events("wf-1") is events
    assert await api.cancel("wf-1") == {"workflowId": "wf-1", "status": "cancelled"}
    assert rest.calls[0] == {
        "method": "GET",
        "path": "/v1/creative-agent/workflows",
        "params": {"limit": 20, "offset": 5},
    }
    assert rest.calls[-1]["body"] == {}


@pytest.mark.asyncio
async def test_workflow_methods_reject_malformed_success_envelopes() -> None:
    rest = FakeRest([{}, {"data": {}}, {"data": {}}])
    api = CreativeWorkflowsApi(fake_client(rest))

    with pytest.raises(ValueError, match="did not include data"):
        await api.list()
    with pytest.raises(ValueError, match=r"data\.workflow"):
        await api.resume("wf")
    with pytest.raises(ValueError, match=r"data\.workflow"):
        await api.reseed("wf")


@pytest.mark.asyncio
async def test_workflow_event_stream_sends_resume_state_and_stops_at_terminal_event() -> None:
    lines = [
        "id: 8",
        "event: workflow_event",
        'data: {"status":"running","step":"image"}',
        "",
        "id: 9",
        "event: workflow_event",
        'data: {"status":"completed"}',
        "",
        "id: 10",
        'data: {"status":"running","unexpected":true}',
        "",
    ]
    rest = FakeRest(stream_lines=lines)
    api = CreativeWorkflowsApi(fake_client(rest))

    frames = [frame async for frame in api.stream_events("wf / one", last_event_id=7)]

    assert [frame["id"] for frame in frames] == ["8", "9"]
    assert frames[-1]["data"] == {"status": "completed"}
    assert rest.calls == [
        {
            "method": "STREAM",
            "path": "/v1/creative-agent/workflows/wf%20%2F%20one/events/stream",
            "params": {"after": 7},
            "headers": {"Accept": "text/event-stream", "Last-Event-ID": "7"},
            "timeout": None,
        }
    ]


@pytest.mark.asyncio
async def test_workflow_event_stream_after_can_differ_from_last_event_header() -> None:
    rest = FakeRest(stream_lines=[])
    api = CreativeWorkflowsApi(fake_client(rest))

    assert [frame async for frame in api.streamEvents("wf", after=10, lastEventId=7)] == []
    assert rest.calls[0]["params"] == {"after": 10}
    assert rest.calls[0]["headers"]["Last-Event-ID"] == "7"


@pytest.mark.asyncio
async def test_template_list_clamps_pagination_and_filters_invalid_records() -> None:
    valid = {"id": "template-1", "name": "Image to video"}
    rest = FakeRest(
        [
            {
                "status": "success",
                "data": {
                    "templates": [valid, {"id": "missing-name"}, None],
                    "next": 200,
                },
            }
        ]
    )
    templates = CreativeWorkflowsApi(fake_client(rest)).templates

    result = await templates.list(visibility="public", offset=3.9, limit=999)

    assert result == {"templates": [valid], "next_cursor": 200}
    assert rest.calls[0]["params"] == {"visibility": "public", "offset": 3, "limit": 200}


@pytest.mark.asyncio
async def test_template_crud_and_fork_preserve_raw_templates() -> None:
    created = {"id": "new", "name": "Created"}
    updated = {"id": "template / 1", "name": "Updated"}
    forked = {"id": "fork", "name": "Copy"}
    rest = FakeRest(
        [
            {"status": "success", "data": {"template": created}},
            {"template": updated},
            {"deleted": True},
            {"status": "success", "data": {"template": forked}},
            {"status": "success", "data": {"template": updated}},
        ]
    )
    templates = CreativeWorkflowsApi(fake_client(rest)).templates

    assert await templates.create(created) is created
    assert await templates.update("template / 1", {"name": "Updated"}) is updated
    assert await templates.delete("template / 1") is None
    assert await templates.fork("template / 1", new_name="Copy") is forked
    assert await templates.get("template / 1") is updated

    assert rest.calls[1] == {
        "method": "PATCH",
        "path": "/v1/creative-agent/workflows/templates/template%20%2F%201",
        "body": {"name": "Updated"},
    }
    assert rest.calls[3]["body"] == {"newName": "Copy"}


@pytest.mark.asyncio
async def test_template_missing_required_fields_raises_typed_api_error() -> None:
    rest = FakeRest([{"status": "success", "data": {"template": {"id": "no-name"}}}])
    templates = CreativeWorkflowsApi(fake_client(rest)).templates

    with pytest.raises(ApiError) as raised:
        await templates.get("no-name")

    assert raised.value.status == 500
    assert "missing template field" in str(raised.value)


@pytest.mark.asyncio
async def test_replay_api_normalizes_wrapper_metadata_and_preserves_record_payloads() -> None:
    record = {"run_id": "run-1", "rounds": []}
    summary = {"runId": "run-1", "rounds": 0}
    rest = FakeRest(
        [
            {
                "runId": "run-1",
                "schemaVersion": "1",
                "redacted": True,
                "createTime": 100,
                "updateTime": 200,
            },
            {"records": [summary]},
            {"record": record, "createTime": 100},
        ]
    )
    replay = ReplayApi(fake_client(rest))

    assert await replay.write(record) == {
        "run_id": "run-1",
        "schema_version": "1",
        "redacted": True,
        "create_time": 100,
        "update_time": 200,
    }
    assert await replay.list(limit=10.8) == {"records": [summary]}
    assert await replay.get("run / 1") == {"record": record, "create_time": 100}
    assert rest.calls[1]["params"] == {"limit": 10}
    assert rest.calls[2]["path"] == "/v1/replay/records/run%20%2F%201"


@pytest.mark.asyncio
async def test_replay_api_rejects_missing_required_response_fields() -> None:
    rest = FakeRest([{}, {}])
    replay = ReplayApi(fake_client(rest))

    with pytest.raises(ApiError, match="missing runId"):
        await replay.write({"run_id": "run-1"})
    with pytest.raises(ApiError, match="missing record"):
        await replay.get("run-1")


@pytest.mark.asyncio
async def test_stats_leaderboard_merges_mapping_and_keyword_query() -> None:
    rows = [{"rank": 1, "username": "artist", "value": "42"}]
    rest = FakeRest([{"status": "success", "data": rows}])
    stats = StatsApi(fake_client(rest))

    result = await stats.leaderboard(
        {"type": "projectCompleteArtist", "period": "day", "page": 1},
        page=2,
        limit=10,
    )

    assert result is rows
    assert rest.calls == [
        {
            "method": "GET",
            "path": "/v1/leaderboard/",
            "params": {
                "type": "projectCompleteArtist",
                "period": "day",
                "page": 2,
                "limit": 10,
            },
        }
    ]
