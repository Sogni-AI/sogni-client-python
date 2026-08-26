"""Durable creative workflows and saved workflow templates.

The APIs in this module use the REST transport because workflow execution is
persisted server-side.  A caller can therefore submit a workflow, disconnect,
and later recover either its snapshot or its replayable SSE event stream.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import quote

from .errors import ApiError
from .transport import ApiClient, RestClient
from .utils import new_id, parse_sse_chunk

_MISSING = object()
_TERMINAL_WORKFLOW_STATUSES = {"completed", "failed", "cancelled"}


def _attribution_headers(
    client: Any, app_source: str | None, override: Any, operation_id: str
) -> dict[str, str]:
    builder = getattr(client, "attribution_headers", None)
    return builder(app_source, override, operation_id) if callable(builder) else {}


def _values(params: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(params or {})
    values.update(kwargs)
    return values


def _pick(values: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if name in values:
            return values[name]
    return default


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _workflow_data(response: Any) -> Mapping[str, Any]:
    if not isinstance(response, Mapping) or not isinstance(response.get("data"), Mapping):
        raise ValueError("Creative workflow response did not include data")
    return response["data"]


def _workflow_field(response: Any, key: str) -> Any:
    data = _workflow_data(response)
    if key not in data:
        raise ValueError(f"Creative workflow response did not include data.{key}")
    return data[key]


def _is_workflow_template(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("id"), str)
        and isinstance(value.get("name"), str)
    )


def _template_data(response: Any) -> Mapping[str, Any]:
    """Unwrap the current success envelope while accepting legacy responses."""

    if not isinstance(response, Mapping):
        return {}
    data = response.get("data")
    if response.get("status") == "success" and isinstance(data, Mapping):
        return data
    return response


def _required_template(data: Mapping[str, Any], operation: str) -> Mapping[str, Any]:
    template = data.get("template")
    if _is_workflow_template(template):
        return template
    operation_label = f" {operation}" if operation else ""
    raise ApiError(
        500,
        {
            "status": "error",
            "message": f"Workflow template{operation_label} response missing template field",
            "errorCode": 0,
        },
    )


def _assert_external_media(media_references: Any) -> None:
    """Durable runs may only persist retrievable HTTP(S) media references."""

    if not isinstance(media_references, list):
        return
    violations: list[str] = []
    for index, reference in enumerate(media_references):
        if not isinstance(reference, Mapping):
            continue
        url = reference.get("url")
        if isinstance(url, str) and url.strip():
            normalized = url.strip().lower()
            if normalized.startswith("data:") or not normalized.startswith(("http://", "https://")):
                violations.append(f"mediaReferences[{index}].url")
        for field in ("dataUri", "data_uri"):
            value = reference.get(field)
            if isinstance(value, str) and value.strip():
                violations.append(f"mediaReferences[{index}].{field}")
    if violations:
        fields = ", ".join(violations)
        raise ValueError(
            "Durable creative workflows do not support inline base64/data URI media. "
            "Upload media first and pass HTTP(S) URLs instead. "
            f"Offending field(s): {fields}"
        )


async def _sse_frames(
    rest: RestClient,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Turn the transport's decoded SSE lines into complete event frames."""

    lines: list[str] = []
    async for raw_line in rest.stream_lines(
        path,
        params=params,
        headers=headers,
        timeout=None,
    ):
        line = raw_line.removesuffix("\r")
        if line:
            lines.append(line)
            continue
        if not lines:
            continue
        for frame in parse_sse_chunk("\n".join(lines)):
            yield frame
        lines.clear()
    if lines:
        for frame in parse_sse_chunk("\n".join(lines)):
            yield frame


class CreativeWorkflowTemplatesApi:
    """CRUD and fork operations for saved, parameterized workflow recipes."""

    def __init__(self, client: ApiClient) -> None:
        self.client = client

    async def list(
        self,
        options: Mapping[str, Any] | None = None,
        *,
        visibility: str | None = None,
        offset: int | float | None = None,
        limit: int | float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        values = _values(options, kwargs)
        if visibility is not None:
            values["visibility"] = visibility
        if offset is not None:
            values["offset"] = offset
        if limit is not None:
            values["limit"] = limit

        query: dict[str, Any] = {}
        selected_visibility = _pick(values, "visibility", default=None)
        if selected_visibility:
            query["visibility"] = selected_visibility
        selected_offset = _pick(values, "offset", default=None)
        if _number(selected_offset) and selected_offset >= 0:
            query["offset"] = math.floor(selected_offset)
        selected_limit = _pick(values, "limit", default=None)
        if _number(selected_limit) and selected_limit > 0:
            query["limit"] = min(max(math.floor(selected_limit), 1), 200)

        response = await self.client.rest.get(
            "/v1/creative-agent/workflows/templates", query or None
        )
        data = _template_data(response)
        raw_templates = data.get("templates")
        templates = (
            [template for template in raw_templates if _is_workflow_template(template)]
            if isinstance(raw_templates, list)
            else []
        )
        next_value = data.get("next")
        next_cursor = next_value if _number(next_value) else None
        return {"templates": templates, "next_cursor": next_cursor}

    async def get(self, template_id: str) -> Mapping[str, Any]:
        response = await self.client.rest.get(
            f"/v1/creative-agent/workflows/templates/{quote(str(template_id), safe='')}"
        )
        return _required_template(_template_data(response), "")

    async def create(self, template: Mapping[str, Any]) -> Mapping[str, Any]:
        response = await self.client.rest.post(
            "/v1/creative-agent/workflows/templates", dict(template)
        )
        return _required_template(_template_data(response), "create")

    async def update(self, template_id: str, patch: Mapping[str, Any]) -> Mapping[str, Any]:
        response = await self.client.rest.patch(
            f"/v1/creative-agent/workflows/templates/{quote(str(template_id), safe='')}",
            dict(patch),
        )
        return _required_template(_template_data(response), "update")

    async def delete(self, template_id: str) -> None:
        await self.client.rest.delete(
            f"/v1/creative-agent/workflows/templates/{quote(str(template_id), safe='')}"
        )

    async def fork(
        self,
        template_id: str,
        body: Mapping[str, Any] | None = None,
        *,
        new_name: str | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        payload = dict(body or {})
        payload.update(kwargs)
        snake_name = payload.pop("new_name", _MISSING)
        if new_name is not None:
            payload["newName"] = new_name
        elif snake_name is not _MISSING and "newName" not in payload:
            payload["newName"] = snake_name
        response = await self.client.rest.post(
            f"/v1/creative-agent/workflows/templates/{quote(str(template_id), safe='')}/fork",
            payload,
        )
        return _required_template(_template_data(response), "fork")


class CreativeWorkflowsApi:
    """Durable, deterministic multi-step creative workflow operations."""

    def __init__(self, client: ApiClient) -> None:
        self.client = client
        self.templates = CreativeWorkflowTemplatesApi(client)

    async def start(
        self, params: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> Mapping[str, Any]:
        values = _values(params, kwargs)
        input_plan = _pick(values, "input", default=None)
        workflow_id = _pick(values, "workflow_id", "workflowId", default=None)
        if input_plan is None and not workflow_id:
            raise ValueError(
                "CreativeWorkflowsApi.start requires either `input` (inline plan) or "
                "`workflow_id` (saved template id)"
            )
        if input_plan is not None and workflow_id:
            raise ValueError(
                "CreativeWorkflowsApi.start accepts `input` or `workflow_id`, not both"
            )

        media_references = _pick(values, "media_references", "mediaReferences", default=_MISSING)
        if media_references is not _MISSING:
            _assert_external_media(media_references)

        body: dict[str, Any] = {}
        if input_plan is not None:
            body["input"] = input_plan
        if workflow_id:
            body["workflow_id"] = workflow_id
            inputs = _pick(values, "inputs", default=_MISSING)
            if inputs is not _MISSING and inputs is not None:
                body["inputs"] = inputs

        token_type = _pick(values, "token_type", "tokenType", default=None)
        billing_mode = _pick(values, "billing_mode", "billingMode", default=None)
        app_source = _pick(values, "app_source", "appSource", default=None)
        if app_source is None:
            app_source = self.client.app_source
        max_units = _pick(
            values,
            "max_estimated_capacity_units",
            "maxEstimatedCapacityUnits",
            default=_MISSING,
        )
        confirm_cost = _pick(values, "confirm_cost", "confirmCost", default=_MISSING)
        if token_type:
            body["token_type"] = token_type
        if billing_mode:
            body["billing_mode"] = billing_mode
        if app_source:
            body["app_source"] = app_source
        if max_units is not _MISSING and max_units is not None:
            body["max_estimated_capacity_units"] = max_units
        if confirm_cost is not _MISSING and confirm_cost is not None:
            body["confirm_cost"] = confirm_cost
        if media_references is not _MISSING and media_references is not None:
            body["media_references"] = media_references

        headers = _attribution_headers(
            self.client,
            app_source,
            _pick(values, "attribution", default=None),
            new_id(),
        )
        idempotency_key = _pick(values, "idempotency_key", "idempotencyKey", default=None)
        if idempotency_key:
            headers["Idempotency-Key"] = str(idempotency_key)
        response = await self.client.rest.post(
            "/v1/creative-agent/workflows", body, headers=headers or None
        )
        return _workflow_field(response, "workflow")

    async def resume(
        self,
        workflow_id: str,
        params: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        values = _values(params, kwargs)
        body = self._billing_body(values)
        app_source = _pick(values, "app_source", "appSource", default=None)
        if app_source is None:
            app_source = self.client.app_source
        response = await self.client.rest.post(
            f"/v1/creative-agent/workflows/{quote(str(workflow_id), safe='')}/resume",
            body,
            headers=_attribution_headers(
                self.client,
                app_source,
                _pick(values, "attribution", default=None),
                new_id(),
            ),
        )
        data = _workflow_data(response)
        if "workflow" not in data:
            raise ValueError("Creative workflow response did not include data.workflow")
        return {"workflow": data["workflow"], "resumed": data.get("resumed") is True}

    async def reseed(
        self,
        workflow_id: str,
        params: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        values = _values(params, kwargs)
        body = self._billing_body(values)
        app_source = _pick(values, "app_source", "appSource", default=None)
        if app_source is None:
            app_source = self.client.app_source
        seed_overrides = _pick(values, "seed_overrides", "seedOverrides", default=_MISSING)
        if seed_overrides is not _MISSING and seed_overrides is not None:
            body["seed_overrides"] = seed_overrides
        response = await self.client.rest.post(
            f"/v1/creative-agent/workflows/{quote(str(workflow_id), safe='')}/reseed",
            body,
            headers=_attribution_headers(
                self.client,
                app_source,
                _pick(values, "attribution", default=None),
                new_id(),
            ),
        )
        data = _workflow_data(response)
        if "workflow" not in data:
            raise ValueError("Creative workflow response did not include data.workflow")
        reseed = data.get("reseed") if isinstance(data.get("reseed"), Mapping) else {}
        cloned_from = reseed.get("cloned_from_run_id")
        steps = reseed.get("steps") if isinstance(reseed.get("steps"), list) else []
        return {
            "workflow": data["workflow"],
            "reseed": {
                "cloned_from_run_id": cloned_from if isinstance(cloned_from, str) else "",
                "steps": steps,
            },
        }

    async def list(
        self,
        options: Mapping[str, Any] | None = None,
        *,
        limit: int | float | None = None,
        offset: int | float | None = None,
        **kwargs: Any,
    ) -> list[Mapping[str, Any]]:
        values = _values(options, kwargs)
        if limit is not None:
            values["limit"] = limit
        if offset is not None:
            values["offset"] = offset
        query = {
            "limit": _pick(values, "limit", default=None),
            "offset": _pick(values, "offset", default=None),
        }
        response = await self.client.rest.get("/v1/creative-agent/workflows", query)
        return _workflow_field(response, "workflows")

    async def get(self, workflow_id: str) -> Mapping[str, Any]:
        response = await self.client.rest.get(
            f"/v1/creative-agent/workflows/{quote(str(workflow_id), safe='')}"
        )
        return _workflow_field(response, "workflow")

    async def events(self, workflow_id: str) -> list[Mapping[str, Any]]:
        response = await self.client.rest.get(
            f"/v1/creative-agent/workflows/{quote(str(workflow_id), safe='')}/events"
        )
        return _workflow_field(response, "events")

    async def cancel(self, workflow_id: str) -> Mapping[str, Any]:
        response = await self.client.rest.post(
            f"/v1/creative-agent/workflows/{quote(str(workflow_id), safe='')}/cancel",
            {},
        )
        return _workflow_field(response, "workflow")

    async def stream_events(
        self,
        workflow_id: str,
        options: Mapping[str, Any] | None = None,
        *,
        after: str | int | None = None,
        last_event_id: str | int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        values = _values(options, kwargs)
        if after is not None:
            values["after"] = after
        if last_event_id is not None:
            values["last_event_id"] = last_event_id
        selected_last_id = _pick(values, "last_event_id", "lastEventId", default=None)
        selected_after = _pick(values, "after", default=None)
        if selected_after is None:
            selected_after = selected_last_id
        headers = {"Accept": "text/event-stream"}
        if selected_last_id is not None:
            headers["Last-Event-ID"] = str(selected_last_id)
        path = f"/v1/creative-agent/workflows/{quote(str(workflow_id), safe='')}/events/stream"
        async for frame in _sse_frames(
            self.client.rest,
            path,
            params={"after": selected_after},
            headers=headers,
        ):
            yield frame
            data = frame.get("data")
            if isinstance(data, Mapping) and data.get("status") in _TERMINAL_WORKFLOW_STATUSES:
                return

    def _billing_body(self, values: Mapping[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {}
        token_type = _pick(values, "token_type", "tokenType", default=None)
        billing_mode = _pick(values, "billing_mode", "billingMode", default=None)
        app_source = _pick(values, "app_source", "appSource", default=None)
        if app_source is None:
            app_source = self.client.app_source
        if token_type:
            body["token_type"] = token_type
        if billing_mode:
            body["billing_mode"] = billing_mode
        if app_source:
            body["app_source"] = app_source
        return body

    streamEvents = stream_events


# Short aliases are convenient when composing the root client.
WorkflowsApi = CreativeWorkflowsApi
WorkflowTemplatesApi = CreativeWorkflowTemplatesApi


__all__ = [
    "CreativeWorkflowTemplatesApi",
    "CreativeWorkflowsApi",
    "WorkflowTemplatesApi",
    "WorkflowsApi",
]
