"""RunRecord ingestion and replay-viewer read APIs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from .errors import ApiError
from .transport import ApiClient


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _options(options: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(options or {})
    values.update(kwargs)
    return values


class ReplayApi:
    """Write, list, and retrieve replayable chat/harness run records."""

    def __init__(self, client: ApiClient) -> None:
        self.client = client

    async def write(self, record: Mapping[str, Any]) -> dict[str, Any]:
        body = await self.client.rest.post("/v1/replay/records", dict(record))
        if not isinstance(body, Mapping) or not isinstance(body.get("runId"), str):
            raise ApiError(
                500,
                {
                    "status": "error",
                    "message": "Replay write response missing runId",
                    "errorCode": 0,
                },
            )
        create_time = body.get("createTime")
        update_time = body.get("updateTime")
        schema_version = body.get("schemaVersion")
        return {
            "run_id": body["runId"],
            "schema_version": schema_version if schema_version is not None else 0,
            "redacted": body.get("redacted") is True,
            "create_time": create_time if _number(create_time) else 0,
            "update_time": update_time if _number(update_time) else 0,
        }

    async def list(
        self,
        options: Mapping[str, Any] | None = None,
        *,
        limit: int | float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        values = _options(options, kwargs)
        if limit is not None:
            values["limit"] = limit
        selected_limit = values.get("limit")
        params = (
            {"limit": math.floor(selected_limit)}
            if _number(selected_limit) and selected_limit > 0
            else None
        )
        body = await self.client.rest.get("/v1/replay/records", params)
        records = body.get("records") if isinstance(body, Mapping) else None
        return {"records": records if isinstance(records, list) else []}

    async def get(self, run_id: str) -> dict[str, Any]:
        body = await self.client.rest.get(f"/v1/replay/records/{quote(str(run_id), safe='')}")
        record = body.get("record") if isinstance(body, Mapping) else None
        if not isinstance(record, Mapping):
            raise ApiError(
                500,
                {
                    "status": "error",
                    "message": "Replay get response missing record field",
                    "errorCode": 0,
                },
            )
        create_time = body.get("createTime")
        return {
            "record": record,
            "create_time": create_time if _number(create_time) else 0,
        }


__all__ = ["ReplayApi"]
