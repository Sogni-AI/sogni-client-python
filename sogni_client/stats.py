"""Network statistics APIs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .transport import ApiClient


class StatsApi:
    """Leaderboard queries for artist, worker, volume, and referral stats."""

    def __init__(self, client: ApiClient) -> None:
        self.client = client

    async def leaderboard(self, params: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        query = dict(params or {})
        query.update(kwargs)
        response = await self.client.rest.get("/v1/leaderboard/", query)
        return response.get("data") if isinstance(response, Mapping) else None


__all__ = ["StatsApi"]
