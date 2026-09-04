from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sogni_client.announcements import AnnouncementsApi


class FakeRest:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
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
        self.calls.append({"method": "POST", "path": path, "body": body})
        return self._response()


def make_api(responses: list[Any] | None = None) -> tuple[AnnouncementsApi, FakeRest]:
    rest = FakeRest(responses)
    return AnnouncementsApi(SimpleNamespace(rest=rest)), rest


@pytest.mark.asyncio
async def test_active_returns_announcements() -> None:
    api, rest = make_api([{"announcements": [{"id": "a1", "kind": "banner", "title": "Hi"}]}])

    result = await api.active()

    assert [item["id"] for item in result] == ["a1"]
    assert rest.calls[0] == {"method": "GET", "path": "/v1/announcements/active", "params": None}


@pytest.mark.asyncio
async def test_active_forwards_the_platform_so_scoped_announcements_match() -> None:
    api, rest = make_api([{"announcements": []}])

    await api.active("sogni-mac")

    assert rest.calls[0]["params"] == {"platform": "sogni-mac"}


@pytest.mark.asyncio
async def test_active_tolerates_a_missing_or_malformed_list() -> None:
    # An announcement read must never be the thing that breaks a client's
    # startup, so an unexpected envelope degrades to "nothing to show".
    api, _ = make_api([{}])
    assert await api.active() == []

    api, _ = make_api([{"announcements": "nope"}])
    assert await api.active() == []


@pytest.mark.asyncio
async def test_dismiss_posts_to_the_escaped_id() -> None:
    api, rest = make_api([{"dismissed": True, "id": "a/1"}])

    await api.dismiss("a/1")

    assert rest.calls[0]["method"] == "POST"
    assert rest.calls[0]["path"] == "/v1/announcements/a%2F1/dismiss"
