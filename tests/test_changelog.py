"""Transport checks for ``client.changelog`` (``/v1/changelog``).

Mirrors ``scripts/check-changelog-api.cjs`` in the TypeScript client: a client
that never signed in reads the public changelog without credentials, sends
every filter under its wire name, unwraps each response, and surfaces the API's
errors as ``ApiError``, including the 404 / errorCode 204 of an unknown entry.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

import sogni_client
from sogni_client import ApiError, ChangelogApi, SogniClient

BASE = "https://api.changelog.test"

ENTRY = {
    "id": "2026-11-10-sdk-js-javascript-sdk-5-80-0",
    "url": "https://www.sogni.ai/changelog/2026-11-10-sdk-js-javascript-sdk-5-80-0",
    "title": "JavaScript SDK 5.80.0",
    "summary": "projects: type and send opt-in SAM 3 video selection options",
    "bodyMarkdown": "### Features",
    "type": "release",
    "audiences": ["developer"],
    "products": ["sdk-js"],
    "version": "5.80.0",
    "links": [
        {
            "label": "GitHub release",
            "url": "https://github.com/Sogni-AI/sogni-client/releases/tag/v5.80.0",
        }
    ],
    "actionRequired": False,
    "effectiveAt": None,
    "publishedAt": "2026-11-10T12:00:00.000Z",
    "updatedAt": "2026-11-10T12:00:00.000Z",
}

PRODUCTS = [
    {"id": "create", "label": "Sogni Create", "audiences": ["artist"]},
    {"id": "sdk-python", "label": "Python SDK", "audiences": ["developer"]},
]


class Api:
    """Answers every request with the next queued response and records it."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.responses: list[httpx.Response] = []

    def respond(self, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
        self.responses.append(httpx.Response(status, json=payload, headers=headers))

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"Unexpected request {request.method} {request.url}")
        return self.responses.pop(0)


@pytest.fixture
async def changelog() -> AsyncIterator[tuple[ChangelogApi, Api]]:
    api = Api()
    async with httpx.AsyncClient(transport=httpx.MockTransport(api.handle)) as http:
        # A REST-only client that never signed in: the changelog needs no account.
        sdk = await SogniClient.create(disable_socket=True, rest_endpoint=BASE, http_client=http)
        assert sdk.api_client.auth.is_authenticated is False
        yield sdk.changelog, api
        await sdk.aclose()


def query(request: httpx.Request) -> dict[str, str]:
    return dict(request.url.params)


@pytest.mark.asyncio
async def test_list_reads_the_first_page_without_credentials_and_unwraps_data(
    changelog: tuple[ChangelogApi, Api],
) -> None:
    sdk, api = changelog
    api.respond(200, {"status": "success", "data": {"entries": [ENTRY], "nextCursor": "next-1"}})

    page = await sdk.list()

    assert page == {"entries": [ENTRY], "nextCursor": "next-1"}
    assert len(api.requests) == 1
    request = api.requests[0]
    assert request.method == "GET"
    assert str(request.url) == f"{BASE}/v1/changelog"
    for header in ("authorization", "api-key", "cookie"):
        assert header not in request.headers


@pytest.mark.asyncio
async def test_list_sends_every_filter_under_its_wire_name(
    changelog: tuple[ChangelogApi, Api],
) -> None:
    sdk, api = changelog
    api.respond(200, {"status": "success", "data": {"entries": [], "nextCursor": None}})

    page = await sdk.list(
        audience=["developer", "worker"],
        product="sdk-python",
        type=("release", "deprecated"),
        since="2026-11-01",
        until="2026-11-13T09:00:00-08:00",
        action_required=True,
        limit=3,
        cursor="eyJwIjoiMjAyNi0xMS0xMFQxMjowMDowMC4wMDBaIn0",
    )

    assert page == {"entries": [], "nextCursor": None}
    assert query(api.requests[0]) == {
        "audience": "developer,worker",
        "product": "sdk-python",
        "type": "release,deprecated",
        "since": "2026-11-01",
        "until": "2026-11-13T09:00:00-08:00",
        "actionRequired": "true",
        "limit": "3",
        "cursor": "eyJwIjoiMjAyNi0xMS0xMFQxMjowMDowMC4wMDBaIn0",
    }


@pytest.mark.asyncio
async def test_list_sends_single_values_as_given_and_leaves_action_required_out_unless_true(
    changelog: tuple[ChangelogApi, Api],
) -> None:
    sdk, api = changelog
    api.respond(200, {"status": "success", "data": {"entries": [], "nextCursor": None}})
    api.respond(200, {"status": "success", "data": {"entries": [], "nextCursor": None}})

    await sdk.list(audience="artist", product=["create", "api"], action_required=False)
    await sdk.list(since="2026-11-01T17:00:00+00:00")

    assert query(api.requests[0]) == {"audience": "artist", "product": "create,api"}
    # The offset sign survives encoding; a raw "+" would read as a space.
    assert query(api.requests[1]) == {"since": "2026-11-01T17:00:00+00:00"}
    assert "%2B00%3A00" in str(api.requests[1].url)


@pytest.mark.asyncio
async def test_get_reads_one_entry_by_its_encoded_id(
    changelog: tuple[ChangelogApi, Api],
) -> None:
    sdk, api = changelog
    api.respond(200, {"status": "success", "data": {"entry": ENTRY}})
    api.respond(200, {"status": "success", "data": {"entry": ENTRY}})

    assert await sdk.get(ENTRY["id"]) == ENTRY
    await sdk.get("a/b c")

    assert str(api.requests[0].url) == f"{BASE}/v1/changelog/{ENTRY['id']}"
    assert api.requests[1].url.raw_path == b"/v1/changelog/a%2Fb%20c"


@pytest.mark.asyncio
async def test_an_unknown_entry_surfaces_the_api_404_with_error_code_204(
    changelog: tuple[ChangelogApi, Api],
) -> None:
    sdk, api = changelog
    api.respond(404, {"status": "error", "errorCode": 204, "message": "Changelog entry not found"})

    with pytest.raises(ApiError) as raised:
        await sdk.get("does-not-exist")

    assert raised.value.status == 404
    assert raised.value.error_code == 204
    assert str(raised.value) == "Changelog entry not found"


@pytest.mark.asyncio
async def test_an_empty_id_reaches_the_list_not_an_entry_and_reads_as_not_found(
    changelog: tuple[ChangelogApi, Api],
) -> None:
    sdk, api = changelog
    api.respond(200, {"status": "success", "data": {"entries": [ENTRY], "nextCursor": None}})

    with pytest.raises(ApiError) as raised:
        await sdk.get("")

    assert raised.value.status == 404
    assert raised.value.error_code == 204
    assert str(api.requests[0].url) == f"{BASE}/v1/changelog/"


@pytest.mark.asyncio
async def test_products_reads_the_product_list(changelog: tuple[ChangelogApi, Api]) -> None:
    sdk, api = changelog
    api.respond(200, {"status": "success", "data": {"products": PRODUCTS}})

    assert await sdk.products() == PRODUCTS
    assert str(api.requests[0].url) == f"{BASE}/v1/changelog/products"


@pytest.mark.asyncio
async def test_the_api_message_status_and_retry_wait_survive_an_error(
    changelog: tuple[ChangelogApi, Api],
) -> None:
    sdk, api = changelog
    api.respond(
        400, {"status": "error", "errorCode": 102, "message": "limit must be a positive integer"}
    )
    api.respond(
        503,
        {
            "status": "error",
            "errorCode": 135,
            "message": "The changelog is temporarily unavailable",
        },
        {"Retry-After": "60"},
    )

    with pytest.raises(ApiError) as bad_limit:
        await sdk.list(limit=0)
    with pytest.raises(ApiError) as unavailable:
        await sdk.products()

    assert query(api.requests[0]) == {"limit": "0"}
    assert bad_limit.value.status == 400
    assert bad_limit.value.error_code == 102
    assert str(bad_limit.value) == "limit must be a positive integer"
    assert unavailable.value.status == 503
    assert unavailable.value.retry_after == 60


def test_the_changelog_api_and_types_are_exported_from_the_package() -> None:
    names: list[str] = [
        "ChangelogApi",
        "ChangelogAudience",
        "ChangelogEntry",
        "ChangelogEntryType",
        "ChangelogLink",
        "ChangelogProduct",
        "ListChangelogResult",
    ]
    for name in names:
        assert name in sogni_client.__all__
        assert getattr(sogni_client, name) is not None
    # The entry type names the same wire fields the API sends.
    assert set(sogni_client.ChangelogEntry.__annotations__) == set(ENTRY)
