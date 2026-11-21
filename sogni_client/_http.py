"""HTTP redirect handling shared by authenticated and caller-header requests."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx


async def request_with_redirects(
    client: httpx.AsyncClient, method: str, url: str, *, stream: bool = False, **kwargs: Any
) -> httpx.Response:
    request = client.build_request(method, url, **kwargs)
    hop = 0
    while True:
        # Only the initial request uses client auth; redirected requests already
        # carry the appropriate headers. An explicit no-op also skips URL auth.
        auth = {} if hop == 0 else {"auth": httpx.Auth()}
        response = await client.send(request, stream=stream, follow_redirects=False, **auth)
        if (
            response.status_code not in (301, 302, 303, 307, 308)
            or "location" not in response.headers
        ):
            return response
        await response.aclose()
        if hop == 5:
            raise httpx.TooManyRedirects("Too many redirects", request=request)
        request = response.request
        target = request.url.join(response.headers["location"])
        headers = request.headers.copy()
        current_origin = (request.url.scheme, request.url.host, request.url.port)
        target_origin = (target.scheme, target.host, target.port)
        if current_origin != target_origin:
            for name in ("api-key", "authorization", "cookie"):
                headers.pop(name, None)
        headers.pop("host", None)
        method = request.method
        content = request.content
        if (response.status_code == 303 and method != "HEAD") or (
            response.status_code in (301, 302) and method == "POST"
        ):
            method, content = "GET", b""
            for name in ("content-length", "transfer-encoding", "content-type"):
                headers.pop(name, None)
        request = httpx.Request(
            method, target, headers=headers, content=content, extensions=request.extensions
        )
        hop += 1


@asynccontextmanager
async def stream_with_redirects(
    client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
) -> AsyncIterator[httpx.Response]:
    response = await request_with_redirects(client, method, url, stream=True, **kwargs)
    try:
        yield response
    finally:
        await response.aclose()
