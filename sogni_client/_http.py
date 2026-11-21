"""HTTP redirect handling shared by authenticated and caller-header requests."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

_CREDENTIAL_HEADERS = ("api-key", "authorization", "cookie")


def _port_or_default(url: httpx.URL) -> int | None:
    return url.port if url.port is not None else {"http": 80, "https": 443}.get(url.scheme)


def _keeps_credentials(current: httpx.URL, target: httpx.URL) -> bool:
    """Same origin, or an http -> https upgrade of the same host.

    The upgrade keeps credentials between the default ports (as httpx does for
    Authorization) or on the same port, the cases the JavaScript SDK keeps.
    """

    if current.host != target.host:
        return False
    current_port, target_port = _port_or_default(current), _port_or_default(target)
    if current.scheme == target.scheme:
        return current_port == target_port
    return (
        current.scheme == "http"
        and target.scheme == "https"
        and ((current_port, target_port) == (80, 443) or current_port == target_port)
    )


# The client methods this module or the code before it calls.
_DISPATCH_METHODS = ("build_request", "send", "request", "stream", "get", "post")


def uses_own_dispatch(client: Any) -> bool:
    """Whether a supplied client sends requests its own way.

    A client that is not an ``httpx.AsyncClient``, or a subclass that overrides
    how requests are sent (retries, logging, auth), keeps the calls the SDK made
    before redirect handling, so its overrides still run.
    """

    if not isinstance(client, httpx.AsyncClient):
        return True
    cls = type(client)
    return any(
        getattr(cls, name) is not getattr(httpx.AsyncClient, name) for name in _DISPATCH_METHODS
    )


async def request_with_redirects(
    client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
) -> httpx.Response:
    if uses_own_dispatch(client):
        return await client.request(method, url, **kwargs)
    return await _follow(client, method, url, stream=False, **kwargs)


async def get_with_redirects(client: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
    if uses_own_dispatch(client):
        return await client.get(url, **kwargs)
    return await _follow(client, "GET", url, stream=False, **kwargs)


async def _follow(
    client: httpx.AsyncClient, method: str, url: str, *, stream: bool, **kwargs: Any
) -> httpx.Response:
    request = client.build_request(method, url, **kwargs)
    hop = 0
    while True:
        # Only the initial request uses client auth; redirected requests already
        # carry the appropriate headers. An explicit no-op also skips URL auth.
        auth = {} if hop == 0 else {"auth": httpx.Auth()}
        response = await client.send(request, stream=stream, follow_redirects=False, **auth)
        # A client built with follow_redirects=False gets the redirect itself, as before.
        if (
            response.status_code not in (301, 302, 303, 307, 308)
            or "location" not in response.headers
            or not client.follow_redirects
        ):
            return response
        await response.aclose()
        if hop >= client.max_redirects:
            raise httpx.TooManyRedirects("Too many redirects", request=request)
        request = response.request
        target = request.url.join(response.headers["location"])
        if target.scheme not in ("http", "https"):
            raise httpx.UnsupportedProtocol(
                f"Request URL has an unsupported protocol '{target.scheme}://'.", request=request
            )
        headers = request.headers.copy()
        if not _keeps_credentials(request.url, target):
            for name in _CREDENTIAL_HEADERS:
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
    if uses_own_dispatch(client):
        async with client.stream(method, url, **kwargs) as response:
            yield response
        return
    response = await _follow(client, method, url, stream=True, **kwargs)
    try:
        yield response
    finally:
        await response.aclose()
