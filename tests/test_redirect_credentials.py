from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import pytest

from sogni_client.auth import ApiKeyAuthManager
from sogni_client.errors import ApiError
from sogni_client.transport import RestClient


@asynccontextmanager
async def rest_client(handler, supplied, base_url="https://api.example", **options):
    auth = ApiKeyAuthManager()
    await auth.authenticate("example-key")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer example", "Cookie": "session=example"},
        cookies={"client-cookie": "example"},
        auth=httpx.BasicAuth("example", "example"),
        **{"follow_redirects": True, **options},
    ) as http:
        if supplied:
            rest = RestClient(base_url, auth, http_client=http)
        else:
            with patch("sogni_client.transport.httpx.AsyncClient", return_value=http):
                rest = RestClient(base_url, auth)
        yield rest


async def read(rest, stream, path="/start"):
    if stream:
        return [line async for line in rest.stream_lines(path)]
    return await rest.get(path)


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "target",
    [
        "/next",
        "https://other.example/next",
        "https://api.example:444/next",
        "http://api.example/next",
    ],
)
async def test_redirect_headers_follow_origin(supplied, stream, target):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(302, headers={"Location": target})
        return httpx.Response(200, text='{"ok":true}\n')

    async with rest_client(handle, supplied) as rest:
        if stream:
            assert [line async for line in rest.stream_lines("/start")] == ['{"ok":true}']
        else:
            assert await rest.get("/start") == {"ok": True}
    assert len(requests) == 2
    assert requests[0].headers["api-key"] == "example-key"
    for name in ("api-key", "authorization", "cookie"):
        if target == "/next":
            assert requests[1].headers[name] == requests[0].headers[name]
        else:
            assert name not in requests[1].headers


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("same_origin", [False, True])
async def test_redirect_userinfo_preserves_origin_headers(supplied, stream, same_origin):
    requests = []
    host = "api.example" if same_origin else "other.example"

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(308, headers={"Location": f"https://name:example@{host}/next"})
        return httpx.Response(200, text='{"ok":true}\n')

    async with rest_client(handle, supplied) as rest:
        if stream:
            assert [line async for line in rest.stream_lines("/start")] == ['{"ok":true}']
        else:
            assert await rest.get("/start") == {"ok": True}
    assert len(requests) == 2
    for name in ("api-key", "authorization", "cookie"):
        if same_origin:
            assert requests[1].headers[name] == requests[0].headers[name]
        else:
            assert name not in requests[1].headers


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_redirect_post_method_and_body(supplied, status):
    requests = []

    def handle(request):
        requests.append(request)
        return (
            httpx.Response(status, headers={"Location": "../next?new=1"})
            if len(requests) == 1
            else httpx.Response(200, json={})
        )

    async with rest_client(handle, supplied) as rest:
        await rest.request("POST", "/folder/start", json_body={"value": 1}, params={"old": "1"})
    assert str(requests[1].url) == "https://api.example/next?new=1"
    if status in (301, 302, 303):
        assert requests[1].method == "GET"
        assert requests[1].content == b""
        assert "content-length" not in requests[1].headers
    else:
        assert requests[1].method == "POST"
        assert requests[1].content == requests[0].content


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("limit", [None, 2])
async def test_redirect_limit_is_the_clients(supplied, stream, limit):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "/again"})

    options = {} if limit is None else {"max_redirects": limit}
    async with rest_client(handle, supplied, **options) as rest:
        with pytest.raises(httpx.TooManyRedirects, match="Too many redirects"):
            await read(rest, stream)
    # httpx's default is 20: that many redirects are followed, the next one fails.
    assert len(requests) == (20 if limit is None else limit) + 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "origin,target,kept",
    [
        ("http://api.example", "https://api.example/next", True),
        ("http://api.example", "https://api.example:443/next", True),
        ("http://api.example:8080", "https://api.example:8080/next", True),
        ("http://api.example:443", "https://api.example/next", True),
        ("http://api.example", "https://api.example:8443/next", False),
        ("http://api.example:8080", "https://api.example/next", False),
        ("http://api.example", "https://other.example/next", False),
        ("http://api.example", "http://api.example:8080/next", False),
        ("https://api.example", "http://api.example/next", False),
    ],
)
async def test_https_upgrade_keeps_credentials(stream, origin, target, kept):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(301, headers={"Location": target})
        return httpx.Response(200, text='{"ok":true}\n')

    async with rest_client(handle, True, base_url=origin) as rest:
        await read(rest, stream)
    assert len(requests) == 2
    for name in ("api-key", "authorization", "cookie"):
        if kept:
            assert requests[1].headers[name] == requests[0].headers[name]
        else:
            assert name not in requests[1].headers


@pytest.mark.parametrize("stream", [False, True])
async def test_client_without_redirects_gets_the_redirect(stream):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://other.example/next"})

    async with rest_client(handle, True, follow_redirects=False) as rest:
        with pytest.raises(ApiError) as caught:
            await read(rest, stream)
    assert caught.value.status == 302
    assert len(requests) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "target,scheme",
    [
        ("ftp://other.example/file", "ftp"),
        ("file:///etc/passwd", "file"),
        ("ws://api.example/socket", "ws"),
        ("wss://other.example/socket", "wss"),
    ],
)
async def test_redirect_to_other_protocols_is_refused(stream, target, scheme):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(307, headers={"Location": target})

    async with rest_client(handle, True) as rest:
        message = f"Request URL has an unsupported protocol '{scheme}://'."
        with pytest.raises(httpx.UnsupportedProtocol) as caught:
            await read(rest, stream)
    assert str(caught.value) == message
    assert len(requests) == 1


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_websocket_does_not_follow_redirect(status):
    from websockets.asyncio.server import serve
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    from sogni_client.transport import WebSocketClient

    seen = []

    async def handler(connection):
        await connection.close()

    def destination(connection, request):
        seen.append(request)

    auth = ApiKeyAuthManager()
    await auth.authenticate("example-key")
    async with serve(handler, "127.0.0.1", 0, process_request=destination) as target:
        port = target.sockets[0].getsockname()[1]

        def redirect(connection, request):
            return Response(
                status,
                "Redirect",
                Headers({"Location": f"ws://127.0.0.1:{port}/next", "Content-Length": "0"}),
            )

        async with serve(handler, "127.0.0.1", 0, process_request=redirect) as origin:
            origin_port = origin.sockets[0].getsockname()[1]
            socket = WebSocketClient(f"ws://127.0.0.1:{origin_port}", auth, "test", "fast")
            try:
                with pytest.raises(InvalidStatus):
                    await socket.connect()
                assert seen == []
            finally:
                await socket.aclose()


@pytest.mark.parametrize("ranged", [False, True])
async def test_media_caller_headers_follow_origin(ranged):
    from sogni_client import SogniClient
    from sogni_client.errors import MediaEncryptionError

    requests = []

    def handle(request):
        requests.append(request)
        if request.url.host == "api.example":
            return httpx.Response(302, headers={"Location": "https://media.example/file"})
        return httpx.Response(200, content=b"plain bytes")

    async with rest_client(handle, True) as rest:
        client = await SogniClient.create(disable_socket=True, http_client=rest._client)
        headers = {
            "api-key": "example",
            "Authorization": "Bearer example",
            "Cookie": "session=example",
        }
        try:
            if ranged:
                with pytest.raises(MediaEncryptionError):
                    await client.media_encryption.open_decrypted(
                        "https://api.example/file", headers=headers
                    )
            else:
                assert (
                    await client.media_encryption.fetch_decrypted(
                        "https://api.example/file", headers=headers
                    )
                    == b"plain bytes"
                )
            assert len(requests) == 2
            for name in ("api-key", "authorization", "cookie"):
                assert name not in requests[1].headers
        finally:
            await client.aclose()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_refresh_stays_on_configured_endpoint(status):
    import time

    from sogni_client.auth import TokenAuthManager

    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(status, headers={"Location": "https://other.example/refresh"})
        return httpx.Response(400, json={"message": "No token pair"})

    async with rest_client(handle, True) as rest:
        auth = TokenAuthManager("https://api.example", refresh_client=rest._client)
        auth._refresh_token = "example-refresh"
        auth._refresh_expires_at = time.time() + 3600
        with pytest.raises(ApiError):
            await auth.headers()
    assert len(requests) == 1


@pytest.mark.parametrize("supplied", [False, True])
async def test_credentials_stay_removed_when_redirect_returns(supplied):
    requests = []
    targets = ["https://other.example/next", "https://api.example/back", "/end"]

    def handle(request):
        requests.append(request)
        if targets:
            return httpx.Response(
                302, headers={"Location": targets.pop(0), "Set-Cookie": "new=example; Path=/"}
            )
        return httpx.Response(200, json={})

    async with rest_client(handle, supplied) as rest:
        await rest.get("/start")
    for request in requests[1:]:
        assert not {"api-key", "authorization", "cookie"}.intersection(request.headers)


async def test_default_port_is_same_origin():
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(302, headers={"Location": "https://API.example:443/end"})
        return httpx.Response(200, json={})

    async with rest_client(handle, True) as rest:
        await rest.get("/start")
    assert requests[1].headers["api-key"] == "example-key"


async def test_client_redirect_limit_is_allowed():
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) <= 20:
            return httpx.Response(302, headers={"Location": "/next"})
        return httpx.Response(200, json={})

    async with rest_client(handle, True) as rest:
        assert await rest.get("/start") == {}
    assert len(requests) == 21


async def test_stream_remains_lazy_and_closes_responses():
    class Body(httpx.AsyncByteStream):
        def __init__(self):
            self.read = False
            self.closed = False

        async def __aiter__(self):
            self.read = True
            yield b"first\n"
            raise AssertionError("The stream should stop after the first line")

        async def aclose(self):
            self.closed = True

    redirect_body, final_body = Body(), Body()

    def handle(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/end"}, stream=redirect_body)
        return httpx.Response(200, stream=final_body)

    async with rest_client(handle, True) as rest:
        lines = rest.stream_lines("/start")
        assert await anext(lines) == "first"
        await lines.aclose()
    assert not redirect_body.read
    assert redirect_body.closed and final_body.closed
