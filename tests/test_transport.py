from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from sogni_client.auth import ApiKeyAuthManager
from sogni_client.errors import ApiError
from sogni_client.transport import SWITCH_CONNECTION, ApiClient, RestClient, WebSocketClient
from sogni_client.utils import b64_json_decode, b64_json_encode


class FakeHttpClient:
    def __init__(self, responses: list[httpx.Response] | None = None) -> None:
        self.responses = list(responses or [])
        self.requests: list[dict[str, Any]] = []
        self.puts: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []
        self.closed = False

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"Unexpected request: {method} {url}")
        return self.responses.pop(0)

    async def put(self, url: str, **kwargs: Any) -> httpx.Response:
        self.puts.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"Unexpected PUT: {url}")
        return self.responses.pop(0)

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.gets.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"Unexpected GET: {url}")
        return self.responses.pop(0)

    async def aclose(self) -> None:
        self.closed = True


class FakeSocket:
    def __init__(self) -> None:
        self.messages: asyncio.Queue[str | bytes] = asyncio.Queue()
        self.sent: list[str] = []
        self.closed: tuple[int, str] | None = None

    async def recv(self) -> str | bytes:
        return await self.messages.get()

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def close(self, *, code: int, reason: str) -> None:
        self.closed = (code, reason)


class FakeSocketFactory:
    def __init__(self, socket: FakeSocket | None = None) -> None:
        self.socket = socket or FakeSocket()
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, url: str, **kwargs: Any) -> FakeSocket:
        self.calls.append({"url": url, **kwargs})
        return self.socket


def response(status: int, *, json_body: Any = None, text: str | None = None) -> httpx.Response:
    kwargs: dict[str, Any] = {"request": httpx.Request("GET", "https://api.sogni.ai/test")}
    if text is not None:
        kwargs["text"] = text
    elif json_body is not None:
        kwargs["json"] = json_body
    return httpx.Response(status, **kwargs)


@pytest.mark.asyncio
async def test_rest_client_serializes_auth_query_and_nested_json_without_none() -> None:
    auth = ApiKeyAuthManager()
    await auth.authenticate("secret")
    fake = FakeHttpClient([response(200, json_body={"status": "success", "data": {"ok": True}})])
    client = RestClient("https://api.sogni.ai/base/", auth, http_client=fake, timeout=12)

    result = await client.request(
        "POST",
        "/v1/example",
        params={"keep": 0, "drop": None},
        json_body={"keep": False, "drop": None, "nested": {"drop": None, "keep": 1}},
        headers={"X-Test": "yes"},
    )

    assert result == {"status": "success", "data": {"ok": True}}
    assert fake.requests == [
        {
            "method": "POST",
            "url": "https://api.sogni.ai/base/v1/example",
            "params": {"keep": 0},
            "json": {"keep": False, "nested": {"keep": 1}},
            "content": None,
            "headers": {"api-key": "secret", "X-Test": "yes"},
            "timeout": 12,
        }
    ]


@pytest.mark.asyncio
async def test_rest_client_explicit_headers_can_override_auth_header() -> None:
    auth = ApiKeyAuthManager()
    await auth.authenticate("default")
    fake = FakeHttpClient([response(200, json_body={})])
    client = RestClient("https://api.sogni.ai", auth, http_client=fake)

    await client.request("GET", "/test", headers={"api-key": "per-request"})

    assert fake.requests[0]["headers"]["api-key"] == "per-request"


@pytest.mark.asyncio
async def test_rest_client_non_json_error_preserves_status_and_body_excerpt() -> None:
    auth = ApiKeyAuthManager()
    fake = FakeHttpClient([response(502, text="<html> upstream   unavailable </html>")])
    client = RestClient("https://api.sogni.ai", auth, http_client=fake)

    with pytest.raises(ApiError) as raised:
        await client.get("/gateway")

    assert raised.value.status == 502
    assert raised.value.error_code == 502
    assert "upstream unavailable" in str(raised.value)


@pytest.mark.asyncio
async def test_rest_client_clears_authentication_on_401_before_raising() -> None:
    auth = ApiKeyAuthManager()
    await auth.authenticate("secret")
    fake = FakeHttpClient(
        [response(401, json_body={"status": "error", "message": "Unauthorized", "errorCode": 1})]
    )
    client = RestClient("https://api.sogni.ai", auth, http_client=fake)

    with pytest.raises(ApiError):
        await client.get("/private")

    assert auth.is_authenticated is False


@pytest.mark.asyncio
async def test_rest_client_rejects_non_json_success_and_accepts_empty_success() -> None:
    auth = ApiKeyAuthManager()
    fake = FakeHttpClient([response(200, text="not json"), response(204, text="")])
    client = RestClient("https://api.sogni.ai", auth, http_client=fake)

    with pytest.raises(ValueError, match=r"HTTP 200"):
        await client.get("/invalid")
    assert await client.get("/empty") is None


@pytest.mark.asyncio
async def test_rest_client_presigned_upload_and_download_helpers() -> None:
    auth = ApiKeyAuthManager()
    fake = FakeHttpClient(
        [
            response(200, text=""),
            httpx.Response(
                200,
                content=b"result bytes",
                request=httpx.Request("GET", "https://cdn.example/result"),
            ),
        ]
    )
    client = RestClient("https://api.sogni.ai", auth, http_client=fake)

    await client.put_bytes("https://upload.example/signed", b"image", content_type="image/png")
    data = await client.get_bytes("https://cdn.example/result")

    assert fake.puts[0]["headers"] == {"Content-Type": "image/png"}
    assert fake.puts[0]["content"] == b"image"
    assert data == b"result bytes"


@pytest.mark.asyncio
async def test_websocket_connect_builds_protocol_query_and_exact_auth_headers() -> None:
    auth = ApiKeyAuthManager()
    await auth.authenticate("socket-secret")
    factory = FakeSocketFactory()
    socket = WebSocketClient(
        "https://socket.sogni.ai/connect?existing=1",
        auth,
        "APP-123",
        "fast",
        app_source="  pytest  ",
        socket_event_subscriptions={"modelAvailability": False, "ignored": "no"},
        connect_factory=factory,
    )

    await socket.connect()
    try:
        call = factory.calls[0]
        query = parse_qs(urlsplit(call["url"]).query, keep_blank_values=True)
        assert urlsplit(call["url"]).scheme == "wss"
        assert query["existing"] == ["1"]
        assert query["appId"] == ["APP-123"]
        assert query["appSource"] == ["pytest"]
        assert query["clientType"] == ["artist"]
        assert query["forceWorkerId"] == ["fast"]
        assert json.loads(query["socketEventSubscriptions"][0]) == {"modelAvailability": False}
        assert query["clientName"] == ["Sogni/3.0.0 (sogni-client) 5.21.3"]
        assert call["additional_headers"] == {"api-key": "socket-secret"}
        assert call["ping_interval"] == call["ping_timeout"] == 15
        assert call["max_size"] is None
    finally:
        await socket.aclose()


@pytest.mark.asyncio
async def test_websocket_connect_serializes_connection_attribution_query() -> None:
    auth = ApiKeyAuthManager()
    await auth.authenticate("socket-secret")
    factory = FakeSocketFactory()
    socket = WebSocketClient(
        "wss://socket.sogni.ai/connect",
        auth,
        "APP-123",
        "fast",
        connection_attribution={
            "interaction_kind": "external_agent",
            "agent_framework": "codex",
            "agent_surface": "sdk",
            "execution_mode": "server",
        },
        connect_factory=factory,
    )

    await socket.connect()
    try:
        query = parse_qs(urlsplit(factory.calls[0]["url"]).query)
        assert query["interactionKind"] == ["external_agent"]
        assert query["agentFramework"] == ["codex"]
        assert query["agentSurface"] == ["sdk"]
        assert query["executionMode"] == ["server"]
    finally:
        await socket.aclose()


@pytest.mark.asyncio
async def test_relaxed_websocket_connection_sends_blank_force_worker_id() -> None:
    auth = ApiKeyAuthManager()
    await auth.authenticate("key")
    factory = FakeSocketFactory()
    socket = WebSocketClient("ws://localhost:9000", auth, "APP", "relaxed", connect_factory=factory)

    await socket.connect()
    try:
        query = parse_qs(urlsplit(factory.calls[0]["url"]).query, keep_blank_values=True)
        assert query["forceWorkerId"] == [""]
    finally:
        await socket.aclose()


@pytest.mark.asyncio
async def test_websocket_send_uses_base64_json_envelope() -> None:
    auth = ApiKeyAuthManager()
    await auth.authenticate("key")
    factory = FakeSocketFactory()
    client = WebSocketClient("wss://socket.sogni.ai", auth, "APP", "fast", connect_factory=factory)

    await client.send("jobRequest", {"jobID": "ABC", "prompt": "moon 🌙"})
    try:
        envelope = json.loads(factory.socket.sent[0])
        assert envelope["type"] == "jobRequest"
        assert b64_json_decode(envelope["data"]) == {"jobID": "ABC", "prompt": "moon 🌙"}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_websocket_reader_decodes_payload_and_uppercases_job_identifiers() -> None:
    auth = ApiKeyAuthManager()
    await auth.authenticate("key")
    factory = FakeSocketFactory()
    client = WebSocketClient("wss://socket.sogni.ai", auth, "APP", "fast", connect_factory=factory)
    received: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
    client.on("jobProgress", lambda data: received.set_result(data))
    await client.connect()
    await factory.socket.messages.put(
        json.dumps(
            {
                "type": "jobProgress",
                "data": b64_json_encode({"jobID": "abc-def", "imgID": "xyz", "step": 2}),
            }
        ).encode()
    )

    try:
        assert await asyncio.wait_for(received, 1) == {
            "jobID": "ABC-DEF",
            "imgID": "XYZ",
            "step": 2,
        }
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_websocket_reader_drops_malformed_frame_without_disconnecting() -> None:
    auth = ApiKeyAuthManager()
    await auth.authenticate("key")
    factory = FakeSocketFactory()
    client = WebSocketClient("wss://socket.sogni.ai", auth, "APP", "fast", connect_factory=factory)
    received: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
    disconnected: list[Any] = []
    client.on("jobProgress", lambda data: received.set_result(data))
    client.on("disconnected", disconnected.append)
    await client.connect()
    await factory.socket.messages.put('{"type":"jobProgress","data":"not-base64"}')
    await factory.socket.messages.put(
        json.dumps(
            {
                "type": "jobProgress",
                "data": b64_json_encode({"jobID": "valid", "step": 3}),
            }
        )
    )

    try:
        assert await asyncio.wait_for(received, 1) == {"jobID": "VALID", "step": 3}
        assert disconnected == []
        assert client.is_connected is True
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_socket_subscription_update_supports_shorthand_and_full_shape() -> None:
    auth = ApiKeyAuthManager()
    await auth.authenticate("key")
    factory = FakeSocketFactory()
    client = WebSocketClient("wss://socket.sogni.ai", auth, "APP", "fast", connect_factory=factory)

    await client.set_socket_event_subscriptions({"modelAvailability": False})
    await client.setSocketEventSubscriptions({"reset": True, "subscribe": ["swarmModels"]})
    try:
        first, second = [json.loads(message) for message in factory.socket.sent]
        assert b64_json_decode(first["data"]) == {"subscriptions": {"modelAvailability": False}}
        assert b64_json_decode(second["data"]) == {
            "reset": True,
            "subscribe": ["swarmModels"],
        }
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_api_client_duplicate_app_id_disconnect_preserves_api_key() -> None:
    fake_http = FakeHttpClient()
    client = ApiClient(
        base_url="https://api.sogni.ai",
        socket_url="wss://socket.sogni.ai",
        app_id="APP",
        network="fast",
        auth_type="apiKey",
        disable_socket=True,
        http_client=fake_http,
        socket_http_client=fake_http,
    )
    await client.auth.authenticate("secret")

    client._on_socket_disconnected({"code": SWITCH_CONNECTION, "reason": "duplicate"})

    assert client.auth.is_authenticated is True
    await client.aclose()


@pytest.mark.asyncio
async def test_api_client_nonrecoverable_socket_error_clears_auth() -> None:
    fake_http = FakeHttpClient()
    client = ApiClient(
        base_url="https://api.sogni.ai",
        socket_url="wss://socket.sogni.ai",
        app_id="APP",
        network="fast",
        auth_type="apiKey",
        disable_socket=True,
        http_client=fake_http,
        socket_http_client=fake_http,
    )
    await client.auth.authenticate("secret")

    client._on_socket_disconnected({"code": 4001, "reason": "unauthorized"})

    assert client.auth.is_authenticated is False
    await client.aclose()
