from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

from sogni_client.auth import ApiKeyAuthManager
from sogni_client.errors import ApiError
from sogni_client.transport import (
    SWITCH_CONNECTION,
    ApiClient,
    RestClient,
    WebSocketClient,
    _log_connection_error,
)
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
        self.messages: asyncio.Queue[str | bytes | Exception] = asyncio.Queue()
        self.sent: list[str] = []
        self.closed: tuple[int, str] | None = None

    async def recv(self) -> str | bytes:
        message = await self.messages.get()
        if isinstance(message, Exception):
            raise message
        return message

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def close(self, *, code: int, reason: str) -> None:
        self.closed = (code, reason)


def frame(message_type: str, data: Any) -> str:
    return json.dumps({"type": message_type, "data": b64_json_encode(data)})


class FakeSocketFactory:
    """Hands out one fake socket. Like a real server it accepts the connection
    for work with an ``authenticated`` frame unless ``authenticate`` is off."""

    def __init__(self, socket: FakeSocket | None = None, *, authenticate: bool = True) -> None:
        self.socket = socket or FakeSocket()
        self.authenticate = authenticate
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, url: str, **kwargs: Any) -> FakeSocket:
        self.calls.append({"url": url, **kwargs})
        if self.authenticate:
            self.socket.messages.put_nowait(frame("authenticated", {"clientType": "artist"}))
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


# When `WebSocketClient.send` puts a frame on the wire (mirrors sogni-client
# scripts/check-socket-send-readiness.cjs).
#
# The socket server drops any frame that arrives before its `authenticated`
# handshake, and during a socket deploy there is a gap in which connections are
# refused or accepted and immediately closed with 1001. Work submitted in that
# window must go out on the next authenticated connection, not fail and not
# vanish. These run a real local WebSocket server.


class ReadinessServer:
    """A local socket server whose per-connection behaviour the test controls."""

    def __init__(self) -> None:
        self.mode = "auth"
        self.auth_delay = 0.0
        self.received: list[dict[str, Any]] = []
        self.connections = 0
        self.live: set[ServerConnection] = set()
        self._server: Any = None

    async def __aenter__(self) -> ReadinessServer:
        self._server = await serve(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def close_all(self, code: int, reason: str) -> None:
        await asyncio.gather(*(ws.close(code, reason) for ws in list(self.live)))

    async def _handle(self, ws: ServerConnection) -> None:
        self.connections += 1
        if self.mode == "restarting":
            await ws.close(1001, "Server is restarting")
            return
        self.live.add(ws)
        state = {"authenticated": False}

        async def authenticate() -> None:
            await asyncio.sleep(self.auth_delay)
            state["authenticated"] = True
            await ws.send(frame("authenticated", {"clientType": "artist", "activeProjects": []}))

        authenticating = asyncio.create_task(authenticate())
        try:
            async for raw in ws:
                message = json.loads(raw)
                self.received.append(
                    {
                        "type": message["type"],
                        "authenticated": state["authenticated"],
                        "data": json.loads(base64.b64decode(message["data"])),
                    }
                )
        finally:
            self.live.discard(ws)
            authenticating.cancel()
            with contextlib.suppress(BaseException):
                await authenticating


async def authenticated_socket_client(url: str, app_id: str) -> WebSocketClient:
    auth = ApiKeyAuthManager()
    await auth.authenticate("key")
    return WebSocketClient(url, auth, app_id, "fast")


@pytest.mark.asyncio
async def test_send_right_after_connect_waits_for_authenticated() -> None:
    async with ReadinessServer() as server:
        server.auth_delay = 0.15
        client = await authenticated_socket_client(server.url, "APP-1")
        await client.connect()
        await client.send("jobRequest", {"jobID": "P1"})
        await asyncio.sleep(0.05)
        try:
            assert len(server.received) == 1
            assert server.received[0]["authenticated"] is True, "sent only after authentication"
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_send_issued_during_a_socket_restart_goes_out_after_reconnect() -> None:
    # Socket deploy: the connection closes with 1001 and the next attempts are
    # turned away. A send issued in the gap waits and goes out on the connection
    # that authenticates, without opening connections of its own.
    async with ReadinessServer() as server:
        client = await authenticated_socket_client(server.url, "APP-2")
        authenticated = asyncio.get_running_loop().create_future()
        client.once("authenticated", lambda data: authenticated.set_result(data))
        await client.connect()
        await asyncio.wait_for(authenticated, 1)
        server.mode = "restarting"

        # Emulate the ApiClient: reconnect after each recoverable close.
        async def reconnect() -> None:
            with contextlib.suppress(Exception):
                await client.connect()

        def on_disconnected(data: Any) -> None:
            if data.get("code") in {1001, 1006}:
                asyncio.get_running_loop().call_later(
                    0.04, lambda: asyncio.ensure_future(reconnect())
                )

        remove_reconnect = client.on("disconnected", on_disconnected)
        try:
            await server.close_all(1001, "Server is restarting")
            await asyncio.sleep(0.02)
            assert client._reconnect_expected is True
            connections_before = server.connections
            sending = asyncio.create_task(client.send("jobRequest", {"jobID": "P2"}))
            await asyncio.sleep(0.15)
            assert server.received == [], "nothing sent into the gap"
            assert server.connections - connections_before <= 4, (
                "send did not add its own connection attempts on top of the reconnect loop"
            )
            server.mode = "auth"
            await asyncio.wait_for(sending, 5)
            await asyncio.sleep(0.05)
            assert [(r["data"]["jobID"], r["authenticated"]) for r in server.received] == [
                ("P2", True)
            ], "delivered once, after the restart, on an authenticated connection"
        finally:
            remove_reconnect()
            await client.aclose()


@pytest.mark.asyncio
async def test_terminal_close_ends_the_send_wait_with_an_error() -> None:
    async with ReadinessServer() as server:
        server.auth_delay = 10
        client = await authenticated_socket_client(server.url, "APP-3")
        await client.connect()
        sending = asyncio.create_task(client.send("jobRequest", {"jobID": "P3"}))
        await asyncio.sleep(0.08)
        await server.close_all(4021, "Authentication error")
        try:
            with pytest.raises(ConnectionError, match="connection failed"):
                await asyncio.wait_for(sending, 1)
            assert server.received == []
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_send_falls_back_to_an_open_but_silent_socket_from_an_older_server() -> None:
    auth = ApiKeyAuthManager()
    await auth.authenticate("key")
    factory = FakeSocketFactory(authenticate=False)
    client = WebSocketClient("wss://socket.sogni.ai", auth, "APP", "fast", connect_factory=factory)
    client._send_ready_tuning["authenticated_fallback_seconds"] = 0.05
    loop = asyncio.get_running_loop()
    started = loop.time()

    await client.send("jobRequest", {"jobID": "OLD"})
    try:
        assert loop.time() - started >= 0.05
        assert len(factory.socket.sent) == 1
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_send_waits_for_the_reconnect_loop_and_times_out_if_it_never_comes() -> None:
    auth = ApiKeyAuthManager()
    await auth.authenticate("key")
    factory = FakeSocketFactory()
    client = WebSocketClient("wss://socket.sogni.ai", auth, "APP", "fast", connect_factory=factory)
    client._send_ready_tuning["timeout_seconds"] = 0.1
    disconnected = asyncio.get_running_loop().create_future()
    client.once("disconnected", lambda data: disconnected.set_result(data))
    await client.connect()
    # A recoverable close: the ApiClient owns the reconnect, which never comes here.
    factory.socket.messages.put_nowait(ConnectionClosed(Close(1001, "Server is restarting"), None))
    assert await asyncio.wait_for(disconnected, 1) == {
        "code": 1001,
        "reason": "Server is restarting",
    }
    assert client.is_connected is False
    assert client._reconnect_expected is True

    try:
        with pytest.raises(TimeoutError, match="WebSocket connection timeout"):
            await client.send("jobRequest", {"jobID": "LATE"})
        assert len(factory.calls) == 1, "send never opened a competing connection"
        assert factory.socket.sent == []
    finally:
        await client.aclose()


# Transport diagnostics never reach a logger with credentials (mirrors
# sogni-client scripts/check-websocket-diagnostics.cjs). A connection failure can
# carry the upgrade request, its URL and its `api-key` / `Authorization` headers
# in its message, chained exceptions or traceback; only the bounded HTTP upgrade
# status is kept.

CREDENTIAL = "test-credential-never-log"


def _sogni_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "sogni_client"]


def _assert_no_credential(records: list[logging.LogRecord]) -> None:
    for record in records:
        assert record.exc_info is None
        assert record.exc_text is None
        assert CREDENTIAL not in record.getMessage()
        assert CREDENTIAL not in repr(record.args)
        assert "api-key" not in record.getMessage()


def _rejected_upgrade(status: int) -> InvalidStatus:
    error = InvalidStatus(
        Response(status, "Bad Gateway", Headers({"api-key": CREDENTIAL}), CREDENTIAL.encode())
    )
    error.__cause__ = RuntimeError(f"GET wss://socket.sogni.ai/?token={CREDENTIAL}")
    return error


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_rejected_upgrade(502), {"status": 502}),
        (RuntimeError(f"Failed request https://example.test/?token={CREDENTIAL}"), {}),
        (OSError(), {}),
        (None, {}),
    ],
)
def test_connection_error_logs_only_bounded_status(
    caplog: pytest.LogCaptureFixture, error: BaseException | None, expected: dict[str, int]
) -> None:
    caplog.set_level(logging.DEBUG, logger="sogni_client")

    _log_connection_error(error)

    [record] = _sogni_records(caplog)
    assert record.levelno == logging.ERROR
    assert record.getMessage() == f"WebSocket connection error {expected}"
    _assert_no_credential([record])


@pytest.mark.asyncio
async def test_failed_reconnect_attempt_logs_status_without_request_or_headers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="sogni_client")
    attempts: list[dict[str, Any]] = []

    async def rejecting_factory(url: str, **kwargs: Any) -> FakeSocket:
        attempts.append({"url": url, **kwargs})
        raise _rejected_upgrade(502)

    fake_http = FakeHttpClient()
    client = ApiClient(
        base_url="https://api.sogni.ai",
        socket_url="wss://socket.sogni.ai",
        app_id="APP",
        network="fast",
        auth_type="apiKey",
        http_client=fake_http,
        socket_http_client=fake_http,
        websocket_factory=rejecting_factory,
    )
    await client.auth.authenticate(CREDENTIAL)
    client._schedule_reconnect = lambda: None  # type: ignore[method-assign]
    try:
        await client._reconnect(0)
        assert attempts[0]["additional_headers"] == {"api-key": CREDENTIAL}
        records = _sogni_records(caplog)
        assert [record.getMessage() for record in records] == [
            "WebSocket connection error {'status': 502}"
        ]
        _assert_no_credential(records)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_receive_loop_failure_logs_without_raw_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="sogni_client")
    auth = ApiKeyAuthManager()
    await auth.authenticate(CREDENTIAL)
    factory = FakeSocketFactory(authenticate=False)
    client = WebSocketClient("wss://socket.sogni.ai", auth, "APP", "fast", connect_factory=factory)
    disconnected: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    client.on("disconnected", lambda data: disconnected.set_result(data))
    await client.connect()
    try:
        await factory.socket.messages.put(
            RuntimeError(f"Failed request https://example.test/?token={CREDENTIAL}")
        )
        await asyncio.wait_for(disconnected, 1)
        records = _sogni_records(caplog)
        assert [record.getMessage() for record in records] == ["WebSocket connection error {}"]
        _assert_no_credential(records)
    finally:
        await client.aclose()
