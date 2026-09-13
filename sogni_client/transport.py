"""Async HTTP and WebSocket communication for the Sogni wire protocol."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
from collections.abc import AsyncIterator, Callable
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit

import httpx
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from .attribution import (
    build_sogni_attribution_headers,
    connection_attribution_query,
    normalize_connection_attribution,
    resolve_workload_attribution,
)
from .auth import ApiKeyAuthManager, AuthManager, CookieAuthManager, TokenAuthManager
from .errors import ApiError
from .events import EventEmitter
from .utils import b64_json_decode, b64_json_encode, drop_none

LIB_VERSION = "5.21.3"
PROTOCOL_VERSION = "3.0.0"
SWITCH_CONNECTION = 4015
# Reconnect backoff for recoverable socket drops. Attempts continue for as long
# as the session stays authenticated: an in-flight generation survives a network
# blip, a sleeping laptop, or a socket deploy, and the server hands it back on
# reconnect.
WS_RECONNECT_BASE_DELAY = 1.0
WS_RECONNECT_MAX_DELAY = 15.0
# How long `send` waits for a socket that can carry work. Covers a socket deploy
# (a ~6 s gap plus reconnect backoff) without hanging the caller on a transport
# that is not coming back.
SEND_READY_TIMEOUT_SECONDS = 30.0
# The server drops frames that arrive before its `authenticated` handshake. Every
# current server sends that frame within milliseconds; if an open socket stays
# silent this long, send anyway rather than stall on an older server.
AUTHENTICATED_FALLBACK_SECONDS = 10.0
READY_POLL_SECONDS = 0.1


def _is_not_recoverable(code: int) -> bool:
    return 4000 <= code < 5000


def _connection_error_diagnostics(error: BaseException | None) -> dict[str, int]:
    """The only facts about a connection failure that are safe to log.

    Connection errors can carry the upgrade request, its URL, and its
    authentication headers (``api-key`` / ``Authorization``) in their message,
    chained exceptions, or traceback. Keep only the bounded HTTP upgrade status
    when the server rejected the handshake.
    """

    if not isinstance(error, InvalidStatus):
        return {}
    status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 999:
        return {"status": status}
    return {}


def _log_connection_error(error: BaseException | None) -> None:
    # Never pass the exception, its message, or exc_info to the logger.
    logging.getLogger("sogni_client").error(
        "WebSocket connection error %s", _connection_error_diagnostics(error)
    )


class RestClient:
    def __init__(
        self,
        base_url: str,
        auth: AuthManager,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.auth = auth
        self.timeout = timeout
        self._client = http_client or httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self._owns_client = http_client is None

    def url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        content: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        response = await self.raw_request(
            method,
            path,
            params=params,
            json_body=json_body,
            content=content,
            headers=headers,
            timeout=timeout,
        )
        return await self.process_response(response)

    async def raw_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        content: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        auth_headers = await self.auth.headers()
        request_headers = {**auth_headers, **(headers or {})}
        clean_params = {key: value for key, value in (params or {}).items() if value is not None}
        return await self._client.request(
            method,
            self.url(path),
            params=clean_params,
            json=drop_none(json_body) if json_body is not None else None,
            content=content,
            headers=request_headers,
            timeout=timeout if timeout is not None else self.timeout,
        )

    async def process_response(self, response: httpx.Response) -> Any:
        if response.status_code == 401 and self.auth.is_authenticated:
            self.auth.clear()
        text = response.text
        parsed: Any = None
        parse_error: ValueError | None = None
        if text:
            try:
                parsed = response.json()
            except ValueError as error:
                parse_error = error
        if not response.is_success:
            if isinstance(parsed, dict):
                payload = parsed
            else:
                excerpt = " ".join(text[:200].split())
                message = response.reason_phrase or f"HTTP {response.status_code}"
                if excerpt:
                    message = f"{message}: {excerpt}"
                payload = {
                    "status": "error",
                    "message": message,
                    "errorCode": response.status_code,
                }
            raise ApiError(response.status_code, payload)
        if parse_error is not None:
            raise ValueError(
                f"Failed to parse response body (HTTP {response.status_code}): {parse_error}"
            ) from parse_error
        return parsed

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return await self.request("GET", path, params=params)

    async def post(
        self,
        path: str,
        body: Any = None,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return await self.request(
            "POST", path, json_body=body or {}, headers=headers, timeout=timeout
        )

    async def patch(self, path: str, body: Any = None) -> Any:
        return await self.request("PATCH", path, json_body=body or {})

    async def delete(self, path: str) -> Any:
        return await self.request("DELETE", path)

    async def stream_lines(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        auth_headers = await self.auth.headers()
        request_headers = {**auth_headers, **(headers or {})}
        async with self._client.stream(
            "GET",
            self.url(path),
            params={k: v for k, v in (params or {}).items() if v is not None},
            headers=request_headers,
            timeout=timeout,
        ) as response:
            if not response.is_success:
                await response.aread()
                await self.process_response(response)
            async for line in response.aiter_lines():
                yield line

    async def put_bytes(self, url: str, data: bytes, *, content_type: str | None = None) -> None:
        headers = {"Content-Type": content_type} if content_type else {}
        response = await self._client.put(url, content=data, headers=headers, timeout=300)
        if not response.is_success:
            raise ApiError(
                response.status_code,
                {
                    "status": "error",
                    "message": response.reason_phrase or "Failed to upload media",
                    "errorCode": 0,
                },
            )

    async def put_signed(self, url: str, data: bytes, *, headers: dict[str, str]) -> int:
        """PUT to a presigned URL with its signed headers; returns the HTTP status.

        No auth headers or cookies are sent and redirects are not followed.
        """
        response = await self._client.put(
            url, content=data, headers=headers, timeout=300, follow_redirects=False
        )
        return response.status_code

    async def get_bytes(self, url: str) -> bytes:
        response = await self._client.get(url, timeout=300)
        response.raise_for_status()
        return response.content

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class WebSocketClient(EventEmitter):
    def __init__(
        self,
        base_url: str,
        auth: AuthManager,
        app_id: str,
        network: str,
        *,
        app_source: str | None = None,
        connection_attribution: dict[str, Any] | None = None,
        socket_event_subscriptions: dict[str, bool] | None = None,
        socket_http_client: httpx.AsyncClient | None = None,
        connect_factory: Callable[..., Any] = websocket_connect,
    ) -> None:
        super().__init__()
        self.base_url = base_url
        self.auth = auth
        self.app_id = app_id
        self.app_source = app_source.strip() if app_source and app_source.strip() else None
        self.connection_attribution = normalize_connection_attribution(connection_attribution)
        self.socket_event_subscriptions = socket_event_subscriptions
        self.supernet_type = network
        http_scheme = "http" if urlsplit(base_url).scheme in {"http", "ws"} else "https"
        parts = urlsplit(base_url)
        http_url = urlunsplit((http_scheme, parts.netloc, parts.path, "", ""))
        self.rest = RestClient(http_url, auth, http_client=socket_http_client)
        self._connect_factory = connect_factory
        self._socket: Any = None
        self._reader_task: asyncio.Task[None] | None = None
        self._connect_lock = asyncio.Lock()
        self._intentional_close = False
        # The socket the server has sent `authenticated` on, i.e. one that
        # accepts work.
        self._authenticated_socket: Any = None
        self._opened_at = 0.0
        # Set when the last close was recoverable while the session is
        # authenticated: the ApiClient owns the reconnect, so `send` waits for it
        # instead of racing it with a connection of its own (which would also
        # reset the reconnect backoff on every failed attempt).
        self._reconnect_expected = False
        # Send-readiness timing. Overridable so regression tests can run the
        # flow in fractions of a second.
        self._send_ready_tuning = {
            "timeout_seconds": SEND_READY_TIMEOUT_SECONDS,
            "authenticated_fallback_seconds": AUTHENTICATED_FALLBACK_SECONDS,
            "poll_seconds": READY_POLL_SECONDS,
        }

        def remember_subscriptions(payload: Any) -> None:
            if isinstance(payload, dict) and isinstance(
                payload.get("socketEventSubscriptions"), dict
            ):
                self.socket_event_subscriptions = dict(payload["socketEventSubscriptions"])

        self.on("socketEventSubscriptionsUpdated", remember_subscriptions)

    @property
    def is_connected(self) -> bool:
        return self._socket is not None

    @property
    def isConnected(self) -> bool:
        return self.is_connected

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._socket is not None:
                return
            self._intentional_close = False
            parts = urlsplit(self.base_url)
            scheme = "ws" if parts.scheme in {"http", "ws"} else "wss"
            query: dict[str, str] = {
                "appId": self.app_id,
                "clientName": f"Sogni/{PROTOCOL_VERSION} (sogni-client) {LIB_VERSION}",
                "clientType": "artist",
                "forceWorkerId": "fast" if self.supernet_type == "fast" else "",
            }
            if self.app_source:
                query["appSource"] = self.app_source
            query.update(connection_attribution_query(self.connection_attribution))
            subscriptions = {
                key: enabled
                for key, enabled in (self.socket_event_subscriptions or {}).items()
                if isinstance(enabled, bool)
            }
            if subscriptions:
                query["socketEventSubscriptions"] = json.dumps(subscriptions, separators=(",", ":"))
            existing = parts.query
            encoded = urlencode(query)
            url = urlunsplit(
                (
                    scheme,
                    parts.netloc,
                    parts.path,
                    f"{existing}&{encoded}" if existing else encoded,
                    "",
                )
            )
            headers = await self.auth.headers()
            self._socket = await self._connect_factory(
                url,
                additional_headers=headers or None,
                ping_interval=15,
                ping_timeout=15,
                close_timeout=5,
                max_size=None,
            )
            # Cleared only once a socket is open: a failed attempt leaves the
            # reconnect loop in charge, so `send` keeps waiting for it.
            self._reconnect_expected = False
            self._authenticated_socket = None
            self._opened_at = asyncio.get_running_loop().time()
            self._reader_task = asyncio.create_task(self._read_loop())
            self.emit("connected", {"network": self.supernet_type})

    async def disconnect(self, code: int = 1000, reason: str = "Client disconnected") -> None:
        self._intentional_close = True
        self._reconnect_expected = False
        self._authenticated_socket = None
        socket, self._socket = self._socket, None
        reader, self._reader_task = self._reader_task, None
        if socket is not None:
            await socket.close(code=code, reason=reason)
        current = asyncio.current_task()
        if reader is not None and reader is not current and not reader.done():
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    async def _read_loop(self) -> None:
        socket = self._socket
        close_code = 0
        close_reason = ""
        try:
            while socket is self._socket:
                message = await socket.recv()
                try:
                    if isinstance(message, bytes):
                        message = message.decode()
                    envelope = json.loads(message)
                    if not isinstance(envelope, dict) or not isinstance(envelope.get("type"), str):
                        raise ValueError("WebSocket envelope must include a string type")
                    payload = b64_json_decode(envelope["data"]) if envelope.get("data") else None
                    if isinstance(payload, dict):
                        for key in ("jobID", "imgID"):
                            if payload.get(key):
                                payload[key] = str(payload[key]).upper()
                    if envelope["type"] == "authenticated" and socket is self._socket:
                        self._authenticated_socket = socket
                    self.emit(envelope["type"], payload)
                except (KeyError, TypeError, UnicodeDecodeError, ValueError):
                    # A malformed application frame must not tear down a healthy
                    # socket or clear otherwise-valid authentication state.
                    logging.getLogger("sogni_client").warning(
                        "Dropped malformed WebSocket frame", exc_info=True
                    )
        except ConnectionClosed as closed:
            # `ConnectionClosed.code` / `.reason` are deprecated since websockets
            # 13.1; read the received close frame (none means 1006, as before).
            received = closed.rcvd
            close_code = int(received.code) if received is not None else 1006
            close_reason = (received.reason if received is not None else "") or ""
        except asyncio.CancelledError:
            return
        except Exception as error:
            _log_connection_error(error)
        finally:
            if socket is self._socket:
                self._socket = None
            if not self._intentional_close and self._socket is None:
                self._authenticated_socket = None
                self._reconnect_expected = (
                    self.auth.is_authenticated
                    and bool(close_code)
                    and close_code != 1000
                    and not _is_not_recoverable(close_code)
                )
            if not self._intentional_close:
                self.emit("disconnected", {"code": close_code, "reason": close_reason})

    def _is_ready_for_work(self) -> bool:
        """The current socket is open and the server has accepted it for work."""

        return self._socket is not None and self._authenticated_socket is self._socket

    def _expecting_reconnect(self) -> bool:
        return self._reconnect_expected and self.auth.is_authenticated

    async def _wait_for_connection(self, timeout: float | None = None) -> None:
        """Wait until the socket can carry work.

        An open socket is not enough: the server drops frames that arrive before
        its ``authenticated`` handshake. A recoverable close (a socket deploy, a
        network blip) keeps the wait alive while the ApiClient reconnects, so
        work submitted during the gap goes out on the next connection instead of
        failing. A terminal close, a signed-out session, or the deadline ends it.
        """

        if self._is_ready_for_work():
            return
        if (
            self._socket is None
            and not self._expecting_reconnect()
            and not self._connect_lock.locked()
        ):
            raise ConnectionError("WebSocket not connected")
        tuning = self._send_ready_tuning
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (
            timeout if timeout is not None else float(tuning["timeout_seconds"])
        )
        fallback_seconds = float(tuning["authenticated_fallback_seconds"])
        wake = asyncio.Event()
        failure: list[Exception] = []

        def on_authenticated(_data: Any) -> None:
            wake.set()

        def on_disconnected(_data: Any) -> None:
            if not self._expecting_reconnect():
                failure.append(ConnectionError("WebSocket connection failed"))
            wake.set()

        remove_authenticated = self.on("authenticated", on_authenticated)
        remove_disconnected = self.on("disconnected", on_disconnected)
        try:
            while True:
                wake.clear()
                if failure:
                    raise failure[0]
                if self._is_ready_for_work():
                    return
                open_seconds = loop.time() - self._opened_at
                if self._socket is not None and open_seconds >= fallback_seconds:
                    # An older server that never sends `authenticated`.
                    return
                if (
                    self._socket is None
                    and not self._expecting_reconnect()
                    and not self._connect_lock.locked()
                ):
                    # Closed on purpose, or signed out while waiting.
                    raise ConnectionError("WebSocket connection failed")
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError("WebSocket connection timeout")
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        wake.wait(), min(float(tuning["poll_seconds"]), remaining)
                    )
        finally:
            remove_authenticated()
            remove_disconnected()

    async def send(self, message_type: str, data: Any) -> None:
        # While a recoverable close is being handled the ApiClient reconnects;
        # opening a connection here would race it.
        if self._socket is None and not self._expecting_reconnect():
            await self.connect()
        await self._wait_for_connection()
        socket = self._socket
        if socket is None:  # pragma: no cover - guarded by the wait above
            raise ConnectionError("WebSocket not connected")
        envelope = json.dumps(
            {"type": message_type, "data": b64_json_encode(data)}, separators=(",", ":")
        )
        await socket.send(envelope)

    async def switch_network(self, network: str) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        def changed(data: Any) -> None:
            resolved = data.get("network") if isinstance(data, dict) else data
            self.supernet_type = str(resolved)
            if not future.done():
                future.set_result(self.supernet_type)

        remove = self.once("changeNetwork", changed)
        try:
            await self.send("changeNetwork", network)
            return await asyncio.wait_for(future, timeout=30)
        finally:
            remove()

    switchNetwork = switch_network

    async def set_socket_event_subscriptions(self, update: dict[str, Any]) -> None:
        update_keys = {"subscriptions", "subscribe", "unsubscribe", "reset", "event", "enabled"}
        normalized = update if update_keys.intersection(update) else {"subscriptions": update}
        await self.send("setSocketEventSubscriptions", normalized)

    setSocketEventSubscriptions = set_socket_event_subscriptions

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return await self.rest.get(path, params)

    async def aclose(self) -> None:
        await self.disconnect()
        await self.rest.aclose()


class ApiClient(EventEmitter):
    """Coordinates authentication, REST, socket connection, and reconnects."""

    def __init__(
        self,
        *,
        base_url: str,
        socket_url: str,
        app_id: str,
        network: str,
        auth_type: str,
        app_source: str | None = None,
        attribution: dict[str, Any] | None = None,
        socket_event_subscriptions: dict[str, bool] | None = None,
        disable_socket: bool = False,
        http_client: httpx.AsyncClient | None = None,
        socket_http_client: httpx.AsyncClient | None = None,
        websocket_factory: Callable[..., Any] = websocket_connect,
    ) -> None:
        super().__init__()
        self.app_id = app_id
        self.app_source = app_source.strip() if app_source and app_source.strip() else None
        raw_attribution = attribution if isinstance(attribution, dict) else {}
        connection = raw_attribution.get("connection")
        workload = raw_attribution.get("workload")
        self.attribution: dict[str, dict[str, Any]] = {}
        normalized_connection = normalize_connection_attribution(connection)
        if normalized_connection:
            self.attribution["connection"] = normalized_connection
        if isinstance(workload, dict):
            self.attribution["workload"] = dict(workload)
        if auth_type == "apiKey":
            self.auth: AuthManager = ApiKeyAuthManager()
        elif auth_type == "cookies":
            self.auth = CookieAuthManager()
        else:
            self.auth = TokenAuthManager(base_url, refresh_client=http_client)
        self.rest = RestClient(base_url, self.auth, http_client=http_client)
        self.socket = WebSocketClient(
            socket_url,
            self.auth,
            app_id,
            network,
            app_source=self.app_source,
            connection_attribution=self.attribution.get("connection"),
            socket_event_subscriptions=socket_event_subscriptions,
            socket_http_client=socket_http_client,
            connect_factory=websocket_factory,
        )
        self.socket_enabled = not disable_socket
        self._disposed = False
        self._reconnect_attempt = 0
        self._reconnect_task: asyncio.Task[None] | None = None
        self.auth.on("updated", self._on_auth_updated)
        self.socket.on("connected", self._on_socket_connected)
        self.socket.on("disconnected", self._on_socket_disconnected)

    @property
    def is_authenticated(self) -> bool:
        return self.auth.is_authenticated

    @property
    def isAuthenticated(self) -> bool:
        return self.is_authenticated

    def resolve_workload_attribution(
        self,
        override: dict[str, Any] | None = None,
        fallback_operation_id: str | None = None,
    ) -> dict[str, str] | None:
        return resolve_workload_attribution(
            self.attribution.get("workload"), override, fallback_operation_id
        )

    resolveWorkloadAttribution = resolve_workload_attribution

    def attribution_headers(
        self,
        app_source: str | None,
        override: dict[str, Any] | None = None,
        fallback_operation_id: str | None = None,
    ) -> dict[str, str]:
        return build_sogni_attribution_headers(
            app_source=app_source,
            connection=self.attribution.get("connection"),
            workload=self.resolve_workload_attribution(override, fallback_operation_id),
        )

    attributionHeaders = attribution_headers

    async def start(self) -> None:
        if self.socket_enabled and self.auth.is_authenticated and not self.socket.is_connected:
            self.emit("connecting", {"network": self.socket.supernet_type})
            await self.socket.connect()

    def _on_auth_updated(self, authenticated: bool) -> None:
        if self._disposed:
            return
        if authenticated:
            if self.socket_enabled and not self.socket.is_connected:
                asyncio.create_task(self.start())
        elif self.socket.is_connected:
            asyncio.create_task(self.socket.disconnect())

    def _on_socket_connected(self, data: Any) -> None:
        self._reconnect_attempt = 0
        self._clear_reconnect()
        self.emit("connected", data)

    def _on_socket_disconnected(self, data: Any) -> None:
        code = int(data.get("code") or 0) if isinstance(data, dict) else 0
        if self._disposed or not self.auth.is_authenticated or code == 1000:
            self._clear_reconnect()
            self.emit("disconnected", data)
            return
        if code == SWITCH_CONNECTION:
            self._clear_reconnect()
            self.emit("disconnected", data)
            return
        if code == 0 or 4000 <= code < 5000:
            self._clear_reconnect()
            self.auth.clear()
            self.emit("disconnected", data)
            return
        # Recoverable drop: keep trying with capped exponential backoff while the
        # session is authenticated. Generation keeps running on the Supernet while
        # the socket is down, and the server hands the project back on reconnect,
        # so consumers see `connecting` here; `disconnected` is reserved for
        # terminal outcomes.
        self._schedule_reconnect()

    def _clear_reconnect(self) -> None:
        task, self._reconnect_task = self._reconnect_task, None
        if task is not None and not task.done():
            task.cancel()

    def _schedule_reconnect(self) -> None:
        self._clear_reconnect()
        attempt = self._reconnect_attempt
        self._reconnect_attempt += 1
        base = min(WS_RECONNECT_BASE_DELAY * 2**attempt, WS_RECONNECT_MAX_DELAY)
        delay = base * (0.8 + random.random() * 0.4)
        self.emit("connecting", {"network": self.socket.supernet_type})
        self._reconnect_task = asyncio.create_task(self._reconnect(delay))

    async def _reconnect(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        if self._disposed or not self.auth.is_authenticated or not self.socket_enabled:
            return
        try:
            await self.socket.connect()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _log_connection_error(error)
            self._schedule_reconnect()

    async def set_socket_event_subscriptions(self, update: dict[str, Any]) -> None:
        await self.socket.set_socket_event_subscriptions(update)

    setSocketEventSubscriptions = set_socket_event_subscriptions

    async def aclose(self) -> None:
        self._disposed = True
        task, self._reconnect_task = self._reconnect_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.socket.aclose()
        await self.rest.aclose()
        self.auth.clear()
        self.remove_all_listeners()

    async def dispose(self) -> None:
        await self.aclose()
