"""Async HTTP and WebSocket communication for the Sogni wire protocol."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit

import httpx
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed

from .auth import ApiKeyAuthManager, AuthManager, CookieAuthManager, TokenAuthManager
from .errors import ApiError
from .events import EventEmitter
from .utils import b64_json_decode, b64_json_encode, drop_none

LIB_VERSION = "5.1.0a24"
PROTOCOL_VERSION = "3.0.0"
SWITCH_CONNECTION = 4015


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
        socket_event_subscriptions: dict[str, bool] | None = None,
        socket_http_client: httpx.AsyncClient | None = None,
        connect_factory: Callable[..., Any] = websocket_connect,
    ) -> None:
        super().__init__()
        self.base_url = base_url
        self.auth = auth
        self.app_id = app_id
        self.app_source = app_source.strip() if app_source and app_source.strip() else None
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
            self._reader_task = asyncio.create_task(self._read_loop())
            self.emit("connected", {"network": self.supernet_type})

    async def disconnect(self, code: int = 1000, reason: str = "Client disconnected") -> None:
        self._intentional_close = True
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
                    self.emit(envelope["type"], payload)
                except (KeyError, TypeError, UnicodeDecodeError, ValueError):
                    # A malformed application frame must not tear down a healthy
                    # socket or clear otherwise-valid authentication state.
                    logging.getLogger("sogni_client").warning(
                        "Dropped malformed WebSocket frame", exc_info=True
                    )
        except ConnectionClosed as closed:
            close_code = int(closed.code or 0)
            close_reason = closed.reason or ""
        except asyncio.CancelledError:
            return
        except Exception:
            logging.getLogger("sogni_client").exception("WebSocket receive loop failed")
        finally:
            if socket is self._socket:
                self._socket = None
            if not self._intentional_close:
                self.emit("disconnected", {"code": close_code, "reason": close_reason})

    async def send(self, message_type: str, data: Any) -> None:
        if self._socket is None:
            await self.connect()
        envelope = json.dumps(
            {"type": message_type, "data": b64_json_encode(data)}, separators=(",", ":")
        )
        await self._socket.send(envelope)

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
        socket_event_subscriptions: dict[str, bool] | None = None,
        disable_socket: bool = False,
        http_client: httpx.AsyncClient | None = None,
        socket_http_client: httpx.AsyncClient | None = None,
        websocket_factory: Callable[..., Any] = websocket_connect,
    ) -> None:
        super().__init__()
        self.app_id = app_id
        self.app_source = app_source.strip() if app_source and app_source.strip() else None
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
            socket_event_subscriptions=socket_event_subscriptions,
            socket_http_client=socket_http_client,
            connect_factory=websocket_factory,
        )
        self.socket_enabled = not disable_socket
        self._disposed = False
        self._reconnect_attempts = 5
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
        self._reconnect_attempts = 5
        self.emit("connected", data)

    def _on_socket_disconnected(self, data: Any) -> None:
        code = int(data.get("code") or 0) if isinstance(data, dict) else 0
        if self._disposed or not self.auth.is_authenticated or code == 1000:
            self.emit("disconnected", data)
            return
        if code == SWITCH_CONNECTION:
            self.emit("disconnected", data)
            return
        if code == 0 or 4000 <= code < 5000:
            self.auth.clear()
            self.emit("disconnected", data)
            return
        if self._reconnect_attempts <= 0:
            self._reconnect_attempts = 5
            self.emit("disconnected", data)
            return
        self._reconnect_attempts -= 1
        self.emit("connecting", {"network": self.socket.supernet_type})
        self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        await asyncio.sleep(1)
        if not self._disposed and self.auth.is_authenticated and self.socket_enabled:
            try:
                await self.socket.connect()
            except Exception:
                self._on_socket_disconnected({"code": 1006, "reason": "Reconnect failed"})

    async def set_socket_event_subscriptions(self, update: dict[str, Any]) -> None:
        await self.socket.set_socket_event_subscriptions(update)

    setSocketEventSubscriptions = set_socket_event_subscriptions

    async def aclose(self) -> None:
        self._disposed = True
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            await asyncio.gather(self._reconnect_task, return_exceptions=True)
        await self.socket.aclose()
        await self.rest.aclose()
        self.auth.clear()
        self.remove_all_listeners()

    async def dispose(self) -> None:
        await self.aclose()
