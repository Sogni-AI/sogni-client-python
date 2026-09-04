"""Top-level Sogni SDK client."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from .account import AccountApi
from .announcements import AnnouncementsApi
from .auth import ApiKeyAuthManager, CookieAuthManager, TokenAuthManager
from .chat import ChatApi
from .projects import ProjectsApi
from .replay import ReplayApi
from .stats import StatsApi
from .transport import ApiClient
from .workflows import CreativeWorkflowsApi


class SogniClient:
    """Async entry point for the Sogni Supernet.

    Prefer :meth:`create` or use the client as an async context manager::

        async with await SogniClient.create(api_key="...", app_id="...") as sogni:
            project = await sogni.projects.create(...)
            urls = await project.wait_for_completion()
    """

    def __init__(self, api_client: ApiClient, *, testnet: bool = False) -> None:
        self.api_client = api_client
        self.apiClient = api_client
        self.account = AccountApi(api_client, testnet=testnet)
        self.projects = ProjectsApi(api_client)
        self.stats = StatsApi(api_client)
        self.chat = ChatApi(api_client, self.projects)
        self.workflows = CreativeWorkflowsApi(api_client)
        self.replay = ReplayApi(api_client)
        self.announcements = AnnouncementsApi(api_client)
        self._closed = False

    @classmethod
    async def create(
        cls,
        config: Mapping[str, Any] | None = None,
        *,
        app_id: str | None = None,
        app_source: str | None = None,
        attribution: dict[str, Any] | None = None,
        network: str = "fast",
        api_key: str | None = None,
        auth_type: str = "token",
        rest_endpoint: str = "https://api.sogni.ai",
        socket_endpoint: str = "wss://socket.sogni.ai",
        socket_event_subscriptions: dict[str, bool] | None = None,
        disable_socket: bool = False,
        testnet: bool = False,
        http_client: httpx.AsyncClient | None = None,
        socket_http_client: httpx.AsyncClient | None = None,
        websocket_factory: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> SogniClient:
        """Create and authenticate a client.

        Both Python names (``api_key``) and JavaScript SDK names
        (``apiKey``) are accepted. If ``app_id`` is omitted, a UUID is created;
        callers that need a stable server identity should pass one explicitly.
        """

        values = dict(config or {})
        values.update(kwargs)

        def pick(current: Any, *names: str) -> Any:
            for name in names:
                if name in values:
                    return values[name]
            return current

        app_id = pick(app_id, "app_id", "appId") or str(uuid.uuid4())
        app_source = pick(app_source, "app_source", "appSource")
        attribution = pick(attribution, "attribution")
        network = pick(network, "network")
        api_key = pick(api_key, "api_key", "apiKey")
        auth_type = pick(auth_type, "auth_type", "authType")
        rest_endpoint = pick(rest_endpoint, "rest_endpoint", "restEndpoint")
        socket_endpoint = pick(socket_endpoint, "socket_endpoint", "socketEndpoint")
        socket_event_subscriptions = pick(
            socket_event_subscriptions,
            "socket_event_subscriptions",
            "socketEventSubscriptions",
        )
        disable_socket = bool(pick(disable_socket, "disable_socket", "disableSocket"))
        testnet = bool(pick(testnet, "testnet"))
        if network not in {"fast", "relaxed"}:
            raise ValueError("network must be 'fast' or 'relaxed'")
        if api_key:
            auth_type = "apiKey"
        if auth_type == "cookie":
            auth_type = "cookies"
        if auth_type not in {"token", "cookies", "apiKey"}:
            raise ValueError("auth_type must be 'token', 'cookies', or 'apiKey'")

        transport_kwargs: dict[str, Any] = {
            "base_url": rest_endpoint,
            "socket_url": socket_endpoint,
            "app_id": app_id,
            "app_source": app_source,
            "socket_event_subscriptions": socket_event_subscriptions,
            "network": network,
            "auth_type": auth_type,
            "disable_socket": disable_socket,
            "http_client": http_client,
            "socket_http_client": socket_http_client,
        }
        if attribution is not None:
            transport_kwargs["attribution"] = attribution
        if websocket_factory is not None:
            transport_kwargs["websocket_factory"] = websocket_factory
        api_client = ApiClient(**transport_kwargs)
        client = cls(api_client, testnet=testnet)
        if api_key:
            auth = api_client.auth
            if not isinstance(auth, ApiKeyAuthManager):  # pragma: no cover - guarded above
                raise RuntimeError("API key auth manager was not configured")
            await auth.authenticate(api_key)
            # Authentication starts the socket in the background, matching the
            # JS SDK. Socket-backed calls also lazy-connect before sending.
        return client

    create_instance = create
    createInstance = create

    @property
    def current_account(self):
        return self.account.current_account

    @property
    def currentAccount(self):
        return self.current_account

    async def set_tokens(
        self,
        tokens: Mapping[str, str] | None = None,
        *,
        token: str | None = None,
        refresh_token: str | None = None,
        refreshToken: str | None = None,
    ) -> None:
        auth = self.api_client.auth
        if not isinstance(auth, TokenAuthManager):
            raise RuntimeError("set_tokens can only be used with token authentication")
        await auth.authenticate(
            dict(tokens or {}),
            token=token,
            refresh_token=refresh_token,
            refreshToken=refreshToken,
        )
        await self.account.me()

    setTokens = set_tokens

    async def check_auth(self) -> bool:
        auth = self.api_client.auth
        if not isinstance(auth, CookieAuthManager):
            raise RuntimeError("check_auth should only be called with cookie auth")
        try:
            await self.account.me()
            await auth.authenticate()
            return True
        except Exception:
            return False

    checkAuth = check_auth

    async def set_socket_event_subscriptions(self, update: dict[str, Any]) -> None:
        await self.api_client.set_socket_event_subscriptions(update)

    setSocketEventSubscriptions = set_socket_event_subscriptions

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.api_client.aclose()

    async def dispose(self) -> None:
        await self.aclose()

    async def __aenter__(self) -> SogniClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.aclose()


AsyncSogniClient = SogniClient
