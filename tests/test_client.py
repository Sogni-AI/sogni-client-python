from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from typing import Any

import pytest

import sogni_client.client as client_module
from sogni_client import (
    AccountApi,
    ApiKeyAuthManager,
    AsyncSogniClient,
    ChatApi,
    CookieAuthManager,
    CreativeWorkflowsApi,
    ProjectsApi,
    ReplayApi,
    SogniClient,
    StatsApi,
    TokenAuthManager,
)
from sogni_client.errors import ApiError
from sogni_client.events import EventEmitter


def jwt(payload: dict[str, Any]) -> str:
    def encode(value: dict[str, Any]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode(payload)}.signature"


class FakeRest:
    def __init__(self) -> None:
        self.gets: list[tuple[str, dict[str, Any] | None]] = []
        self.me_response: Any = {
            "data": {
                "username": "ada",
                "currentEmail": "ada@example.test",
                "walletAddress": "0xabc",
            }
        }

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.gets.append((path, params))
        if path != "/v1/account/me":
            raise AssertionError(f"Unexpected GET {path}")
        if isinstance(self.me_response, Exception):
            raise self.me_response
        return self.me_response


class FakeSocket(EventEmitter):
    def __init__(self, network: str) -> None:
        super().__init__()
        self.is_connected = False
        self.supernet_type = network
        self.subscription_updates: list[dict[str, Any]] = []

    async def set_socket_event_subscriptions(self, update: dict[str, Any]) -> None:
        self.subscription_updates.append(update)


class FakeApiClient(EventEmitter):
    instances: list[FakeApiClient] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.app_id = kwargs["app_id"]
        source = kwargs.get("app_source")
        self.app_source = source.strip() if source and source.strip() else None
        auth_type = kwargs["auth_type"]
        if auth_type == "apiKey":
            self.auth = ApiKeyAuthManager()
        elif auth_type == "cookies":
            self.auth = CookieAuthManager()
        else:
            self.auth = TokenAuthManager(kwargs["base_url"])
        self.rest = FakeRest()
        self.socket = FakeSocket(kwargs["network"])
        self.closed = 0
        self.subscription_updates: list[dict[str, Any]] = []
        self.instances.append(self)

    async def set_socket_event_subscriptions(self, update: dict[str, Any]) -> None:
        self.subscription_updates.append(update)
        await self.socket.set_socket_event_subscriptions(update)

    async def aclose(self) -> None:
        self.closed += 1


@pytest.fixture(autouse=True)
def fake_api_client(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeApiClient.instances.clear()
    monkeypatch.setattr(client_module, "ApiClient", FakeApiClient)


@pytest.mark.asyncio
async def test_create_maps_javascript_config_and_wires_all_public_api_groups() -> None:
    http_client = object()
    socket_http_client = object()
    sdk = await SogniClient.create(
        {
            "appId": "app-123",
            "appSource": "  desktop  ",
            "network": "relaxed",
            "apiKey": "  secret-key  ",
            "authType": "token",
            "restEndpoint": "https://rest.example/base",
            "socketEndpoint": "wss://socket.example/connect",
            "socketEventSubscriptions": {"modelAvailability": False},
            "disableSocket": True,
            "testnet": True,
        },
        http_client=http_client,  # type: ignore[arg-type]
        socket_http_client=socket_http_client,  # type: ignore[arg-type]
    )
    await asyncio.sleep(0)

    transport = FakeApiClient.instances[0]
    assert sdk.api_client is sdk.apiClient is transport
    assert transport.kwargs == {
        "base_url": "https://rest.example/base",
        "socket_url": "wss://socket.example/connect",
        "app_id": "app-123",
        "app_source": "  desktop  ",
        "socket_event_subscriptions": {"modelAvailability": False},
        "network": "relaxed",
        "auth_type": "apiKey",
        "disable_socket": True,
        "http_client": http_client,
        "socket_http_client": socket_http_client,
    }
    assert await transport.auth.headers() == {"api-key": "secret-key"}
    assert isinstance(sdk.account, AccountApi)
    assert isinstance(sdk.projects, ProjectsApi)
    assert isinstance(sdk.stats, StatsApi)
    assert isinstance(sdk.chat, ChatApi)
    assert isinstance(sdk.workflows, CreativeWorkflowsApi)
    assert isinstance(sdk.replay, ReplayApi)
    assert sdk.chat.projects is sdk.projects
    assert sdk.chat.tools.projects is sdk.projects
    assert sdk.current_account is sdk.currentAccount is sdk.account.current_account
    assert sdk.account._eip712_domain["chainId"] == 84532
    assert sdk.current_account.username == "ada"

    await sdk.aclose()


@pytest.mark.asyncio
async def test_create_generates_uuid_accepts_camel_keywords_and_forwards_socket_factory() -> None:
    async def socket_factory(*_args: Any, **_kwargs: Any) -> None:
        return None

    sdk = await SogniClient.createInstance(
        appSource="cli",
        authType="cookie",
        disableSocket=True,
        websocket_factory=socket_factory,
    )

    transport = FakeApiClient.instances[0]
    assert uuid.UUID(transport.kwargs["app_id"])
    assert transport.kwargs["app_source"] == "cli"
    assert transport.kwargs["auth_type"] == "cookies"
    assert transport.kwargs["disable_socket"] is True
    assert transport.kwargs["websocket_factory"] is socket_factory
    assert isinstance(transport.auth, CookieAuthManager)

    await sdk.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"network": "turbo"}, "network must be"),
        ({"auth_type": "basic"}, "auth_type must be"),
    ],
)
async def test_create_validates_configuration_before_transport_construction(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await SogniClient.create(**kwargs)

    assert FakeApiClient.instances == []


@pytest.mark.asyncio
async def test_set_tokens_authenticates_refresh_pair_and_populates_current_account() -> None:
    sdk = await SogniClient.create(app_id="token-app", auth_type="token", disable_socket=True)
    access = jwt({"exp": time.time() + 600, "kind": "access"})
    refresh = jwt({"exp": time.time() + 3600, "kind": "refresh"})

    await sdk.setTokens({"token": access, "refresh_token": refresh})
    await asyncio.sleep(0)

    assert isinstance(sdk.api_client.auth, TokenAuthManager)
    assert await sdk.api_client.auth.backup() == {
        "token": access,
        "refreshToken": refresh,
    }
    assert sdk.current_account.username == "ada"
    assert sdk.current_account.email == "ada@example.test"
    assert sdk.current_account.wallet_address == "0xabc"
    assert sdk.api_client.rest.gets
    assert all(path == "/v1/account/me" for path, _params in sdk.api_client.rest.gets)

    await sdk.aclose()


@pytest.mark.asyncio
async def test_set_tokens_rejects_non_token_authentication() -> None:
    sdk = await SogniClient.create(app_id="key-app", api_key="secret", disable_socket=True)

    with pytest.raises(RuntimeError, match="token authentication"):
        await sdk.set_tokens(token="unused", refresh_token="unused")

    await sdk.aclose()


@pytest.mark.asyncio
async def test_check_auth_populates_cookie_account_and_returns_false_on_failure() -> None:
    sdk = await SogniClient.create(app_id="cookie-app", auth_type="cookies", disable_socket=True)

    assert await sdk.checkAuth() is True
    await asyncio.sleep(0)
    assert isinstance(sdk.api_client.auth, CookieAuthManager)
    assert sdk.api_client.auth.is_authenticated is True
    assert sdk.current_account.username == "ada"

    failed = await SogniClient.create(
        app_id="failed-cookie-app", auth_type="cookie", disable_socket=True
    )
    failed.api_client.rest.me_response = ApiError(401, {"message": "Unauthorized"})
    assert await failed.check_auth() is False
    assert failed.api_client.auth.is_authenticated is False
    assert failed.current_account.wallet_address is None

    await sdk.aclose()
    await failed.aclose()


@pytest.mark.asyncio
async def test_check_auth_rejects_non_cookie_authentication() -> None:
    sdk = await SogniClient.create(app_id="token-app", disable_socket=True)

    with pytest.raises(RuntimeError, match="cookie auth"):
        await sdk.check_auth()

    await sdk.aclose()


@pytest.mark.asyncio
async def test_socket_subscription_alias_and_close_are_forwarded_once() -> None:
    sdk = await SogniClient.create(app_id="app", disable_socket=True)
    update = {"subscriptions": {"modelAvailability": False}}

    await sdk.setSocketEventSubscriptions(update)
    assert sdk.api_client.subscription_updates == [update]
    assert sdk.api_client.socket.subscription_updates == [update]

    await sdk.dispose()
    await sdk.aclose()
    assert sdk.api_client.closed == 1


@pytest.mark.asyncio
async def test_async_context_manager_returns_client_and_closes_on_exit() -> None:
    sdk = await SogniClient.create(app_id="context-app", disable_socket=True)

    async with sdk as entered:
        assert entered is sdk
        assert sdk.api_client.closed == 0

    assert sdk.api_client.closed == 1


def test_async_client_name_is_a_compatibility_alias() -> None:
    assert AsyncSogniClient is SogniClient
