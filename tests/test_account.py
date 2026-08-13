from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from sogni_client.account import AccountApi, CurrentAccount, _parse_ether
from sogni_client.auth import ApiKeyAuthManager
from sogni_client.errors import ApiError
from sogni_client.events import EventEmitter


class FakeRest:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def _response(self) -> Any:
        if not self.responses:
            raise AssertionError("Unexpected REST call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append({"method": "GET", "path": path, "params": params})
        return self._response()

    async def post(self, path: str, body: Any = None) -> Any:
        self.calls.append({"method": "POST", "path": path, "body": body})
        return self._response()


class FakeSocket(EventEmitter):
    def __init__(self, *, connected: bool = False, network: str | None = None) -> None:
        super().__init__()
        self.is_connected = connected
        self.supernet_type = network
        self.switches: list[str] = []

    async def switch_network(self, network: str) -> str:
        self.switches.append(network)
        self.supernet_type = network
        return network


class FakeClient(EventEmitter):
    def __init__(
        self,
        rest: FakeRest,
        *,
        connected: bool = False,
        network: str | None = None,
        app_id: str = "pytest-app",
        app_source: str | None = "pytest-source",
    ) -> None:
        super().__init__()
        self.rest = rest
        self.socket = FakeSocket(connected=connected, network=network)
        self.auth = ApiKeyAuthManager()
        self.app_id = app_id
        self.app_source = app_source


def test_current_account_defaults_aliases_and_unlimited_entitlement() -> None:
    account = CurrentAccount(
        {
            "networkStatus": "connected",
            "network": "fast",
            "walletAddress": "0xabc",
            "username": "Ada",
            "subscription": {"active": True, "tier": "unlimited_pro"},
        }
    )

    assert account.network_status == account.networkStatus == "connected"
    assert account.wallet_address == account.walletAddress == "0xabc"
    assert account.is_authenticated is True
    assert account.isAuthenticated is True
    assert account.is_authenicated is True
    assert account.isAuthenicated is True
    assert account.is_unlimited is account.isUnlimited is True

    snapshot = account.to_dict()
    snapshot["balance"]["sogni"]["net"] = "999"
    assert account.balance["sogni"]["net"] == "0"

    account._clear()
    assert account.is_authenticated is False
    assert account.is_unlimited is False
    assert account.network_status == "disconnected"
    assert account.subscription is None


def test_account_tracks_client_and_socket_state_events() -> None:
    client = FakeClient(FakeRest(), connected=True, network="fast")
    api = AccountApi(client)

    assert api.current_account.network_status == "connected"
    assert api.currentAccount is api.current_account

    client.emit("connecting", {"network": "relaxed"})
    assert api.current_account.network_status == "connecting"
    assert api.current_account.network == "relaxed"

    client.emit("connected", {"network": "relaxed"})
    assert api.current_account.network_status == "connected"
    client.socket.emit("balanceUpdate", {"sogni": {"net": "12"}})
    assert api.current_account.balance == {"sogni": {"net": "12"}}

    client.socket.emit("changeNetwork", {"network": "fast"})
    assert api.current_account.network == "fast"
    client.emit("disconnected")
    assert api.current_account.network_status == "disconnected"
    assert api.current_account.network is None


def test_socket_authentication_and_entitlement_mapping_reject_stale_versions() -> None:
    client = FakeClient(FakeRest())
    api = AccountApi(client)
    entitlement = {
        "active": True,
        "subscription": {
            "status": "grace",
            "tier": "unlimited",
            "term": "monthly",
            "version": "8",
            "periodStart": 1_000,
            "periodEnd": 2_000,
            "graceEnd": 3_000,
            "scheduledTier": "unlimited_pro",
            "scheduledChangeAt": 4_000,
            "paymentPending": True,
        },
    }

    client.socket.emit(
        "authenticated",
        {
            "username": "ada",
            "address": "0xabc",
            "subscriptionEntitlement": entitlement,
        },
    )

    subscription = api.current_account.subscription
    assert api.current_account.username == "ada"
    assert api.current_account.wallet_address == "0xabc"
    assert subscription == {
        "active": True,
        "status": "grace_period",
        "tier": "unlimited",
        "term": "monthly",
        "scheduledTier": "unlimited_pro",
        "currentPeriodStart": "1970-01-01T00:00:01.000Z",
        "currentPeriodEnd": "1970-01-01T00:00:03.000Z",
        "scheduledChangeAt": "1970-01-01T00:00:04.000Z",
        "paymentPending": True,
        "capabilities": {"unlimited": True},
    }

    client.socket.emit(
        "subscriptionEntitlementUpdated",
        {
            "active": False,
            "subscription": {"status": "expired", "tier": "unlimited", "version": 7},
        },
    )
    assert api.current_account.subscription is subscription

    client.socket.emit(
        "subscriptionEntitlementUpdated",
        {
            "active": True,
            "subscription": {"status": "cancelled", "tier": "unlimited", "version": 9},
        },
    )
    assert api.current_account.subscription == {
        "active": True,
        "status": "cancel_at_period_end",
        "tier": "unlimited",
        "cancelAtPeriodEnd": True,
        "capabilities": {"unlimited": True},
    }


@pytest.mark.asyncio
async def test_login_and_create_serialize_aliases_and_optional_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rest = FakeRest(
        [
            {"data": {"nonce": "login-nonce"}},
            {"data": {"token": "t", "refreshToken": "r", "user": "ada"}},
            {"data": {"nonce": "signup-nonce"}},
            {"data": {"token": "t2", "refreshToken": "r2", "user": "grace"}},
        ]
    )
    api = AccountApi(FakeClient(rest))
    wallet = SimpleNamespace(address="0x0000000000000000000000000000000000000abc")
    monkeypatch.setattr(AccountApi, "get_wallet", staticmethod(lambda _u, _p: wallet))
    monkeypatch.setattr(AccountApi, "_sign_authentication", lambda _self, _w, _n: "login-sig")
    monkeypatch.setattr(AccountApi, "_sign_signup", lambda _self, _w, _p, _n: "signup-sig")

    login = await api.login("Ada", "secret", rememberMe=True, appSource="desktop")
    created = await api.create(
        username="Grace",
        email="grace@example.test",
        password="secret",
        subscribe=True,
        turnstileToken="turnstile",
        referralCode="REF",
        rememberMe=True,
        appSource="mobile",
    )

    assert login["user"] == "ada"
    assert created["user"] == "grace"
    assert rest.calls == [
        {
            "method": "POST",
            "path": "/v1/account/nonce",
            "body": {"walletAddress": wallet.address},
        },
        {
            "method": "POST",
            "path": "/v1/account/login",
            "body": {
                "walletAddress": wallet.address,
                "signature": "login-sig",
                "rememberMe": True,
                "appSource": "desktop",
            },
        },
        {
            "method": "POST",
            "path": "/v1/account/nonce",
            "body": {"walletAddress": wallet.address},
        },
        {
            "method": "POST",
            "path": "/v1/account/create",
            "body": {
                "appid": "pytest-app",
                "username": "Grace",
                "email": "grace@example.test",
                "subscribe": 1,
                "walletAddress": wallet.address,
                "turnstileToken": "turnstile",
                "signature": "signup-sig",
                "appSource": "mobile",
                "referralCode": "REF",
                "rememberMe": True,
            },
        },
    ]


@pytest.mark.asyncio
async def test_account_queries_update_entity_and_preserve_wire_names() -> None:
    balance = {"sogni": {"net": "7"}, "spark": {"net": "3"}}
    rest = FakeRest(
        [
            {"data": balance},
            {"data": {"sogni": "22"}},
            {
                "data": {
                    "username": "ada",
                    "currentEmail": "new@example.test",
                    "walletAddress": "0xabc",
                }
            },
            {"status": "success", "data": {"available": True}},
            {"status": "success"},
        ]
    )
    api = AccountApi(FakeClient(rest))

    assert await api.refresh_balance() is balance
    assert api.current_account.balance is balance
    assert await api.walletBalance(walletAddress="0xabc", provider="etherlink") == {"sogni": "22"}
    assert (await api.me())["username"] == "ada"
    assert api.current_account.email == "new@example.test"
    assert api.current_account.wallet_address == "0xabc"
    assert await api.validateUsername("available-name") == {
        "status": "success",
        "data": {"available": True},
    }
    await api.claimRewards(rewardIds=["daily", "referral"], turnstileToken="token", provider="base")

    assert rest.calls == [
        {"method": "GET", "path": "/v4/account/balance", "params": None},
        {
            "method": "GET",
            "path": "/v2/wallet/balance",
            "params": {"walletAddress": "0xabc", "provider": "etherlink"},
        },
        {"method": "GET", "path": "/v1/account/me", "params": None},
        {
            "method": "POST",
            "path": "/v1/account/username/validate",
            "body": {"username": "available-name"},
        },
        {
            "method": "POST",
            "path": "/v3/account/reward/claim",
            "body": {
                "claims": ["daily", "referral"],
                "provider": "base",
                "turnstileToken": "token",
            },
        },
    ]


@pytest.mark.asyncio
async def test_validate_username_returns_structured_expected_conflict() -> None:
    payload = {
        "status": "error",
        "message": "Username is unavailable",
        "errorCode": 108,
        "data": {"available": False},
    }
    api = AccountApi(FakeClient(FakeRest([ApiError(409, payload)])))

    assert await api.validate_username("taken") is payload


@pytest.mark.asyncio
async def test_transaction_history_serializes_query_and_converts_millisecond_dates() -> None:
    raw = {
        "id": "tx-1",
        "address": "0xabc",
        "status": "complete",
        "role": "sender",
        "amount": "4",
        "tokenType": "spark",
        "description": "Render",
        "source": "generation",
        "type": "debit",
        "createTime": 1_000,
        "updateTime": 2_000,
        "endTime": 3_000,
    }
    rest = FakeRest([{"data": {"transactions": [raw], "next": 40}}])
    api = AccountApi(FakeClient(rest))

    result = await api.transactionHistory(
        status="complete", address="0xabc", limit=20, offset=20, provider="base"
    )

    assert result["entries"][0]["createTime"] == datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    assert result["entries"][0]["endTime"] == datetime(1970, 1, 1, 0, 0, 3, tzinfo=timezone.utc)
    assert result["next"] == {
        "status": "complete",
        "address": "0xabc",
        "limit": 20,
        "offset": 40,
        "provider": "base",
    }
    assert rest.calls == [
        {
            "method": "GET",
            "path": "/v1/transactions/list",
            "params": {
                "status": "complete",
                "address": "0xabc",
                "limit": "20",
                "provider": "base",
                "offset": "20",
            },
        }
    ]


@pytest.mark.asyncio
async def test_subscription_endpoints_unwrap_and_serialize_false_and_empty_values() -> None:
    subscription = {"active": True, "status": "active", "tier": "unlimited"}
    usage = {"jobs": 3, "trialCreditsUsed": 2}
    plans = [{"id": "unlimited", "priceUsd": 20}]
    rest = FakeRest(
        [
            {"data": {"subscription": subscription}},
            {"data": {"usage": usage}},
            {"data": {"eligible": False, "reasonCode": "wallet_already_used"}},
            {"status": "success"},
            {"data": {"plans": plans}},
            {"data": {"url": "https://checkout.example/session"}},
            {"data": {"url": "https://portal.example/session"}},
        ]
    )
    api = AccountApi(FakeClient(rest))

    assert await api.getSubscriptionStatus() is subscription
    assert api.current_account.subscription is subscription
    assert await api.getSubscriptionUsage() is usage
    assert await api.getTrialEligibility() == {
        "eligible": False,
        "reasonCode": "wallet_already_used",
    }
    await api.setDeviceId(deviceId="device-1")
    assert await api.getSubscriptionPlans() is plans
    assert await api.createSubscriptionCheckout(
        planId="unlimited",
        term="monthly",
        redirectType="app",
        appSource="ios",
        startTrial=False,
        deviceId="",
    ) == {"url": "https://checkout.example/session"}
    assert await api.createSubscriptionPortalSession() == {"url": "https://portal.example/session"}

    assert rest.calls[-4:] == [
        {
            "method": "POST",
            "path": "/v1/account/device-id",
            "body": {"deviceId": "device-1"},
        },
        {"method": "GET", "path": "/v1/subscriptions/plans", "params": None},
        {
            "method": "POST",
            "path": "/v1/iap/stripe/subscribe",
            "body": {
                "planId": "unlimited",
                "term": "monthly",
                "redirectType": "app",
                "appSource": "ios",
                "startTrial": False,
                "deviceId": "",
            },
        },
        {
            "method": "POST",
            "path": "/v1/subscriptions/stripe/portal",
            "body": {},
        },
    ]


class DeferredSubscriptionRest(FakeRest):
    def __init__(self, response: Any) -> None:
        super().__init__()
        self.response = response
        self.request_started = asyncio.Event()
        self.release_response = asyncio.Event()

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append({"method": "GET", "path": path, "params": params})
        self.request_started.set()
        await self.release_response.wait()
        return self.response


@pytest.mark.asyncio
async def test_in_flight_rest_subscription_cannot_overwrite_newer_socket_push() -> None:
    stale = {"active": False, "status": "none"}
    rest = DeferredSubscriptionRest({"data": {"subscription": stale}})
    client = FakeClient(rest)
    api = AccountApi(client)

    request = asyncio.create_task(api.get_subscription_status())
    await rest.request_started.wait()
    client.socket.emit(
        "subscriptionEntitlementUpdated",
        {
            "active": True,
            "subscription": {
                "status": "active",
                "tier": "unlimited",
                # Deliberately omit a version to exercise the logical write clock.
            },
        },
    )
    fresh = api.current_account.subscription
    rest.release_response.set()

    assert await request is fresh
    assert api.current_account.subscription is fresh
    assert fresh == {
        "active": True,
        "status": "active",
        "tier": "unlimited",
        "capabilities": {"unlimited": True},
    }


@pytest.mark.asyncio
async def test_switch_network_and_logout_keep_current_entity_consistent() -> None:
    rest = FakeRest([ApiError(401, {"message": "already logged out"})])
    client = FakeClient(rest, connected=True, network="fast")
    client.auth._api_key = "secret"
    api = AccountApi(client)
    api.current_account._update({"wallet_address": "0xabc", "username": "ada"})

    assert await api.switchNetwork("relaxed") == "relaxed"
    assert client.socket.switches == ["relaxed"]
    assert api.current_account.network_status == "connected"
    assert api.current_account.network == "relaxed"
    with pytest.raises(ValueError, match="fast.*relaxed"):
        await api.switch_network("turbo")

    await api.logout()
    assert client.auth.is_authenticated is False
    assert api.current_account.wallet_address is None
    assert rest.calls == [{"method": "POST", "path": "/v1/account/logout", "body": None}]


@pytest.mark.parametrize(
    ("value", "wei"),
    [
        (1, "1000000000000000000"),
        ("0.000000000000000001", "1"),
        (Decimal("1.25"), "1250000000000000000"),
        ("-.5", "-500000000000000000"),
    ],
)
def test_parse_ether_matches_decimal_ether_units(value: Any, wei: str) -> None:
    assert _parse_ether(value) == wei


@pytest.mark.parametrize("value", ["", "1e3", "nan", "0.0000000000000000001"])
def test_parse_ether_rejects_non_decimal_or_overprecise_values(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_ether(value)
