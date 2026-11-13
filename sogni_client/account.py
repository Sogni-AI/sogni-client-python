"""Account, wallet, balance, reward, and subscription APIs."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from .auth import ApiKeyAuthManager, CookieAuthManager, TokenAuthManager
from .errors import ApiError, TwoFactorRequiredError
from .events import DataEntity, EventEmitter
from .transport import ApiClient

_LOGGER = logging.getLogger("sogni_client")
_MAX_DEPOSIT_ATTEMPTS = 4
_INSUFFICIENT_ALLOWANCE = 149
_VERIFYING_CONTRACT = "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"


def _account_defaults() -> dict[str, Any]:
    return {
        "network_status": "disconnected",
        "network": None,
        "balance": {
            "sogni": {"credit": "0", "debit": "0", "net": "0", "settled": "0"},
            "spark": {
                "credit": "0",
                "debit": "0",
                "net": "0",
                "settled": "0",
                "premiumCredit": "0",
            },
        },
        "wallet_address": None,
        "username": None,
        "email": None,
        "subscription": None,
        "free_spark_locked": None,
        "free_spark_unlock_path": None,
    }


def _unwrap_data(response: Any) -> Any:
    if isinstance(response, dict) and "data" in response:
        return response["data"]
    return response


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _iso_from_milliseconds(value: Any) -> str:
    return (
        datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _date_from_milliseconds(value: Any) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)


def _parse_ether(value: int | float | str | Decimal) -> str:
    """Match ethers.parseEther for ordinary decimal input."""

    text = (format(value, "f") if isinstance(value, Decimal) else str(value)).strip()
    if not re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)", text):
        raise ValueError("amount must be a base-10 decimal with at most 18 decimal places")
    negative = text.startswith("-")
    unsigned = text.removeprefix("-")
    whole, _, fraction = unsigned.partition(".")
    if len(fraction.rstrip("0")) > 18:
        raise ValueError("amount has more than 18 decimal places")
    fraction = fraction[:18].ljust(18, "0")
    wei = int(whole or "0") * 10**18 + int(fraction or "0")
    return str(-wei if negative else wei)


def _load_eth_account() -> tuple[Any, Any]:
    try:
        from eth_account import Account
        from eth_account.messages import encode_typed_data
    except ImportError as error:  # pragma: no cover - dependency failure is environment-specific
        raise RuntimeError("Account login and signing require the 'eth-account' package") from error
    return Account, encode_typed_data


def _factor_proof(value: str | Mapping[str, Any]) -> dict[str, Any]:
    """An authenticator or backup code as the API expects it."""

    if isinstance(value, str):
        return {"code": value.strip().upper()}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("proof must be an authenticator or backup code")


def _signature_hex(signature: Any) -> str:
    value = signature.hex()
    return value if value.startswith("0x") else f"0x{value}"


def _sign_typed_data(
    wallet: Any,
    domain: Mapping[str, Any],
    types: Mapping[str, Any],
    message: Mapping[str, Any],
) -> str:
    _, encode_typed_data = _load_eth_account()
    message_types = {key: value for key, value in types.items() if key != "EIP712Domain"}
    signable = encode_typed_data(
        domain_data=dict(domain),
        message_types=message_types,
        message_data=dict(message),
    )
    return _signature_hex(wallet.sign_message(signable).signature)


class CurrentAccount(DataEntity):
    """Observable snapshot of the authenticated account."""

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        initial = _account_defaults()
        if data:
            supplied = dict(data)
            aliases = {
                "networkStatus": "network_status",
                "walletAddress": "wallet_address",
                "freeSparkLocked": "free_spark_locked",
                "freeSparkUnlockPath": "free_spark_unlock_path",
            }
            for source, target in aliases.items():
                if source in supplied and target not in supplied:
                    supplied[target] = supplied.pop(source)
            initial.update(supplied)
        super().__init__(initial)

    def _clear(self) -> None:
        self._update(_account_defaults())

    @property
    def is_authenticated(self) -> bool:
        return bool(self._data.get("wallet_address"))

    @property
    def is_authenicated(self) -> bool:
        """Compatibility for the misspelled property in the JavaScript SDK."""

        return self.is_authenticated

    @property
    def network_status(self) -> str:
        return str(self._data["network_status"])

    @property
    def network(self) -> str | None:
        return self._data.get("network")

    @property
    def balance(self) -> dict[str, Any]:
        return self._data["balance"]

    @property
    def wallet_address(self) -> str | None:
        return self._data.get("wallet_address")

    @property
    def username(self) -> str | None:
        return self._data.get("username")

    @property
    def email(self) -> str | None:
        return self._data.get("email")

    @property
    def subscription(self) -> dict[str, Any] | None:
        return self._data.get("subscription")

    @property
    def free_spark_locked(self) -> bool | None:
        return self._data.get("free_spark_locked")

    @property
    def free_spark_unlock_path(self) -> str | None:
        return self._data.get("free_spark_unlock_path")

    @property
    def is_unlimited(self) -> bool:
        subscription = self.subscription
        return bool(
            subscription
            and subscription.get("active")
            and subscription.get("tier") in {"unlimited", "unlimited_pro"}
        )

    # JavaScript SDK compatibility spellings.
    isAuthenticated = property(lambda self: self.is_authenticated)
    isAuthenicated = property(lambda self: self.is_authenicated)
    networkStatus = property(lambda self: self.network_status)
    walletAddress = property(lambda self: self.wallet_address)
    isUnlimited = property(lambda self: self.is_unlimited)
    freeSparkLocked = property(lambda self: self.free_spark_locked)
    freeSparkUnlockPath = property(lambda self: self.free_spark_unlock_path)


class AccountApi(EventEmitter):
    """Async account API exposed as :attr:`SogniClient.account`."""

    def __init__(self, client: ApiClient, *, testnet: bool = False) -> None:
        super().__init__()
        self.client = client
        self.testnet = testnet
        self.current_account = CurrentAccount()
        self.currentAccount = self.current_account
        self._last_subscription_version: int | float | None = None
        self._subscription_socket_writes = 0
        self._account_session_version = self.client.auth.session_version
        self._socket_priority_availability: tuple[int, bool] | None = None

        self.current_account._update(
            {
                "network_status": (
                    "connected" if self.client.socket.is_connected else "disconnected"
                ),
                "network": self.client.socket.supernet_type,
            }
        )
        self.client.socket.on("balanceUpdate", self._on_balance_update)
        self.client.socket.on("changeNetwork", self._on_change_network)
        self.client.socket.on("authenticated", self._on_socket_authenticated)
        self.client.socket.on(
            "subscriptionEntitlementUpdated", self._on_subscription_entitlement_updated
        )
        self.client.on("connecting", self._on_connecting)
        self.client.on("connected", self._on_connected)
        self.client.on("disconnected", self._on_disconnected)
        self.client.auth.on("updated", self._on_auth_updated)
        self.client.on("sessionChanged", lambda _data: self._on_auth_updated(False))
        self.client.auth.on("sessionChanged", self._clear_changed_account_session)

    @property
    def _eip712_domain(self) -> dict[str, Any]:
        return {
            "name": "Sogni-testnet" if self.testnet else "Sogni AI",
            "version": "1",
            "chainId": 84532 if self.testnet else 8453,
            "verifyingContract": _VERIFYING_CONTRACT,
        }

    def _on_balance_update(self, data: Any) -> None:
        if isinstance(data, dict):
            self.current_account._update({"balance": data})

    def _on_change_network(self, data: Any) -> None:
        if isinstance(data, dict):
            network = data.get("network")
        else:
            network = data
        self.current_account._update({"network": network, "network_status": "connected"})

    def _on_connecting(self, data: Any) -> None:
        self._clear_socket_priority_availability()
        network = data.get("network") if isinstance(data, dict) else data
        self.current_account._update({"network": network, "network_status": "connecting"})

    def _on_connected(self, data: Any) -> None:
        self._clear_socket_priority_availability()
        network = data.get("network") if isinstance(data, dict) else data
        self.current_account._update({"network": network, "network_status": "connected"})

    def _on_disconnected(self, _data: Any) -> None:
        self._clear_socket_priority_availability()
        self.current_account._update({"network": None, "network_status": "disconnected"})

    def _clear_socket_priority_availability(self) -> None:
        self._socket_priority_availability = None
        subscription = self.current_account.subscription
        capabilities = subscription.get("capabilities") if subscription else None
        if not isinstance(capabilities, dict) or not capabilities.get("priorityMode"):
            return
        capabilities = {key: value for key, value in capabilities.items() if key != "priorityMode"}
        self.current_account._update(
            {"subscription": {**subscription, "capabilities": capabilities}}
        )

    def _with_socket_priority_availability(self, subscription: dict[str, Any]) -> dict[str, Any]:
        """A REST plan refresh cannot replace the current connection's Priority verdict."""
        verdict = self._socket_priority_availability
        available = bool(
            verdict
            and verdict[0] == self.client.auth.session_version
            and verdict[1]
            and self.client.socket.is_connected
            and subscription.get("active")
            and subscription.get("status") != "trialing"
            and subscription.get("tier") in {"unlimited", "unlimited_pro"}
        )
        raw_capabilities = subscription.get("capabilities")
        capabilities = raw_capabilities if isinstance(raw_capabilities, dict) else {}
        if available == (capabilities.get("priorityMode") is True):
            return subscription
        capabilities = dict(capabilities)
        if available:
            capabilities["priorityMode"] = True
        else:
            capabilities.pop("priorityMode", None)
        return {**subscription, "capabilities": capabilities}

    def _clear_changed_account_session(self, _data: Any = None) -> None:
        if self._account_session_version == self.client.auth.session_version:
            return
        self._account_session_version = self.client.auth.session_version
        self._last_subscription_version = None
        self._subscription_socket_writes = 0
        self._socket_priority_availability = None
        self.current_account._clear()

    def _on_auth_updated(self, authenticated: bool) -> Any:
        if not authenticated:
            self._last_subscription_version = None
            self._subscription_socket_writes = 0
            self._socket_priority_availability = None
            self.current_account._clear()
            return None
        self._clear_changed_account_session()
        return self.me()

    @staticmethod
    def _parse_subscription_version(value: Any) -> int | float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        if not number.is_integer():
            return number
        return int(number)

    def _map_socket_subscription(self, data: Any) -> dict[str, Any] | None:
        if not isinstance(data, dict):
            return None
        subscription = data.get("subscription")
        if not isinstance(subscription, dict) or not subscription.get("status"):
            return {"active": False, "status": "none"}

        producer_status = subscription.get("status")
        active = data.get("active") is True
        if producer_status == "grace":
            status = "grace_period"
        elif producer_status == "cancelled":
            status = "cancel_at_period_end" if active else "canceled"
        elif producer_status == "revoked":
            status = "canceled"
        elif producer_status == "needs_reconciliation":
            status = "past_due"
        elif producer_status == "expired":
            status = "expired"
        elif producer_status == "trialing":
            status = "trialing"
        elif producer_status == "active":
            status = "active"
        else:
            status = "active" if active else "none"

        period_end = (
            subscription.get("graceEnd")
            if status == "grace_period" and subscription.get("graceEnd")
            else subscription.get("periodEnd")
        )
        cancel_at_period_end = subscription.get("cancelAtPeriodEnd")
        if not isinstance(cancel_at_period_end, bool):
            cancel_at_period_end = True if producer_status == "cancelled" else None

        mapped: dict[str, Any] = {"active": active, "status": status}
        direct_fields = {
            "tier": "tier",
            "term": "term",
            "provider": "provider",
            "scheduledTier": "scheduledTier",
            "scheduledTerm": "scheduledTerm",
        }
        for source, target in direct_fields.items():
            if subscription.get(source):
                mapped[target] = subscription[source]
        if subscription.get("periodStart"):
            mapped["currentPeriodStart"] = _iso_from_milliseconds(subscription["periodStart"])
        if period_end:
            mapped["currentPeriodEnd"] = _iso_from_milliseconds(period_end)
        if cancel_at_period_end is not None:
            mapped["cancelAtPeriodEnd"] = cancel_at_period_end
        if subscription.get("scheduledChangeAt"):
            mapped["scheduledChangeAt"] = _iso_from_milliseconds(subscription["scheduledChangeAt"])
        if subscription.get("paymentPending"):
            mapped["paymentPending"] = True
        vendor_discount_bps = subscription.get("vendorDiscountBps")
        if (
            active
            and isinstance(vendor_discount_bps, (int, float))
            and not isinstance(vendor_discount_bps, bool)
            and vendor_discount_bps > 0
        ):
            mapped["frontierDiscountBps"] = vendor_discount_bps
        capabilities = subscription.get("capabilities")
        if isinstance(capabilities, dict):
            mapped["capabilities"] = capabilities
        elif active and subscription.get("tier") in {"unlimited", "unlimited_pro"}:
            mapped["capabilities"] = {"unlimited": True}
        else:
            mapped["capabilities"] = {}
        fair_use = subscription.get("fairUse")
        if (
            isinstance(fair_use, dict)
            and fair_use.get("limited") is True
            and isinstance(fair_use.get("resetAt"), (int, float))
            and fair_use["resetAt"] > datetime.now(timezone.utc).timestamp() * 1000
        ):
            mapped["fairUse"] = {
                **fair_use,
                "resetAt": _iso_from_milliseconds(fair_use["resetAt"]),
            }
        return mapped

    def _apply_free_spark_locked(self, locked: Any, unlock_path: Any) -> None:
        if not isinstance(locked, bool):
            return
        path = unlock_path if unlock_path in {"trial", "purchase"} else None
        self.current_account._update({"free_spark_locked": locked, "free_spark_unlock_path": path})

    def _apply_subscription(
        self,
        subscription: dict[str, Any],
        *,
        source: str,
        version: int | float | None = None,
        socket_writes_at_start: int | None = None,
    ) -> bool:
        if (
            version is not None
            and self._last_subscription_version is not None
            and version < self._last_subscription_version
        ):
            return False
        if (
            source == "rest"
            and version is None
            and socket_writes_at_start is not None
            and self._subscription_socket_writes != socket_writes_at_start
        ):
            return False
        if version is not None:
            self._last_subscription_version = version
        if source == "socket":
            self._subscription_socket_writes += 1
            capabilities = subscription.get("capabilities")
            self._socket_priority_availability = (
                self.client.auth.session_version,
                isinstance(capabilities, dict) and capabilities.get("priorityMode") is True,
            )
        self.current_account._update(
            {"subscription": self._with_socket_priority_availability(subscription)}
        )
        return True

    def _on_subscription_entitlement_updated(self, data: Any) -> None:
        if isinstance(data, dict):
            self._apply_free_spark_locked(
                data.get("freeSparkLocked"), data.get("freeSparkUnlockPath")
            )
        mapped = self._map_socket_subscription(data)
        if mapped is None:
            return
        raw = data.get("subscription") if isinstance(data, dict) else None
        version = self._parse_subscription_version(
            raw.get("version") if isinstance(raw, dict) else None
        )
        self._apply_subscription(mapped, source="socket", version=version)

    def _on_socket_authenticated(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return None
        self._apply_free_spark_locked(data.get("freeSparkLocked"), data.get("freeSparkUnlockPath"))
        if isinstance(self.client.auth, ApiKeyAuthManager):
            self.current_account._update(
                {"username": data.get("username"), "wallet_address": data.get("address")}
            )
        entitlement = data.get("subscriptionEntitlement")
        mapped = self._map_socket_subscription(entitlement)
        if mapped is None:
            return self._refresh_subscription_best_effort()
        raw = entitlement.get("subscription") if isinstance(entitlement, dict) else None
        version = self._parse_subscription_version(
            raw.get("version") if isinstance(raw, dict) else None
        )
        self._apply_subscription(mapped, source="socket", version=version)
        return None

    async def _refresh_subscription_best_effort(self) -> None:
        try:
            await self.refresh_subscription()
        except Exception:
            _LOGGER.debug(
                "Failed to refresh the subscription after socket authentication",
                exc_info=True,
            )

    async def get_nonce(self, wallet_address: str | None = None, **kwargs: Any) -> str:
        wallet_address = wallet_address or kwargs.get("walletAddress")
        if not wallet_address:
            raise ValueError("wallet_address is required")
        response = await self.client.rest.post(
            "/v1/account/nonce", {"walletAddress": wallet_address}
        )
        data = _unwrap_data(response)
        if not isinstance(data, dict) or not isinstance(data.get("nonce"), str):
            raise ValueError("Nonce response did not include data.nonce")
        return data["nonce"]

    @staticmethod
    def get_wallet(username: str, password: str) -> Any:
        if not isinstance(username, str) or not isinstance(password, str):
            raise TypeError("username and password must be strings")
        Account, _ = _load_eth_account()
        private_key = hashlib.pbkdf2_hmac(
            "sha256",
            f"{username.lower()}{password}".encode(),
            b"sogni-salt-value",
            10_000,
            dklen=32,
        )
        return Account.from_key(private_key)

    def _sign_authentication(self, wallet: Any, nonce: str) -> str:
        types = {
            "Authentication": [
                {"name": "walletAddress", "type": "address"},
                {"name": "nonce", "type": "string"},
            ]
        }
        return _sign_typed_data(
            wallet,
            self._eip712_domain,
            types,
            {"walletAddress": wallet.address, "nonce": nonce},
        )

    def _sign_signup(self, wallet: Any, payload: Mapping[str, Any], nonce: str) -> str:
        types = {
            "Signup": [
                {"name": "appid", "type": "string"},
                {"name": "username", "type": "string"},
                {"name": "email", "type": "string"},
                {"name": "subscribe", "type": "uint256"},
                {"name": "walletAddress", "type": "address"},
                {"name": "nonce", "type": "string"},
            ]
        }
        message = {
            "appid": payload["appid"],
            "username": payload["username"],
            "email": payload["email"],
            "subscribe": payload["subscribe"],
            "walletAddress": payload["walletAddress"],
            "nonce": nonce,
        }
        return _sign_typed_data(wallet, self._eip712_domain, types, message)

    async def create(
        self,
        params: Mapping[str, Any] | None = None,
        remember_me: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        values = {**dict(params or {}), **kwargs}
        username = _first(values, "username")
        email = _first(values, "email")
        password = _first(values, "password")
        subscribe = bool(_first(values, "subscribe", default=False))
        turnstile_token = _first(values, "turnstile_token", "turnstileToken")
        referral_code = _first(values, "referral_code", "referralCode")
        app_source = _first(values, "app_source", "appSource")
        remember_me = bool(_first(values, "remember_me", "rememberMe", default=remember_me))
        if not all(isinstance(value, str) and value for value in (username, email, password)):
            raise ValueError("username, email, and password are required")

        wallet = self.get_wallet(username, password)
        nonce = await self.get_nonce(wallet.address)
        payload: dict[str, Any] = {
            "appid": self.client.app_id,
            "username": username,
            "email": email,
            "subscribe": 1 if subscribe else 0,
            "walletAddress": wallet.address,
            "turnstileToken": turnstile_token,
        }
        resolved_source = str(app_source).strip() if app_source is not None else ""
        resolved_source = resolved_source or self.client.app_source
        payload["signature"] = self._sign_signup(wallet, payload, nonce)
        if resolved_source:
            payload["appSource"] = resolved_source
        if referral_code is not None:
            payload["referralCode"] = referral_code
        payload["rememberMe"] = remember_me
        response = await self.client.rest.post("/v1/account/create", payload)
        data = _unwrap_data(response)
        if not isinstance(data, dict):
            raise ValueError("Account create response did not include data")
        await self._authenticate_from_response(data)
        return data

    async def _authenticate_from_response(self, data: Mapping[str, Any]) -> None:
        if isinstance(self.client.auth, TokenAuthManager):
            await self.client.auth.authenticate(
                {
                    "token": data.get("token"),
                    "refreshToken": data.get("refreshToken"),
                }
            )
        elif isinstance(self.client.auth, CookieAuthManager):
            await self.client.auth.authenticate()

    async def login(
        self,
        username: str,
        password: str,
        remember_me: bool = False,
        app_source: str | None = None,
        *,
        restore_deleted_account: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Sign in with a username and password.

        Pass ``restore_deleted_account=True`` (``restoreDeletedAccount``) to
        restore a deleted account and sign in, after the person confirms. A
        deleted account that can be restored makes ``login()`` raise an
        :class:`ApiError` whose ``error_code`` is
        :data:`~sogni_client.errors.ACCOUNT_RESTORABLE_ERROR_CODE` (196) and whose
        ``details["deletedAt"]`` is when it was deleted, in Unix seconds. The
        option changes nothing for an account that has not been deleted.

        For an account with two-factor authentication, ``login()`` raises
        :class:`~sogni_client.errors.TwoFactorRequiredError` without starting a
        session. Ask the person for an authenticator or backup code and pass both
        to :meth:`verify_two_factor_login`.
        """

        remember_me = bool(kwargs.get("rememberMe", remember_me))
        app_source = kwargs.get("appSource", app_source)
        restore_deleted_account = kwargs.get("restoreDeletedAccount", restore_deleted_account)
        wallet = self.get_wallet(username, password)
        nonce = await self.get_nonce(wallet.address)
        resolved_source = (
            app_source.strip() if app_source and app_source.strip() else self.client.app_source
        )
        body: dict[str, Any] = {
            "walletAddress": wallet.address,
            "signature": self._sign_authentication(wallet, nonce),
            "rememberMe": remember_me,
        }
        if resolved_source:
            body["appSource"] = resolved_source
        if restore_deleted_account:
            body["restoreDeletedAccount"] = True
        response = await self.client.rest.post("/v1/account/login", body)
        data = _unwrap_data(response)
        if not isinstance(data, dict):
            raise ValueError("Login response did not include data")
        if data.get("requires2FA"):
            temp_token = data.get("tempToken")
            if not temp_token:
                raise ValueError("Unable to start verification. Please sign in again.")
            raise TwoFactorRequiredError(temp_token, wallet.address, data.get("username"))
        await self._authenticate_from_response(data)
        return data

    async def verify_two_factor_login(
        self, challenge: TwoFactorRequiredError, proof: str | Mapping[str, Any]
    ) -> dict[str, Any]:
        """Finish the sign-in that raised :class:`TwoFactorRequiredError`.

        ``proof`` is the person's authenticator code or one of their backup codes.
        Keep the challenge in memory; do not log or store it.
        """

        response = await self.client.rest.post(
            "/v1/account/2fa/verify-login",
            {"tempToken": challenge.temp_token, **_factor_proof(proof)},
        )
        data = _unwrap_data(response)
        if not isinstance(data, dict):
            raise ValueError("Two-factor verification response did not include data")
        await self._authenticate_from_response(data)
        return {**data, "username": challenge.username}

    verifyTwoFactorLogin = verify_two_factor_login

    async def get_two_factor_status(self) -> dict[str, Any]:
        """Two-factor settings of the signed-in account."""

        data = _unwrap_data(await self.client.rest.get("/v1/account/2fa/status"))
        if not isinstance(data, dict):
            raise ValueError("Two-factor status response did not include data")
        return data

    getTwoFactorStatus = get_two_factor_status

    async def _two_factor_signature(self, username: str, password: str) -> str:
        wallet = self.get_wallet(username, password)
        current = self.current_account.wallet_address
        if not current or wallet.address.lower() != current.lower():
            raise ValueError("Incorrect password. Please try again.")
        nonce = await self.get_nonce(wallet.address)
        return self._sign_authentication(wallet, nonce)

    async def setup_two_factor(
        self,
        username: str,
        password: str,
        proof: str | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start authenticator setup. Needs a password session.

        Returns ``setupId``, the authenticator ``secret`` and ``qrData``, and
        ``backupCodes``. Show them to the person, then call
        :meth:`confirm_two_factor_setup` with a code from their authenticator.
        Pass ``proof`` when two-factor authentication is already on.
        """

        body: dict[str, Any] = {"signature": await self._two_factor_signature(username, password)}
        if proof:
            body["proof"] = _factor_proof(proof)
        data = _unwrap_data(await self.client.rest.post("/v1/account/2fa/setup", body))
        if not isinstance(data, dict):
            raise ValueError("Two-factor setup response did not include data")
        return data

    setupTwoFactor = setup_two_factor

    async def confirm_two_factor_setup(self, setup_id: str, code: str) -> None:
        """Turn on two-factor authentication with a code from the new authenticator."""

        await self.client.rest.post(
            "/v1/account/2fa/verify-setup", {"setupId": setup_id, "code": code.strip()}
        )

    confirmTwoFactorSetup = confirm_two_factor_setup

    async def disable_two_factor(
        self, username: str, password: str, proof: str | Mapping[str, Any]
    ) -> None:
        """Turn off two-factor authentication. Needs a password session and a code."""

        await self.client.rest.post(
            "/v1/account/2fa/disable",
            {
                "signature": await self._two_factor_signature(username, password),
                **_factor_proof(proof),
            },
        )

    disableTwoFactor = disable_two_factor

    async def regenerate_two_factor_backup_codes(
        self, username: str, password: str, proof: str | Mapping[str, Any]
    ) -> list[str]:
        """Replace the account's backup codes. Needs a password session and a code."""

        response = await self.client.rest.post(
            "/v1/account/2fa/backup-codes",
            {
                "signature": await self._two_factor_signature(username, password),
                **_factor_proof(proof),
            },
        )
        data = _unwrap_data(response)
        codes = data.get("backupCodes") if isinstance(data, dict) else None
        if not isinstance(codes, list):
            raise ValueError("Backup code response did not include data.backupCodes")
        return codes

    regenerateTwoFactorBackupCodes = regenerate_two_factor_backup_codes

    async def logout(self) -> None:
        try:
            await self.client.rest.post("/v1/account/logout")
        except ApiError as error:
            if error.status != 401:
                raise
        self.client.auth.clear()

    async def refresh_balance(self) -> dict[str, Any]:
        balance = await self.account_balance()
        self.current_account._update({"balance": balance})
        return balance

    async def account_balance(self) -> dict[str, Any]:
        data = _unwrap_data(await self.client.rest.get("/v4/account/balance"))
        if not isinstance(data, dict):
            raise ValueError("Balance response did not include data")
        return data

    async def wallet_balance(
        self, wallet_address: str | None = None, provider: str = "base", **kwargs: Any
    ) -> dict[str, Any]:
        wallet_address = wallet_address or kwargs.get("walletAddress")
        if not wallet_address:
            raise ValueError("wallet_address is required")
        data = _unwrap_data(
            await self.client.rest.get(
                "/v2/wallet/balance",
                {"walletAddress": wallet_address, "provider": provider},
            )
        )
        if not isinstance(data, dict):
            raise ValueError("Wallet balance response did not include data")
        return data

    async def me(self) -> dict[str, Any]:
        data = _unwrap_data(await self.client.rest.get("/v1/account/me"))
        if not isinstance(data, dict):
            raise ValueError("Account response did not include data")
        self.current_account._update(
            {
                "username": data.get("username"),
                "email": data.get("currentEmail", data.get("email")),
                "wallet_address": data.get("walletAddress", data.get("wallet_address")),
            }
        )
        return data

    async def validate_username(self, username: str) -> dict[str, Any]:
        try:
            response = await self.client.rest.post(
                "/v1/account/username/validate", {"username": username}
            )
            return (
                response if isinstance(response, dict) else {"status": "success", "data": response}
            )
        except ApiError as error:
            if error.error_code == 108:
                return error.payload
            raise

    async def switch_network(self, network: str) -> str:
        if network not in {"fast", "relaxed"}:
            raise ValueError("network must be 'fast' or 'relaxed'")
        self.current_account._update({"network_status": "switching", "network": None})
        resolved = await self.client.socket.switch_network(network)
        self.current_account._update({"network_status": "connected", "network": resolved})
        return resolved

    async def transaction_history(
        self, params: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        values = {**dict(params or {}), **kwargs}
        status = _first(values, "status")
        address = _first(values, "address")
        limit = _first(values, "limit")
        if status is None or address is None or limit is None:
            raise ValueError("status, address, and limit are required")
        query = {"status": str(status), "address": str(address), "limit": str(limit)}
        provider = _first(values, "provider")
        offset = _first(values, "offset")
        if provider:
            query["provider"] = str(provider)
        if offset:
            query["offset"] = str(offset)
        include_priority = (
            _first(values, "includePriorityAccounting", "include_priority_accounting") is True
        )
        if include_priority:
            query["includePriorityAccounting"] = "true"
        data = _unwrap_data(await self.client.rest.get("/v1/transactions/list", query))
        if not isinstance(data, dict):
            raise ValueError("Transaction response did not include data")
        entries: list[dict[str, Any]] = []
        for raw in data.get("transactions", []):
            if not isinstance(raw, dict):
                continue
            entry = {
                key: raw.get(key)
                for key in (
                    "id",
                    "address",
                    "status",
                    "role",
                    "amount",
                    "tokenType",
                    "description",
                    "source",
                    "type",
                    "billingMode",
                    "paymentModel",
                    "subscriptionTier",
                    "subscriptionTrialing",
                    "subscriptionThrottled",
                )
            }
            entry["createTime"] = _date_from_milliseconds(raw.get("createTime", 0))
            entry["updateTime"] = _date_from_milliseconds(raw.get("updateTime", 0))
            entry["endTime"] = _date_from_milliseconds(raw.get("endTime", 0))
            if include_priority and raw.get("priorityUsage"):
                entry["priorityUsage"] = raw["priorityUsage"]
            entries.append(entry)
        next_params = dict(values)
        next_params["offset"] = data.get("next")
        return {"entries": entries, "next": next_params}

    async def rewards(
        self, query: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        values = {**dict(query or {}), **kwargs}
        data = _unwrap_data(await self.client.rest.get("/v4/account/rewards", values))
        raw_rewards = data.get("rewards", []) if isinstance(data, dict) else []
        result: list[dict[str, Any]] = []
        for raw in raw_rewards:
            if not isinstance(raw, dict):
                continue
            timestamp = raw.get("lastClaimTimestamp", 0)
            frequency = raw.get("claimResetFrequencySec", -1)
            next_claim = None
            if timestamp and isinstance(frequency, (int, float)) and frequency > -1:
                next_claim = datetime.fromtimestamp(timestamp + frequency, tz=timezone.utc)
            result.append(
                {
                    "id": raw.get("id"),
                    "type": raw.get("type"),
                    "title": raw.get("title"),
                    "description": raw.get("description"),
                    "amount": raw.get("amount"),
                    "tokenType": raw.get("tokenType"),
                    "claimed": bool(raw.get("claimed")),
                    "canClaim": bool(raw.get("canClaim")),
                    "cantClaimReason": raw.get("cantClaimReason"),
                    "lastClaim": datetime.fromtimestamp(float(timestamp), tz=timezone.utc),
                    "nextClaim": next_claim,
                    "provider": values.get("provider") or "base",
                }
            )
        return result

    async def claim_rewards(
        self,
        reward_ids: list[str] | None = None,
        options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if reward_ids is None:
            reward_ids = kwargs.pop("rewardIds", None)
        if reward_ids is None:
            raise ValueError("reward_ids is required")
        values = {**dict(options or {}), **kwargs}
        provider = _first(values, "provider") or "base"
        body: dict[str, Any] = {
            "claims": reward_ids,
            "provider": provider,
        }
        turnstile = _first(values, "turnstile_token", "turnstileToken")
        if turnstile:
            body["turnstileToken"] = turnstile
        await self.client.rest.post("/v3/account/reward/claim", body)

    def _require_account_username(self) -> str:
        username = self.current_account.username
        if not username:
            raise ValueError("Current account username is not available; call me() first")
        return username

    def _assert_wallet_matches(self, wallet: Any) -> None:
        expected = self.current_account.wallet_address
        if not expected or wallet.address.lower() != expected.lower():
            raise ApiError(
                400,
                {"status": "error", "message": "Incorrect password", "errorCode": 0},
            )

    async def withdraw(
        self, password: str, amount: int | float | str | Decimal, provider: str = "base"
    ) -> None:
        wallet = self.get_wallet(self._require_account_username(), password)
        self._assert_wallet_matches(wallet)
        payload = {
            "walletAddress": wallet.address,
            "amount": _parse_ether(amount),
            "provider": provider,
        }
        permit = _unwrap_data(
            await self.client.rest.post("/v1/account/token/withdraw/permit", payload)
        )
        if not isinstance(permit, dict):
            raise ValueError("Withdrawal permit response did not include data")
        signature = _sign_typed_data(wallet, permit["domain"], permit["types"], permit["message"])
        await self.client.rest.post(
            "/v2/account/token/withdraw", {**payload, "signature": signature}
        )

    async def deposit(
        self, password: str, amount: int | float | str | Decimal, provider: str = "base"
    ) -> None:
        await self._deposit(password, amount, provider, attempt=1)

    async def _deposit(
        self,
        password: str,
        amount: int | float | str | Decimal,
        provider: str,
        *,
        attempt: int,
    ) -> None:
        wallet = self.get_wallet(self._require_account_username(), password)
        self._assert_wallet_matches(wallet)
        try:
            await self.client.rest.post(
                "/v3/account/token/deposit",
                {
                    "walletAddress": wallet.address,
                    "amount": _parse_ether(amount),
                    "provider": provider,
                },
            )
        except ApiError as error:
            if error.error_code != _INSUFFICIENT_ALLOWANCE:
                raise
            if attempt == 1:
                await self.approve_token_usage(password, "account", provider)
            if attempt >= _MAX_DEPOSIT_ATTEMPTS:
                raise
            await asyncio.sleep(10)
            await self._deposit(password, amount, provider, attempt=attempt + 1)

    async def approve_token_usage(
        self, password: str, spender: str, provider: str = "base"
    ) -> None:
        if spender not in {"account", "staker"}:
            raise ValueError("spender must be 'account' or 'staker'")
        wallet = self.get_wallet(self._require_account_username(), password)
        permit = _unwrap_data(
            await self.client.rest.post(
                "/v1/contract/token/approve/permit",
                {"walletAddress": wallet.address, "spender": spender, "provider": provider},
            )
        )
        if not isinstance(permit, dict):
            raise ValueError("Approval permit response did not include data")
        signature = _sign_typed_data(wallet, permit["domain"], permit["types"], permit["message"])
        await self.client.rest.post(
            "/v1/contract/token/approve",
            {
                "walletAddress": wallet.address,
                "spender": spender,
                "provider": provider,
                "deadline": permit["message"]["deadline"],
                "approveSignature": signature,
            },
        )

    async def get_subscription_status(self) -> dict[str, Any]:
        socket_writes = self._subscription_socket_writes
        data = _unwrap_data(await self.client.rest.get("/v1/subscriptions/status"))
        subscription = data.get("subscription") if isinstance(data, dict) else None
        if not isinstance(subscription, dict):
            raise ValueError("Subscription status response did not include data.subscription")
        version = self._parse_subscription_version(subscription.get("version"))
        applied = self._apply_subscription(
            subscription,
            source="rest",
            version=version,
            socket_writes_at_start=socket_writes,
        )
        if not applied and self.current_account.subscription is not None:
            return self.current_account.subscription
        return self.current_account.subscription or subscription

    async def get_subscription_usage(self) -> dict[str, Any]:
        data = _unwrap_data(await self.client.rest.get("/v1/subscriptions/usage"))
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            raise ValueError("Subscription usage response did not include data.usage")
        return usage

    async def get_trial_eligibility(self) -> dict[str, Any]:
        data = _unwrap_data(await self.client.rest.get("/v1/subscriptions/trial-eligibility"))
        if not isinstance(data, dict):
            raise ValueError("Trial eligibility response did not include data")
        return {"eligible": bool(data.get("eligible")), "reasonCode": data.get("reasonCode")}

    async def set_device_id(self, device_id: str | None = None, **kwargs: Any) -> None:
        device_id = device_id or kwargs.get("deviceId")
        if not device_id:
            raise ValueError("device_id is required")
        await self.client.rest.post("/v1/account/device-id", {"deviceId": device_id})

    async def get_subscription_plans(self) -> list[dict[str, Any]]:
        data = _unwrap_data(await self.client.rest.get("/v1/subscriptions/plans"))
        plans = data.get("plans") if isinstance(data, dict) else None
        if not isinstance(plans, list):
            raise ValueError("Subscription plans response did not include data.plans")
        return plans

    async def create_subscription_checkout(
        self,
        plan_id: str | None = None,
        term: str | None = None,
        options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        values = {**dict(options or {}), **kwargs}
        plan_id = plan_id or _first(values, "plan_id", "planId")
        term = term or values.get("term")
        if not plan_id or not term:
            raise ValueError("plan_id and term are required")
        body: dict[str, Any] = {
            "planId": plan_id,
            "term": term,
            "redirectType": _first(values, "redirect_type", "redirectType", default="web"),
        }
        app_source = _first(values, "app_source", "appSource")
        if app_source:
            body["appSource"] = app_source
        start_trial = _first(values, "start_trial", "startTrial")
        if start_trial is not None:
            body["startTrial"] = start_trial
        device_id = _first(values, "device_id", "deviceId")
        if device_id is not None:
            body["deviceId"] = device_id
        data = _unwrap_data(await self.client.rest.post("/v1/iap/stripe/subscribe", body))
        if not isinstance(data, dict):
            raise ValueError("Subscription checkout response did not include data")
        return data

    async def create_subscription_portal_session(self) -> dict[str, Any]:
        data = _unwrap_data(await self.client.rest.post("/v1/subscriptions/stripe/portal", {}))
        if not isinstance(data, dict):
            raise ValueError("Subscription portal response did not include data")
        return data

    async def refresh_subscription(self) -> dict[str, Any]:
        return await self.get_subscription_status()

    # JavaScript SDK compatibility aliases.
    getNonce = get_nonce
    getWallet = get_wallet
    refreshBalance = refresh_balance
    accountBalance = account_balance
    walletBalance = wallet_balance
    validateUsername = validate_username
    switchNetwork = switch_network
    transactionHistory = transaction_history
    claimRewards = claim_rewards
    approveTokenUsage = approve_token_usage
    getSubscriptionStatus = get_subscription_status
    getSubscriptionUsage = get_subscription_usage
    getTrialEligibility = get_trial_eligibility
    setDeviceId = set_device_id
    getSubscriptionPlans = get_subscription_plans
    createSubscriptionCheckout = create_subscription_checkout
    createSubscriptionPortalSession = create_subscription_portal_session
    refreshSubscription = refresh_subscription


__all__ = ["AccountApi", "CurrentAccount"]
