"""Authentication strategies shared by REST and WebSocket transports."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .errors import ApiError
from .events import EventEmitter


def _decode_jwt(token: str) -> dict[str, Any]:
    raw = token.removeprefix("Bearer ").strip()
    try:
        payload = raw.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Invalid JWT") from error
    if not isinstance(decoded, dict):
        raise ValueError("Invalid JWT payload")
    return decoded


class AuthManager(EventEmitter, ABC):
    @property
    @abstractmethod
    def is_authenticated(self) -> bool: ...

    @abstractmethod
    async def headers(self) -> dict[str, str]: ...

    @abstractmethod
    async def backup(self) -> Any: ...

    @abstractmethod
    def clear(self) -> None: ...

    @property
    def isAuthenticated(self) -> bool:
        return self.is_authenticated


class ApiKeyAuthManager(AuthManager):
    def __init__(self) -> None:
        super().__init__()
        self._api_key: str | None = None

    @property
    def is_authenticated(self) -> bool:
        return bool(self._api_key)

    async def authenticate(self, api_key: str) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        self._api_key = api_key.strip()
        self.emit("updated", True)

    async def headers(self) -> dict[str, str]:
        return {"api-key": self._api_key} if self._api_key else {}

    async def backup(self) -> str | None:
        return self._api_key

    def clear(self) -> None:
        if self._api_key is None:
            return
        self._api_key = None
        self.emit("updated", False)


class TokenAuthManager(AuthManager):
    def __init__(self, base_url: str, *, refresh_client: httpx.AsyncClient | None = None) -> None:
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._refresh_token: str | None = None
        self._refresh_expires_at = 0.0
        self._renew_lock = asyncio.Lock()
        self._refresh_client = refresh_client

    @property
    def is_authenticated(self) -> bool:
        return bool(self._refresh_token and self._refresh_expires_at > time.time())

    async def authenticate(
        self,
        tokens: dict[str, str] | None = None,
        *,
        token: str | None = None,
        refresh_token: str | None = None,
        refreshToken: str | None = None,
    ) -> None:
        tokens = tokens or {}
        token = token or tokens.get("token")
        refresh_token = (
            refresh_token
            or refreshToken
            or tokens.get("refreshToken")
            or tokens.get("refresh_token")
        )
        if not token or not refresh_token:
            raise ValueError("Both token and refresh_token are required")
        token_exp = float(_decode_jwt(token).get("exp", 0))
        refresh_exp = float(_decode_jwt(refresh_token).get("exp", 0))
        self._refresh_token = refresh_token
        self._refresh_expires_at = refresh_exp
        if token_exp > time.time():
            self._set_tokens(token, refresh_token, token_exp, refresh_exp)
        else:
            await self._renew_token_safe()

    async def headers(self) -> dict[str, str]:
        token = await self._get_token()
        return {"Authorization": token} if token else {}

    async def backup(self) -> dict[str, str] | None:
        if self._token and self._refresh_token:
            return {"token": self._token, "refreshToken": self._refresh_token}
        return None

    def clear(self) -> None:
        if not self._token and not self._refresh_token:
            return
        self._token = None
        self._token_expires_at = 0
        self._refresh_token = None
        self._refresh_expires_at = 0
        self.emit("updated", False)

    async def _get_token(self) -> str | None:
        if self._token and self._token_expires_at > time.time():
            return self._token
        if not self._refresh_token:
            return None
        return await self._renew_token_safe()

    async def _renew_token_safe(self) -> str:
        async with self._renew_lock:
            if self._token and self._token_expires_at > time.time():
                return self._token
            return await self._renew_token()

    async def _renew_token(self) -> str:
        if not self._refresh_token or self._refresh_expires_at <= time.time():
            self.clear()
            raise ValueError("Refresh token expired")
        owns_client = self._refresh_client is None
        client = self._refresh_client or httpx.AsyncClient(timeout=30)
        try:
            response = await client.post(
                f"{self._base_url}/v1/account/refresh-token",
                json={"refreshToken": self._refresh_token},
            )
        finally:
            if owns_client:
                await client.aclose()
        try:
            body = response.json()
        except ValueError:
            body = {
                "status": "error",
                "message": response.reason_phrase or f"HTTP {response.status_code}",
                "errorCode": response.status_code,
            }
        if not response.is_success:
            self.clear()
            raise ApiError(response.status_code, body if isinstance(body, dict) else None)
        data = body.get("data", {})
        token = data.get("token")
        refresh_token = data.get("refreshToken")
        if not token or not refresh_token:
            self.clear()
            raise ValueError("Token refresh response did not include both tokens")
        token_exp = float(_decode_jwt(token).get("exp", 0))
        refresh_exp = float(_decode_jwt(refresh_token).get("exp", 0))
        self._set_tokens(token, refresh_token, token_exp, refresh_exp)
        return token

    def _set_tokens(
        self, token: str, refresh_token: str, token_exp: float, refresh_exp: float
    ) -> None:
        changed = token != self._token or refresh_token != self._refresh_token
        self._token = token
        self._token_expires_at = token_exp
        self._refresh_token = refresh_token
        self._refresh_expires_at = refresh_exp
        if changed:
            self.emit("updated", True)


class CookieAuthManager(AuthManager):
    """Compatibility strategy for callers that inject a cookie-enabled HTTP client."""

    def __init__(self) -> None:
        super().__init__()
        self._authenticated = False

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    async def authenticate(self) -> None:
        self._authenticated = True
        self.emit("updated", True)

    async def headers(self) -> dict[str, str]:
        return {}

    async def backup(self) -> None:
        raise NotImplementedError("Cookie authentication cannot be backed up")

    def clear(self) -> None:
        if not self._authenticated:
            return
        self._authenticated = False
        self.emit("updated", False)


AuthData = dict[str, str]
