from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Iterable
from typing import Any

import httpx
import pytest

from sogni_client.auth import ApiKeyAuthManager, CookieAuthManager, TokenAuthManager, _decode_jwt
from sogni_client.errors import ApiError


def make_jwt(exp: float, **claims: Any) -> str:
    def segment(value: dict[str, Any]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{segment({'alg': 'none'})}.{segment({'exp': exp, **claims})}.signature"


def json_response(status: int, body: Any) -> httpx.Response:
    return httpx.Response(
        status,
        json=body,
        request=httpx.Request("POST", "https://api.sogni.ai/v1/account/refresh-token"),
    )


class FakeRefreshClient:
    def __init__(
        self,
        responses: httpx.Response | Iterable[httpx.Response],
        *,
        pause: bool = False,
    ) -> None:
        if isinstance(responses, httpx.Response):
            self.responses = [responses]
        else:
            self.responses = list(responses)
        self.calls: list[tuple[str, Any]] = []
        self.pause = pause
        self.release = asyncio.Event()

    async def post(self, url: str, *, json: Any) -> httpx.Response:
        self.calls.append((url, json))
        if self.pause:
            await self.release.wait()
        if not self.responses:
            raise AssertionError("Unexpected refresh request")
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_api_key_authentication_uses_exact_wire_header_and_emits_updates() -> None:
    auth = ApiKeyAuthManager()
    updates: list[bool] = []
    auth.on("updated", updates.append)

    await auth.authenticate("  secret-key  ")

    assert auth.is_authenticated is True
    assert auth.isAuthenticated is True
    assert await auth.headers() == {"api-key": "secret-key"}
    assert await auth.backup() == "secret-key"

    auth.clear()
    auth.clear()

    assert await auth.headers() == {}
    assert updates == [True, False]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "   "])
async def test_api_key_authentication_rejects_empty_values(value: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        await ApiKeyAuthManager().authenticate(value)


def test_decode_jwt_accepts_raw_or_bearer_token_and_rejects_bad_data() -> None:
    token = make_jwt(time.time() + 60, subject="artist")

    assert _decode_jwt(token)["subject"] == "artist"
    assert _decode_jwt(f"Bearer {token}")["subject"] == "artist"
    with pytest.raises(ValueError, match="Invalid JWT"):
        _decode_jwt("not-a-jwt")


@pytest.mark.asyncio
async def test_token_authentication_preserves_raw_authorization_value_and_backup_shape() -> None:
    now = time.time()
    access = make_jwt(now + 3600)
    refresh = make_jwt(now + 7200)
    auth = TokenAuthManager("https://api.sogni.ai")

    await auth.authenticate(token=access, refresh_token=refresh)

    assert auth.is_authenticated is True
    assert await auth.headers() == {"Authorization": access}
    assert await auth.backup() == {"token": access, "refreshToken": refresh}


@pytest.mark.asyncio
async def test_token_authentication_accepts_javascript_and_python_token_mapping_keys() -> None:
    now = time.time()
    access = make_jwt(now + 3600)
    refresh = make_jwt(now + 7200)

    js_auth = TokenAuthManager("https://api.sogni.ai")
    await js_auth.authenticate({"token": access, "refreshToken": refresh})
    python_auth = TokenAuthManager("https://api.sogni.ai")
    await python_auth.authenticate({"token": access, "refresh_token": refresh})

    assert await js_auth.headers() == await python_auth.headers() == {"Authorization": access}


@pytest.mark.asyncio
async def test_expired_access_token_is_refreshed_with_canonical_request() -> None:
    now = time.time()
    expired_access = make_jwt(now - 1)
    old_refresh = make_jwt(now + 7200, generation=1)
    new_access = make_jwt(now + 3600, generation=2)
    new_refresh = make_jwt(now + 10800, generation=2)
    refresh_client = FakeRefreshClient(
        json_response(
            200,
            {"status": "success", "data": {"token": new_access, "refreshToken": new_refresh}},
        )
    )
    auth = TokenAuthManager("https://api.sogni.ai/", refresh_client=refresh_client)

    await auth.authenticate(token=expired_access, refresh_token=old_refresh)

    assert refresh_client.calls == [
        (
            "https://api.sogni.ai/v1/account/refresh-token",
            {"refreshToken": old_refresh},
        )
    ]
    assert await auth.headers() == {"Authorization": new_access}
    assert await auth.backup() == {"token": new_access, "refreshToken": new_refresh}


@pytest.mark.asyncio
async def test_concurrent_header_requests_share_one_refresh() -> None:
    now = time.time()
    access = make_jwt(now + 3600, generation=1)
    old_refresh = make_jwt(now + 7200, generation=1)
    new_access = make_jwt(now + 3600, generation=2)
    new_refresh = make_jwt(now + 7200, generation=2)
    refresh_client = FakeRefreshClient(
        json_response(
            200,
            {"status": "success", "data": {"token": new_access, "refreshToken": new_refresh}},
        ),
        pause=True,
    )
    auth = TokenAuthManager("https://api.sogni.ai", refresh_client=refresh_client)
    await auth.authenticate(token=access, refresh_token=old_refresh)
    auth._token_expires_at = 0

    requests = [asyncio.create_task(auth.headers()) for _ in range(8)]
    for _ in range(100):
        if refresh_client.calls:
            break
        await asyncio.sleep(0)
    assert len(refresh_client.calls) == 1
    refresh_client.release.set()

    assert await asyncio.gather(*requests) == [{"Authorization": new_access}] * 8
    assert len(refresh_client.calls) == 1


@pytest.mark.asyncio
async def test_refresh_http_error_raises_api_error_and_clears_authentication() -> None:
    now = time.time()
    access = make_jwt(now + 3600)
    refresh = make_jwt(now + 7200)
    refresh_client = FakeRefreshClient(
        json_response(401, {"status": "error", "message": "Invalid refresh", "errorCode": 104})
    )
    auth = TokenAuthManager("https://api.sogni.ai", refresh_client=refresh_client)
    await auth.authenticate(token=access, refresh_token=refresh)
    auth._token_expires_at = 0

    with pytest.raises(ApiError) as raised:
        await auth.headers()

    assert raised.value.status == 401
    assert raised.value.error_code == 104
    assert auth.is_authenticated is False
    assert await auth.backup() is None


@pytest.mark.asyncio
async def test_non_json_refresh_failure_keeps_real_http_status() -> None:
    now = time.time()
    access = make_jwt(now + 3600)
    refresh = make_jwt(now + 7200)
    response = httpx.Response(
        502,
        text="<html>bad gateway</html>",
        request=httpx.Request("POST", "https://api.sogni.ai/v1/account/refresh-token"),
    )
    auth = TokenAuthManager("https://api.sogni.ai", refresh_client=FakeRefreshClient(response))
    await auth.authenticate(token=access, refresh_token=refresh)
    auth._token_expires_at = 0

    with pytest.raises(ApiError) as raised:
        await auth.headers()

    assert raised.value.status == 502
    assert raised.value.error_code == 502


@pytest.mark.asyncio
async def test_refresh_success_without_both_tokens_is_rejected_and_cleared() -> None:
    now = time.time()
    access = make_jwt(now + 3600)
    refresh = make_jwt(now + 7200)
    auth = TokenAuthManager(
        "https://api.sogni.ai",
        refresh_client=FakeRefreshClient(
            json_response(200, {"status": "success", "data": {"token": access}})
        ),
    )
    await auth.authenticate(token=access, refresh_token=refresh)
    auth._token_expires_at = 0

    with pytest.raises(ValueError, match="both tokens"):
        await auth.headers()

    assert auth.is_authenticated is False


@pytest.mark.asyncio
async def test_expired_refresh_token_is_rejected_without_an_http_call() -> None:
    now = time.time()
    expired_access = make_jwt(now - 60)
    expired_refresh = make_jwt(now - 1)
    refresh_client = FakeRefreshClient([])
    auth = TokenAuthManager("https://api.sogni.ai", refresh_client=refresh_client)

    with pytest.raises(ValueError, match="Refresh token expired"):
        await auth.authenticate(token=expired_access, refresh_token=expired_refresh)

    assert refresh_client.calls == []
    assert auth.is_authenticated is False


@pytest.mark.asyncio
async def test_cookie_auth_manager_is_explicit_and_not_backupable() -> None:
    auth = CookieAuthManager()
    updates: list[bool] = []
    auth.on("updated", updates.append)

    await auth.authenticate()

    assert auth.is_authenticated is True
    assert await auth.headers() == {}
    with pytest.raises(NotImplementedError):
        await auth.backup()
    auth.clear()
    assert updates == [True, False]
