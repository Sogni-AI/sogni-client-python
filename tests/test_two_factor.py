"""Two-factor sign-in, mirroring sogni-client's ``scripts/check-two-factor.cjs``."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from wire_harness import jwt

from sogni_client import TwoFactorRequiredError
from sogni_client.account import AccountApi
from sogni_client.auth import CookieAuthManager, TokenAuthManager
from sogni_client.errors import ApiError
from sogni_client.transport import ApiClient

EXP = 4102444800


class Server:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail = False
        self.login: dict[str, Any] = {
            "requires2FA": True,
            "tempToken": "pending",
            "username": "artist",
            "token": "",
            "refreshToken": "",
        }
        self.wallet = ""

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append({"method": request.method, "path": request.url.path, "body": body})
        path = request.url.path
        if path == "/v1/account/nonce":
            return httpx.Response(200, json={"data": {"nonce": "nonce"}})
        if path == "/v1/account/login":
            return httpx.Response(200, json={"status": "success", "data": self.login})
        if path == "/v1/account/2fa/verify-login":
            if self.fail:
                return httpx.Response(
                    400, json={"status": "error", "message": "Invalid code", "errorCode": 400}
                )
            return httpx.Response(
                200,
                json={"data": {"token": jwt(EXP), "refreshToken": jwt(EXP + 60)}},
            )
        if path == "/v1/account/me":
            return httpx.Response(200, json={"data": {"walletAddress": self.wallet}})
        if path == "/v1/account/2fa/status":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "enabled": True,
                        "backupCodesRemaining": 8,
                        "canManage": True,
                        "authenticatorEnabled": True,
                        "passkeys": [],
                    }
                },
            )
        if path == "/v1/account/2fa/setup":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "setupId": "setup",
                        "secret": "SECRET",
                        "qrData": "otpauth://",
                        "backupCodes": ["A"],
                    }
                },
            )
        if path == "/v1/account/2fa/backup-codes":
            return httpx.Response(200, json={"data": {"backupCodes": ["NEW1", "NEW2"]}})
        return httpx.Response(200, json={"status": "success", "data": {}})


def make_account(auth_type: str) -> tuple[AccountApi, ApiClient, Server]:
    server = Server()
    http = httpx.AsyncClient(transport=httpx.MockTransport(server.handle))
    client = ApiClient(
        base_url="https://api.two-factor.test",
        socket_url="wss://socket.two-factor.test",
        app_id="two-factor-test",
        network="fast",
        auth_type=auth_type,
        disable_socket=True,
        http_client=http,
    )
    account = AccountApi(client)
    server.wallet = account.get_wallet("artist", "password").address
    return account, client, server


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type", ["token", "cookies"])
async def test_two_factor_sign_in(auth_type: str) -> None:
    account, client, server = make_account(auth_type)
    sessions: list[int] = []
    client.auth.on("sessionChanged", lambda _d: sessions.append(client.auth.session_version))

    with pytest.raises(TwoFactorRequiredError) as raised:
        await account.login("artist", "password", True)
    challenge = raised.value
    assert sessions == [], "a pending sign-in cannot create a session"
    assert client.auth.is_authenticated is False
    assert server.requests[1]["body"]["rememberMe"] is True
    assert challenge.temp_token == challenge.tempToken == "pending"
    assert challenge.wallet_address == challenge.walletAddress == server.wallet
    assert challenge.username == "artist"
    assert str(challenge) == "Enter your authenticator code or a backup code to finish signing in."
    assert "pending" not in repr(challenge)

    server.fail = True
    with pytest.raises(ApiError):
        await account.verify_two_factor_login(challenge, "123456")
    assert sessions == [], "a rejected code cannot create a session"

    server.fail = False
    result = await account.verifyTwoFactorLogin(challenge, " abcdef12 ")
    assert server.requests[-1]["body"] == {"tempToken": "pending", "code": "ABCDEF12"}
    assert result == {"token": jwt(EXP), "refreshToken": jwt(EXP + 60), "username": "artist"}
    assert client.auth.is_authenticated is True
    assert len(sessions) == 1
    if auth_type == "token":
        assert isinstance(client.auth, TokenAuthManager)
        assert await client.auth.backup() == {"token": jwt(EXP), "refreshToken": jwt(EXP + 60)}
    else:
        assert isinstance(client.auth, CookieAuthManager)
    await client.aclose()


@pytest.mark.asyncio
async def test_a_challenge_without_a_token_asks_to_sign_in_again() -> None:
    account, client, server = make_account("token")
    server.login = {"requires2FA": True, "username": "artist", "token": "", "refreshToken": ""}
    with pytest.raises(ValueError, match="Unable to start verification. Please sign in again."):
        await account.login("artist", "password")
    assert client.auth.is_authenticated is False
    await client.aclose()


@pytest.mark.asyncio
async def test_accounts_without_two_factor_sign_in_as_before() -> None:
    account, client, server = make_account("token")
    server.login = {"token": jwt(EXP), "refreshToken": jwt(EXP + 60), "username": "artist"}
    data = await account.login("artist", "password")
    assert data == server.login
    assert client.auth.is_authenticated is True
    assert not [r for r in server.requests if "/2fa/" in r["path"]]
    await client.aclose()


async def signed_in() -> tuple[AccountApi, ApiClient, Server]:
    account, client, server = make_account("token")
    server.login = {"token": jwt(EXP), "refreshToken": jwt(EXP + 60), "username": "artist"}
    await account.login("artist", "password")
    for _ in range(50):
        if account.current_account.wallet_address:
            break
        await asyncio.sleep(0.01)
    assert account.current_account.wallet_address == server.wallet
    return account, client, server


@pytest.mark.asyncio
async def test_authenticator_management() -> None:
    account, client, server = await signed_in()
    status = await account.get_two_factor_status()
    assert status["enabled"] is True and server.requests[-1]["path"] == "/v1/account/2fa/status"

    with pytest.raises(ValueError, match="Incorrect password. Please try again."):
        await account.setup_two_factor("artist", "wrong")

    setup = await account.setup_two_factor("artist", "password", " 123456 ")
    assert setup["setupId"] == "setup"
    body = server.requests[-1]["body"]
    assert body["proof"] == {"code": "123456"} and body["signature"].startswith("0x")
    await account.setup_two_factor("artist", "password")
    assert "proof" not in server.requests[-1]["body"]

    await account.confirm_two_factor_setup("setup", " 654321 ")
    assert server.requests[-1] == {
        "method": "POST",
        "path": "/v1/account/2fa/verify-setup",
        "body": {"setupId": "setup", "code": "654321"},
    }

    await account.disable_two_factor("artist", "password", "abcd-1234")
    body = server.requests[-1]["body"]
    assert server.requests[-1]["path"] == "/v1/account/2fa/disable"
    assert body["code"] == "ABCD-1234" and "signature" in body

    codes = await account.regenerate_two_factor_backup_codes("artist", "password", {"code": "1"})
    assert codes == ["NEW1", "NEW2"]
    assert server.requests[-1]["body"]["code"] == "1"
    await client.aclose()
