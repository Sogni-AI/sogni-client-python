"""Public-client account changes during asynchronous submission."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from sogni_client import SogniClient
from sogni_client.auth import ApiKeyAuthManager
from sogni_client.transport import RestClient, WebSocketClient
from sogni_client.utils import b64_json_encode


class SessionSocket:
    def __init__(self, credential: str) -> None:
        self.credential = credential
        self.sent: list[dict[str, Any]] = []
        self.messages: asyncio.Queue[str] = asyncio.Queue()
        self.messages.put_nowait(
            json.dumps({"type": "authenticated", "data": b64_json_encode({"clientType": "artist"})})
        )
        self.closed = False

    async def recv(self) -> str:
        return await self.messages.get()

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def close(self, **_kwargs: Any) -> None:
        self.closed = True


@pytest.fixture
async def sessions():
    sockets: list[SessionSocket] = []

    async def connect(_url: str, **kwargs: Any) -> SessionSocket:
        socket = SessionSocket(kwargs["additional_headers"]["api-key"])
        sockets.append(socket)
        return socket

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))
    ) as http:
        client = await SogniClient.create(
            app_id="offline-session-regression",
            api_key="account-a",
            http_client=http,
            socket_http_client=http,
            websocket_factory=connect,
        )
        try:
            await client.api_client.socket.send("probe", {})
            yield client, sockets
        finally:
            await client.aclose()


async def test_create_does_not_submit_after_account_changes_during_asset_upload(
    sessions, monkeypatch
):
    client, sockets = sessions
    entered = asyncio.Event()
    release = asyncio.Event()

    async def upload(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return "image/png"

    monkeypatch.setattr(
        client.projects,
        "get_model_options",
        AsyncMock(
            return_value={
                "type": "image",
                "sampler": {"allowed": [], "default": None},
                "scheduler": {"allowed": [], "default": None},
            }
        ),
    )
    monkeypatch.setattr(client.projects, "_upload_asset", upload)
    creation = asyncio.create_task(
        client.projects.create(
            type="image",
            modelId="flux1-schnell-fp8",
            positivePrompt="A ceramic mug",
            numberOfMedia=1,
            startingImage=b"image",
        )
    )
    await entered.wait()
    await client.api_client.auth.authenticate("account-b")
    release.set()
    with pytest.raises(RuntimeError, match="account changed"):
        await creation
    assert not any(message["type"] == "jobRequest" for socket in sockets for message in socket.sent)


async def test_new_account_request_uses_a_connection_authenticated_for_that_account(sessions):
    client, sockets = sessions
    await client.api_client.auth.authenticate("account-b")
    await client.api_client.socket.send("jobRequest", {"jobID": "new-account-work"})
    owners = [
        socket.credential
        for socket in sockets
        for message in socket.sent
        if message["type"] == "jobRequest"
    ]
    assert owners == ["account-b"]
    assert sockets[0].closed


async def test_waiting_send_cannot_follow_the_new_account_connection(sessions):
    client, sockets = sessions
    client.api_client.socket._authenticated_socket = None
    pending = asyncio.create_task(
        client.api_client.socket.send("jobRequest", {"jobID": "old-work"})
    )
    await asyncio.sleep(0)
    await client.api_client.auth.authenticate("account-b")
    with pytest.raises(RuntimeError, match="account changed"):
        await pending
    assert not any(message["type"] == "jobRequest" for socket in sockets for message in socket.sent)


async def test_queued_logout_cleanup_cannot_disconnect_a_new_signin(sessions):
    client, sockets = sessions
    client.api_client.auth.clear()
    await client.api_client.auth.authenticate("account-b")
    await client.api_client.socket.send("jobRequest", {})
    await asyncio.sleep(0)
    assert client.api_client.socket.is_connected
    assert sockets[-1].credential == "account-b"
    assert not sockets[-1].closed


async def test_logout_rejects_a_pending_create_without_sending_work(sessions, monkeypatch):
    client, sockets = sessions
    entered = asyncio.Event()
    release = asyncio.Event()

    async def options(_model):
        entered.set()
        await release.wait()
        return {
            "type": "image",
            "sampler": {"allowed": [], "default": None},
            "scheduler": {"allowed": [], "default": None},
        }

    monkeypatch.setattr(client.projects, "get_model_options", options)
    pending = asyncio.create_task(
        client.projects.create(
            type="image", modelId="flux1-schnell-fp8", positivePrompt="A mug", numberOfMedia=1
        )
    )
    await entered.wait()
    client.api_client.auth.clear()
    release.set()
    with pytest.raises(RuntimeError, match="account changed"):
        await pending
    assert not any(message["type"] == "jobRequest" for socket in sockets for message in socket.sent)


async def test_stale_upgrade_closes_old_socket_and_new_account_can_connect():
    auth = ApiKeyAuthManager()
    await auth.authenticate("account-a")
    entered = asyncio.Event()
    release = asyncio.Event()
    sockets = []

    async def connect(_url, **kwargs):
        socket = SessionSocket(kwargs["additional_headers"]["api-key"])
        sockets.append(socket)
        if socket.credential == "account-a":
            entered.set()
            await release.wait()
        return socket

    client = WebSocketClient(
        "wss://socket.sogni.ai", auth, "offline", "fast", connect_factory=connect
    )
    try:
        pending = asyncio.create_task(client.connect())
        await entered.wait()
        await auth.authenticate("account-b")
        current = asyncio.create_task(client.connect())
        release.set()
        with pytest.raises(RuntimeError, match="account changed"):
            await pending
        await current
        await client.send("jobRequest", {})
        assert sockets[0].closed
        assert not sockets[0].sent
        assert sockets[1].credential == "account-b"
        assert len(sockets[1].sent) == 1
    finally:
        await client.aclose()


async def test_old_rest_unauthorized_response_cannot_clear_new_credentials():
    auth = ApiKeyAuthManager()
    await auth.authenticate("account-a")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(_request):
        entered.set()
        await release.wait()
        return httpx.Response(401, json={"message": "Unauthorized"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = RestClient("https://api.sogni.ai", auth, http_client=http)
        pending = asyncio.create_task(client.get("/v1/account"))
        await entered.wait()
        await auth.authenticate("account-b")
        release.set()
        with pytest.raises(RuntimeError, match="account changed"):
            await pending
        assert await auth.headers() == {"api-key": "account-b"}


async def test_queued_recovery_snapshot_cannot_enter_the_new_account(sessions):
    client, _sockets = sessions
    lock = client.projects._sync_lock
    await lock.acquire()
    pending = asyncio.create_task(client.projects._queue_sync({"activeProjects": []}, "manual", 0))
    await asyncio.sleep(0)
    await client.api_client.auth.authenticate("account-b")
    lock.release()
    with pytest.raises(RuntimeError, match="account changed"):
        await pending
    assert client.projects.tracked_projects == []


async def test_account_switch_forgets_unadmitted_requests_without_server_cancellation(
    sessions, monkeypatch
):
    client, sockets = sessions
    monkeypatch.setattr(
        client.projects,
        "get_model_options",
        AsyncMock(
            return_value={
                "type": "image",
                "sampler": {"allowed": [], "default": None},
                "scheduler": {"allowed": [], "default": None},
            }
        ),
    )
    project = await client.projects.create(
        type="image", modelId="flux1-schnell-fp8", positivePrompt="A ceramic mug", numberOfMedia=1
    )
    assert project.id in client.projects._unadmitted_requests
    await client.api_client.auth.authenticate("account-b")
    assert client.projects.tracked_projects == []
    assert client.projects._unadmitted_requests == {}
    assert project.finished
    assert not any(message["type"] == "jobError" for socket in sockets for message in socket.sent)


@pytest.mark.parametrize("delay_cancellation", [False, True])
async def test_delayed_result_url_cannot_complete_in_a_new_account(
    sessions, monkeypatch, delay_cancellation
):
    client, _sockets = sessions
    entered = asyncio.Event()
    release = asyncio.Event()
    events = []
    client.projects.on("job", events.append)

    async def download(_params):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            if not delay_cancellation:
                raise
            await release.wait()
        return "https://example.test/old-result.png"

    monkeypatch.setattr(client.projects, "download_url", download)
    client.api_client.socket.emit("jobResult", {"jobID": "old-project", "imgID": "old-job"})
    await entered.wait()
    await client.api_client.auth.authenticate("account-b")
    release.set()
    for _ in range(5):
        await asyncio.sleep(0)
    assert events == []


async def test_pending_upload_cannot_follow_expired_token_signin(monkeypatch):
    def token(account, expires):
        payload = (
            base64.urlsafe_b64encode(json.dumps({"addr": account, "exp": expires}).encode())
            .decode()
            .rstrip("=")
        )
        return f"e30.{payload}.signature"

    now = time.time()
    new_access = token("account-b", now + 3600)
    new_refresh = token("account-b", now + 7200)
    renewing = asyncio.Event()
    finish_renewal = asyncio.Event()
    uploading = asyncio.Event()
    finish_upload = asyncio.Event()
    sockets = []

    async def respond(request):
        if request.url.path.endswith("/refresh-token"):
            renewing.set()
            await finish_renewal.wait()
            return httpx.Response(
                200, json={"data": {"token": new_access, "refreshToken": new_refresh}}
            )
        return httpx.Response(200, json={})

    async def connect(_url, **kwargs):
        socket = SessionSocket(kwargs["additional_headers"]["Authorization"])
        sockets.append(socket)
        return socket

    async def upload(*_args, **_kwargs):
        uploading.set()
        await finish_upload.wait()
        return "image/png"

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = await SogniClient.create(
            app_id="offline-expired-session",
            http_client=http,
            socket_http_client=http,
            websocket_factory=connect,
        )
        try:
            await client.set_tokens(
                token=token("account-a", now + 3600), refresh_token=token("account-a", now + 7200)
            )
            await client.api_client.socket.send("probe", {})
            monkeypatch.setattr(
                client.projects,
                "get_model_options",
                AsyncMock(
                    return_value={
                        "type": "image",
                        "sampler": {"allowed": [], "default": None},
                        "scheduler": {"allowed": [], "default": None},
                    }
                ),
            )
            monkeypatch.setattr(client.projects, "_upload_asset", upload)
            creation = asyncio.create_task(
                client.projects.create(
                    type="image",
                    modelId="flux1-schnell-fp8",
                    positivePrompt="A mug",
                    numberOfMedia=1,
                    startingImage=b"image",
                )
            )
            await uploading.wait()
            signin = asyncio.create_task(
                client.set_tokens(token=token("account-b", now - 60), refresh_token=new_refresh)
            )
            await renewing.wait()
            finish_upload.set()
            for _ in range(10):
                await asyncio.sleep(0)
            finish_renewal.set()
            await signin
            with pytest.raises(RuntimeError, match="account changed"):
                await creation
            assert not any(
                message["type"] == "jobRequest" for socket in sockets for message in socket.sent
            )
        finally:
            await client.aclose()


async def test_background_connection_switch_has_no_unhandled_stale_exception():
    entered = asyncio.Event()
    release = asyncio.Event()
    errors = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))

    async def connect(_url, **kwargs):
        socket = SessionSocket(kwargs["additional_headers"]["api-key"])
        if socket.credential == "account-a":
            entered.set()
            await release.wait()
        return socket

    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))
        ) as http:
            client = await SogniClient.create(
                app_id="offline-background-session",
                api_key="account-a",
                http_client=http,
                socket_http_client=http,
                websocket_factory=connect,
            )
            try:
                await entered.wait()
                await client.api_client.auth.authenticate("account-b")
                release.set()
                await client.api_client.socket.send("probe", {})
                for _ in range(5):
                    await asyncio.sleep(0)
                assert errors == []
            finally:
                await client.aclose()
    finally:
        loop.set_exception_handler(previous_handler)
