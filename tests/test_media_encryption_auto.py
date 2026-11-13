"""``media_encryption="auto"`` through a real client, mirroring sogni-client's
``scripts/check-media-encryption-auto.cjs`` case by case."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import os
from typing import Any

import httpx
import pytest
from Crypto.Cipher import AES
from wire_harness import (
    REST,
    STORAGE,
    FakeApi,
    Session,
    pattern,
    settle,
    wait_for,
)

from sogni_client import _media_encryption as me
from sogni_client.errors import MediaEncryptionError

AUTO = {"media_encryption": "auto"}


def key_response(current: bytes, previous: list[bytes] | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "status": "success",
            "data": {
                "version": 1,
                "current": {
                    "id": "mk1_000000000001",
                    "key": base64.b64encode(current).decode(),
                    "createdAt": 1,
                },
                "previous": [
                    {
                        "id": f"mk1_00000000000{index + 2}",
                        "key": base64.b64encode(key).decode(),
                        "retiredAt": 1,
                    }
                    for index, key in enumerate(previous or [])
                ],
            },
        },
    )


def unavailable() -> httpx.Response:
    return httpx.Response(
        503, json={"status": "error", "errorCode": 135, "message": "Service Unavailable"}
    )


def turned_off() -> httpx.Response:
    return httpx.Response(
        409, json={"status": "error", "errorCode": 205, "message": "Media key deleted"}
    )


def serve(handler: Any) -> Any:
    def configure(api: FakeApi) -> None:
        api.media_key = handler

    return configure


def hkdf(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out, block, counter = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes((counter,)), hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def unwrap_stanza_independently(account_key: bytes, stanza: bytes) -> bytes:
    """An independent check of the SDK's output."""

    check_secret = hkdf(account_key, b"sogni-media-encryption", b"key-check-secret-v1", 32)
    nonce = stanza[9:21]
    assert stanza[0] == 1
    assert stanza[1:9] == hkdf(check_secret, nonce, b"key-check-v1", 8)
    cipher = AES.new(account_key, AES.MODE_GCM, nonce=nonce)
    cipher.update(b"sogni-media-recipient-v1")
    return cipher.decrypt_and_verify(stanza[21:53], stanza[53:])


def query_names(handshake: dict[str, Any]) -> list[str]:
    return sorted(name for name, _value in handshake["query"])


async def default_query_names() -> list[str]:
    session = await Session.start()
    try:
        return query_names(session.server.handshakes[0])
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_auto_encrypts_once_admitted() -> None:
    default_query = await default_query_names()
    account_key, retired_key = os.urandom(32), os.urandom(32)
    session = await Session.start(
        options=AUTO, configure=serve(lambda _n: key_response(account_key, [retired_key]))
    )
    try:
        await wait_for(lambda: len(session.key_requests()) == 1, "the key request")
        key_request = session.key_requests()[0]
        assert key_request["headers"]["api-key"] == "wire-harness-key"
        assert "x-sogni-media" not in key_request["headers"]

        # The handshake came before the key: it declared nothing; the open
        # connection declares in-band instead of reconnecting.
        assert query_names(session.server.handshakes[0]) == default_query
        await wait_for(lambda: len(session.declarations()) == 1, "the in-band declaration")
        assert session.declarations()[0] == {
            "connection": 0,
            "type": "mediaCapability",
            "data": {"sogniMedia": "ciphertext-v1"},
        }

        project, request = await session.create()
        assert project.encrypted is True
        field = request["mediaEncryption"]
        assert field["version"] == 1
        project_key = base64.b64decode(field["key"])
        assert (
            unwrap_stanza_independently(account_key, base64.b64decode(field["recipient"]))
            == project_key
        )
        assert len(field["inputKeys"]) == 1

        upload_url = next(
            r for r in session.api.requests if r["url"].endswith("/v1/image/uploadUrl")
        )
        assert upload_url["headers"]["x-sogni-media"] == "ciphertext-v1"
        upload = session.uploads()[0]
        assert "x-sogni-media" not in upload["headers"]
        assert upload["headers"]["content-type"] == "image/png"
        uploaded = upload["raw"]
        assert me.is_encrypted_media(uploaded)
        input_key = me.unwrap_input_key(project_key, base64.b64decode(field["inputKeys"][0]))
        assert me.decrypt_media(uploaded, lambda _h: input_key)[0] == pattern(3000)

        # Results: encrypted and plain results both come back as the media.
        sealed = me.encrypt_media(
            pattern(500, 9),
            key=project_key,
            content_type="image/png",
            recipient=base64.b64decode(field["recipient"]),
        )
        assert await session.result(project, "/sealed", sealed) == pattern(500, 9)
        assert await session.result(project, "/plain", pattern(500, 9)) == pattern(500, 9)

        # Older media opens too.
        older_key = os.urandom(32)
        older = me.encrypt_media(
            pattern(40, 5),
            key=older_key,
            content_type="image/webp",
            recipient=me.wrap_project_key(me.import_account_key(retired_key), older_key),
        )
        assert await session.sogni.media_encryption.decrypt(older) == pattern(40, 5)
        stranger = me.encrypt_media(
            pattern(40, 5),
            key=older_key,
            recipient=me.wrap_project_key(me.import_account_key(os.urandom(32)), older_key),
        )
        with pytest.raises(MediaEncryptionError) as raised:
            await session.sogni.media_encryption.decrypt(stranger)
        assert raised.value.code == "wrong_account_key"

        # A project opts out; ineligible projects go plain rather than fail.
        for label, params in (
            ("media_encryption=False", {"media_encryption": False}),
            ("a model that cannot render encrypted", {"modelId": "plain-model"}),
            ("media uploaded outside the SDK", {"startingImage": True}),
        ):
            plain_project, plain_request = await session.create(**params)
            assert "mediaEncryption" not in plain_request, label
            assert plain_project.encrypted is False, label

        # Relaxed sends projects unencrypted for now.
        uploads_before = len(session.uploads())
        relaxed_project, relaxed_request = await session.create(network="relaxed")
        assert "mediaEncryption" not in relaxed_request
        assert relaxed_request["network"] == "relaxed"
        assert relaxed_project.encrypted is False
        assert session.uploads()[uploads_before]["raw"] == pattern(3000)
        with pytest.raises(MediaEncryptionError) as raised:
            await session.sogni.projects.create(
                type="image",
                modelId="encryptable-model",
                positivePrompt="x",
                numberOfMedia=1,
                steps=20,
                guidance=7,
                network="relaxed",
                startingImage=pattern(10),
                media_encryption=True,
            )
        assert raised.value.code == "unsupported" and "network" in str(raised.value)
        assert len(session.uploads()) == uploads_before + 1, "nothing uploaded for the refusal"
        with pytest.raises(MediaEncryptionError) as raised:
            await session.sogni.projects.create(
                type="image",
                modelId="plain-model",
                positivePrompt="x",
                numberOfMedia=1,
                steps=20,
                guidance=7,
                mediaEncryption=True,
            )
        assert raised.value.code == "unsupported"
        assert len(session.key_requests()) == 1, "the key is fetched once per sign-in"

        # A key fetched again on a connection that already declared adds nothing.
        await session.sogni.media_encryption.lock()
        assert await session.sogni.media_encryption.is_unlocked() is True
        assert len(session.key_requests()) == 2
        await settle()
        assert len(session.declarations()) == 1, "a declaration is made once per connection"

        # A later connection declares at its handshake.
        session.server.drop()
        await wait_for(lambda: len(session.server.handshakes) == 2, "the reconnect", 10)
        assert dict(session.server.handshakes[1]["query"])["sogniMedia"] == "ciphertext-v1"
        assert [
            name for name in query_names(session.server.handshakes[1]) if name != "sogniMedia"
        ] == default_query
        await wait_for(
            lambda: session.sogni.api_client.socket._authenticated_socket is not None,
            "authentication",
        )
        _, again = await session.create()
        assert again["mediaEncryption"]["version"] == 1
        assert len(session.declarations()) == 1, "no in-band declaration on that connection"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_auto_waits_for_admission() -> None:
    default_query = await default_query_names()
    session = await Session.start(
        options=AUTO,
        authenticated={"mediaEncryptionAvailable": False},
        configure=serve(lambda _n: key_response(os.urandom(32))),
    )
    try:
        project, request = await session.create()
        assert "mediaEncryption" not in request
        assert project.encrypted is False
        assert session.uploads()[0]["raw"] == pattern(3000)
        assert await session.sogni.media_encryption.is_unlocked() is False
        await settle()
        assert session.key_requests() == [], "no key request, so no key is created"
        assert session.declared_rest() == []
        assert query_names(session.server.handshakes[0]) == default_query
        assert session.declarations() == []

        changes: list[bool] = []
        session.sogni.media_encryption.on("availabilityChanged", changes.append)
        session.server.push("mediaEncryptionAvailabilityUpdated", True)
        await wait_for(lambda: len(session.key_requests()) == 1, "the key request")
        await wait_for(lambda: len(session.declarations()) == 1, "the in-band declaration")
        _, admitted = await session.create()
        assert admitted["mediaEncryption"]["version"] == 1
        assert session.declared_rest(), "REST declares from then on"

        session.server.push("mediaEncryptionAvailabilityUpdated", False)
        await wait_for(lambda: not session.sogni.media_encryption.available, "admission to close")
        _, closed = await session.create()
        assert "mediaEncryption" not in closed
        assert await session.sogni.media_encryption.is_unlocked() is True, "the key is kept"
        assert len(session.key_requests()) == 1
        assert len(session.server.handshakes) == 1
        assert changes == [True, False]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_relaxed_connection_sends_plain() -> None:
    session = await Session.start(
        options={**AUTO, "network": "relaxed"},
        configure=serve(lambda _n: key_response(os.urandom(32))),
    )
    try:
        assert await session.sogni.media_encryption.is_unlocked() is True
        project, request = await session.create()
        assert "mediaEncryption" not in request
        assert project.encrypted is False
        assert session.uploads()[0]["raw"] == pattern(3000)
        _, fast = await session.create(network="fast")
        assert fast["mediaEncryption"]["version"] == 1
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_rest_only_client_never_fetches_or_declares() -> None:
    session = await Session.start(
        options=AUTO,
        disable_socket=True,
        configure=serve(lambda _n: key_response(os.urandom(32))),
    )
    try:
        await session.sogni.api_client.rest.get("/v1/account/balance")
        assert await session.sogni.media_encryption.is_unlocked() is False
        assert session.server.handshakes == []
        assert session.key_requests() == []
        assert session.declared_rest() == []
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unknown_key_format_is_not_used(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, "sogni_client")
    response = httpx.Response(
        200,
        json={
            "status": "success",
            "data": {
                "version": 2,
                "current": {"id": "mk2_x", "key": base64.b64encode(os.urandom(32)).decode()},
            },
        },
    )
    session = await Session.start(options=AUTO, configure=serve(lambda _n: response))
    try:
        await wait_for(lambda: len(session.key_requests()) == 1, "the key request")
        _, request = await session.create()
        assert "mediaEncryption" not in request
        assert "Media encryption is unavailable right now" in caplog.text
        assert session.declarations() == []
        assert session.declared_rest() == []
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_key_unavailable_sends_plain_and_retries_later(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, "sogni_client")
    account_key = os.urandom(32)
    answer = {"respond": unavailable}
    session = await Session.start(options=AUTO, configure=serve(lambda _n: answer["respond"]()))
    media = session.sogni.media_encryption
    now = {"t": 1000.0}
    media._clock = lambda: now["t"]
    try:
        await wait_for(lambda: len(session.key_requests()) == 1, "the key request")
        assert "sogniMedia" not in dict(session.server.handshakes[0]["query"])
        _, first = await session.create()
        assert "mediaEncryption" not in first
        assert not me.is_encrypted_media(session.uploads()[0]["raw"])
        assert session.declared_rest() == []
        assert session.declarations() == []
        warning = next(r for r in caplog.records if "unavailable right now" in r.getMessage())
        assert "{'status': 503, 'errorCode': 135}" in warning.getMessage()
        assert "key" not in caplog.text.lower().replace("media key", "")

        # The first failure was stamped with the real clock; restart the wait on ours.
        media._key_retry_at = now["t"] + media._tuning["key_retry_seconds"]
        answer["respond"] = lambda: key_response(account_key)
        now["t"] += 4 * 60
        await session.create()
        assert len(session.key_requests()) == 1, "not asked again within the wait"

        now["t"] += 61
        _, later = await session.create()
        assert len(session.key_requests()) == 2, "asked again after the wait"
        assert later["mediaEncryption"]["version"] == 1
        await wait_for(lambda: len(session.declarations()) == 1, "the in-band declaration")
        assert len(session.server.handshakes) == 1, "declared without reconnecting"

        # A new sign-in asks at once, without waiting out a failure.
        answer["respond"] = unavailable
        auth = session.sogni.api_client.auth
        auth.clear()
        assert await media.is_unlocked() is False, "sign-out drops the key"
        await auth.authenticate("wire-harness-key")
        session.server.push("mediaEncryptionAvailabilityUpdated", True)
        await wait_for(lambda: len(session.key_requests()) == 3, "key request after sign-in")
        await wait_for(lambda: media._key_request is None, "the failure to be handled")
        answer["respond"] = lambda: key_response(account_key)
        auth.clear()
        await auth.authenticate("wire-harness-key")
        session.server.push("mediaEncryptionAvailabilityUpdated", True)
        await wait_for(lambda: len(session.key_requests()) == 4, "key request after next sign-in")
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_key_turned_off_is_respected(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, "sogni_client")
    session = await Session.start(options=AUTO, configure=serve(lambda _n: turned_off()))
    media = session.sogni.media_encryption
    now = {"t": 1000.0}
    media._clock = lambda: now["t"]
    try:
        await wait_for(lambda: len(session.key_requests()) == 1, "the key request")
        _, request = await session.create()
        assert "mediaEncryption" not in request
        assert "turned off for this account" in caplog.text
        now["t"] += 10 * 60
        _, later = await session.create()
        assert "mediaEncryption" not in later
        assert await media.is_unlocked() is False
        session.server.push("mediaEncryptionAvailabilityUpdated", False)
        session.server.push("mediaEncryptionAvailabilityUpdated", True)
        await settle()
        assert len(session.key_requests()) == 1, "not asked again this session"
        assert [r for r in session.key_requests() if r["method"] != "GET"] == []
        assert session.declarations() == [] and session.declared_rest() == []
        auth = session.sogni.api_client.auth
        auth.clear()
        await auth.authenticate("wire-harness-key")
        session.server.push("mediaEncryptionAvailabilityUpdated", True)
        await wait_for(lambda: len(session.key_requests()) == 2, "the key request after sign-in")
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_slow_key_server_holds_a_project_briefly() -> None:
    release = asyncio.Event()
    account_key = os.urandom(32)

    class Slow(httpx.AsyncBaseTransport):
        def __init__(self, inner: httpx.MockTransport) -> None:
            self.inner = inner

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/account/media-key":
                await release.wait()
            return await self.inner.handle_async_request(request)

    session = await Session.start(
        options=AUTO, configure=serve(lambda _n: key_response(account_key))
    )
    media = session.sogni.media_encryption
    media._tuning["key_wait_seconds"] = 0.3
    rest = session.sogni.api_client.rest
    rest._client = httpx.AsyncClient(transport=Slow(httpx.MockTransport(session.api.handle)))
    try:
        await session.sogni.media_encryption.lock()
        loop = asyncio.get_running_loop()
        started = loop.time()
        _, slow = await session.create()
        waited = loop.time() - started
        assert 0.28 <= waited < 3, waited
        assert "mediaEncryption" not in slow

        waiting = asyncio.create_task(session.create())
        await asyncio.sleep(0.05)
        session.server.push("mediaEncryptionAvailabilityUpdated", False)
        await wait_for(lambda: not media.available, "admission to close")
        release.set()
        _, plain = await waiting
        assert "mediaEncryption" not in plain
        session.server.push("mediaEncryptionAvailabilityUpdated", True)
        await wait_for(lambda: media.available, "admission to open")
        _, encrypted = await session.create()
        assert encrypted["mediaEncryption"]["version"] == 1
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_without_declaring_the_capability() -> None:
    session = await Session.start(
        options={**AUTO, "declare_media_capability": False},
        configure=serve(lambda _n: key_response(os.urandom(32))),
    )
    try:
        await wait_for(lambda: len(session.key_requests()) == 1, "the key request")
        project, request = await session.create()
        assert request["mediaEncryption"]["version"] == 1
        assert project.encrypted is True
        assert me.is_encrypted_media(session.uploads()[0]["raw"])
        session.server.drop()
        await wait_for(lambda: len(session.server.handshakes) == 2, "the reconnect", 10)
        await settle()
        assert session.declarations() == []
        assert session.declared_rest() == []
        assert all(
            name != "sogniMedia" for h in session.server.handshakes for name, _ in h["query"]
        )
        # Its results come back as plain media and pass through.
        assert await session.result(project, "/plain", pattern(64, 2)) == pattern(64, 2)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_apis_reached_without_the_capability_in_sogni_client_stay_undeclared() -> None:
    session = await Session.start(
        options=AUTO, configure=serve(lambda _n: key_response(os.urandom(32)))
    )
    try:
        await wait_for(lambda: len(session.declarations()) == 1, "the declaration")
        rest = session.sogni.api_client.rest
        for path in (
            "/v1/world-builds",
            "/v1/world-builds?worldId=w",
            "/v1/creative-agent/workflows/templates",
            "/v1/chat/runs/run-1",
            "/v1/replay/records",
        ):
            await rest.get(path)
        await rest.get("/v1/projects/list")
        declared = {r["url"].removeprefix(REST) for r in session.declared_rest()}
        assert "/v1/projects/list" in declared
        assert not {
            p
            for p in declared
            if p.startswith(("/v1/world", "/v1/creative", "/v1/chat/runs", "/v1/replay"))
        }
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_job_error_4098_refreshes_the_model_list() -> None:
    session = await Session.start(options=AUTO)
    try:
        projects = session.sogni.projects
        await projects.get_supported_models()
        assert projects._supported_models_at > 0
        projects._handle_job_error({"jobID": "P-1", "error": 4098, "error_message": "x"})
        assert projects._supported_models_at == 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_open_decrypted_reads_verified_ranges() -> None:
    account_key = os.urandom(32)
    session = await Session.start(
        options=AUTO, configure=serve(lambda _n: key_response(account_key))
    )
    try:
        media = session.sogni.media_encryption
        assert await media.is_unlocked()
        project_key = os.urandom(32)
        plaintext = os.urandom(150000)
        sealed = me.encrypt_media(
            plaintext,
            key=project_key,
            content_type="video/mp4",
            recipient=me.wrap_project_key(me.import_account_key(account_key), project_key),
            segment_size_log2=16,
        )
        session.api.results["/video"] = sealed
        async with await media.open_decrypted(f"{STORAGE}/video") as reader:
            assert reader.size == len(plaintext)
            assert reader.content_type == "video/mp4"
            assert await reader.read(0, 10) == plaintext[:10]
            assert await reader.read(65530, 65550) == plaintext[65530:65550]
            assert await reader.read(149990, 150000) == plaintext[149990:]
            chunks = [chunk async for chunk in reader.stream(1000, 140000)]
            assert len(chunks) == 3
            assert b"".join(chunks) == plaintext[1000:140000]
            with pytest.raises(ValueError):
                await reader.read(0, len(plaintext) + 1)
        assert reader.closed
        with pytest.raises(ValueError):
            await reader.read(0, 1)
        ranged = [r for r in session.api.requests if r["url"].endswith("/video")]
        assert all(r["headers"]["range"].startswith("bytes=") for r in ranged)
        assert all("x-sogni-media" not in r["headers"] for r in ranged)

        # Locking closes open readers.
        reader = await media.open_decrypted(f"{STORAGE}/video")
        await media.lock()
        assert reader.closed

        # Changed bytes never reach the caller.
        assert await media.is_unlocked()
        damaged = bytearray(sealed)
        damaged[len(sealed) // 2] ^= 1
        session.api.results["/damaged"] = bytes(damaged)
        reader = await media.open_decrypted(f"{STORAGE}/damaged")
        with pytest.raises(MediaEncryptionError) as raised:
            await reader.read(70000, 80000)
        assert raised.value.code == "authentication_failed"
        reader.close()

        session.api.serve_ranges = False
        with pytest.raises(MediaEncryptionError) as raised:
            await media.open_decrypted(f"{STORAGE}/video")
        assert raised.value.code == "unsupported"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_fetch_decrypted_passes_plain_media_through() -> None:
    account_key = os.urandom(32)
    session = await Session.start(
        options=AUTO, configure=serve(lambda _n: key_response(account_key))
    )
    try:
        media = session.sogni.media_encryption
        session.api.results["/plain"] = b"plain bytes"
        assert await media.fetch_decrypted(f"{STORAGE}/plain") == b"plain bytes"
        with pytest.raises(MediaEncryptionError) as raised:
            await media.fetch_decrypted(f"{STORAGE}/plain", require_encrypted=True)
        assert raised.value.code == "not_encrypted"
        project_key = os.urandom(32)
        session.api.results["/sealed"] = me.encrypt_media(
            b"secret frame",
            key=project_key,
            recipient=me.wrap_project_key(me.import_account_key(account_key), project_key),
        )
        assert await media.fetch_decrypted(f"{STORAGE}/sealed") == b"secret frame"
        assert media.is_encrypted(session.api.results["/sealed"][:8])
        assert not media.is_encrypted(b"plain")
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_recovered_encrypted_project_opens_in_an_auto_client() -> None:
    import wire_harness

    account_key = os.urandom(32)
    session = await Session.start(
        options=AUTO,
        authenticated={"activeProjects": [wire_harness._recovered_encrypted_project()]},
        configure=serve(lambda _n: key_response(account_key)),
    )
    try:
        await wait_for(
            lambda: any(p.id == "RECOVERED-1" for p in session.sogni.projects.tracked_projects),
            "the recovered project",
        )
        project = next(p for p in session.sogni.projects.tracked_projects if p.id == "RECOVERED-1")
        assert project.encrypted
        project_key = os.urandom(32)
        sealed = me.encrypt_media(
            b"recovered result",
            key=project_key,
            recipient=me.wrap_project_key(me.import_account_key(account_key), project_key),
        )
        assert await session.result(project, "/recovered", sealed) == b"recovered result"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_set_account_key_needs_a_session_and_32_bytes() -> None:
    session = await Session.start(disable_socket=True, api_key=False)
    try:
        with pytest.raises(MediaEncryptionError) as raised:
            await session.sogni.media_encryption.set_account_key(os.urandom(32))
        assert raised.value.code == "locked"
    finally:
        await session.close()
    session = await Session.start()
    try:
        media = session.sogni.media_encryption
        with pytest.raises(MediaEncryptionError):
            await media.set_account_key(b"short")
        events: list[str] = []
        media.on("unlocked", lambda _d: events.append("unlocked"))
        media.on("locked", lambda _d: events.append("locked"))
        await media.set_account_key(os.urandom(32))
        assert await media.is_unlocked()
        await media.lock()
        assert events == ["unlocked", "locked"]
        # Supplying the key never declares the capability for a client that is not 'auto'.
        assert session.declarations() == [] and session.declared_rest() == []
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_enhancing_an_explicitly_encrypted_result_stays_encrypted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wire_harness

    monkeypatch.setattr(
        wire_harness,
        "MODELS",
        [
            *wire_harness.MODELS,
            {
                "id": "krea2_turbo_fp8_scaled",
                "name": "Krea",
                "SID": 3,
                "tier": "image-tier",
                "media": "image",
                "supportsMediaEncryption": True,
            },
        ],
    )
    account_key = os.urandom(32)
    session = await Session.start(
        options=AUTO, configure=serve(lambda _n: key_response(account_key))
    )
    try:
        assert await session.sogni.media_encryption.is_unlocked()
        project, request = await session.create(media_encryption=True)
        field = request["mediaEncryption"]
        project_key = base64.b64decode(field["key"])
        sealed = me.encrypt_media(
            pattern(64, 7),
            key=project_key,
            content_type="image/png",
            recipient=base64.b64decode(field["recipient"]),
        )
        assert await session.result(project, "/to-enhance", sealed) == pattern(64, 7)
        before = len(session.job_requests())
        enhancing = asyncio.create_task(project.jobs[0].enhance("light"))
        await wait_for(lambda: len(session.job_requests()) > before, "the enhancement request")
        enhancing.cancel()
        enhancement = session.job_requests()[-1]
        assert enhancement["mediaEncryption"]["version"] == 1
        upload = session.uploads()[-1]["raw"]
        assert me.is_encrypted_media(upload), "the decrypted result is re-encrypted for upload"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_models_that_take_inputs_as_they_are(monkeypatch: pytest.MonkeyPatch) -> None:
    import wire_harness

    monkeypatch.setattr(
        wire_harness,
        "MODELS",
        [
            *wire_harness.MODELS,
            {
                "id": "inputs-as-they-are-model",
                "name": "Encrypted results, plain inputs",
                "SID": 3,
                "tier": "image-tier",
                "media": "image",
                "supportsMediaEncryption": True,
                "mediaEncryptionInputs": False,
            },
        ],
    )
    session = await Session.start(
        options=AUTO, configure=serve(lambda _n: key_response(os.urandom(32)))
    )
    try:
        await wait_for(lambda: len(session.key_requests()) == 1, "the key request")
        for choice in (None, True):
            before = len(session.uploads())
            project, request = await session.create(
                modelId="inputs-as-they-are-model", media_encryption=choice
            )
            field = request["mediaEncryption"]
            assert field["version"] == 1 and field["key"] and field["recipient"]
            assert "inputKeys" not in field
            assert project.encrypted is True
            assert session.uploads()[before]["raw"] == pattern(3000), "plain input"
        _, elsewhere = await session.create(modelId="inputs-as-they-are-model", startingImage=True)
        assert elsewhere["mediaEncryption"]["version"] == 1, "inputs uploaded elsewhere are fine"
    finally:
        await session.close()
