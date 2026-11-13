"""With the 5.82.0 options unset, the client behaves exactly as the published 5.81.0.

``fixtures/wire-baseline-5.81.0.json`` was recorded by running the scenarios in
``wire_harness.py`` against the sogni-client 5.81.0 wheel from PyPI (sha256
06187832e400c59258edcf543920a8e61c06afa087bbfad8161b6b7e8087c77f). The server in
every scenario offers media encryption (the account is admitted, the model
supports it, the key endpoint answers), so anything the new code would send
because of it shows up as a difference. The only normalization is the SDK
version in the socket handshake's ``clientName``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import wire_harness

import sogni_client

BASELINE = json.loads(
    (Path(__file__).parent / "fixtures" / "wire-baseline-5.81.0.json").read_text()
)


def _without_version(value: Any, version: str) -> Any:
    return json.loads(
        json.dumps(value).replace(f"(sogni-client) {version}", "(sogni-client) <version>")
    )


def test_baseline_was_recorded_from_5_81_0() -> None:
    assert BASELINE["sogniClientVersion"] == "5.81.0"
    assert set(BASELINE["scenarios"]) == set(wire_harness.SCENARIOS)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(wire_harness.SCENARIOS))
async def test_unset_options_reproduce_5_81_0_exactly(name: str) -> None:
    current = await wire_harness.SCENARIOS[name]()
    expected = _without_version(BASELINE["scenarios"][name], "5.81.0")
    actual = _without_version(current, sogni_client.__version__)
    assert actual["outcome"] == expected["outcome"]
    assert actual["wire"]["handshakes"] == expected["wire"]["handshakes"]
    assert actual["wire"]["frames"] == expected["wire"]["frames"]
    assert actual["wire"]["requests"] == expected["wire"]["requests"]


@pytest.mark.asyncio
async def test_unset_client_sends_nothing_new() -> None:
    session = await wire_harness.Session.start()
    try:
        await session.create()
        session.server.push("mediaEncryptionAvailabilityUpdated", True)
        await wire_harness.settle()
        _, request = await session.create()
        assert session.sogni.media_encryption.available is True
        assert session.key_requests() == []
        assert session.declared_rest() == []
        assert session.declarations() == []
        assert all(name != "sogniMedia" for name, _value in session.server.handshakes[0]["query"])
        assert "mediaEncryption" not in request
        assert all(not upload["raw"].startswith(b"SOGNIE2E") for upload in session.uploads())
        assert await session.sogni.media_encryption.is_unlocked() is False
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unset_client_returns_recovered_ciphertext_unchanged() -> None:
    """An unset client never decrypts or raises for media it cannot open, as in 5.81.0."""

    session = await wire_harness.Session.start(
        authenticated={"activeProjects": [wire_harness._recovered_encrypted_project()]}
    )
    try:
        await wire_harness.wait_for(
            lambda: any(p.id == "RECOVERED-1" for p in session.sogni.projects.tracked_projects),
            "the recovered project",
        )
        project = next(p for p in session.sogni.projects.tracked_projects if p.id == "RECOVERED-1")
        assert project.encrypted is True
        assert "mediaEncryption" not in project.params
        data = await session.result(project, "/recovered", wire_harness.ENCRYPTED_RESULT)
        assert data == wire_harness.ENCRYPTED_RESULT
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, "on", "manual", 1])
async def test_other_client_values_leave_encryption_off(
    value: Any, caplog: pytest.LogCaptureFixture
) -> None:
    session = await wire_harness.Session.start(options={"media_encryption": value})
    try:
        _, request = await session.create()
        assert session.sogni.media_encryption.automatic is False
        assert "mediaEncryption" not in request
        assert session.key_requests() == []
        assert "accepts only 'auto'" in caplog.text
    finally:
        await session.close()


def test_api_client_defaults_are_unchanged() -> None:
    from sogni_client.transport import ApiClient

    client = ApiClient(
        base_url="https://api.example.test",
        socket_url="wss://socket.example.test",
        app_id="defaults",
        network="fast",
        auth_type="apiKey",
    )
    assert client.media_encryption is None
    assert client.rest._media_capability_declared is None
    assert client.socket._media_connection_capability is None
