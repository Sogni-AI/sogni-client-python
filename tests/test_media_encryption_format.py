"""Encrypted media, against the test vectors shared with sogni-client.

``fixtures/media-encryption-vectors.json`` is byte-identical to sogni-client's
``scripts/fixtures/media-encryption-vectors.json`` and sogni-api's
``test/fixtures/media-encryption-vectors.json``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest
from Crypto.Cipher import AES

from sogni_client import _media_encryption as me
from sogni_client.errors import MediaEncryptionError

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "media-encryption-vectors.json").read_text()
)
VECTORS_SHA256 = "5a6b25b2779ecef1ec4eeaa2ddd537a0461e3aac5626c40ec71b6cafcbf94ed2"


def pattern(length: int) -> bytes:
    return bytes(((i * 31 + (i >> 8) * 7 + 3) & 0xFF) for i in range(length))


def hkdf(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 on the standard library, independent of the SDK's implementation."""

    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out, block, counter = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes((counter,)), hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def vector_account_secret() -> bytes:
    """The test vectors' account secret, computed as sogni-client's check does."""

    wallet = bytes.fromhex(VECTORS["walletPrivateKey"])
    return hkdf(wallet, b"sogni-media-encryption", b"account-key-v1", 32)


ACCOUNT = me.import_account_key(vector_account_secret())
PROJECT_KEY = bytes.fromhex(VECTORS["projectKey"])


def test_fixture_is_the_shared_file() -> None:
    raw = (Path(__file__).parent / "fixtures" / "media-encryption-vectors.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == VECTORS_SHA256
    assert VECTORS["format"] == "SOGNIE2E" and VECTORS["version"] == 1


def test_recipient_matches_the_vector() -> None:
    stanza = me.wrap_project_key(ACCOUNT, PROJECT_KEY, bytes.fromhex(VECTORS["recipientNonce"]))
    assert stanza.hex() == VECTORS["recipient"]
    assert len(stanza) == me.RECIPIENT_BYTES == 69


def test_recipient_opens_with_the_contract_derivations() -> None:
    stanza = bytes.fromhex(VECTORS["recipient"])
    secret = vector_account_secret()
    check_secret = hkdf(secret, b"sogni-media-encryption", b"key-check-secret-v1", 32)
    nonce = stanza[9:21]
    assert stanza[0] == 1
    assert stanza[1:9] == hkdf(check_secret, nonce, b"key-check-v1", 8)
    cipher = AES.new(secret, AES.MODE_GCM, nonce=nonce)
    cipher.update(b"sogni-media-recipient-v1")
    assert cipher.decrypt_and_verify(stanza[21:53], stanza[53:]) == PROJECT_KEY
    assert me.unwrap_project_key(ACCOUNT, stanza) == PROJECT_KEY


@pytest.mark.parametrize("case", VECTORS["files"], ids=lambda case: case["name"])
def test_files_match_the_vectors_byte_for_byte(case: dict) -> None:
    recipient = bytes.fromhex(VECTORS["recipient"]) if case["hasRecipient"] else b""
    plaintext = pattern(case["plaintextLength"])
    sealed = me.encrypt_media(
        plaintext,
        key=PROJECT_KEY,
        content_type=case["contentType"],
        recipient=recipient,
        segment_size_log2=case["segmentSizeLog2"],
        salt=bytes.fromhex(VECTORS["salt"]),
        nonce_prefix=bytes.fromhex(VECTORS["noncePrefix"]),
    )
    assert len(sealed) == case["ciphertextLength"]
    assert hashlib.sha256(sealed).hexdigest() == case["ciphertextSha256"]
    if "ciphertext" in case:
        assert sealed.hex() == case["ciphertext"]
    opened, header = me.decrypt_media(sealed, lambda _header: PROJECT_KEY)
    assert opened == plaintext
    assert header.content_type == case["contentType"]
    assert header.recipient == recipient
    if case["hasRecipient"]:
        via_account, _ = me.decrypt_media(
            sealed, lambda header: me.unwrap_project_key(ACCOUNT, header.recipient)
        )
        assert via_account == plaintext


@pytest.mark.parametrize("case", VECTORS["inputs"], ids=lambda case: case["name"])
def test_inputs_match_the_vectors_byte_for_byte(case: dict) -> None:
    plaintext = pattern(case["plaintextLength"])
    assert hashlib.sha256(plaintext).hexdigest() == case["plaintextSha256"]
    material = me.input_material_for_media(ACCOUNT, plaintext, case["contentType"])
    assert material.key.hex() == case["inputKey"]
    assert material.salt.hex() == case["salt"]
    assert material.nonce_prefix.hex() == case["noncePrefix"]
    assert material.recipient_nonce.hex() == case["recipientNonce"]
    wrapped = me.wrap_input_key(PROJECT_KEY, material.key, bytes.fromhex(VECTORS["inputWrapNonce"]))
    assert wrapped.hex() == case["wrappedInputKey"]
    assert me.unwrap_input_key(PROJECT_KEY, wrapped) == material.key
    sealed = me.encrypt_media(
        plaintext,
        key=material.key,
        content_type=case["contentType"],
        recipient=me.wrap_project_key(ACCOUNT, material.key, material.recipient_nonce),
        salt=material.salt,
        nonce_prefix=material.nonce_prefix,
    )
    assert len(sealed) == case["ciphertextLength"]
    assert hashlib.sha256(sealed).hexdigest() == case["ciphertextSha256"]
    if "ciphertext" in case:
        assert sealed.hex() == case["ciphertext"]
        for_worker, _ = me.decrypt_media(sealed, lambda _header: material.key)
        assert for_worker == plaintext
        for_artist, _ = me.decrypt_media(
            sealed, lambda header: me.unwrap_project_key(ACCOUNT, header.recipient)
        )
        assert for_artist == plaintext


def test_another_account_derives_different_input_bytes() -> None:
    stranger = me.import_account_key(vector_account_secret()[::-1])
    case = VECTORS["inputs"][0]
    theirs = me.input_material_for_media(
        stranger, pattern(case["plaintextLength"]), case["contentType"]
    )
    assert theirs.key.hex() != case["inputKey"]


def test_large_inputs_use_fresh_material(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(me, "MAX_CONTENT_KEYED_INPUT_BYTES", 10)
    first = me.input_material_for_media(ACCOUNT, b"x" * 11, "video/mp4")
    second = me.input_material_for_media(ACCOUNT, b"x" * 11, "video/mp4")
    assert first.key != second.key


def sealed_vector(name: str) -> bytes:
    return bytes.fromhex(
        next(case for case in VECTORS["files"] if case["name"] == name)["ciphertext"]
    )


def test_changed_or_truncated_media_is_refused() -> None:
    sealed = sealed_vector("small-with-recipient")
    for damaged in (
        sealed[:-1] + bytes((sealed[-1] ^ 1,)),
        sealed[:120] + bytes((sealed[120] ^ 1,)) + sealed[121:],
    ):
        with pytest.raises(MediaEncryptionError) as raised:
            me.decrypt_media(damaged, lambda _header: PROJECT_KEY)
        assert raised.value.code == "authentication_failed"
    with pytest.raises(MediaEncryptionError) as raised:
        me.decrypt_media(sealed[:-30], lambda _header: PROJECT_KEY)
    assert raised.value.code in {"authentication_failed", "malformed"}


def test_header_problems_have_their_own_codes() -> None:
    sealed = sealed_vector("small-with-recipient")
    with pytest.raises(MediaEncryptionError) as raised:
        me.parse_header(b"\x89PNG plain image bytes")
    assert raised.value.code == "not_encrypted"
    with pytest.raises(MediaEncryptionError) as raised:
        me.parse_header(sealed[:8] + b"\x02" + sealed[9:])
    assert raised.value.code == "unsupported_version"
    with pytest.raises(MediaEncryptionError) as raised:
        me.parse_header(sealed[:30])
    assert raised.value.code == "malformed"


def test_recipient_checks() -> None:
    stanza = bytes.fromhex(VECTORS["recipient"])
    stranger = me.import_account_key(bytes(32))
    with pytest.raises(MediaEncryptionError) as raised:
        me.unwrap_project_key(stranger, stanza)
    assert raised.value.code == "wrong_account_key"
    assert me.unwrap_project_key([stranger, ACCOUNT], stanza) == PROJECT_KEY
    with pytest.raises(MediaEncryptionError) as raised:
        me.unwrap_project_key(ACCOUNT, b"")
    assert raised.value.code == "no_recipient"
    with pytest.raises(MediaEncryptionError) as raised:
        me.unwrap_project_key(ACCOUNT, stanza[:-1])
    assert raised.value.code == "malformed"


def test_encryption_is_randomized_without_fixed_material() -> None:
    first = me.encrypt_media(b"same", key=PROJECT_KEY)
    second = me.encrypt_media(b"same", key=PROJECT_KEY)
    assert first != second
    assert me.decrypt_media(first, lambda _h: PROJECT_KEY)[0] == b"same"


def test_streaming_writer_yields_the_same_bytes() -> None:
    options = {
        "key": PROJECT_KEY,
        "content_type": "video/mp4",
        "segment_size_log2": 16,
        "salt": bytes.fromhex(VECTORS["salt"]),
        "nonce_prefix": bytes.fromhex(VECTORS["noncePrefix"]),
    }
    parts = list(me.iter_encrypted_media(pattern(150000), **options))
    assert len(parts) == 1 + 3
    assert b"".join(parts) == me.encrypt_media(pattern(150000), **options)


def test_key_and_header_limits() -> None:
    with pytest.raises(MediaEncryptionError):
        me.encrypt_media(b"x", key=b"short")
    with pytest.raises(MediaEncryptionError):
        me.encrypt_media(b"x", key=PROJECT_KEY, segment_size_log2=12)
    with pytest.raises(MediaEncryptionError):
        me.encrypt_media(b"x", key=PROJECT_KEY, content_type="x" * 256)
    assert me.MAX_HEADER_BYTES == 819
