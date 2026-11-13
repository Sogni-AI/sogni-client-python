"""Internal helpers for encrypted project media. Not part of the public API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from .errors import MediaEncryptionError

VERSION = 1
MAGIC = b"SOGNIE2E"
KEY_BYTES = 32
SALT_BYTES = 32
NONCE_PREFIX_BYTES = 7
NONCE_BYTES = 12
TAG_BYTES = 16
DEFAULT_SEGMENT_SIZE_LOG2 = 20
MIN_SEGMENT_SIZE_LOG2 = 16
MAX_SEGMENT_SIZE_LOG2 = 24
MAX_CONTENT_TYPE_BYTES = 255
MAX_RECIPIENT_BYTES = 512
_FIXED_HEADER_BYTES = 50
MAX_HEADER_BYTES = _FIXED_HEADER_BYTES + MAX_CONTENT_TYPE_BYTES + 2 + MAX_RECIPIENT_BYTES

_DERIVATION_SALT = b"sogni-media-encryption"
_KEY_CHECK_SECRET_INFO = b"key-check-secret-v1"
_INPUT_SECRET_INFO = b"input-secret-v1"
_KEY_CHECK_INFO = b"key-check-v1"
_RECIPIENT_AAD = b"sogni-media-recipient-v1"
_RECIPIENT_VERSION = 1
_KEY_CHECK_BYTES = 8
RECIPIENT_BYTES = 1 + _KEY_CHECK_BYTES + NONCE_BYTES + KEY_BYTES + TAG_BYTES

_INPUT_MATERIAL_INFO_PREFIX = "input-media-v1\0"
_INPUT_KEY_AAD = b"sogni-media-input-key-v1"
_INPUT_MATERIAL_BYTES = KEY_BYTES + SALT_BYTES + NONCE_PREFIX_BYTES + NONCE_BYTES
WRAPPED_INPUT_KEY_BYTES = NONCE_BYTES + KEY_BYTES + TAG_BYTES
MAX_INPUT_KEYS = 32
MAX_CONTENT_KEYED_INPUT_BYTES = 100 * 1024 * 1024

_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)", re.ASCII)
_MAX_SAFE_INTEGER = 2**53 - 1

_UNREADABLE = "Encrypted media could not be read"


def _crypto() -> tuple[Any, Any, Any]:
    try:
        from Crypto.Cipher import AES
        from Crypto.Hash import SHA256
        from Crypto.Protocol.KDF import HKDF
    except ImportError as error:  # pragma: no cover - installed with eth-account
        raise MediaEncryptionError(
            "unsupported", "Media encryption requires the 'pycryptodome' package"
        ) from error
    return AES, HKDF, SHA256


def random_bytes(length: int) -> bytes:
    return os.urandom(length)


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(value: str) -> bytes:
    return base64.b64decode(value, validate=True)


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    _, hkdf, sha256 = _crypto()
    return hkdf(ikm, length, salt, sha256, context=info)


def _require_key(key: bytes) -> None:
    if len(key) != KEY_BYTES:
        raise MediaEncryptionError("malformed", "Media encryption keys must be 32 bytes")


def seal(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes | None = None) -> bytes:
    _require_key(key)
    aes, _, _ = _crypto()
    cipher = aes.new(key, aes.MODE_GCM, nonce=nonce, mac_len=TAG_BYTES)
    if aad:
        cipher.update(aad)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return ciphertext + tag


def open_sealed(key: bytes, nonce: bytes, sealed: bytes, aad: bytes | None = None) -> bytes:
    _require_key(key)
    aes, _, _ = _crypto()
    if len(sealed) < TAG_BYTES:
        raise MediaEncryptionError(
            "authentication_failed",
            "Encrypted media could not be verified; it may be incomplete or changed",
        )
    cipher = aes.new(key, aes.MODE_GCM, nonce=nonce, mac_len=TAG_BYTES)
    if aad:
        cipher.update(aad)
    try:
        return cipher.decrypt_and_verify(sealed[:-TAG_BYTES], sealed[-TAG_BYTES:])
    except ValueError as error:
        raise MediaEncryptionError(
            "authentication_failed",
            "Encrypted media could not be verified; it may be incomplete or changed",
        ) from error


@dataclass(frozen=True)
class MediaHeader:
    version: int
    segment_size_log2: int
    salt: bytes
    nonce_prefix: bytes
    content_type: str
    recipient: bytes
    raw: bytes


def is_encrypted_media(data: bytes | bytearray | memoryview) -> bool:
    return len(data) >= len(MAGIC) and bytes(data[: len(MAGIC)]) == MAGIC


def encode_header(
    *,
    segment_size_log2: int,
    salt: bytes,
    nonce_prefix: bytes,
    content_type: str,
    recipient: bytes,
) -> bytes:
    if (
        isinstance(segment_size_log2, bool)
        or not isinstance(segment_size_log2, int)
        or not MIN_SEGMENT_SIZE_LOG2 <= segment_size_log2 <= MAX_SEGMENT_SIZE_LOG2
    ):
        raise MediaEncryptionError("malformed", "Unsupported media encryption option")
    if len(salt) != SALT_BYTES or len(nonce_prefix) != NONCE_PREFIX_BYTES:
        raise MediaEncryptionError("malformed", _UNREADABLE)
    content = content_type.encode("utf-8")
    if len(content) > MAX_CONTENT_TYPE_BYTES:
        raise MediaEncryptionError("malformed", "Media content type is too long to encrypt")
    if len(recipient) > MAX_RECIPIENT_BYTES:
        raise MediaEncryptionError("malformed", _UNREADABLE)
    return b"".join(
        (
            MAGIC,
            bytes((VERSION, segment_size_log2)),
            bytes(salt),
            bytes(nonce_prefix),
            bytes((len(content),)),
            content,
            len(recipient).to_bytes(2, "big"),
            bytes(recipient),
        )
    )


def parse_header(data: bytes | bytearray | memoryview) -> MediaHeader:
    if not is_encrypted_media(data):
        raise MediaEncryptionError("not_encrypted", "This media is not encrypted")
    data = bytes(data[:MAX_HEADER_BYTES])
    if len(data) < _FIXED_HEADER_BYTES:
        raise MediaEncryptionError("malformed", _UNREADABLE)
    offset = len(MAGIC)
    version = data[offset]
    offset += 1
    if version != VERSION:
        raise MediaEncryptionError(
            "unsupported_version", "This encrypted media needs a newer version of the SDK"
        )
    segment_size_log2 = data[offset]
    offset += 1
    if not MIN_SEGMENT_SIZE_LOG2 <= segment_size_log2 <= MAX_SEGMENT_SIZE_LOG2:
        raise MediaEncryptionError("malformed", _UNREADABLE)
    salt = data[offset : offset + SALT_BYTES]
    offset += SALT_BYTES
    nonce_prefix = data[offset : offset + NONCE_PREFIX_BYTES]
    offset += NONCE_PREFIX_BYTES
    content_type_length = data[offset]
    offset += 1
    if len(data) < offset + content_type_length + 2:
        raise MediaEncryptionError("malformed", _UNREADABLE)
    try:
        content_type = data[offset : offset + content_type_length].decode("utf-8")
    except UnicodeDecodeError as error:
        raise MediaEncryptionError("malformed", _UNREADABLE) from error
    offset += content_type_length
    recipient_length = int.from_bytes(data[offset : offset + 2], "big")
    offset += 2
    if recipient_length > MAX_RECIPIENT_BYTES or len(data) < offset + recipient_length:
        raise MediaEncryptionError("malformed", _UNREADABLE)
    recipient = data[offset : offset + recipient_length]
    offset += recipient_length
    return MediaHeader(
        version=version,
        segment_size_log2=segment_size_log2,
        salt=salt,
        nonce_prefix=nonce_prefix,
        content_type=content_type,
        recipient=recipient,
        raw=data[:offset],
    )


def segment_nonce(nonce_prefix: bytes, index: int, is_last: bool) -> bytes:
    return (
        bytes(nonce_prefix)
        + (index & 0xFFFFFFFF).to_bytes(4, "big")
        + (b"\x01" if is_last else b"\x00")
    )


def derive_file_key(project_key: bytes, header: bytes, salt: bytes) -> bytes:
    _require_key(project_key)
    return hkdf_sha256(project_key, salt, header, KEY_BYTES)


def iter_encrypted_media(
    source: bytes | bytearray | memoryview,
    *,
    key: bytes,
    content_type: str = "",
    recipient: bytes = b"",
    segment_size_log2: int = DEFAULT_SEGMENT_SIZE_LOG2,
    salt: bytes | None = None,
    nonce_prefix: bytes | None = None,
) -> Iterator[bytes]:
    _require_key(key)
    salt = random_bytes(SALT_BYTES) if salt is None else salt
    nonce_prefix = random_bytes(NONCE_PREFIX_BYTES) if nonce_prefix is None else nonce_prefix
    header = encode_header(
        segment_size_log2=segment_size_log2,
        salt=salt,
        nonce_prefix=nonce_prefix,
        content_type=content_type,
        recipient=recipient,
    )
    file_key = derive_file_key(key, header, salt)
    view = memoryview(source)
    segment_size = 2**segment_size_log2
    total = len(view)
    segment_count = max(1, -(-total // segment_size))
    yield header
    for index in range(segment_count):
        start = index * segment_size
        plaintext = bytes(view[start : min(total, start + segment_size)])
        is_last = index == segment_count - 1
        yield seal(file_key, segment_nonce(nonce_prefix, index, is_last), plaintext)


def encrypt_media(source: bytes | bytearray | memoryview, **options: Any) -> bytes:
    return b"".join(iter_encrypted_media(source, **options))


def decrypt_with_key(
    source: bytes | bytearray | memoryview, header: MediaHeader, project_key: bytes
) -> bytes:
    file_key = derive_file_key(project_key, header.raw, header.salt)
    view = memoryview(source)
    sealed_segment_size = 2**header.segment_size_log2 + TAG_BYTES
    total = len(view)
    body_length = total - len(header.raw)
    if body_length < TAG_BYTES:
        raise MediaEncryptionError("malformed", _UNREADABLE)
    segment_count = -(-body_length // sealed_segment_size)
    last_length = body_length - (segment_count - 1) * sealed_segment_size
    if last_length < TAG_BYTES:
        raise MediaEncryptionError("malformed", _UNREADABLE)
    parts: list[bytes] = []
    for index in range(segment_count):
        start = len(header.raw) + index * sealed_segment_size
        sealed = bytes(view[start : min(total, start + sealed_segment_size)])
        is_last = index == segment_count - 1
        parts.append(
            open_sealed(file_key, segment_nonce(header.nonce_prefix, index, is_last), sealed)
        )
    return b"".join(parts)


def decrypt_media(
    source: bytes | bytearray | memoryview, resolve_key: Callable[[MediaHeader], bytes]
) -> tuple[bytes, MediaHeader]:
    header = parse_header(source)
    return decrypt_with_key(source, header, resolve_key(header)), header


class AccountKey:
    __slots__ = ("_check_secret", "_input_secret", "_key")

    def __init__(self, secret: bytes) -> None:
        if len(secret) != KEY_BYTES:
            raise MediaEncryptionError("malformed", "Media encryption keys must be 32 bytes")
        self._key = bytes(secret)
        self._check_secret = hkdf_sha256(
            self._key, _DERIVATION_SALT, _KEY_CHECK_SECRET_INFO, KEY_BYTES
        )
        self._input_secret = hkdf_sha256(self._key, _DERIVATION_SALT, _INPUT_SECRET_INFO, KEY_BYTES)

    def __repr__(self) -> str:
        return "AccountKey(...)"

    def key_check(self, nonce: bytes) -> bytes:
        return hkdf_sha256(self._check_secret, nonce, _KEY_CHECK_INFO, _KEY_CHECK_BYTES)


def import_account_key(secret: bytes) -> AccountKey:
    return AccountKey(bytes(secret))


def wrap_project_key(account: AccountKey, project_key: bytes, nonce: bytes | None = None) -> bytes:
    _require_key(project_key)
    nonce = random_bytes(NONCE_BYTES) if nonce is None else bytes(nonce)
    wrapped = seal(account._key, nonce, project_key, _RECIPIENT_AAD)
    return bytes((_RECIPIENT_VERSION,)) + account.key_check(nonce) + nonce + wrapped


def unwrap_project_key(accounts: AccountKey | Sequence[AccountKey], stanza: bytes) -> bytes:
    if len(stanza) == 0:
        raise MediaEncryptionError("no_recipient", "This media can only be opened with its own key")
    if len(stanza) != RECIPIENT_BYTES or stanza[0] != _RECIPIENT_VERSION:
        raise MediaEncryptionError("malformed", _UNREADABLE)
    key_check = bytes(stanza[1 : 1 + _KEY_CHECK_BYTES])
    nonce = bytes(stanza[1 + _KEY_CHECK_BYTES : 1 + _KEY_CHECK_BYTES + NONCE_BYTES])
    wrapped = bytes(stanza[1 + _KEY_CHECK_BYTES + NONCE_BYTES :])
    candidates = [accounts] if isinstance(accounts, AccountKey) else list(accounts)
    for account in candidates:
        if hmac.compare_digest(account.key_check(nonce), key_check):
            return open_sealed(account._key, nonce, wrapped, _RECIPIENT_AAD)
    raise MediaEncryptionError("wrong_account_key", "This media belongs to a different account")


@dataclass(frozen=True)
class InputMaterial:
    key: bytes
    salt: bytes
    nonce_prefix: bytes
    recipient_nonce: bytes


def _split_material(material: bytes) -> InputMaterial:
    offset = 0

    def take(length: int) -> bytes:
        nonlocal offset
        part = material[offset : offset + length]
        offset += length
        return part

    return InputMaterial(
        key=take(KEY_BYTES),
        salt=take(SALT_BYTES),
        nonce_prefix=take(NONCE_PREFIX_BYTES),
        recipient_nonce=take(NONCE_BYTES),
    )


def derive_input_material(
    account: AccountKey, plaintext_sha256: bytes, content_type: str
) -> InputMaterial:
    if len(plaintext_sha256) != 32:
        raise MediaEncryptionError("malformed", _UNREADABLE)
    info = (_INPUT_MATERIAL_INFO_PREFIX + content_type).encode("utf-8")
    return _split_material(
        hkdf_sha256(account._input_secret, plaintext_sha256, info, _INPUT_MATERIAL_BYTES)
    )


def random_input_material() -> InputMaterial:
    return _split_material(random_bytes(_INPUT_MATERIAL_BYTES))


def input_material_for_media(
    account: AccountKey, media: bytes | bytearray | memoryview, content_type: str
) -> InputMaterial:
    if len(media) > MAX_CONTENT_KEYED_INPUT_BYTES:
        return random_input_material()
    return derive_input_material(account, hashlib.sha256(media).digest(), content_type)


def wrap_input_key(project_key: bytes, input_key: bytes, nonce: bytes | None = None) -> bytes:
    _require_key(project_key)
    _require_key(input_key)
    nonce = random_bytes(NONCE_BYTES) if nonce is None else bytes(nonce)
    return nonce + seal(project_key, nonce, input_key, _INPUT_KEY_AAD)


def unwrap_input_key(project_key: bytes, wrapped: bytes) -> bytes:
    if len(wrapped) != WRAPPED_INPUT_KEY_BYTES:
        raise MediaEncryptionError("malformed", _UNREADABLE)
    return open_sealed(project_key, wrapped[:NONCE_BYTES], wrapped[NONCE_BYTES:], _INPUT_KEY_AAD)


class DecryptedMediaReader:
    """Seekable reads of encrypted video or audio, without downloading it whole.

    Offsets refer to the media itself; ``end`` is exclusive. Each chunk is
    verified before it is returned. Close the reader when it is no longer
    needed, or use it as an async context manager.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        url: str,
        resolve_key: Callable[[MediaHeader], Awaitable[bytes]],
        *,
        headers: dict[str, str] | None = None,
        assert_current: Callable[[], None] | None = None,
        on_close: Callable[[DecryptedMediaReader], None] | None = None,
    ) -> None:
        self._http = http
        self._url = url
        self._resolve_key = resolve_key
        self._headers = dict(headers or {})
        self._assert_current = assert_current
        self._on_close = on_close
        self._closed = False
        self._file_key: bytes | None = None
        self._header: MediaHeader | None = None
        self._total = 0
        self._segment_count = 0
        self.content_type = ""
        self.size = 0

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._file_key = None
        if self._on_close is not None:
            self._on_close(self)

    async def aclose(self) -> None:
        self.close()

    async def __aenter__(self) -> DecryptedMediaReader:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        self.close()

    def _assert_open(self) -> None:
        if self._closed:
            raise ValueError("Encrypted media reader is closed")
        if self._assert_current is not None:
            self._assert_current()

    async def _fetch_range(
        self, start: int, end: int, expected_total: int | None = None
    ) -> tuple[bytes, int]:
        self._assert_open()
        headers = {
            **self._headers,
            "Range": f"bytes={start}-{end - 1}",
            "Accept-Encoding": "identity",
        }
        async with self._http.stream("GET", self._url, headers=headers) as response:
            self._assert_open()
            match = _CONTENT_RANGE.fullmatch(response.headers.get("content-range", ""))
            encoding = response.headers.get("content-encoding")
            if (
                response.status_code != 206
                or not match
                or (encoding and encoding.lower() != "identity")
            ):
                raise MediaEncryptionError(
                    "unsupported", "Reading encrypted media in ranges needs HTTP byte-range support"
                )
            first, last, complete = (int(value) for value in match.groups())
            if (
                complete <= 0
                or complete > _MAX_SAFE_INTEGER
                or first != start
                or last != min(end, complete) - 1
                or (expected_total is not None and complete != expected_total)
            ):
                raise MediaEncryptionError("malformed", _UNREADABLE)
            expected = last - first + 1
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > expected:
                    raise MediaEncryptionError("malformed", _UNREADABLE)
                body += chunk
            if len(body) != expected:
                raise MediaEncryptionError("malformed", _UNREADABLE)
        self._assert_open()
        return bytes(body), complete

    async def _open(self) -> None:
        try:
            prefix, total = await self._fetch_range(0, MAX_HEADER_BYTES)
            header = parse_header(prefix)
            segment_size = 2**header.segment_size_log2
            sealed_segment_size = segment_size + TAG_BYTES
            body_length = total - len(header.raw)
            segment_count = -(-body_length // sealed_segment_size)
            final_length = body_length - (segment_count - 1) * sealed_segment_size
            if body_length < TAG_BYTES or final_length < TAG_BYTES or segment_count > 2**32:
                raise MediaEncryptionError("malformed", _UNREADABLE)
            project_key = await self._resolve_key(header)
            self._assert_open()
            self._file_key = derive_file_key(project_key, header.raw, header.salt)
            self._header = header
            self._total = total
            self._segment_count = segment_count
            self.content_type = header.content_type
            self.size = body_length - segment_count * TAG_BYTES
            # Verify the start and the end first, so the reported type and size
            # can be trusted before any read.
            await self._read_segment(0)
            if segment_count > 1:
                await self._read_segment(segment_count - 1)
        except BaseException:
            self.close()
            raise

    async def _read_segment(self, index: int) -> bytes:
        self._assert_open()
        header = self._header
        file_key = self._file_key
        if header is None or file_key is None:
            raise ValueError("Encrypted media reader is closed")
        sealed_segment_size = 2**header.segment_size_log2 + TAG_BYTES
        start = len(header.raw) + index * sealed_segment_size
        sealed, _ = await self._fetch_range(
            start, min(self._total, start + sealed_segment_size), self._total
        )
        plaintext = open_sealed(
            file_key,
            segment_nonce(header.nonce_prefix, index, index == self._segment_count - 1),
            sealed,
        )
        self._assert_open()
        return plaintext

    def _validate_range(self, start: int, end: int) -> None:
        self._assert_open()
        for value in (start, end):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("Decrypted byte range is outside the media")
        if start < 0 or end < start or end > self.size:
            raise ValueError("Decrypted byte range is outside the media")

    async def stream(self, start: int = 0, end: int | None = None) -> AsyncIterator[bytes]:
        """Yield ``[start, end)`` one verified piece at a time."""

        end = self.size if end is None else end
        self._validate_range(start, end)
        header = self._header
        assert header is not None
        segment_size = 2**header.segment_size_log2
        offset = start
        while offset < end:
            index = offset // segment_size
            plaintext = await self._read_segment(index)
            begin = offset % segment_size
            count = min(len(plaintext) - begin, end - offset)
            if count <= 0:
                raise MediaEncryptionError("malformed", _UNREADABLE)
            yield plaintext[begin : begin + count]
            offset += count

    async def read(self, start: int, end: int) -> bytes:
        """Return ``[start, end)``. Prefer :meth:`stream` for long ranges."""

        self._validate_range(start, end)
        return b"".join([chunk async for chunk in self.stream(start, end)])
