"""Encrypted project media: ``sogni.media_encryption``."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from ._media_encryption import (
    KEY_BYTES,
    MAX_INPUT_KEYS,
    VERSION,
    AccountKey,
    DecryptedMediaReader,
    MediaHeader,
    b64decode,
    b64encode,
    decrypt_with_key,
    encrypt_media,
    import_account_key,
    input_material_for_media,
    is_encrypted_media,
    parse_header,
    random_bytes,
    unwrap_project_key,
    wrap_input_key,
    wrap_project_key,
)
from .auth import RequestSessionError
from .errors import ApiError, MediaEncryptionError
from .events import EventEmitter
from .transport import ApiClient

_LOGGER = logging.getLogger("sogni_client")
_ACCOUNT_MEDIA_KEY_PATH = "/v1/account/media-key"
_ACCOUNT_MEDIA_KEY_VERSION = 1

__all__ = ["DecryptedMediaReader", "MediaEncryptionApi"]


class _ProjectMediaEncryption:
    """Encrypts one project's uploads while it is being submitted."""

    def __init__(self, account: AccountKey) -> None:
        self._account = account
        self._project_key = random_bytes(KEY_BYTES)
        self._recipient = wrap_project_key(account, self._project_key)
        self._input_keys: dict[str, str] = {}
        self._sealed: list[tuple[object, str, bytes]] = []

    def wire(self) -> dict[str, Any]:
        field: dict[str, Any] = {
            "version": VERSION,
            "key": b64encode(self._project_key),
            "recipient": b64encode(self._recipient),
        }
        if self._input_keys:
            field["inputKeys"] = list(self._input_keys.values())
        return field

    async def encrypt(self, media: bytes, content_type: str = "") -> bytes:
        for source, kind, sealed in self._sealed:
            if source is media and kind == content_type:
                return sealed
        material = input_material_for_media(self._account, media, content_type)
        key_id = b64encode(material.key)
        if key_id not in self._input_keys:
            if len(self._input_keys) >= MAX_INPUT_KEYS:
                raise MediaEncryptionError(
                    "unsupported",
                    f"An encrypted project can carry at most {MAX_INPUT_KEYS} different input files",
                )
            self._input_keys[key_id] = b64encode(wrap_input_key(self._project_key, material.key))
        sealed = encrypt_media(
            media,
            key=material.key,
            content_type=content_type,
            recipient=wrap_project_key(self._account, material.key, material.recipient_nonce),
            salt=material.salt,
            nonce_prefix=material.nonce_prefix,
        )
        self._sealed.append((media, content_type, sealed))
        return sealed


class MediaEncryptionApi(EventEmitter):
    """Optional encryption of project media, available as ``sogni.media_encryption``.

    Create the client with ``media_encryption="auto"`` to encrypt the media of
    every eligible project. Read results with ``job.get_result_data()``,
    :meth:`fetch_decrypted` or :meth:`open_decrypted`, which return the media
    decrypted. Without the option nothing changes.

    Events: ``availabilityChanged`` (bool), ``unlocked`` and ``locked``.
    """

    def __init__(self, client: ApiClient) -> None:
        super().__init__()
        self.client = client
        self._keyring: tuple[list[AccountKey], int] | None = None
        self._key_generation = 0
        self._readers: set[DecryptedMediaReader] = set()
        self._server_available = False
        self._key_request: tuple[int, asyncio.Future[bool]] | None = None
        self._key_retry_at = 0.0
        self._key_turned_off = False
        self._key_supplied = False
        self._tasks: set[asyncio.Task[Any]] = set()
        # Overridable so regression tests can run the flow quickly.
        self._tuning = {"key_retry_seconds": 300.0, "key_wait_seconds": 5.0}
        self._clock: Callable[[], float] = time.monotonic
        client.auth.on("updated", self._on_auth_updated)
        client.auth.on("sessionChanged", self._on_session_changed)
        client.socket.on("authenticated", self._on_socket_authenticated)
        client.socket.on("mediaEncryptionAvailabilityUpdated", self._on_availability_updated)
        if self.automatic and self.declares_capability:
            client._set_media_capability(self._holds_key)

    @property
    def automatic(self) -> bool:
        """``True`` when the client was created with ``media_encryption="auto"``."""

        return getattr(self.client, "media_encryption", None) == "auto"

    @property
    def declares_capability(self) -> bool:
        return getattr(self.client, "declare_media_capability", True) is not False

    @property
    def available(self) -> bool:
        """Whether media encryption is available for this account right now.

        A client created with ``media_encryption="auto"`` encrypts new projects
        only while this is ``True``.
        """

        return self._server_available

    def _on_auth_updated(self, authenticated: bool) -> None:
        if authenticated:
            return
        self._set_server_available(False)
        self._forget_keys()

    def _on_session_changed(self, _data: Any = None) -> None:
        self._set_server_available(False)
        self._forget_keys()

    def _on_socket_authenticated(self, data: Any) -> None:
        self._set_server_available(
            isinstance(data, dict) and data.get("mediaEncryptionAvailable") is True
        )

    def _on_availability_updated(self, available: Any) -> None:
        self._set_server_available(available is True)

    def _set_server_available(self, available: bool) -> None:
        if self._server_available == available:
            return
        self._server_available = available
        self.emit("availabilityChanged", available)
        if available and self.automatic:
            self._account_key_request()

    def is_encrypted(self, data: bytes | bytearray | memoryview) -> bool:
        """``True`` when ``data`` is encrypted media. Needs only the first 8 bytes."""

        return is_encrypted_media(data)

    isEncrypted = is_encrypted

    async def is_unlocked(self) -> bool:
        """Whether this client can open the signed-in account's encrypted media."""

        if self.automatic:
            await self._ensure_account_key()
        return self._holds_key()

    isUnlocked = is_unlocked

    async def set_account_key(self, secret: bytes) -> None:
        """Use these 32 bytes as the signed-in account's media key.

        A client created with ``media_encryption="auto"`` manages the key itself.
        The key is dropped when the account signs out or changes.
        """

        if not self.client.auth.is_authenticated:
            raise MediaEncryptionError("locked", "Sign in before setting the media key")
        is_current = self._begin_key_change()
        key = import_account_key(bytes(secret))
        if not is_current():
            raise MediaEncryptionError(
                "locked", "Media encryption was locked or the account changed"
            )
        self._key_supplied = True
        self._set_keyring([key])

    setAccountKey = set_account_key

    async def lock(self) -> None:
        """Stop opening encrypted media in this client until it is unlocked again."""

        self._forget_keys()

    async def decrypt(
        self, media: bytes | bytearray | memoryview, *, key: bytes | None = None
    ) -> bytes:
        """Return encrypted media decrypted. Pass ``key`` only when the media needs its own key."""

        assert_current = self._capture_media_session()
        data = bytes(media)
        header = parse_header(data)
        project_key = await self._resolve_project_key(header, key)
        plaintext = decrypt_with_key(data, header, project_key)
        assert_current()
        return plaintext

    async def fetch_decrypted(
        self,
        url: str,
        *,
        require_encrypted: bool = False,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        """Download ``url`` and return its media decrypted.

        Media that is not encrypted is returned as downloaded, so one call reads
        both. Pass ``require_encrypted=True`` to raise for unencrypted media instead.
        """

        assert_current = self._capture_media_session()
        response = await self.client.rest._client.get(url, headers=headers, timeout=300)
        response.raise_for_status()
        assert_current()
        return await self._open_if_encrypted(response.content, require_encrypted)

    fetchDecrypted = fetch_decrypted

    async def open_decrypted(
        self, url: str, *, headers: dict[str, str] | None = None
    ) -> DecryptedMediaReader:
        """Open encrypted video or audio for seekable reads without downloading all of it.

        The URL must support HTTP byte ranges. Offsets refer to the media itself.
        Close the reader when finished (or use ``async with``); locking or
        changing the account also closes it.
        """

        reader = DecryptedMediaReader(
            self.client.rest._client,
            url,
            lambda header: self._resolve_project_key(header, None),
            headers=headers,
            assert_current=self._capture_media_session(),
            on_close=self._readers.discard,
        )
        self._readers.add(reader)
        await reader._open()
        return reader

    openDecrypted = open_decrypted

    @property
    def _opted_in(self) -> bool:
        """Whether this client encrypts or opens media at all."""

        return self.automatic or self._key_supplied

    async def _open_if_encrypted(self, data: bytes, require_encrypted: bool = False) -> bytes:
        if is_encrypted_media(data):
            return await self.decrypt(data)
        if require_encrypted:
            raise MediaEncryptionError("not_encrypted", "This media is not encrypted")
        return data

    async def _can_encrypt_automatically(self) -> bool:
        """Whether a project submitted now can be encrypted. Never raises."""

        if not self.automatic or not self._server_available:
            return False
        request = self._account_key_request()
        if not request.done():
            await asyncio.wait({request}, timeout=self._tuning["key_wait_seconds"])
        return self._holds_key() and self._server_available

    async def _create_project_encryption(self) -> _ProjectMediaEncryption:
        account = (await self._require_account_keys())[0]
        return _ProjectMediaEncryption(account)

    def _holds_key(self) -> bool:
        return (
            self._keyring is not None
            and self._keyring[1] == self.client.auth.session_version
            and self.client.auth.is_authenticated
        )

    def _settled(self, value: bool) -> asyncio.Future[bool]:
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        future.set_result(value)
        return future

    def _account_key_request(self) -> asyncio.Future[bool]:
        """Started when needed. Never fails."""

        if self._holds_key():
            return self._settled(True)
        auth = self.client.auth
        if not self.automatic or not auth.is_authenticated or not self._server_available:
            return self._settled(False)
        session = auth.session_version
        if self._key_request is not None and self._key_request[0] == session:
            return self._key_request[1]
        if self._key_turned_off or self._clock() < self._key_retry_at:
            return self._settled(False)
        task = asyncio.create_task(self._fetch_account_key())
        self._key_request = (session, task)
        self._tasks.add(task)

        def finished(done: asyncio.Task[bool]) -> None:
            self._tasks.discard(done)
            if self._key_request is not None and self._key_request[1] is done:
                self._key_request = None

        task.add_done_callback(finished)
        return task

    async def _ensure_account_key(self) -> bool:
        return await asyncio.shield(self._account_key_request())

    async def _fetch_account_key(self) -> bool:
        assert_session = self.client.auth.capture_session()
        generation = self._key_generation
        try:
            response = await self.client.rest.get(_ACCOUNT_MEDIA_KEY_PATH)
            assert_session()
            keys = _import_account_keys(
                response.get("data") if isinstance(response, dict) else None
            )
            assert_session()
            if generation != self._key_generation:
                return self._holds_key()
            self._key_retry_at = 0.0
            self._set_keyring(keys)
            return True
        except RequestSessionError:
            return False
        except Exception as error:
            status = error.status if isinstance(error, ApiError) else None
            if status == 409:
                self._key_turned_off = True
                _LOGGER.warning(
                    "Media encryption is turned off for this account; projects are sent unencrypted"
                )
                return False
            self._key_retry_at = self._clock() + self._tuning["key_retry_seconds"]
            details: Any = (
                {"status": status, "errorCode": error.error_code}
                if isinstance(error, ApiError)
                else type(error).__name__
            )
            _LOGGER.warning(
                "Media encryption is unavailable right now; projects are sent unencrypted %s",
                details,
            )
            return False

    async def _resolve_project_key(self, header: MediaHeader, key: bytes | None) -> bytes:
        if key is not None:
            return bytes(key)
        return unwrap_project_key(await self._require_account_keys(), header.recipient)

    async def _require_account_keys(self) -> list[AccountKey]:
        if self.automatic:
            await self._ensure_account_key()
        if not self._holds_key() or self._keyring is None:
            raise MediaEncryptionError(
                "locked",
                "Media encryption is not available for this account right now"
                if self.automatic
                else "Create the client with media_encryption='auto' to use encrypted media",
            )
        return self._keyring[0]

    def _set_keyring(self, keys: list[AccountKey]) -> None:
        self._keyring = (keys, self.client.auth.session_version)
        self.emit("unlocked", None)
        if self.automatic and self.declares_capability:
            self.client._declare_media_capability()

    def _forget_keys(self) -> None:
        self._close_readers()
        self._key_generation += 1
        self._key_request = None
        self._key_retry_at = 0.0
        self._key_turned_off = False
        had_key = self._keyring is not None
        self._keyring = None
        if had_key:
            self.emit("locked", None)

    def _begin_key_change(self) -> Callable[[], bool]:
        self._close_readers()
        self._key_generation += 1
        generation = self._key_generation
        session = self.client.auth.session_version
        return lambda: (
            generation == self._key_generation and session == self.client.auth.session_version
        )

    def _close_readers(self) -> None:
        for reader in list(self._readers):
            reader.close()
        self._readers.clear()

    def _capture_media_session(self) -> Callable[[], None]:
        generation = self._key_generation
        session = self.client.auth.session_version

        def check() -> None:
            if generation != self._key_generation or session != self.client.auth.session_version:
                raise MediaEncryptionError(
                    "locked", "Media encryption was locked or the account changed"
                )

        return check


def _import_account_keys(data: Any) -> list[AccountKey]:
    current = data.get("current") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or data.get("version") != _ACCOUNT_MEDIA_KEY_VERSION
        or not isinstance(current, dict)
        or not isinstance(current.get("key"), str)
    ):
        raise MediaEncryptionError("malformed", "The media encryption response is not recognized")
    previous = data.get("previous")
    encoded: list[Any] = [current["key"]]
    if isinstance(previous, list):
        encoded.extend(entry.get("key") if isinstance(entry, dict) else None for entry in previous)
    keys: list[AccountKey] = []
    for value in encoded:
        if not isinstance(value, str):
            raise MediaEncryptionError(
                "malformed", "The media encryption response is not recognized"
            )
        try:
            secret = b64decode(value)
        except ValueError as error:
            raise MediaEncryptionError(
                "malformed", "The media encryption response is not recognized"
            ) from error
        keys.append(import_account_key(secret))
    return keys
