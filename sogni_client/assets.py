"""Private subscriber uploads that can be reused across projects.

Port of sogni-client's ``ReusableUploads`` (5.42.0). Uploads stay private to the
signed-in account; the API verifies the checksum-bound bytes before a saved
upload becomes reusable, and binding copies it into a project's input slot.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import weakref
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar
from urllib.parse import quote

from .errors import ApiError, SogniError

T = TypeVar("T")

MAX_SAVED_UPLOAD_BYTES = 100 * 1024 * 1024
SUPPORTED_SAVED_UPLOAD_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "video/mp4",
        "video/quicktime",
        "video/webm",
        "audio/mp4",
        "audio/mpeg",
        "audio/flac",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
    }
)
# Automatic reuse may fall back to an ordinary project upload only when saved
# storage could not be prepared. Later failures must surface before submission.
_FALLBACK_STATUSES = frozenset({400, 403, 404, 409, 410, 503})
_CAPABILITY_UNAVAILABLE_STATUSES = frozenset({403, 404, 503})


def _upload_error(status: int, message: str) -> ApiError:
    return ApiError(status, {"status": "error", "errorCode": 0, "message": message})


class ReusableUploads:
    """Manage subscriber uploads once and reuse them across projects."""

    def __init__(self, rest: Any) -> None:
        self._rest = rest
        self._pending: dict[str, asyncio.Task[dict[str, Any]]] = {}
        # Bound hashing/upload memory even when a project has many reference files.
        self._lanes = asyncio.Semaphore(2)
        self._session = 0
        self._availability: tuple[float, asyncio.Task[bool]] | None = None
        self._preparation_failures: weakref.WeakSet[BaseException] = weakref.WeakSet()
        auth = getattr(rest, "auth", None)
        if auth is not None and callable(getattr(auth, "on", None)):
            auth.on("updated", self._reset)

    def _reset(self, _data: Any = None) -> None:
        self._session += 1
        self._pending.clear()
        self._availability = None

    def _assert_session(self, session: int) -> None:
        if session != self._session:
            raise SogniError("The account changed. Select the upload again.")

    async def _can_automatically_save(self) -> bool:
        if self._availability and self._availability[0] > time.monotonic():
            return await self._availability[1]

        async def lookup() -> bool:
            try:
                response = await self._rest.get("/v1/assets/capabilities")
            except ApiError as error:
                if error.status in _CAPABILITY_UNAVAILABLE_STATUSES:
                    return False
                self._availability = None
                raise
            return (response or {}).get("data", {}).get("enabled") is True

        task = asyncio.ensure_future(lookup())
        self._availability = (time.monotonic() + 60, task)
        return await task

    async def list(self) -> dict[str, Any]:
        """Return ``{"assets": [...], "limits": {...}}`` for the signed-in account."""
        response = await self._rest.get("/v1/assets")
        return response["data"]

    async def remove(self, asset_id: str) -> None:
        await self._rest.delete(f"/v1/assets/{quote(asset_id, safe='')}")

    async def bind(self, asset_id: str, binding: dict[str, Any]) -> None:
        """Copy a saved upload into a project input slot (``projectId``, ``type``, ``id``)."""
        session = self._session

        async def post() -> Any:
            self._assert_session(session)
            return await self._rest.post(
                f"/v1/assets/{quote(asset_id, safe='')}/bind", dict(binding)
            )

        await self._retry_busy(post)
        self._assert_session(session)

    async def _retry_busy(self, operation: Callable[[], Awaitable[T]]) -> T:
        attempt = 0
        while True:
            try:
                return await operation()
            except ApiError as error:
                if error.status != 423 or attempt >= 4:
                    raise
                await asyncio.sleep(0.25 * 2**attempt)
                attempt += 1

    async def upload(
        self, data: bytes, content_type: str, name: str = "Saved upload"
    ) -> dict[str, Any]:
        """Upload once; the API verifies the file before making it reusable."""
        session = self._session
        if not data or len(data) > MAX_SAVED_UPLOAD_BYTES:
            raise _upload_error(400, "Choose a saved upload no larger than 100 MiB.")
        async with self._lanes:
            self._assert_session(session)
            sha256 = (
                await asyncio.to_thread(lambda: hashlib.sha256(data).hexdigest())
                if len(data) > 8 * 1024 * 1024
                else hashlib.sha256(data).hexdigest()
            )
            self._assert_session(session)
            key = f"{sha256}:{content_type}"
            existing = self._pending.get(key)
            if existing is not None:
                return await asyncio.shield(existing)
            task = asyncio.ensure_future(
                self._perform_upload(data, sha256, content_type, name, session)
            )
            self._pending[key] = task
            try:
                return await asyncio.shield(task)
            finally:
                if self._pending.get(key) is task:
                    del self._pending[key]

    async def _perform_upload(
        self, data: bytes, sha256: str, content_type: str, name: str, session: int
    ) -> dict[str, Any]:
        async def prepare() -> Any:
            self._assert_session(session)
            return await self._rest.post(
                "/v1/assets/prepare",
                {"sha256": sha256, "bytes": len(data), "contentType": content_type, "name": name},
            )

        try:
            result = await self._retry_busy(prepare)
        except ApiError as error:
            self._preparation_failures.add(error)
            raise
        self._assert_session(session)
        prepared = result["data"]
        if prepared.get("state") == "ready":
            return prepared
        if not prepared.get("uploadUrl") or not prepared.get("uploadHeaders"):
            raise SogniError("The saved upload could not be prepared.")
        status = await self._rest.put_signed(
            prepared["uploadUrl"], data, headers=dict(prepared["uploadHeaders"])
        )
        self._assert_session(session)
        # A concurrent/retried write-once upload may already have finished. The
        # server still verifies the stored bytes before acknowledging completion.
        if not 200 <= status < 300 and status != 412:
            raise _upload_error(status, "Could not upload the selected file.")

        async def finalize() -> Any:
            self._assert_session(session)
            return await self._rest.post(f"/v1/assets/{quote(prepared['id'], safe='')}/finalize")

        finalized = await self._retry_busy(finalize)
        self._assert_session(session)
        return finalized["data"]

    async def try_bind_file(
        self,
        data: bytes,
        content_type: str | None,
        binding: dict[str, Any],
        name: str = "Saved upload",
    ) -> bool:
        """Existing project uploads remain available when saved uploads cannot be used."""
        if (
            not content_type
            or content_type not in SUPPORTED_SAVED_UPLOAD_TYPES
            or not data
            or len(data) > MAX_SAVED_UPLOAD_BYTES
        ):
            return False
        session = self._session
        if not await self._can_automatically_save():
            return False
        self._assert_session(session)
        try:
            saved = await self.upload(data, content_type, name)
            self._assert_session(session)
            await self.bind(saved["id"], binding)
            return True
        except ApiError as error:
            # Fallback is permitted only before a transfer was prepared. Checksum,
            # storage-access and binding failures must surface before job submission.
            if error in self._preparation_failures and error.status in _FALLBACK_STATUSES:
                return False
            raise

    tryBindFile = try_bind_file
