"""Manage the authenticated account's personal LoRA library."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from .errors import SogniError
from .utils import normalize_params


class PersonalLoras:
    """Import, inspect, discover, and remove personal LoRAs without caching private data."""

    def __init__(self, rest: Any) -> None:
        self._rest = rest
        self._session = 0
        auth = getattr(rest, "auth", None)
        if auth is not None:
            auth.on("updated", self._reset)

    def _reset(self, _data: Any = None) -> None:
        self._session += 1

    def _assert_session(self, session: int) -> None:
        if session != self._session:
            raise SogniError("The account changed. Refresh your LoRA library.")

    async def _read(self, path: str) -> dict[str, Any]:
        session = self._session
        response = await self._rest.get(path)
        self._assert_session(session)
        return response["data"]

    async def list(self) -> dict[str, Any]:
        """Return imports, supported models, and limits, including after a subscription lapses."""
        return await self._read("/v1/loras/personal")

    async def get(self, lora_id: str) -> dict[str, Any]:
        """Read one owned import; unavailable IDs return an API error."""
        return await self._read(f"/v1/loras/personal/{quote(lora_id, safe='')}")

    async def import_lora(
        self, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Start an import with URL, name, model_id, and explicit rights_confirmed consent.

        Poll :meth:`get` until its status is ready, rejected, or revoked.
        """
        session = self._session
        response = await self._rest.post("/v1/loras/personal", normalize_params(params, **kwargs))
        self._assert_session(session)
        return response["data"]

    importLora = import_lora

    async def remove(self, lora_id: str) -> None:
        """Remove an owned entry, including after a subscription lapses."""
        await self._rest.delete(f"/v1/loras/personal/{quote(lora_id, safe='')}")

    async def catalog(
        self, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Read ready imports, compatibility, and strength ranges; requires entitlement."""
        model_id = normalize_params(params, **kwargs).get("modelId")
        payload = await self._read("/v1/loras/personal/catalog")
        return {
            "loras": [
                row for row in payload["loras"]
                if not model_id or model_id in row["modelIds"]
            ]
        }
