"""Admin-authored in-app announcements (pinned banners and toasts).

Port of `src/Announcements/index.ts` in the TypeScript client.

Live delivery is the ``appAlert`` socket event. Opt in when constructing the
client (``socket_event_subscriptions={"appAlert": True}``) and listen with
``client.on("appAlert", handler)``; without the opt-in the server sends the same
announcement as a plain ``toastMessage`` instead, so no existing integration
breaks. A client never receives both renderings of one announcement.

``appAlert`` is NOT at-most-once: a live pinned announcement is re-sent on every
reconnect while its window is open, so a user who was offline when it published
still receives it. Deduplicate on ``id``.

This module covers the two moments the socket cannot: reading what is live at
startup before the socket has authenticated, and dismissing an announcement per
ACCOUNT so it stays dismissed on every device.

Full contract: ``docs/app-alert-contract.md`` in sogni-socket.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from .transport import ApiClient


class AnnouncementsApi:
    """Read and dismiss in-app announcements for the signed-in account."""

    def __init__(self, client: ApiClient) -> None:
        self.client = client

    async def active(self, platform: str | None = None) -> list[dict[str, Any]]:
        """Announcements live for this account, minus anything already dismissed.

        :param platform: This client's ``appSource`` (e.g. ``"sogni-mac"``).
            Platform-scoped announcements only match when it is supplied — one
            narrowed to macOS is deliberately withheld from a caller that cannot
            say what it is, rather than sent to everyone.
        """
        params = {"platform": platform} if platform else None
        body = await self.client.rest.get("/v1/announcements/active", params)
        announcements = body.get("announcements") if isinstance(body, Mapping) else None
        return announcements if isinstance(announcements, list) else []

    async def dismiss(self, announcement_id: str) -> None:
        """Dismiss one announcement for this account, on every device.

        Call this when the user closes a pinned announcement.
        """
        await self.client.rest.post(
            f"/v1/announcements/{quote(str(announcement_id), safe='')}/dismiss"
        )


__all__ = ["AnnouncementsApi"]
