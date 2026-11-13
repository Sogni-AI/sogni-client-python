"""The public Sogni changelog (``/v1/changelog``).

Port of ``src/Changelog/index.ts`` in the TypeScript client.

What changed on Sogni for artists, developers and worker operators. These reads
need no sign-in, so they work on a client that is not authenticated. Responses
may be up to five minutes old; poll no more often than that. A refused request
raises :class:`~sogni_client.errors.ApiError` with the API's status and message.

Entries are returned as the API sends them, with the wire field names
(``bodyMarkdown``, ``actionRequired``, ``publishedAt``...). Times are ISO 8601
UTC strings.

Full contract: https://docs.sogni.ai/api-reference/changelog/
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict, cast
from urllib.parse import quote

from .errors import ApiError
from .transport import ApiClient

#: Who a changelog entry is for.
ChangelogAudience = Literal["artist", "developer", "worker"]

#: The kind of change. Every entry has exactly one.
ChangelogEntryType = Literal[
    "new", "improved", "fixed", "release", "deprecated", "removed", "notice"
]


class ChangelogLink(TypedDict):
    """A page related to a changelog entry."""

    label: str
    url: str


class ChangelogEntry(TypedDict):
    """One published changelog entry.

    ``actionRequired`` is ``True`` when you need to change something, such as for
    a model retirement. For a deprecation or removal, ``effectiveAt`` is the day
    the old behavior stops working. Lists sort and filter on ``publishedAt``.
    """

    id: str
    url: str
    title: str
    summary: str
    bodyMarkdown: str | None
    type: ChangelogEntryType
    audiences: list[ChangelogAudience]
    products: list[str]
    version: str | None
    links: list[ChangelogLink]
    actionRequired: bool
    effectiveAt: str | None
    publishedAt: str
    updatedAt: str


class ChangelogProduct(TypedDict):
    """A product entries can name. ``id`` is the value for the ``product`` filter."""

    id: str
    label: str
    audiences: list[ChangelogAudience]


class ListChangelogResult(TypedDict):
    """One page of entries, newest first. ``nextCursor`` is ``None`` on the last page."""

    entries: list[ChangelogEntry]
    nextCursor: str | None


def _joined(value: str | Sequence[str] | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return ",".join(value)


class ChangelogApi:
    """Read the public Sogni changelog: a page of entries, one entry, the products."""

    def __init__(self, client: ApiClient) -> None:
        self.client = client

    async def list(
        self,
        *,
        audience: str | Sequence[str] | None = None,
        product: str | Sequence[str] | None = None,
        type: str | Sequence[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        action_required: bool = False,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> ListChangelogResult:
        """One page of published entries, newest first.

        Every filter is optional, and an entry must match every filter given.
        ``audience`` (``artist``, ``developer``, ``worker``), ``product`` (ids
        from :meth:`products`) and ``type`` (``new``, ``improved``, ``fixed``,
        ``release``, ``deprecated``, ``removed``, ``notice``) take one value or a
        list of up to 20, and match entries that have any of them.

        :param since: Entries published at or after this time: an ISO 8601 date
            (``"2026-11-01"``, read as midnight UTC) or a date-time with a zone
            (``"2026-11-01T17:00:00Z"``).
        :param until: Entries published before this time, in the same format.
        :param action_required: ``True`` lists only the entries you need to act on.
        :param limit: Entries per page. The API defaults to 50 and returns at most 100.
        :param cursor: The previous page's ``nextCursor``. Send the same filters with it.
        """
        params: dict[str, Any] = {
            "audience": _joined(audience),
            "product": _joined(product),
            "type": _joined(type),
            "since": since,
            "until": until,
            "actionRequired": "true" if action_required else None,
            "limit": limit,
            "cursor": cursor,
        }
        body = await self.client.rest.get("/v1/changelog", params)
        return cast(ListChangelogResult, body["data"])

    async def get(self, entry_id: str) -> ChangelogEntry:
        """One published entry by its id.

        An id with no published entry raises :class:`~sogni_client.errors.ApiError`
        with ``status`` 404 and ``error_code`` 204.
        """
        body = await self.client.rest.get(f"/v1/changelog/{quote(str(entry_id), safe='')}")
        data = body.get("data") if isinstance(body, Mapping) else None
        entry = data.get("entry") if isinstance(data, Mapping) else None
        if not isinstance(entry, Mapping):
            # An empty id requests the list itself, and ``products`` or
            # ``feed.json`` the product list or the feed; none of them is an entry.
            raise ApiError(
                404,
                {"status": "error", "message": "Changelog entry not found", "errorCode": 204},
            )
        return cast(ChangelogEntry, entry)

    async def products(self) -> list[ChangelogProduct]:
        """The products entries can name, in display order."""
        body = await self.client.rest.get("/v1/changelog/products")
        return cast("list[ChangelogProduct]", body["data"]["products"])


__all__ = [
    "ChangelogApi",
    "ChangelogAudience",
    "ChangelogEntry",
    "ChangelogEntryType",
    "ChangelogLink",
    "ChangelogProduct",
    "ListChangelogResult",
]
