"""An image slot never uploads text in place of image bytes (port of sogni-client 5.84.2).

Mirrors sogni-client's ``scripts/check-image-upload-body.cjs``: a data: URL, an
HTML page or base64 text in an image slot is refused with a 400 ``ApiError``
naming the slot, before any upload URL, saved-upload preparation, PUT or
jobRequest. Image bytes, SVG markup and anything else upload byte for byte.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from sogni_client.errors import ApiError
from sogni_client.events import EventEmitter
from sogni_client.projects import ProjectsApi
from tests.test_projects import FakeSocket

PNG = bytes.fromhex("89504e470d0a1a0a0000000d494844520000040000000400080200000000") + os.urandom(
    256
)
JPEG = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + os.urandom(256)
# Base64 text, as a canvas export or an encoded image read back as text produces.
BASE64_TEXT = base64.b64encode(os.urandom(3000))
UPLOAD_URL = "https://storage.example.test/presigned-put"


class UploadRest:
    def __init__(self, *, saved_uploads: bool = False) -> None:
        self.saved_uploads = saved_uploads
        self.calls: list[tuple[str, str, Any]] = []
        self.puts: list[bytes] = []

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append(("GET", path, params))
        if path == "/v1/assets/capabilities":
            return {"status": "success", "data": {"enabled": self.saved_uploads}}
        if path in {"/v1/image/uploadUrl", "/v1/media/uploadUrl"}:
            return {"data": {"uploadUrl": UPLOAD_URL}}
        raise AssertionError(f"Unexpected GET {path}")

    async def post(self, path: str, body: Any = None, **_: Any) -> Any:
        self.calls.append(("POST", path, body))
        raise AssertionError(f"Unexpected POST {path}")

    async def put_bytes(self, url: str, data: bytes, *, content_type: str | None = None) -> None:
        assert url == UPLOAD_URL
        self.calls.append(("PUT", url, content_type))
        self.puts.append(data)


class UploadClient(EventEmitter):
    def __init__(self, rest: UploadRest) -> None:
        super().__init__()
        self.rest = rest
        self.socket = FakeSocket()
        self.app_source = "pytest"


def projects(*, saved_uploads: bool = False) -> tuple[ProjectsApi, UploadRest]:
    rest = UploadRest(saved_uploads=saved_uploads)
    api = ProjectsApi(UploadClient(rest))
    api.get_model_options = AsyncMock(
        return_value={
            "type": "image",
            "sampler": {"allowed": [], "default": None},
            "scheduler": {"allowed": [], "default": None},
        }
    )
    return api, rest


def upload_urls(rest: UploadRest) -> list[Any]:
    return [call for call in rest.calls if call[1].endswith("uploadUrl")]


# Every role that fills an image slot, with the label its refusal uses.
IMAGE_SLOTS = [
    ("startingImage", "Guide image"),
    ("cnImage", "ControlNet image"),
    ("contextImage1", "Context image 1"),
    ("contextImage2", "Context image 2"),
    ("keyframeImage1", "Keyframe image 1"),
    ("referenceImage", "Reference image"),
    ("referenceMask", "Reference mask"),
    ("referenceImageEnd", "End reference image"),
]


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


FOUND = {
    "base64 text": "its contents are base64 text",
    "data: URL": "its contents are a data: URL",
    "HTML page": "its contents are an HTML page",
}
UPLOAD_THE_FILE = "Upload the image file itself."

# (name, make value from tmp_path, reason). A str or Path is a file path in this client.
REFUSED = [
    ("base64 text bytes", lambda _: BASE64_TEXT, "base64 text"),
    ("base64 text in a bytearray", lambda _: bytearray(BASE64_TEXT), "base64 text"),
    ("base64 text in a memoryview", lambda _: memoryview(BASE64_TEXT), "base64 text"),
    ("an encoded PNG", lambda _: base64.b64encode(PNG), "base64 text"),
    (
        "a data: URL",
        lambda _: b"data:image/png;base64," + base64.b64encode(PNG),
        "data: URL",
    ),
    ("an upper-case data: URL", lambda _: b"DATA:image/png;base64,AAAA", "data: URL"),
    (
        "an HTML page",
        lambda _: b"<!doctype html><html><head><title>Not found</title>",
        "HTML page",
    ),
    (
        "an HTML page with a BOM",
        lambda _: '﻿\n  <HTML lang="en"><body>Sign in</body></HTML>'.encode(),
        "HTML page",
    ),
    ("an html tag", lambda _: b"\r\n\t<html>", "HTML page"),
    (
        "a .png path holding base64 text",
        lambda tmp: str(write(tmp, "portrait.png", BASE64_TEXT)),
        "base64 text",
    ),
    (
        "a Path holding an HTML page",
        lambda tmp: write(tmp, "portrait.jpg", b"<!DOCTYPE HTML PUBLIC><title>403</title>"),
        "HTML page",
    ),
    (
        "a binary file holding a data: URL",
        lambda _: io.BytesIO(b"data:image/jpeg;base64,/9j/4AAQ"),
        "data: URL",
    ),
    ("a text file holding base64 text", lambda _: io.StringIO(BASE64_TEXT.decode()), "base64 text"),
]

# (name, make value from tmp_path, bytes expected at the PUT), as before the check.
UPLOADED = [
    ("PNG bytes", lambda _: PNG, PNG),
    ("JPEG bytes", lambda _: JPEG, JPEG),
    ("a PNG bytearray", lambda _: bytearray(PNG), PNG),
    ("GIF bytes", lambda _: b"GIF89aAAAAAAAAAAAAAAAAAA" + os.urandom(64), None),
    ("WebP bytes", lambda _: b"RIFF\x24\x00\x00\x00WEBPVP8 " + os.urandom(64), None),
    (
        "SVG markup",
        lambda _: b'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"/>',
        None,
    ),
    (
        "SVG with an XML prolog",
        lambda _: b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"/>',
        None,
    ),
    ("unrecognised bytes", lambda _: bytes([0, 0, 0, 0x0C]) + os.urandom(64), None),
    ("short text", lambda _: b"reference", None),
    ("an html tag past the first 64 bytes", lambda _: b" " * 64 + b"<html>", None),
    ("a not-html tag", lambda _: b"<htmlx>", None),
    ("a .png path", lambda tmp: str(write(tmp, "portrait.png", PNG)), PNG),
    ("a Path", lambda tmp: write(tmp, "portrait.png", PNG), PNG),
    ("a binary file", lambda _: io.BytesIO(PNG), PNG),
]


@pytest.mark.parametrize(("role", "label"), IMAGE_SLOTS)
@pytest.mark.parametrize(("name", "make", "reason"), REFUSED, ids=[case[0] for case in REFUSED])
async def test_image_slot_refuses_text_before_asking_for_an_upload_url(
    tmp_path: Path, role: str, label: str, name: str, make: Any, reason: str
) -> None:
    api, rest = projects()

    with pytest.raises(ApiError) as caught:
        await api._upload_asset("project-1", role, make(tmp_path), media=False)

    assert caught.value.status == 400
    assert str(caught.value) == f"{label} is not image data: {FOUND[reason]}. {UPLOAD_THE_FILE}"
    assert upload_urls(rest) == []
    assert rest.puts == []


@pytest.mark.parametrize(("role", "_label"), IMAGE_SLOTS)
@pytest.mark.parametrize(("name", "make", "expected"), UPLOADED, ids=[case[0] for case in UPLOADED])
async def test_image_slot_uploads_image_bytes_and_markup_unchanged(
    tmp_path: Path, role: str, _label: str, name: str, make: Any, expected: bytes | None
) -> None:
    api, rest = projects()
    value = make(tmp_path)
    if expected is None:
        expected = bytes(value)

    await api._upload_asset("project-1", role, value, media=False)

    assert [call[2]["type"] for call in upload_urls(rest)] == [role]
    assert rest.puts == [expected]


@pytest.mark.parametrize("role", ["referenceAudio", "referenceVideo", "referenceVideo2"])
async def test_audio_and_video_slots_are_not_checked(role: str) -> None:
    api, rest = projects()

    await api._upload_asset("project-1", role, BASE64_TEXT, media=True)

    assert rest.puts == [BASE64_TEXT]


async def test_refusal_comes_before_saved_upload_preparation() -> None:
    api, rest = projects(saved_uploads=True)

    with pytest.raises(ApiError, match="^Guide image is not image data: its contents are base64"):
        await api._upload_asset("project-1", "startingImage", BASE64_TEXT, media=False)

    assert rest.calls == []


async def test_an_unreadable_file_keeps_its_own_error(tmp_path: Path) -> None:
    class Unreadable(io.RawIOBase):
        def read(self, size: int = -1) -> bytes:
            raise OSError("NotReadableError")

    api, rest = projects()

    with pytest.raises(OSError, match="NotReadableError"):
        await api._upload_asset("project-1", "contextImage1", Unreadable(), media=False)
    with pytest.raises(FileNotFoundError):
        await api._upload_asset(
            "project-1", "contextImage1", str(tmp_path / "gone.png"), media=False
        )
    assert rest.calls == []


def krea2(context_images: list[Any]) -> dict[str, Any]:
    return {
        "type": "image",
        "model_id": "krea2_identity_edit_v1_2",
        "positive_prompt": "Same person, studio portrait",
        "number_of_media": 1,
        "width": 1024,
        "height": 1024,
        "token_type": "spark",
        "context_images": context_images,
    }


async def test_create_refuses_before_submitting_so_nothing_is_charged() -> None:
    api, rest = projects()

    with pytest.raises(ApiError) as caught:
        await api.create(**krea2([BASE64_TEXT]))

    assert caught.value.status == 400
    assert str(caught.value) == (
        f"Context image 1 is not image data: {FOUND['base64 text']}. {UPLOAD_THE_FILE}"
    )
    assert upload_urls(rest) == []
    assert rest.puts == []
    assert [kind for kind, _ in api.client.socket.sent if kind == "jobRequest"] == []

    accepted, accepted_rest = projects()
    await accepted.create(**krea2([PNG]))
    assert accepted_rest.puts == [PNG]
    assert [kind for kind, _ in accepted.client.socket.sent if kind == "jobRequest"] == [
        "jobRequest"
    ]
