from __future__ import annotations

import base64
import socket
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from sogni_client.chat import _inline_image, normalize_vision_messages

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
JPEG_MINIMAL = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"

R2_ACCOUNT_HOST = "234df6a88ee221ecac622f8b1a9609e0.r2.cloudflarestorage.com"
ACCEPTED_HOSTS = [
    f"generation-output-production.{R2_ACCOUNT_HOST}",
    f"generation-output-staging.{R2_ACCOUNT_HOST}",
    f"generation-input-production.{R2_ACCOUNT_HOST}",
    f"generation-input-staging.{R2_ACCOUNT_HOST}",
    f"GENERATION-OUTPUT-PRODUCTION.{R2_ACCOUNT_HOST.upper()}",
    "complete-images-production.s3-accelerate.amazonaws.com",
    "complete-images-staging.s3-accelerate.amazonaws.com",
    "artist-upload-production.s3-accelerate.amazonaws.com",
    "artist-upload-staging.s3-accelerate.amazonaws.com",
    "complete-images-production.s3-accelerate.dualstack.amazonaws.com",
    "complete-images-production.s3.amazonaws.com",
    "complete-images-production.s3.us-east-1.amazonaws.com",
    "cdn.sogni.ai",
    "media.sogni.ai",
]

REFUSED_URLS = [
    "https://generation-output-production.ffffffffffffffffffffffffffffffff.r2.cloudflarestorage.com/job/image.png",
    "https://generation-input-production.ffffffffffffffffffffffffffffffff.r2.cloudflarestorage.com/job/image.png",
    f"https://other-bucket.{R2_ACCOUNT_HOST}/job/image.png",
    f"https://generation-output-dev.{R2_ACCOUNT_HOST}/job/image.png",
    f"https://{R2_ACCOUNT_HOST}/job/image.png",
    "https://r2.cloudflarestorage.com/job/image.png",
    "https://pub-0123456789abcdef0123456789abcdef.r2.dev/job/image.png",
    "https://generation-output-production.r2.dev/job/image.png",
    f"https://generation-output-production.{R2_ACCOUNT_HOST}.unlisted.example/job/image.png",
    f"https://other-generation-output-production.{R2_ACCOUNT_HOST}/job/image.png",
    f"https://x.generation-output-production.{R2_ACCOUNT_HOST}/job/image.png",
    "https://complete-images-production.s3-accelerate.amazonaws.com.unlisted.example/job/image.png",
    "https://complete-images-production.s3-accelerate.amazonaws.co/job/image.png",
    "https://s3-accelerate.amazonaws.com/job/image.png",
    "https://other-bucket.s3.amazonaws.com/job/image.png",
    "https://unlisted.example/job/image.png",
    f"https://unlisted.example/generation-output-production.{R2_ACCOUNT_HOST}/image.png",
    f"http://generation-output-production.{R2_ACCOUNT_HOST}/job/image.png",
    "http://complete-images-production.s3-accelerate.amazonaws.com/job/image.png",
    "http://cdn.sogni.ai/job/image.png",
    "http://127.0.0.1:18091/internal/image.png",
    "https://127.0.0.1/image.png",
    "https://[::1]/image.png",
    "https://localhost/image.png",
    f"https://user:pass@generation-output-production.{R2_ACCOUNT_HOST}/job/image.png",
    "https://user@cdn.sogni.ai/job/image.png",
    "https://@cdn.sogni.ai/job/image.png",
    "file:///etc/hosts",
    "file://localhost/tmp/image.png",
    "ftp://cdn.sogni.ai/image.png",
    "gopher://127.0.0.1:70/",
]


def _public_addrinfo(host: str, port: int, **_kwargs: object) -> list[tuple[object, ...]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", port))]


def _mock_async_client(handler):
    real_async_client = httpx.AsyncClient

    def factory(**kwargs):
        assert kwargs.get("follow_redirects") is False
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    return factory


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ACCEPTED_HOSTS)
async def test_inline_image_fetches_allowed_sogni_https_hosts(host: str) -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png", "Content-Length": str(len(PNG_1X1))},
            content=PNG_1X1,
        )

    url = f"https://{host}/2026-09-29/job-1/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    with (
        patch("sogni_client.chat.socket.getaddrinfo", side_effect=_public_addrinfo),
        patch("sogni_client.chat.httpx.AsyncClient", side_effect=_mock_async_client(handle)),
    ):
        data_uri = await _inline_image(url)

    assert data_uri == f"data:image/png;base64,{base64.b64encode(PNG_1X1).decode('ascii')}"
    assert len(seen) == 1
    assert seen[0].url.host == host.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("url", REFUSED_URLS)
async def test_inline_image_refuses_disallowed_urls_before_network_io(url: str) -> None:
    with (
        patch("sogni_client.chat.socket.getaddrinfo") as resolve,
        patch("sogni_client.chat.httpx.AsyncClient") as client_cls,
    ):
        with pytest.raises(ValueError, match="remote URLs are not allowed"):
            await _inline_image(url)

    resolve.assert_not_called()
    client_cls.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("family", "address"),
    [
        (socket.AF_INET, "127.0.0.1"),
        (socket.AF_INET, "10.0.0.8"),
        (socket.AF_INET, "169.254.169.254"),
        (socket.AF_INET, "192.168.1.10"),
        (socket.AF_INET6, "::1"),
        (socket.AF_INET6, "::ffff:127.0.0.1"),
        (socket.AF_INET6, "fe80::1%lo0"),
    ],
)
async def test_inline_image_refuses_non_global_resolved_addresses(
    family: socket.AddressFamily, address: str
) -> None:
    def local_addrinfo(_host: str, port: int, **_kwargs: object) -> list[tuple[object, ...]]:
        sockaddr = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
        return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)]

    with (
        patch("sogni_client.chat.socket.getaddrinfo", side_effect=local_addrinfo),
        patch("sogni_client.chat.httpx.AsyncClient") as client_cls,
    ):
        with pytest.raises(ValueError, match="remote URLs are not allowed"):
            await _inline_image("https://cdn.sogni.ai/job/image.png")

    client_cls.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_inline_image_refuses_remote_redirects(status: int) -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, headers={"Location": "https://cdn.sogni.ai/other.png"})

    with (
        patch("sogni_client.chat.socket.getaddrinfo", side_effect=_public_addrinfo),
        patch("sogni_client.chat.httpx.AsyncClient", side_effect=_mock_async_client(handle)),
    ):
        with pytest.raises(ValueError, match="redirects are not allowed"):
            await _inline_image("https://cdn.sogni.ai/start.png")

    assert len(seen) == 1


@pytest.mark.asyncio
async def test_inline_image_rejects_oversized_content_length_without_reading_body() -> None:
    class UnreadStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.read = False
            self.closed = False

        async def __aiter__(self):
            self.read = True
            yield JPEG_MINIMAL

        async def aclose(self) -> None:
            self.closed = True

    stream = UnreadStream()

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "image/jpeg", "Content-Length": str(10 * 1024 * 1024 + 1)},
            stream=stream,
        )

    with (
        patch("sogni_client.chat.socket.getaddrinfo", side_effect=_public_addrinfo),
        patch("sogni_client.chat.httpx.AsyncClient", side_effect=_mock_async_client(handle)),
    ):
        with pytest.raises(ValueError, match="exceeds 10MB limit"):
            await _inline_image("https://cdn.sogni.ai/large.jpg")

    assert stream.read is False
    assert stream.closed is True


@pytest.mark.asyncio
async def test_inline_image_aborts_streaming_body_as_soon_as_size_limit_is_exceeded() -> None:
    chunk_size = 1024 * 1024

    class LargeStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.chunks_emitted = 0
            self.closed = False

        async def __aiter__(self):
            for index in range(20):
                self.chunks_emitted += 1
                prefix = b"\xff\xd8\xff" if index == 0 else b""
                yield prefix + (b"A" * (chunk_size - len(prefix)))

        async def aclose(self) -> None:
            self.closed = True

    stream = LargeStream()

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "image/jpeg"}, stream=stream)

    with (
        patch("sogni_client.chat.socket.getaddrinfo", side_effect=_public_addrinfo),
        patch("sogni_client.chat.httpx.AsyncClient", side_effect=_mock_async_client(handle)),
    ):
        with pytest.raises(ValueError, match="exceeds 10MB limit"):
            await _inline_image("https://cdn.sogni.ai/streamed.jpg")

    assert stream.chunks_emitted == 11
    assert stream.closed is True


@pytest.mark.asyncio
async def test_inline_image_detects_mime_from_magic_bytes_not_headers_or_extension(
    tmp_path: Path,
) -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "image/png"}, content=b"not-a-png-body")

    with (
        patch("sogni_client.chat.socket.getaddrinfo", side_effect=_public_addrinfo),
        patch("sogni_client.chat.httpx.AsyncClient", side_effect=_mock_async_client(handle)),
    ):
        with pytest.raises(ValueError, match="PNG and JPEG"):
            await _inline_image("https://cdn.sogni.ai/mislabelled.png")

    local_png_name = tmp_path / "mislabelled.png"
    local_png_name.write_bytes(b"plain text with a .png suffix")
    with pytest.raises(ValueError, match="PNG and JPEG"):
        await _inline_image(str(local_png_name))

    local_jpeg = tmp_path / "photo.bin"
    local_jpeg.write_bytes(JPEG_MINIMAL)
    inlined = await _inline_image(str(local_jpeg))
    assert inlined.startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_inline_image_validates_data_uris_and_local_file_size(tmp_path: Path) -> None:
    valid_jpeg = f"  data:image/jpg;base64,{base64.b64encode(JPEG_MINIMAL).decode('ascii')}  "
    normalized = await normalize_vision_messages(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": valid_jpeg}}]}]
    )
    assert normalized[0]["content"][0]["image_url"]["url"] == valid_jpeg.strip()

    for rejected, pattern in [
        ("data:image/gif;base64,R0lGODlhAQABAAAAACw=", "PNG and JPEG"),
        ("data:text/plain;base64,YWJj", "PNG and JPEG"),
        ("data:image/png,raw-bytes", "remote URLs are not allowed"),
        ("data:image/png;base64,", "remote URLs are not allowed"),
        ("data:image/png;base64,not!base64!", "Invalid base64 payload"),
        ("data:image/png;base64," + ("AAAA" * (3_500_000)), "exceeds 10MB limit"),
    ]:
        with pytest.raises(ValueError, match=pattern):
            await _inline_image(rejected)

    oversized_local = tmp_path / "large.png"
    with oversized_local.open("wb") as handle:
        handle.seek(10 * 1024 * 1024 + 1)
        handle.write(b"\x00")
    with pytest.raises(ValueError, match="exceeds 10MB limit"):
        await _inline_image(str(oversized_local))
