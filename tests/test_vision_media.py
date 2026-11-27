from __future__ import annotations

import base64
import socket
from pathlib import Path
from unittest.mock import patch

import httpcore
import httpx
import pytest

from sogni_client.chat import _inline_image, _is_global_ip, normalize_vision_messages

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
    f"sogni-user-assets-production.{R2_ACCOUNT_HOST}",
    f"sogni-user-assets-staging.{R2_ACCOUNT_HOST}",
    f"GENERATION-OUTPUT-PRODUCTION.{R2_ACCOUNT_HOST.upper()}",
    "complete-images-production.s3-accelerate.amazonaws.com",
    "complete-images-staging.s3-accelerate.amazonaws.com",
    "artist-upload-production.s3-accelerate.amazonaws.com",
    "artist-upload-staging.s3-accelerate.amazonaws.com",
    "preview-images-production.s3-accelerate.amazonaws.com",
    "preview-images-staging.s3-accelerate.amazonaws.com",
    "sogni-user-assets-production.s3-accelerate.amazonaws.com",
    "sogni-user-assets-staging.s3-accelerate.amazonaws.com",
    "complete-images-production.s3-accelerate.dualstack.amazonaws.com",
    "sogni-user-assets-production.s3-accelerate.dualstack.amazonaws.com",
    "complete-images-production.s3.amazonaws.com",
    "complete-images-production.s3.us-east-1.amazonaws.com",
    "preview-images-production.s3.amazonaws.com",
    "preview-images-staging.s3.us-east-1.amazonaws.com",
    "sogni-user-assets-production.s3.amazonaws.com",
    "sogni-user-assets-staging.s3.us-east-1.amazonaws.com",
    "cdn.sogni.ai",
    "media.sogni.ai",
]

REFUSED_URLS = [
    "https://generation-output-production.ffffffffffffffffffffffffffffffff.r2.cloudflarestorage.com/job/image.png",
    "https://generation-input-production.ffffffffffffffffffffffffffffffff.r2.cloudflarestorage.com/job/image.png",
    "https://sogni-user-assets-production.ffffffffffffffffffffffffffffffff.r2.cloudflarestorage.com/job/image.png",
    f"https://other-bucket.{R2_ACCOUNT_HOST}/job/image.png",
    f"https://generation-output-dev.{R2_ACCOUNT_HOST}/job/image.png",
    f"https://sogni-user-assets-dev.{R2_ACCOUNT_HOST}/job/image.png",
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


@pytest.fixture(autouse=True)
def _clear_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)


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


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("93.184.216.34", True),
        ("2606:2800:220:1:248:1893:25c8:1946", True),
        ("127.0.0.1", False),
        ("10.0.0.8", False),
        ("100.64.0.1", False),
        ("169.254.169.254", False),
        ("192.168.1.10", False),
        ("::1", False),
        ("::ffff:127.0.0.1", False),
        ("::7f00:1", False),
        ("64:ff9b::7f00:1", False),
        ("64:ff9b:1::7f00:1", False),
        ("2002:7f00:1::1", False),
        ("fe80::1%lo0", False),
    ],
)
def test_is_global_ip_checks_special_and_transition_ranges(address: str, expected: bool) -> None:
    assert _is_global_ip(address) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("family", "address"),
    [
        (socket.AF_INET, "127.0.0.1"),
        (socket.AF_INET, "10.0.0.8"),
        (socket.AF_INET, "100.64.0.1"),
        (socket.AF_INET, "169.254.169.254"),
        (socket.AF_INET, "192.168.1.10"),
        (socket.AF_INET6, "::1"),
        (socket.AF_INET6, "::ffff:127.0.0.1"),
        (socket.AF_INET6, "::7f00:1"),
        (socket.AF_INET6, "64:ff9b::7f00:1"),
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
async def test_inline_image_connects_to_resolved_ip_once_and_preserves_tls_sni() -> None:
    dns_calls: list[tuple[str, int]] = []
    tcp_targets: list[tuple[str, int]] = []
    tls_hostnames: list[str | None] = []
    host_headers: list[bytes] = []

    class FakeStream(httpcore.AsyncNetworkStream):
        def __init__(self) -> None:
            self._replied = False

        async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
            if not self._replied:
                self._replied = True
                return (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: image/png\r\n"
                    + f"Content-Length: {len(PNG_1X1)}\r\n\r\n".encode("ascii")
                    + PNG_1X1
                )
            return b""

        async def write(self, buffer: bytes, timeout: float | None = None) -> None:
            for line in buffer.split(b"\r\n"):
                if line.lower().startswith(b"host:"):
                    host_headers.append(line)

        async def aclose(self) -> None:
            pass

        async def start_tls(
            self,
            ssl_context,
            server_hostname: str | None = None,
            timeout: float | None = None,
        ) -> httpcore.AsyncNetworkStream:
            assert ssl_context is not None
            tls_hostnames.append(server_hostname)
            return self

        def get_extra_info(self, info: str) -> object:
            return None

    def dual_stack_addrinfo(host: str, port: int, **_kwargs: object) -> list[tuple[object, ...]]:
        dns_calls.append((host, port))
        return [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("2606:2800:220:1:248:1893:25c8:1946", port, 0, 0),
            ),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", port)),
        ]

    async def fake_connect_tcp(
        _self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object = None,
    ) -> httpcore.AsyncNetworkStream:
        tcp_targets.append((host, port))
        if len(tcp_targets) == 1:
            raise OSError("IPv6 route unavailable")
        return FakeStream()

    with (
        patch("sogni_client.chat.socket.getaddrinfo", side_effect=dual_stack_addrinfo),
        patch("httpcore._backends.auto.AutoBackend.connect_tcp", new=fake_connect_tcp),
    ):
        data_uri = await _inline_image("https://cdn.sogni.ai/2026-09-29/job-1/image.png")

    assert data_uri == f"data:image/png;base64,{base64.b64encode(PNG_1X1).decode('ascii')}"
    assert dns_calls == [("cdn.sogni.ai", 443)]
    assert tcp_targets == [
        ("2606:2800:220:1:248:1893:25c8:1946", 443),
        ("93.184.216.34", 443),
    ]
    assert tls_hostnames == ["cdn.sogni.ai"]
    assert host_headers == [b"Host: cdn.sogni.ai"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env", "expects_local_dns"),
    [
        ({"HTTPS_PROXY": "http://proxy.corp.example:8080"}, False),
        ({"https_proxy": "http://proxy.corp.example:8080"}, False),
        ({"ALL_PROXY": "http://proxy.corp.example:8080"}, False),
        (
            {
                "HTTPS_PROXY": "http://proxy.corp.example:8080",
                "NO_PROXY": "other.example",
            },
            False,
        ),
        (
            {
                "HTTPS_PROXY": "http://proxy.corp.example:8080",
                "NO_PROXY": ".sogni.ai",
            },
            True,
        ),
        (
            {
                "HTTPS_PROXY": "http://proxy.corp.example:8080",
                "no_proxy": "*",
            },
            True,
        ),
    ],
)
async def test_inline_image_skips_local_dns_only_when_https_proxy_applies(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    expects_local_dns: bool,
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "image/png"}, content=PNG_1X1)

    with (
        patch("sogni_client.chat.socket.getaddrinfo", side_effect=_public_addrinfo) as resolve,
        patch("sogni_client.chat.httpx.AsyncClient", side_effect=_mock_async_client(handle)),
    ):
        data_uri = await _inline_image("https://cdn.sogni.ai/job/image.png")

    assert data_uri == f"data:image/png;base64,{base64.b64encode(PNG_1X1).decode('ascii')}"
    assert resolve.called is expects_local_dns


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
