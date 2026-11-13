"""Full-client wire recorder for compatibility tests.

Drives a real ``SogniClient`` against a mocked REST API (``httpx.MockTransport``)
and an in-process socket, and records everything the client sends: REST
requests with their headers and bodies, socket handshakes, socket frames and
uploads. The same scenarios run against an earlier release produce the baseline
in ``fixtures/``; see ``test_never_break_5_82.py``.

Regenerate a baseline from a checkout of the earlier release, for example::

    PYTHONPATH=/path/to/sogni-client-python-5.81.0 \\
        python tests/wire_harness.py --write tests/fixtures/wire-baseline-5.81.0.json
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx
from websockets.exceptions import ConnectionClosed

REST = "https://api.wire.test"
SOCKET = "wss://socket.wire.test"
STORAGE = "https://storage.wire.test"
API_KEY = "wire-harness-key"
APP_ID = "wire-harness-app"
WALLET = "0x00000000000000000000000000000000000000a1"
IGNORED_HEADERS = {"user-agent", "accept-encoding", "connection", "host", "accept"}

MODELS = [
    {
        "id": "encryptable-model",
        "name": "Encryptable",
        "SID": 1,
        "tier": "image-tier",
        "media": "image",
        "supportsMediaEncryption": True,
    },
    {"id": "plain-model", "name": "Plain", "SID": 2, "tier": "image-tier", "media": "image"},
]
_RANGE = {"min": 1, "max": 50, "default": 20, "decimals": 0}
TIERS = {
    "image-tier": {
        "steps": _RANGE,
        "guidance": _RANGE,
        "scheduler": {"allowed": ["simple"], "default": "simple"},
        "sampler": {"allowed": ["euler"], "default": "euler"},
    }
}


def pattern(length: int, seed: int = 3) -> bytes:
    """PNG-signed, non-repeating bytes so the client types them as image/png."""

    body = bytes(((i * 31 + seed) & 0xFF) for i in range(length))
    return b"\x89PNG\r\n\x1a\n" + body[8:]


def jwt(exp: float, subject: str = "artist") -> str:
    def part(value: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part({'exp': exp, 'sub': subject})}.sig"


def b64_json(value: Any) -> str:
    return base64.b64encode(json.dumps(value, separators=(",", ":")).encode()).decode()


def from_b64_json(value: str) -> Any:
    return json.loads(base64.b64decode(value))


class FakeConnection:
    def __init__(self, server: FakeSocketServer, index: int, url: str) -> None:
        self.server = server
        self.index = index
        self.url = url
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()

    async def recv(self) -> str:
        message = await self.inbox.get()
        if message is None:
            raise ConnectionClosed(None, None)
        return message

    async def send(self, raw: str) -> None:
        frame = json.loads(raw)
        data = from_b64_json(frame["data"]) if frame.get("data") else None
        self.server.frames.append({"connection": self.index, "type": frame["type"], "data": data})

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self in self.server.open:
            self.server.open.remove(self)


class FakeSocketServer:
    def __init__(self, authenticated: dict[str, Any]) -> None:
        self.authenticated = authenticated
        self.handshakes: list[dict[str, Any]] = []
        self.frames: list[dict[str, Any]] = []
        self.open: list[FakeConnection] = []

    async def connect(self, url: str, **kwargs: Any) -> FakeConnection:
        connection = FakeConnection(self, len(self.handshakes), url)
        self.handshakes.append(
            {
                "query": parse_qsl(urlsplit(url).query, keep_blank_values=True),
                "headers": dict(kwargs.get("additional_headers") or {}),
            }
        )
        self.open.append(connection)
        connection.inbox.put_nowait(
            json.dumps({"type": "authenticated", "data": b64_json(self.authenticated)})
        )
        return connection

    def push(self, message_type: str, data: Any) -> None:
        for connection in list(self.open):
            connection.inbox.put_nowait(json.dumps({"type": message_type, "data": b64_json(data)}))

    def drop(self) -> None:
        for connection in list(self.open):
            self.open.remove(connection)
            connection.inbox.put_nowait(None)


class FakeApi:
    """The REST API, the socket's REST endpoints and storage, recorded."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.media_key: Callable[[int], httpx.Response] = lambda _attempt: httpx.Response(
            404, json={"status": "error", "message": "Not found", "errorCode": 404}
        )
        self.media_key_attempts = 0
        self.results: dict[str, bytes] = {}
        self.login: dict[str, Any] = {}
        self.uploads = 0
        self.serve_ranges = True

    def record(self, request: httpx.Request) -> dict[str, Any]:
        headers = {
            name.lower(): value
            for name, value in request.headers.items()
            if name.lower() not in IGNORED_HEADERS
        }
        body: Any = None
        content = request.content
        if content:
            if "json" in headers.get("content-type", ""):
                body = json.loads(content)
            else:
                body = {"sha256": hashlib.sha256(content).hexdigest(), "length": len(content)}
        entry = {
            "method": request.method,
            "url": f"{request.url.scheme}://{request.url.host}{request.url.path}",
            "query": sorted(parse_qsl(request.url.query.decode(), keep_blank_values=True)),
            "headers": headers,
            "body": body,
            "raw": content,
        }
        self.requests.append(entry)
        return entry

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.record(request)
        url = str(request.url)
        path = request.url.path
        if url.startswith(STORAGE):
            if request.method == "PUT":
                return httpx.Response(200)
            data = self.results.get(path, b"")
            byte_range = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers.get("range", ""))
            if byte_range and self.serve_ranges:
                first = int(byte_range.group(1))
                last = min(int(byte_range.group(2)), len(data) - 1)
                return httpx.Response(
                    206,
                    content=data[first : last + 1],
                    headers={"Content-Range": f"bytes {first}-{last}/{len(data)}"},
                )
            return httpx.Response(200, content=data)
        if path == "/v1/account/media-key":
            self.media_key_attempts += 1
            return self.media_key(self.media_key_attempts)
        if path == "/api/v1/models/list":
            return httpx.Response(200, json=MODELS)
        if path == "/api/v2/models/tiers":
            return httpx.Response(200, json=TIERS)
        if path == "/v1/assets/capabilities":
            return httpx.Response(200, json={"status": "success", "data": {"enabled": False}})
        if path in {"/v1/image/uploadUrl", "/v1/media/uploadUrl"}:
            self.uploads += 1
            return httpx.Response(
                200, json={"status": "success", "data": {"uploadUrl": f"{STORAGE}/upload"}}
            )
        if path in {"/v1/image/downloadUrl", "/v1/media/downloadUrl"}:
            return httpx.Response(
                200, json={"status": "success", "data": {"downloadUrl": f"{STORAGE}/minted"}}
            )
        if path == "/v1/account/me":
            return httpx.Response(
                200,
                json={"status": "success", "data": {"walletAddress": WALLET, "username": "artist"}},
            )
        if path == "/v1/account/nonce":
            return httpx.Response(200, json={"status": "success", "data": {"nonce": "nonce-1"}})
        if path == "/v1/account/login":
            return httpx.Response(200, json={"status": "success", "data": self.login})
        return httpx.Response(200, json={"status": "success", "data": {}})


_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def normalize(value: Any, numbered: Any = None) -> Any:
    """Replace generated ids with placeholders.

    Ids that appear in ``numbered`` (the socket frames) are numbered in order of
    first appearance there, so a request's ``jobId`` still names its project.
    Any other id (a fresh upload id) becomes ``<uuid>``.
    """

    names: dict[str, str] = {}
    for match in _UUID.finditer(json.dumps(numbered, sort_keys=True) if numbered else ""):
        key = match.group(0).upper()
        names.setdefault(key, f"<id{len(names) + 1}>")

    def name(match: re.Match[str]) -> str:
        return names.get(match.group(0).upper(), "<uuid>")

    return json.loads(_UUID.sub(name, json.dumps(value, sort_keys=True)))


async def wait_for(condition: Callable[[], bool], label: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {label}")
        await asyncio.sleep(0.005)


async def settle(rounds: int = 20) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)
    await asyncio.sleep(0.02)


class Session:
    """One client against a fresh fake API and socket."""

    def __init__(self, sogni: Any, api: FakeApi, server: FakeSocketServer) -> None:
        self.sogni = sogni
        self.api = api
        self.server = server

    @classmethod
    async def start(
        cls,
        *,
        options: dict[str, Any] | None = None,
        authenticated: dict[str, Any] | None = None,
        configure: Callable[[FakeApi], None] | None = None,
        api_key: bool = True,
        disable_socket: bool = False,
    ) -> Session:
        from sogni_client import SogniClient

        api = FakeApi()
        if configure is not None:
            configure(api)
        payload = {
            "address": WALLET,
            "username": "artist",
            "mediaEncryptionAvailable": True,
            "hasUnclaimedAirdrop": False,
            "firstLoginAfterMigration": False,
        }
        payload.update(authenticated or {})
        server = FakeSocketServer(payload)
        http = httpx.AsyncClient(transport=httpx.MockTransport(api.handle))
        kwargs: dict[str, Any] = {
            "app_id": APP_ID,
            "rest_endpoint": REST,
            "socket_endpoint": SOCKET,
            "http_client": http,
            "socket_http_client": http,
            "websocket_factory": server.connect,
            "disable_socket": disable_socket,
        }
        if api_key:
            kwargs["api_key"] = API_KEY
        kwargs.update(options or {})
        sogni = await SogniClient.create(**kwargs)
        session = cls(sogni, api, server)
        if api_key and not disable_socket:
            await wait_for(lambda: bool(server.handshakes), "the socket handshake")
            await wait_for(
                lambda: sogni.api_client.socket._authenticated_socket is not None,
                "socket authentication",
            )
        await settle()
        return session

    def key_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.api.requests if r["url"].endswith("/v1/account/media-key")]

    def declared_rest(self) -> list[dict[str, Any]]:
        return [r for r in self.api.requests if "x-sogni-media" in r["headers"]]

    def declarations(self) -> list[dict[str, Any]]:
        return [f for f in self.server.frames if f["type"] == "mediaCapability"]

    def job_requests(self) -> list[dict[str, Any]]:
        return [f["data"] for f in self.server.frames if f["type"] == "jobRequest"]

    def uploads(self) -> list[dict[str, Any]]:
        return [r for r in self.api.requests if r["method"] == "PUT"]

    async def create(self, **params: Any) -> tuple[Any, dict[str, Any]]:
        before = len(self.server.frames)
        values: dict[str, Any] = {
            "type": "image",
            "modelId": "encryptable-model",
            "positivePrompt": "a lighthouse",
            "numberOfMedia": 1,
            "steps": 20,
            "guidance": 7,
            "startingImage": pattern(3000),
        }
        values.update(params)
        project = await self.sogni.projects.create(values)
        await wait_for(
            lambda: any(f["type"] == "jobRequest" for f in self.server.frames[before:]),
            "the job request",
        )
        request = next(f for f in self.server.frames[before:] if f["type"] == "jobRequest")
        return project, request["data"]

    async def result(self, project: Any, url_path: str, data: bytes) -> bytes:
        """Deliver a completed job whose result URL serves ``data``; read it back."""

        self.api.results[url_path] = data
        job_id = f"JOB-{len(self.api.results)}"
        self.server.push(
            "jobState",
            {"jobID": project.id, "imgID": job_id, "type": "jobStarted", "workerName": "w"},
        )
        self.server.push(
            "jobResult",
            {
                "jobID": project.id,
                "imgID": job_id,
                "resultUrl": f"{STORAGE}{url_path}",
                "performedStepCount": 20,
            },
        )
        await wait_for(
            lambda: any(j.id == job_id and j.result_url for j in project.jobs), "the job result"
        )
        job = next(j for j in project.jobs if j.id == job_id)
        return await job.get_result_data()

    def transcript(self) -> dict[str, Any]:
        frames = self.server.frames
        requests = [
            normalize({k: v for k, v in r.items() if k != "raw"}, frames) for r in self.api.requests
        ]
        return {
            "handshakes": normalize(self.server.handshakes, frames),
            "frames": normalize(frames, frames),
            # Background requests (account refreshes, recovery syncs) may
            # interleave differently from run to run; compare them as a set.
            "requests": sorted(json.dumps(r, sort_keys=True) for r in requests),
        }

    async def close(self) -> None:
        await self.sogni.aclose()


# ---------------------------------------------------------------------------
# Scenarios with the new options unset. Every one runs unchanged on 5.81.0.


ENCRYPTED_RESULT = b"SOGNIE2E" + bytes(range(1, 120))


def _recovered_encrypted_project() -> dict[str, Any]:
    request = {
        "numberOfImages": 1,
        "network": "fast",
        "tokenType": "spark",
        "previews": 0,
        "keyFrames": [{"modelID": "encryptable-model", "positivePrompt": "recovered", "seed": 9}],
        "mediaEncryption": {"version": 1, "recipient": "AQID"},
    }
    return {
        "id": "RECOVERED-1",
        "appId": APP_ID,
        "modelType": "image",
        "model": {"id": "encryptable-model", "type": "image"},
        "status": "active",
        "stepCount": 20,
        "mediaEncrypted": True,
        "clientRequestData": b64_json(request),
        "workerJobs": [],
        "completedWorkerJobs": [],
    }


async def scenario_api_key_session() -> dict[str, Any]:
    """An API-key client on a server that offers encryption, with the options unset."""

    session = await Session.start(
        authenticated={"activeProjects": [_recovered_encrypted_project()]}
    )
    outcome: dict[str, Any] = {}
    try:
        project, _ = await session.create()
        outcome["first_result_sha256"] = hashlib.sha256(
            await session.result(project, "/result-1", pattern(500, 9))
        ).hexdigest()
        await session.create(contextImages=[pattern(1200, 4)], mediaEncryption=False)
        await session.create(modelId="plain-model")
        session.server.push("mediaEncryptionAvailabilityUpdated", True)
        await settle()
        await session.create()
        await wait_for(
            lambda: any(p.id == "RECOVERED-1" for p in session.sogni.projects.tracked_projects),
            "the recovered project",
        )
        recovered = next(
            p for p in session.sogni.projects.tracked_projects if p.id == "RECOVERED-1"
        )
        outcome["recovered_params"] = recovered.params
        outcome["recovered_result"] = base64.b64encode(
            await session.result(recovered, "/recovered", ENCRYPTED_RESULT)
        ).decode()
        session.server.drop()
        await wait_for(lambda: len(session.server.handshakes) == 2, "the reconnect", timeout=10)
        await settle()
        await session.create()
        await settle()
        return {"outcome": outcome, "wire": session.transcript()}
    finally:
        await session.close()


async def scenario_token_login() -> dict[str, Any]:
    """A password sign-in for an account without two-factor authentication."""

    exp = 4102444800  # 2100-01-01, so the tokens never need renewing.

    def configure(api: FakeApi) -> None:
        api.login = {
            "token": jwt(exp),
            "refreshToken": jwt(exp + 60),
            "username": "artist",
        }

    session = await Session.start(configure=configure, api_key=False)
    try:
        data = await session.sogni.account.login("artist", "correct horse", True)
        await wait_for(lambda: bool(session.server.handshakes), "the socket handshake")
        await wait_for(
            lambda: session.sogni.api_client.socket._authenticated_socket is not None,
            "socket authentication",
        )
        await settle()
        await session.create()
        return {"outcome": {"login": data}, "wire": session.transcript()}
    finally:
        await session.close()


async def scenario_rest_only() -> dict[str, Any]:
    session = await Session.start(disable_socket=True)
    try:
        await session.sogni.api_client.rest.get("/v1/account/balance")
        return {"outcome": {}, "wire": session.transcript()}
    finally:
        await session.close()


SCENARIOS: dict[str, Callable[[], Awaitable[dict[str, Any]]]] = {
    "api_key_session": scenario_api_key_session,
    "token_login": scenario_token_login,
    "rest_only": scenario_rest_only,
}


async def run_all() -> dict[str, Any]:
    return {name: await scenario() for name, scenario in SCENARIOS.items()}


def main(argv: list[str]) -> int:
    import sogni_client

    if len(argv) != 3 or argv[1] != "--write":
        print("usage: wire_harness.py --write <fixture.json>", file=sys.stderr)
        return 2
    results = asyncio.run(run_all())
    document = {"sogniClientVersion": sogni_client.__version__, "scenarios": results}
    with open(argv[2], "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {argv[2]} from sogni-client {sogni_client.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
