from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from sogni_client import PersonalLoras, SogniError
from sogni_client.errors import ApiError
from sogni_client.events import EventEmitter
from sogni_client.projects import ProjectsApi
from tests.test_projects import FakeClient


def make_library():
    rest = AsyncMock()
    rest.auth = EventEmitter()
    return PersonalLoras(rest), rest


async def test_library_crud_preserves_consent_and_encodes_identifiers():
    library, rest = make_library()
    limits = {"entries": 10, "fileBytes": None, "importsPerDay": 5, "perGeneration": 8}
    rest.get.return_value = {"data": {"loras": [], "models": ["krea"], "limits": limits}}
    assert (await library.list())["limits"]["fileBytes"] is None
    rest.get.assert_awaited_with("/v1/loras/personal")
    rest.post.return_value = {"data": {"id": "personal-one", "status": "queued"}}
    imported = await library.import_lora(
        url="https://example.com/model.safetensors", name="My style",
        model_id="krea", rights_confirmed=True,
    )
    assert imported["status"] == "queued"
    rest.post.assert_awaited_with("/v1/loras/personal", {
        "url": "https://example.com/model.safetensors", "name": "My style",
        "modelId": "krea", "rightsConfirmed": True,
    })
    rest.get.return_value = {"data": {"id": "personal/one?", "status": "ready"}}
    assert (await library.get("personal/one?"))["status"] == "ready"
    rest.get.assert_awaited_with("/v1/loras/personal/personal%2Fone%3F")
    await library.remove("personal/one?")
    rest.delete.assert_awaited_with("/v1/loras/personal/personal%2Fone%3F")


async def test_catalog_is_fresh_and_filters_by_compatible_model():
    library, rest = make_library()
    row = {"loraId": "personal-one", "modelIds": ["krea"]}
    rest.get.return_value = {"data": {"loras": [row]}}
    assert await library.catalog(model_id="krea") == {"loras": [row]}
    assert await library.catalog({"modelId": "qwen"}) == {"loras": []}
    assert rest.get.await_count == 2


@pytest.mark.parametrize("method", ["list", "get", "catalog", "import_lora"])
async def test_account_change_rejects_inflight_private_result(method):
    library, rest = make_library()
    started, finish = asyncio.Event(), asyncio.Event()

    async def delayed(*args, **kwargs):
        started.set()
        await finish.wait()
        return {"data": {"loras": []}}

    rest.get.side_effect = rest.post.side_effect = delayed
    args = ("personal-one",) if method == "get" else ()
    task = asyncio.create_task(getattr(library, method)(*args))
    await started.wait()
    rest.auth.emit("updated", {})
    finish.set()
    with pytest.raises(SogniError, match="The account changed"):
        await task


async def test_private_discovery_never_enters_public_catalog_cache():
    public = {"loraId": "public-one", "modelIds": ["krea"]}
    private = {"loraId": "personal-one", "modelIds": ["qwen"]}
    client = FakeClient([
        {"data": {"loras": [public], "models": ["krea"]}},
        {"data": {"loras": [private]}},
        {"data": {"loras": []}},
        {"data": {"loras": [private]}},
    ])
    projects = ProjectsApi(client)
    assert projects.personal_loras is projects.personalLoras
    merged = await projects.available_loras(include_personal=True)
    assert merged["loras"] == [public, private]
    assert merged["models"] == ["krea", "qwen"]
    assert (await projects.available_loras())["loras"] == [public]
    assert (await projects.available_loras(includePersonal=True))["loras"] == [public]
    assert await projects.get_lora("personal-one") == private
    assert len(client.rest.calls) == 4


async def test_private_errors_surface_without_falling_back_to_public_catalog():
    client = FakeClient([
        {"data": {"loras": [], "models": []}},
        ApiError(403, {"message": "An active subscription is required."}),
    ])
    with pytest.raises(ApiError, match="active subscription"):
        await ProjectsApi(client).available_loras(include_personal=True)
