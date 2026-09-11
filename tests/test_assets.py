from __future__ import annotations

import asyncio
import base64
import hashlib
from typing import Any
from unittest.mock import AsyncMock

import pytest

from sogni_client.assets import ReusableUploads
from sogni_client.chat import ChatToolsApi
from sogni_client.errors import ApiError, SogniError
from sogni_client.events import EventEmitter
from sogni_client.projects import (
    ProjectsApi,
    _custom_image_size_bounds,
    _saved_upload_binding,
    create_job_request_message,
)
from tests.test_projects import FakeSocket

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16
MP3 = b"ID3" + b"\0" * 16


class AssetsRest:
    """Scripted REST double for saved uploads (ReusableUploads contract)."""

    def __init__(self, *, enabled: bool = True, prepare: Any = None, put_status: int = 200) -> None:
        self.auth = EventEmitter()
        self.enabled = enabled
        self.prepare = prepare
        self.put_status = put_status
        self.calls: list[tuple[str, str, Any]] = []

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append(("GET", path, params))
        if path == "/v1/assets/capabilities":
            return {"status": "success", "data": {"enabled": self.enabled}}
        if path in {"/v1/image/uploadUrl", "/v1/media/uploadUrl"}:
            return {"data": {"uploadUrl": f"https://upload.example{path}"}}
        raise AssertionError(f"Unexpected GET {path}")

    async def post(self, path: str, body: Any = None, **_: Any) -> Any:
        self.calls.append(("POST", path, body))
        if path == "/v1/assets/prepare":
            if isinstance(self.prepare, Exception):
                raise self.prepare
            prepared = self.prepare or {"id": "asset-1", "state": "ready"}
            return {"status": "success", "data": {"name": body["name"], **prepared}}
        if path.endswith("/finalize"):
            return {"status": "success", "data": {"id": "asset-1", "state": "ready"}}
        if path.endswith("/bind"):
            return {"status": "success", "data": {}}
        raise AssertionError(f"Unexpected POST {path}")

    async def put_signed(self, url: str, data: bytes, *, headers: dict[str, str]) -> int:
        self.calls.append(("PUT_SIGNED", url, headers))
        return self.put_status

    async def put_bytes(self, url: str, data: bytes, *, content_type: str | None = None) -> None:
        self.calls.append(("PUT", url, content_type))

    async def delete(self, path: str) -> Any:
        self.calls.append(("DELETE", path, None))
        return {}


def paths(rest: AssetsRest) -> list[str]:
    return [f"{method} {path}" for method, path, _ in rest.calls]


async def test_reused_saved_upload_binds_without_uploading_bytes_again() -> None:
    rest = AssetsRest(prepare={"id": "asset-1", "state": "ready", "reused": True})
    uploads = ReusableUploads(rest)
    binding = {"projectId": "project-1", "type": "startingImage"}

    assert await uploads.try_bind_file(PNG, "image/png", binding) is True
    prepare = next(body for method, path, body in rest.calls if path == "/v1/assets/prepare")
    assert prepare == {
        "sha256": hashlib.sha256(PNG).hexdigest(),
        "bytes": len(PNG),
        "contentType": "image/png",
        "name": "Saved upload",
    }
    assert ("POST", "/v1/assets/asset-1/bind", binding) in rest.calls
    assert not any(method.startswith("PUT") for method, _, _ in rest.calls)


async def test_new_saved_upload_puts_signed_bytes_then_finalizes_and_binds() -> None:
    headers = {"Content-Type": "image/png", "If-None-Match": "*", "x-amz-checksum-sha256": "abc"}
    rest = AssetsRest(
        prepare={
            "id": "asset-1",
            "state": "uploading",
            "uploadUrl": "https://s3.example/put",
            "uploadHeaders": headers,
        }
    )
    assert await ReusableUploads(rest).try_bind_file(
        PNG, "image/png", {"projectId": "p", "type": "cnImage"}
    )
    assert paths(rest) == [
        "GET /v1/assets/capabilities",
        "POST /v1/assets/prepare",
        "PUT_SIGNED https://s3.example/put",
        "POST /v1/assets/asset-1/finalize",
        "POST /v1/assets/asset-1/bind",
    ]
    assert rest.calls[2][2] == headers


async def test_write_once_conflict_is_verified_by_finalize_not_treated_as_failure() -> None:
    rest = AssetsRest(
        prepare={
            "id": "asset-1",
            "state": "uploading",
            "uploadUrl": "https://s3.example/put",
            "uploadHeaders": {"a": "b"},
        },
        put_status=412,
    )
    assert await ReusableUploads(rest).try_bind_file(
        PNG, "image/png", {"projectId": "p", "type": "cnImage"}
    )


@pytest.mark.parametrize("status", [400, 403, 404, 409, 410, 503])
async def test_unavailable_saved_storage_falls_back_before_any_transfer(status: int) -> None:
    rest = AssetsRest(prepare=ApiError(status, {"message": "unavailable"}))
    assert (
        await ReusableUploads(rest).try_bind_file(
            PNG, "image/png", {"projectId": "p", "type": "cnImage"}
        )
        is False
    )


async def test_transfer_failure_after_preparation_surfaces_instead_of_uploading_elsewhere() -> None:
    rest = AssetsRest(
        prepare={
            "id": "asset-1",
            "state": "uploading",
            "uploadUrl": "https://s3.example/put",
            "uploadHeaders": {"a": "b"},
        },
        put_status=500,
    )
    with pytest.raises(ApiError, match="Could not upload the selected file"):
        await ReusableUploads(rest).try_bind_file(
            PNG, "image/png", {"projectId": "p", "type": "cnImage"}
        )


async def test_disabled_unsupported_or_oversized_inputs_use_ordinary_uploads() -> None:
    disabled = AssetsRest(enabled=False)
    assert (
        await ReusableUploads(disabled).try_bind_file(
            PNG, "image/png", {"projectId": "p", "type": "cnImage"}
        )
        is False
    )
    other = AssetsRest()
    uploads = ReusableUploads(other)
    assert (
        await uploads.try_bind_file(b"GIF89a", "image/gif", {"projectId": "p", "type": "cnImage"})
        is False
    )
    assert (
        await uploads.try_bind_file(
            b"\0" * (100 * 1024 * 1024 + 1), "image/png", {"projectId": "p", "type": "cnImage"}
        )
        is False
    )
    assert other.calls == []
    with pytest.raises(ApiError, match="no larger than 100 MiB"):
        await uploads.upload(b"", "image/png")


async def test_capability_is_cached_and_cleared_when_the_account_changes() -> None:
    rest = AssetsRest(enabled=False)
    uploads = ReusableUploads(rest)
    for _ in range(3):
        await uploads.try_bind_file(PNG, "image/png", {"projectId": "p", "type": "cnImage"})
    assert paths(rest).count("GET /v1/assets/capabilities") == 1
    rest.auth.emit("updated", False)
    await uploads.try_bind_file(PNG, "image/png", {"projectId": "p", "type": "cnImage"})
    assert paths(rest).count("GET /v1/assets/capabilities") == 2


async def test_identical_concurrent_uploads_share_one_preparation() -> None:
    rest = AssetsRest()
    original = rest.post
    gate = asyncio.Event()

    async def slow_post(path: str, body: Any = None, **kwargs: Any) -> Any:
        if path == "/v1/assets/prepare":
            await gate.wait()
        return await original(path, body, **kwargs)

    rest.post = slow_post  # type: ignore[method-assign]
    uploads = ReusableUploads(rest)
    first = asyncio.ensure_future(uploads.upload(PNG, "image/png"))
    second = asyncio.ensure_future(uploads.upload(PNG, "image/png"))
    await asyncio.sleep(0)
    gate.set()
    assert (await first)["id"] == (await second)["id"] == "asset-1"
    assert paths(rest).count("POST /v1/assets/prepare") == 1


async def test_account_change_mid_upload_fails_instead_of_binding_to_the_new_account() -> None:
    rest = AssetsRest()
    uploads = ReusableUploads(rest)
    original = rest.post

    async def switch_accounts(path: str, body: Any = None, **kwargs: Any) -> Any:
        result = await original(path, body, **kwargs)
        if path == "/v1/assets/prepare":
            rest.auth.emit("updated", True)
        return result

    rest.post = switch_accounts  # type: ignore[method-assign]
    with pytest.raises(SogniError, match="The account changed"):
        await uploads.try_bind_file(PNG, "image/png", {"projectId": "p", "type": "cnImage"})
    assert not any(path.endswith("/bind") for _, path, _ in rest.calls)


def test_media_slots_bind_in_the_typescript_client_wire_shape() -> None:
    assert _saved_upload_binding("p", "referenceAudio2") == {
        "projectId": "p",
        "type": "referenceAudio",
        "id": "referenceAudio2",
    }
    assert _saved_upload_binding("p", "referenceVideo") == {
        "projectId": "p",
        "type": "referenceVideo",
    }
    assert _saved_upload_binding("p", "contextImage3") == {
        "projectId": "p",
        "type": "contextImage3",
    }


class FakeClient(EventEmitter):
    def __init__(self, rest: AssetsRest) -> None:
        super().__init__()
        self.rest = rest
        self.socket = FakeSocket()
        self.app_source = "pytest"


async def test_project_inputs_reuse_saved_uploads_instead_of_presigned_uploads() -> None:
    rest = AssetsRest(prepare={"id": "asset-1", "state": "ready"})
    api = ProjectsApi(FakeClient(rest))
    api.get_model_options = AsyncMock(
        return_value={
            "type": "video",
            "sampler": {"allowed": [], "default": None},
            "scheduler": {"allowed": [], "default": None},
        }
    )

    project = await api.create(
        type="video",
        model_id="ltx23-22b-fp8_i2v_distilled",
        positive_prompt="portrait speaking",
        number_of_media=1,
        duration=1,
        reference_image=PNG,
        reference_audio_identity=MP3,
    )

    binds = [body for method, path, body in rest.calls if path.endswith("/bind")]
    assert binds == [
        {"projectId": project.id, "type": "referenceImage"},
        {"projectId": project.id, "type": "referenceAudio"},
    ]
    assert not any(path.endswith("uploadUrl") for _, path, _ in rest.calls)
    keyframe = api.client.socket.sent[-1][1]["keyFrames"][0]
    assert keyframe["referenceImageContentType"] == "image/png"
    assert keyframe["referenceAudioContentType"] == "audio/mpeg"


def gpt_request(**params: Any) -> dict[str, Any]:
    options = {
        "type": "image",
        "sampler": {"allowed": [], "default": None},
        "scheduler": {"allowed": [], "default": None},
    }
    message = create_job_request_message(
        "project-1",
        {"type": "image", "positivePrompt": "a mug", "numberOfMedia": 1, **params},
        options,
    )
    return message["keyFrames"][0]


@pytest.mark.parametrize(
    "model_id", ["gpt-image-2", "gpt-image-2.5-sunburst", "gpt-image-2.5-flare"]
)
def test_gpt_image_models_share_reference_and_size_limits(model_id: str) -> None:
    assert _custom_image_size_bounds(model_id) == (256, 3840)
    with pytest.raises(ApiError, match="up to 16 non-empty references"):
        gpt_request(modelId=model_id, contextImages=[PNG] * 17)


def test_gpt_image_quality_background_and_compression_follow_the_model() -> None:
    with pytest.raises(ApiError, match=r"auto\. Choose low, medium or high, xhigh or max\."):
        gpt_request(modelId="gpt-image-2.5-flare", gptImageQuality="auto")
    with pytest.raises(ApiError, match=r"auto\. Choose low, medium or high\.$"):
        gpt_request(modelId="gpt-image-2", gptImageQuality="auto")
    with pytest.raises(ApiError, match="Unsupported quality for gpt-image-2: xhigh"):
        gpt_request(modelId="gpt-image-2", gptImageQuality="xhigh")
    assert (
        gpt_request(modelId="gpt-image-2.5-sunburst", gptImageQuality="max")["gptImageQuality"]
        == "max"
    )
    with pytest.raises(ApiError, match="Unsupported background for gpt-image-2: transparent"):
        gpt_request(modelId="gpt-image-2", gptImageBackground="transparent")
    with pytest.raises(ApiError, match="requires PNG or WebP"):
        gpt_request(
            modelId="gpt-image-2.5-flare", gptImageBackground="transparent", outputFormat="jpg"
        )
    with pytest.raises(ApiError, match="requires JPEG or WebP"):
        gpt_request(modelId="gpt-image-2.5-flare", gptImageOutputCompression=80, outputFormat="png")
    with pytest.raises(ApiError, match="integer from 0 to 100"):
        gpt_request(
            modelId="gpt-image-2.5-flare", gptImageOutputCompression=101, outputFormat="jpg"
        )
    keyframe = gpt_request(
        modelId="gpt-image-2.5-flare", gptImageOutputCompression=80, outputFormat="webp"
    )
    assert keyframe["gptImageOutputCompression"] == 80


def test_gpt_image_masks_need_a_gpt_model_and_a_first_reference() -> None:
    with pytest.raises(ApiError, match="GPT Image masks require a GPT Image model"):
        gpt_request(modelId="flux1-schnell-fp8", gptImageMaskUrl="https://example.test/mask.png")
    with pytest.raises(ApiError, match="mask requires a mask URL and a first reference image"):
        gpt_request(modelId="gpt-image-2.5-flare", gptImageMaskUrl="https://example.test/mask.png")
    with pytest.raises(ApiError, match="Provide one GPT Image mask"):
        gpt_request(
            modelId="gpt-image-2.5-flare",
            contextImages=[PNG],
            gptImageMask=PNG,
            gptImageMaskUrl="https://x/m.png",
        )
    keyframe = gpt_request(modelId="gpt-image-2.5-flare", contextImages=[PNG], gptImageMask=PNG)
    assert keyframe["hasReferenceMask"] is True
    assert keyframe["referenceMaskContentType"] == "image/png"


async def test_gpt_image_mask_data_uri_uploads_in_the_reference_mask_slot() -> None:
    rest = AssetsRest(enabled=False)
    api = ProjectsApi(FakeClient(rest))
    api.get_model_options = AsyncMock(
        return_value={
            "type": "image",
            "sampler": {"allowed": [], "default": None},
            "scheduler": {"allowed": [], "default": None},
        }
    )
    await api.create(
        type="image",
        model_id="gpt-image-2.5-flare",
        positive_prompt="edit the cup",
        number_of_media=1,
        context_images=[PNG],
        gpt_image_mask_url="data:image/png;base64," + base64.b64encode(PNG).decode(),
    )
    uploads = [
        params["type"] for method, path, params in rest.calls if path == "/v1/image/uploadUrl"
    ]
    assert uploads == ["contextImage1", "referenceMask"]
    keyframe = api.client.socket.sent[-1][1]["keyFrames"][0]
    assert keyframe["hasReferenceMask"] is True
    assert "gptImageMaskUrl" not in keyframe


def test_chat_selectors_route_gpt_image_25_names() -> None:
    selectors = ChatToolsApi._IMAGE_SELECTORS
    assert selectors["gpt-image-2.5"] == selectors["flare"] == "gpt-image-2.5-flare"
    assert selectors["sunburst"] == "gpt-image-2.5-sunburst"
    assert ChatToolsApi._EDIT_SELECTORS["gpt-image-2.5-sunburst"] == "gpt-image-2.5-sunburst"
