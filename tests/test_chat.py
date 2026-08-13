from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from sogni_client.auth import ApiKeyAuthManager
from sogni_client.chat import (
    HOSTED_TOOL_NAMES,
    ChatApi,
    ChatStream,
    ChatToolsApi,
    SogniTools,
    assert_chat_run_external_media,
    is_sogni_tool_call,
    normalize_vision_messages,
    parse_tool_call_arguments,
)
from sogni_client.errors import ApiError, ChatJobError
from sogni_client.events import EventEmitter


class FakeRest:
    def __init__(
        self,
        responses: list[Any] | None = None,
        *,
        stream_lines: list[str] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.lines = list(stream_lines or [])
        self.calls: list[dict[str, Any]] = []

    def _response(self) -> Any:
        if not self.responses:
            raise AssertionError("Unexpected REST call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def post(
        self,
        path: str,
        body: Any = None,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        self.calls.append(
            {
                "method": "POST",
                "path": path,
                "body": body,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self._response()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        self.calls.append(
            {
                "method": method,
                "path": path,
                "body": json_body,
                "headers": headers,
            }
        )
        return self._response()

    async def stream_lines(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ):
        self.calls.append(
            {
                "method": "STREAM",
                "path": path,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        for line in self.lines:
            yield line


class FakeSocket(EventEmitter):
    def __init__(self, responses: list[Any] | None = None) -> None:
        super().__init__()
        self.responses = list(responses or [])
        self.sent: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []

    async def send(self, message_type: str, data: Any) -> None:
        self.sent.append({"type": message_type, "data": data})

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.gets.append({"path": path, "params": params})
        if not self.responses:
            raise AssertionError("Unexpected socket GET")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient(EventEmitter):
    def __init__(
        self,
        *,
        rest: FakeRest | None = None,
        socket: FakeSocket | None = None,
        app_source: str | None = "pytest-chat",
    ) -> None:
        super().__init__()
        self.rest = rest or FakeRest()
        self.socket = socket or FakeSocket()
        self.auth = ApiKeyAuthManager()
        self.app_source = app_source


class FakeProjects:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.project = FakeToolProject()

    async def wait_for_models(self, _timeout: float) -> list[dict[str, Any]]:
        return [
            {"id": "image-slow", "media": "image", "workerCount": 1},
            {"id": "image-fast", "media": "image", "workerCount": 8},
        ]

    async def create(self, params: dict[str, Any]) -> FakeToolProject:
        self.created.append(params)
        return self.project


class FakeToolProject(EventEmitter):
    id = "project-1"

    async def wait_for_completion(self, _timeout: float | None = None) -> list[str]:
        self.emit("progress", 75)
        return ["https://cdn.example/result.png"]


def tool_call(
    name: str = "generate_image",
    arguments: str = '{"prompt":"a moonlit library"}',
) -> dict[str, Any]:
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_sogni_tools_expose_all_canonical_names_and_helpers() -> None:
    definitions = SogniTools.all

    assert len(definitions) == len(HOSTED_TOOL_NAMES) == 24
    assert {item["function"]["name"] for item in definitions} == set(HOSTED_TOOL_NAMES)
    assert SogniTools.generateImage["function"]["name"] == "generate_image"
    assert SogniTools.compose_workflow["function"]["name"] == "compose_workflow"
    assert is_sogni_tool_call(tool_call()) is True
    assert is_sogni_tool_call(tool_call("not_sogni")) is False
    assert parse_tool_call_arguments(tool_call(arguments='{"count":2}')) == {"count": 2}
    assert parse_tool_call_arguments(tool_call(arguments="not-json")) == {}
    assert parse_tool_call_arguments(tool_call(arguments="[]")) == {}
    with pytest.raises(AttributeError):
        _ = SogniTools.not_a_tool


def test_direct_hosted_tool_schemas_match_standalone_golden_fingerprints() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "hosted-tool-alias-parity.generated.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    definitions = {item["function"]["name"]: item for item in SogniTools.all}

    for vector in fixture["tools"]:
        definition = definitions[vector["hostedToolName"]]
        parameters = definition["function"].get("parameters", {})
        value = {"name": definition["function"]["name"], "parameters": parameters}
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
        properties = parameters.get("properties", {})

        assert fingerprint == vector["hostedSchemaSha256"], vector["hostedToolName"]
        assert parameters.get("required", []) == vector["hostedRequired"]
        assert list(properties) == vector["hostedPropertyNames"]
        for alias in vector["argumentAliasTargets"] + vector["mediaAliasTargets"]:
            assert alias in properties, f"{vector['hostedToolName']} is missing {alias}"


@pytest.mark.asyncio
async def test_normalize_vision_messages_inlines_local_png_without_mutating_input(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\npytest")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is here?"},
                {"type": "image_url", "image_url": {"url": str(image), "detail": "low"}},
            ],
        }
    ]

    result = await normalize_vision_messages(messages)

    assert result is not messages
    assert result[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert result[0]["content"][1]["image_url"]["detail"] == "low"
    assert messages[0]["content"][1]["image_url"]["url"] == str(image)


@pytest.mark.asyncio
async def test_normalize_vision_messages_enforces_type_and_count_limits(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "image.bmp"
    invalid.write_bytes(b"BMnot-supported")
    with pytest.raises(ValueError, match="PNG and JPEG"):
        await normalize_vision_messages(
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": str(invalid)}}],
                }
            ]
        )

    parts = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}} for _ in range(21)
    ]
    with pytest.raises(ValueError, match="maximum of 20"):
        await normalize_vision_messages([{"role": "user", "content": parts}])


def test_durable_chat_run_media_must_use_external_urls() -> None:
    assert_chat_run_external_media(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example/input.png"},
                        }
                    ],
                }
            ],
            "mediaReferences": [{"url": "http://cdn.example/reference.png"}],
            "mediaContext": {"images": ["https://cdn.example/context.png"]},
        }
    )

    with pytest.raises(ValueError) as raised:
        assert_chat_run_external_media(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA=="},
                            }
                        ],
                    }
                ],
                "media_references": [{"url": "/tmp/input.png", "data_uri": "data:x"}],
                "media_context": {"uploadedVideos": ["file:///tmp/video.mp4"]},
            }
        )

    message = str(raised.value)
    assert "messages[0].content[0].image_url.url" in message
    assert "mediaReferences[0].url" in message
    assert "mediaReferences[0].dataUri" in message
    assert "mediaContext.uploadedVideos[0]" in message


@pytest.mark.asyncio
async def test_chat_stream_accumulates_chunks_tool_calls_and_final_metadata() -> None:
    stream = ChatStream("job-1")
    chunks = [
        {
            "content": "Hello ",
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call-1",
                    "function": {"name": "generate_image", "arguments": '{"prompt":'},
                }
            ],
        },
        {
            "content": "world",
            "finishReason": "tool_calls",
            "tool_calls": [
                {"index": 0, "function": {"arguments": '"moon"}'}},
            ],
        },
    ]
    for chunk in chunks:
        stream._push_chunk(chunk)
    stream._complete(
        123,
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        worker_name="worker-1",
        cost={"costInUSD": "0.01"},
    )

    yielded = [chunk async for chunk in stream]
    result = await stream.wait()

    assert yielded == chunks
    assert stream.content == "Hello world"
    assert stream.worker_name == stream.workerName == "worker-1"
    assert (
        stream.tool_calls
        == stream.toolCalls
        == [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "generate_image", "arguments": '{"prompt":"moon"}'},
            }
        ]
    )
    assert result == stream.final_result == stream.finalResult
    assert result["finishReason"] == "tool_calls"
    assert result["timeTaken"] == 123
    assert result["cost"] == {"costInUSD": "0.01"}


@pytest.mark.asyncio
async def test_failed_chat_stream_consistently_raises_for_waiters_and_iterators() -> None:
    stream = ChatStream("job-error")
    error = ChatJobError("denied", code=4081, job_id="job-error")
    stream._push_chunk({"content": "discarded"})
    stream._fail(error)

    with pytest.raises(ChatJobError) as waited:
        await stream.wait()
    with pytest.raises(ChatJobError) as first:
        await stream.__anext__()
    with pytest.raises(ChatJobError) as second:
        await asyncio.wait_for(stream.__anext__(), timeout=0.1)

    assert waited.value is first.value is second.value is error
    assert stream.tool_calls == []
    assert stream.final_result is None


@pytest.mark.asyncio
async def test_socket_completion_serializes_aliases_and_emits_stream_lifecycle() -> None:
    socket = FakeSocket()
    api = ChatApi(FakeClient(socket=socket), FakeProjects())
    tokens: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    api.on("token", tokens.append)
    api.on("completed", completed.append)
    api.on("jobState", states.append)

    stream = await api.completions.create(
        model="model-1",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        max_tokens=128,
        top_p=0.8,
        safe_content_filter=False,
        billing_mode="subscription",
        think=False,
        sogni_tools="rich",
        response_format={"type": "json_object"},
    )

    assert isinstance(stream, ChatStream)
    request = socket.sent[0]
    job_id = request["data"]["jobID"]
    assert request["type"] == "llmJobRequest"
    assert request["data"] == {
        "jobID": job_id,
        "type": "llm",
        "model": "model-1",
        "messages": [{"role": "user", "content": "Hello"}],
        "appSource": "pytest-chat",
        "max_tokens": 128,
        "top_p": 0.8,
        "stream": True,
        "safeContentFilter": False,
        "billingMode": "subscription",
        "sogni_tools": "creative-tools",
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }

    socket.emit(
        "jobState",
        {"jobID": job_id, "type": "assigned", "workerName": "worker-a"},
    )
    socket.emit("jobTokens", {"jobID": job_id, "content": "Hi"})
    socket.emit(
        "llmJobResult",
        {
            "jobID": job_id,
            "timeTaken": 9,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "workerName": "worker-final",
            "cost": {"costInToken": "0.2"},
        },
    )

    assert [chunk async for chunk in stream] == [tokens[0]]
    assert (await stream.wait())["content"] == "Hi"
    assert stream.worker_name == "worker-final"
    assert completed == [stream.final_result]
    assert states == [
        {
            "jobID": job_id,
            "type": "assigned",
            "workerName": "worker-a",
            "queuePosition": None,
            "modelId": None,
            "estimatedCost": None,
        }
    ]


@pytest.mark.asyncio
async def test_non_streaming_completion_resolves_from_socket_and_preserves_typed_error() -> None:
    socket = FakeSocket()
    api = ChatApi(FakeClient(socket=socket), FakeProjects())

    success_task = asyncio.create_task(
        api.completions.create(model="model-1", messages=[{"role": "user", "content": "Hello"}])
    )
    await asyncio.sleep(0)
    success_id = socket.sent[-1]["data"]["jobID"]
    socket.emit("jobTokens", {"jobID": success_id, "content": "Done"})
    socket.emit("llmJobResult", {"jobID": success_id, "timeTaken": 4})
    assert (await success_task)["content"] == "Done"

    failure_task = asyncio.create_task(
        api.completions.create(model="model-1", messages=[{"role": "user", "content": "Again"}])
    )
    await asyncio.sleep(0)
    failure_id = socket.sent[-1]["data"]["jobID"]
    socket.emit(
        "llmJobError",
        {
            "jobID": failure_id,
            "error": "subscription",
            "error_code": "4081",
            "error_message": "Upgrade required",
            "subscriptionLimit": True,
            "requiredPlans": ["unlimited"],
            "feature": "creative_tools",
        },
    )
    with pytest.raises(ChatJobError) as raised:
        await failure_task

    assert raised.value.code == "4081"
    assert raised.value.error_type == "subscription"
    assert raised.value.job_id == failure_id
    assert raised.value.subscription_limit is True
    assert raised.value.required_plans == ["unlimited"]


@pytest.mark.asyncio
async def test_chat_model_updates_waiting_and_estimation_match_javascript_encoding() -> None:
    quote_response = {
        "quote": {
            "costInUSD": 0.1,
            "costInSogni": 1,
            "costInSpark": 2,
            "costInToken": 2,
            "inputTokens": 10,
            "outputTokens": 8192,
        }
    }
    socket = FakeSocket([quote_response])
    api = ChatApi(FakeClient(socket=socket), FakeProjects())
    wait = asyncio.create_task(api.wait_for_models(timeout=0.5))
    await asyncio.sleep(0)
    socket.emit(
        "swarmLLMModels",
        {
            "legacy": 3,
            "model/α": {
                "workers": 2,
                "maxOutputTokens": {"default": 4096, "thinkingComplexDefault": 8192},
            },
        },
    )

    models = await wait
    assert models["legacy"] == {"workers": 3}
    models["legacy"]["workers"] = 99
    assert api.models["legacy"]["workers"] == 3

    messages = [{"role": "user", "content": "Moon 🌙"}]
    result = await api.estimate_cost(
        model="model/α",
        messages=messages,
        token_type="spark",
        think=True,
        task_profile="reasoning",
    )

    compact_json = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    javascript_length = len(compact_json.encode("utf-16-le")) // 2
    expected_input_tokens = (javascript_length + 3) // 4
    expected_path = "/api/v1/job-llm/estimate/" + "/".join(
        ["spark", quote("model/α", safe=""), str(expected_input_tokens), "8192"]
    )
    assert socket.gets == [{"path": expected_path, "params": None}]
    assert result == {
        "costInUSD": 0.1,
        "costInSogni": 1,
        "costInSpark": 2,
        "costInToken": 2,
        "inputTokens": 10,
        "outputTokens": 8192,
    }


@pytest.mark.asyncio
async def test_hosted_completion_maps_python_names_and_preserves_false_values() -> None:
    response = {"id": "chat-1", "choices": []}
    rest = FakeRest([response])
    api = ChatApi(FakeClient(rest=rest), FakeProjects())

    result = await api.hosted.create(
        model="hosted-model",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=32,
        top_p=0.5,
        token_type="spark",
        billing_mode="subscription",
        sogni_tools="rich",
        safe_content_filter=False,
        task_profile="coding",
        think=False,
        response_format={"type": "json_object"},
    )

    assert result is response
    call = rest.calls[0]
    assert call["path"] == "/v1/chat/completions"
    assert call["timeout"] == 300
    assert call["body"]["app_source"] == "pytest-chat"
    assert call["body"]["max_tokens"] == 32
    assert call["body"]["top_p"] == 0.5
    assert call["body"]["token_type"] == "spark"
    assert call["body"]["billingMode"] == "subscription"
    assert call["body"]["sogni_tools"] == "creative-tools"
    assert call["body"]["safe_content_filter"] is False
    assert call["body"]["task_profile"] == "coding"
    assert call["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert call["body"]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_hosted_completion_converts_recognized_api_errors_only() -> None:
    recognized = ApiError(
        402,
        {
            "error": {
                "message": "Upgrade required",
                "type": "subscription_error",
                "code": "4081",
                "subscription": {
                    "subscriptionLimit": True,
                    "requiredPlans": ["unlimited"],
                    "feature": "hosted_tools",
                },
            }
        },
    )
    generic = ApiError(500, {"message": "upstream unavailable"})
    rest = FakeRest([recognized, generic])
    api = ChatApi(FakeClient(rest=rest), FakeProjects())
    params = {"model": "model", "messages": [{"role": "user", "content": "Hi"}]}

    with pytest.raises(ChatJobError) as raised:
        await api.hosted.create(params)
    assert raised.value.status == 402
    assert raised.value.code == "4081"
    assert raised.value.subscription_limit is True
    assert raised.value.required_plans == ["unlimited"]

    with pytest.raises(ApiError) as reraised:
        await api.hosted.create(params)
    assert reraised.value is generic


@pytest.mark.asyncio
async def test_durable_run_requests_use_canonical_wire_fields_and_encoded_ids() -> None:
    created = {"id": "run / one", "status": "queued"}
    fetched = {**created, "status": "running"}
    canceled = {**created, "status": "cancelled"}
    confirmed = {**created, "status": "running", "confirmed": True}
    rest = FakeRest(
        [
            {"data": {"run": created}},
            {"data": {"run": fetched}},
            {"data": {"run": canceled}},
            {"data": {"run": confirmed}},
        ]
    )
    api = ChatApi(FakeClient(rest=rest), FakeProjects())

    result = await api.runs.create(
        messages=[{"role": "user", "content": "Create it"}],
        media_references=[{"url": "https://cdn.example/input.png"}],
        media_context={"images": ["https://cdn.example/context.png"]},
        max_estimated_capacity_units=12,
        confirm_cost=False,
        session_id="session-1",
        client_message_id="message-1",
        token_type="spark",
        billing_mode="subscription",
        runtime_config={"qualityTier": "fast"},
        idempotency_key="idem-1",
    )
    assert result is created
    assert await api.runs.get("run / one") is fetched
    assert await api.runs.cancel("run / one", "user request") is canceled
    assert (
        await api.runs.confirmCost(
            "run / one",
            tool_call_id="tool-1",
            decision="confirm",
            overrides={"number_of_variations": 2},
            reason="approved",
        )
        is confirmed
    )

    create_call = rest.calls[0]
    assert create_call["path"] == "/v1/chat/runs"
    assert create_call["headers"] == {"Idempotency-Key": "idem-1"}
    assert {key: value for key, value in create_call["body"].items() if value is not None} == {
        "messages": [{"role": "user", "content": "Create it"}],
        "media_references": [{"url": "https://cdn.example/input.png"}],
        "media_context": {"images": ["https://cdn.example/context.png"]},
        "max_estimated_capacity_units": 12,
        "confirm_cost": False,
        "session_id": "session-1",
        "client_message_id": "message-1",
        "token_type": "spark",
        "billing_mode": "subscription",
        "app_source": "pytest-chat",
        "runtime_config": {"qualityTier": "fast"},
    }
    encoded = "run%20%2F%20one"
    assert rest.calls[1]["path"] == f"/v1/chat/runs/{encoded}"
    assert rest.calls[2]["path"] == f"/v1/chat/runs/{encoded}/cancel"
    assert rest.calls[2]["body"] == {"reason": "user request"}
    assert rest.calls[3]["path"] == f"/v1/chat/runs/{encoded}/confirm-cost"
    assert rest.calls[3]["body"] == {
        "tool_call_id": "tool-1",
        "decision": "confirm",
        "overrides": {"number_of_variations": 2},
        "reason": "approved",
    }


@pytest.mark.asyncio
async def test_durable_run_sse_honors_zero_resume_id_and_skips_status_frames() -> None:
    rest = FakeRest(
        stream_lines=[
            "id: 1",
            "event: run_status",
            'data: {"type":"run_running","sequence":1}',
            "",
            "id: 2",
            "event: tool_started",
            'data: {"type":"tool_started","sequence":2}',
            "",
            # No final blank line: EOF processing must apply the same event filter.
            "event: run_status",
            'data: {"type":"run_completed","sequence":3}',
        ]
    )
    api = ChatApi(FakeClient(rest=rest), FakeProjects())

    events = [
        event
        async for event in api.runs.stream_events(
            "run / one",
            last_event_id=0,
        )
    ]

    assert events == [{"type": "tool_started", "sequence": 2}]
    assert rest.calls == [
        {
            "method": "STREAM",
            "path": "/v1/chat/runs/run%20%2F%20one/events/stream",
            "params": None,
            "headers": {"Accept": "text/event-stream", "Last-Event-ID": "0"},
            "timeout": None,
        }
    ]


@pytest.mark.asyncio
async def test_chat_tools_execute_all_routes_progress_with_python_alias() -> None:
    projects = FakeProjects()
    api = ChatToolsApi(projects)  # type: ignore[arg-type]
    updates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    call = tool_call(arguments='{"prompt":"Moon","number_of_variations":2}')

    results = await api.execute_all(
        [call],
        on_tool_progress=lambda current, progress: updates.append((current, progress)),
        token_type="spark",
    )

    assert results == [
        {
            "toolCallId": "call-1",
            "toolName": "generate_image",
            "success": True,
            "resultUrls": ["https://cdn.example/result.png"],
            "content": json.dumps(
                {
                    "success": True,
                    "media_type": "image",
                    "urls": ["https://cdn.example/result.png"],
                    "model": "image-fast",
                    "prompt": "Moon",
                }
            ),
        }
    ]
    assert projects.created == [
        {
            "type": "image",
            "modelId": "image-fast",
            "positivePrompt": "Moon",
            "numberOfMedia": 2,
            "tokenType": "spark",
        }
    ]
    assert [progress["status"] for _, progress in updates] == [
        "creating",
        "queued",
        "processing",
        "completed",
    ]
    assert all(current is call for current, _ in updates)


@pytest.mark.asyncio
async def test_chat_tools_reject_non_sogni_calls_or_route_custom_handler() -> None:
    api = ChatToolsApi(FakeProjects())  # type: ignore[arg-type]
    custom = tool_call("lookup_weather", '{"city":"Paris"}')

    with pytest.raises(ValueError, match="Not a Sogni tool"):
        await api.execute(custom)

    handled = await api.execute_all(
        [custom], on_tool_call=lambda _call: asyncio.sleep(0, result="sunny")
    )
    assert handled == [
        {
            "toolCallId": "call-1",
            "toolName": "lookup_weather",
            "success": True,
            "resultUrls": [],
            "content": "sunny",
        }
    ]


@pytest.mark.asyncio
async def test_chat_wait_for_models_times_out_and_removes_listener() -> None:
    api = ChatApi(FakeClient(), FakeProjects())

    with pytest.raises(TimeoutError, match="Timeout waiting for LLM models"):
        await api.waitForModels(timeout=0.01)

    assert "modelsUpdated" not in api._listeners
