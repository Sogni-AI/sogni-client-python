"""Socket chat, hosted chat, durable chat runs, and tool helpers."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from collections.abc import AsyncIterator
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .errors import ApiError, ChatJobError, extract_chat_job_error_fields
from .events import EventEmitter
from .projects import ProjectsApi
from .transport import ApiClient
from .utils import new_id, normalize_params, parse_sse_chunk

HOSTED_TOOL_NAMES = (
    "generate_image",
    "generate_video",
    "generate_music",
    "edit_image",
    "apply_style",
    "restore_photo",
    "refine_result",
    "animate_photo",
    "change_angle",
    "video_to_video",
    "stitch_video",
    "orbit_video",
    "dance_montage",
    "sound_to_video",
    "extend_video",
    "replace_video_segment",
    "overlay_video",
    "add_subtitles",
    "enhance_prompt",
    "compose_lyrics",
    "compose_instrumental",
    "compose_script",
    "compose_workflow",
    "compose_workflow_template",
)


def _load_hosted_tools() -> dict[str, dict[str, Any]]:
    resource = files("sogni_client").joinpath("data/hosted_tools.json")
    with resource.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    tools = manifest.get("tools") if isinstance(manifest, dict) else None
    if not isinstance(tools, list):  # pragma: no cover - packaging integrity guard
        raise RuntimeError("Bundled hosted-tools manifest is invalid")
    definitions = {
        tool.get("function", {}).get("name"): tool
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    if set(definitions) != set(HOSTED_TOOL_NAMES):  # pragma: no cover - integrity guard
        raise RuntimeError("Bundled hosted-tools manifest does not match the SDK surface")
    return definitions


_HOSTED_TOOLS = _load_hosted_tools()


def _tool_definition(name: str) -> dict[str, Any]:
    return deepcopy(_HOSTED_TOOLS[name])


class _SogniTools:
    _aliases = {
        "generateImage": "generate_image",
        "editImage": "edit_image",
        "generateVideo": "generate_video",
        "soundToVideo": "sound_to_video",
        "videoToVideo": "video_to_video",
        "generateMusic": "generate_music",
        "applyStyle": "apply_style",
        "restorePhoto": "restore_photo",
        "refineResult": "refine_result",
        "changeAngle": "change_angle",
        "animatePhoto": "animate_photo",
        "stitchVideo": "stitch_video",
        "orbitVideo": "orbit_video",
        "danceMontage": "dance_montage",
        "extendVideo": "extend_video",
        "replaceVideoSegment": "replace_video_segment",
        "overlayVideo": "overlay_video",
        "addSubtitles": "add_subtitles",
        "enhancePrompt": "enhance_prompt",
        "composeLyrics": "compose_lyrics",
        "composeInstrumental": "compose_instrumental",
        "composeScript": "compose_script",
        "composeWorkflow": "compose_workflow",
        "composeWorkflowTemplate": "compose_workflow_template",
    }

    def __getattr__(self, name: str) -> dict[str, Any]:
        resolved = self._aliases.get(name, name)
        if resolved not in HOSTED_TOOL_NAMES:
            raise AttributeError(name)
        return _tool_definition(resolved)

    @property
    def all(self) -> list[dict[str, Any]]:
        return [_tool_definition(name) for name in HOSTED_TOOL_NAMES]


SogniTools = _SogniTools()


def is_sogni_tool_call(tool_call: dict[str, Any]) -> bool:
    return tool_call.get("function", {}).get("name") in HOSTED_TOOL_NAMES


isSogniToolCall = is_sogni_tool_call


def parse_tool_call_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


parseToolCallArguments = parse_tool_call_arguments


async def _inline_image(value: str) -> str:
    if value.startswith("data:"):
        return value
    if value.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(value)
            response.raise_for_status()
            raw = response.content
            mime = response.headers.get("content-type", "").split(";", 1)[0]
    else:
        path = await asyncio.to_thread(Path(value).expanduser)
        raw = await asyncio.to_thread(path.read_bytes)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError("image input exceeds 10MB limit")
    if mime in {"image/jpg", "image/jpeg"} or raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif mime == "image/png" or raw.startswith(b"\x89PNG"):
        mime = "image/png"
    else:
        raise ValueError("Vision chat supports PNG and JPEG images only")
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


async def normalize_vision_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    image_count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            result.append(dict(message))
            continue
        normalized_parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                normalized_parts.append(part)
                continue
            image_count += 1
            if image_count > 20:
                raise ValueError("A maximum of 20 vision images is allowed per request")
            image = part.get("image_url", {})
            normalized_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": await _inline_image(str(image.get("url", ""))),
                        **({"detail": image["detail"]} if image.get("detail") else {}),
                    },
                }
            )
        result.append({**message, "content": normalized_parts})
    return result


def _assert_external_url(value: Any, path: str, violations: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        return
    if not value.strip().lower().startswith(("http://", "https://")):
        violations.append(path)


def assert_chat_run_external_media(params: dict[str, Any]) -> None:
    violations: list[str] = []
    for message_index, message in enumerate(params.get("messages", [])):
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for part_index, part in enumerate(message["content"]):
            if isinstance(part, dict) and part.get("type") == "image_url":
                _assert_external_url(
                    part.get("image_url", {}).get("url"),
                    f"messages[{message_index}].content[{part_index}].image_url.url",
                    violations,
                )
    references = params.get("mediaReferences") or params.get("media_references") or []
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            continue
        _assert_external_url(reference.get("url"), f"mediaReferences[{index}].url", violations)
        if reference.get("dataUri") or reference.get("data_uri"):
            violations.append(f"mediaReferences[{index}].dataUri")
    context = params.get("mediaContext") or params.get("media_context") or {}
    for field in (
        "images",
        "videos",
        "audio",
        "uploadedImages",
        "uploadedVideos",
        "uploadedAudio",
    ):
        for index, value in enumerate(context.get(field, []) if isinstance(context, dict) else []):
            _assert_external_url(value, f"mediaContext.{field}[{index}]", violations)
    if violations:
        raise ValueError(
            "Durable chat runs do not support inline base64/data URI media. Upload media first "
            "and pass HTTP(S) URLs instead. Offending field(s): " + ", ".join(violations)
        )


class ChatStream:
    """Async iterable of socket-native completion chunks."""

    _END = object()

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.jobID = job_id
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._done = False
        self._error: Exception | None = None
        self._content = ""
        self._role = "assistant"
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] | None = None
        self._time_taken = 0
        self._worker_name: str | None = None
        self._cost: dict[str, Any] | None = None
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._finished: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()

    @property
    def content(self) -> str:
        return self._content

    @property
    def worker_name(self) -> str | None:
        return self._worker_name

    workerName = worker_name

    @property
    def cost(self) -> dict[str, Any] | None:
        return self._cost

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return [self._tool_calls[index] for index in sorted(self._tool_calls)]

    toolCalls = tool_calls

    @property
    def final_result(self) -> dict[str, Any] | None:
        if not self._done or self._error:
            return None
        result: dict[str, Any] = {
            "jobID": self.job_id,
            "content": self._content,
            "role": self._role,
            "finishReason": self._finish_reason or "stop",
            "usage": self._usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "timeTaken": self._time_taken,
            "workerName": self._worker_name,
            "cost": self._cost,
        }
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result

    finalResult = final_result

    async def wait(self, timeout: float | None = None) -> dict[str, Any]:
        waiter = asyncio.shield(self._finished)
        return await asyncio.wait_for(waiter, timeout) if timeout is not None else await waiter

    def _push_chunk(self, chunk: dict[str, Any]) -> None:
        if self._done:
            return
        self._content += chunk.get("content") or ""
        self._role = chunk.get("role") or self._role
        self._finish_reason = chunk.get("finishReason") or self._finish_reason
        self._usage = chunk.get("usage") or self._usage
        for delta in chunk.get("tool_calls") or []:
            index = int(delta.get("index", 0))
            current = self._tool_calls.setdefault(
                index,
                {
                    "id": delta.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": delta.get("function", {}).get("name", ""),
                        "arguments": "",
                    },
                },
            )
            if delta.get("id"):
                current["id"] = delta["id"]
            function = delta.get("function") or {}
            if function.get("name"):
                current["function"]["name"] = function["name"]
            if function.get("arguments"):
                current["function"]["arguments"] += function["arguments"]
        self._queue.put_nowait(chunk)

    def _complete(
        self,
        time_taken: int | float = 0,
        usage: dict[str, Any] | None = None,
        *,
        worker_name: str | None = None,
        cost: dict[str, Any] | None = None,
    ) -> None:
        if self._done:
            return
        self._done = True
        self._time_taken = time_taken
        self._usage = usage or self._usage
        self._worker_name = worker_name or self._worker_name
        self._cost = cost or self._cost
        result = self.final_result or {}
        if not self._finished.done():
            self._finished.set_result(result)
        self._queue.put_nowait(self._END)

    def _fail(self, error: Exception) -> None:
        if self._done:
            return
        self._done = True
        self._error = error
        self._tool_calls.clear()
        if not self._finished.done():
            self._finished.set_exception(error)
        self._queue.put_nowait(error)

    def __aiter__(self) -> ChatStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._error is not None:
            raise self._error
        value = await self._queue.get()
        if value is self._END:
            raise StopAsyncIteration
        if isinstance(value, Exception):
            raise value
        return value


class ChatToolsApi:
    """Executes the six tools that directly map to Supernet projects."""

    DIRECT = {
        "generate_image",
        "edit_image",
        "generate_video",
        "sound_to_video",
        "video_to_video",
        "generate_music",
    }

    def __init__(self, projects: ProjectsApi) -> None:
        self.projects = projects

    async def execute(
        self, tool_call: dict[str, Any], options: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        options = normalize_params(options, **kwargs)
        name = tool_call.get("function", {}).get("name", "")
        if not is_sogni_tool_call(tool_call):
            raise ValueError(f"Not a Sogni tool call: {name}. Use is_sogni_tool_call() first.")
        if name not in self.DIRECT:
            return self._error(
                tool_call,
                f"Tool '{name}' must be executed via chat.hosted.create() or chat.runs.create().",
            )
        args = parse_tool_call_arguments(tool_call)
        try:
            models = await self.projects.wait_for_models(10)
            media = (
                "audio"
                if name == "generate_music"
                else "video"
                if name in {"generate_video", "sound_to_video", "video_to_video"}
                else "image"
            )
            requested = args.get("model") or args.get("video_model")
            candidates = [model for model in models if model.get("media", "image") == media]
            model_id = (
                requested
                if requested and any(model.get("id") == requested for model in candidates)
                else max(candidates, key=lambda model: model.get("workerCount", 0))["id"]
            )
            project_params: dict[str, Any] = {
                "type": media,
                "modelId": model_id,
                "positivePrompt": args.get("prompt", ""),
                "numberOfMedia": max(
                    1,
                    min(
                        16, round(args.get("number_of_variations", options.get("numberOfMedia", 1)))
                    ),
                ),
            }
            if args.get("negative_prompt"):
                project_params["negativePrompt"] = args["negative_prompt"]
            for source, target in (
                ("duration", "duration"),
                ("width", "width"),
                ("height", "height"),
                ("seed", "seed"),
                ("bpm", "bpm"),
                ("lyrics", "lyrics"),
                ("keyscale", "keyscale"),
                ("output_format", "outputFormat"),
            ):
                if args.get(source) is not None:
                    project_params[target] = args[source]
            if media == "video":
                project_params.setdefault("fps", 24)
                project_params.setdefault("duration", 5)
                project_params.setdefault("width", 768)
                project_params.setdefault("height", 512)
            if options.get("tokenType"):
                project_params["tokenType"] = options["tokenType"]
            if options.get("network"):
                project_params["network"] = options["network"]
            on_progress = options.get("onProgress")
            if callable(on_progress):
                on_progress({"status": "creating", "percent": 0})
            project = await self.projects.create(project_params)
            if callable(on_progress):
                project.on(
                    "progress",
                    lambda percent: on_progress({"status": "processing", "percent": percent}),
                )
                on_progress({"status": "queued", "percent": 0})
            timeout = float(options.get("timeout", 30 * 60))
            urls = await project.wait_for_completion(timeout)
            if callable(on_progress):
                on_progress({"status": "completed", "percent": 100, "resultUrls": urls})
            content = json.dumps(
                {
                    "success": True,
                    "media_type": media,
                    "urls": urls,
                    "model": model_id,
                    "prompt": args.get("prompt", ""),
                }
            )
            return {
                "toolCallId": tool_call.get("id"),
                "toolName": name,
                "success": True,
                "resultUrls": urls,
                "content": content,
            }
        except Exception as error:
            return self._error(tool_call, str(error))

    async def execute_all(
        self,
        tool_calls: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        options = {**(options or {}), **kwargs}
        if sum(call.get("function", {}).get("name") in self.DIRECT for call in tool_calls) > 8:
            raise ValueError("Too many Sogni tool calls in a single round; maximum is 8")
        results = []
        for tool_call in tool_calls:
            if is_sogni_tool_call(tool_call):
                execute_options = dict(options)
                on_tool_progress = options.get("on_tool_progress") or options.get("onToolProgress")
                if callable(on_tool_progress):
                    execute_options["onProgress"] = (
                        lambda progress, current=tool_call, callback=on_tool_progress: callback(
                            current, progress
                        )
                    )
                results.append(await self.execute(tool_call, execute_options))
            elif callable(options.get("on_tool_call") or options.get("onToolCall")):
                handler = options.get("on_tool_call") or options.get("onToolCall")
                try:
                    content = await handler(tool_call)
                    results.append(
                        {
                            "toolCallId": tool_call.get("id"),
                            "toolName": tool_call.get("function", {}).get("name"),
                            "success": True,
                            "resultUrls": [],
                            "content": content,
                        }
                    )
                except Exception as error:
                    results.append(self._error(tool_call, str(error)))
            else:
                results.append(self._error(tool_call, "No handler for non-Sogni tool call"))
        return results

    executeAll = execute_all

    @staticmethod
    def _error(tool_call: dict[str, Any], error: str) -> dict[str, Any]:
        return {
            "toolCallId": tool_call.get("id"),
            "toolName": tool_call.get("function", {}).get("name"),
            "success": False,
            "resultUrls": [],
            "content": json.dumps({"success": False, "error": error}),
            "error": error,
        }


class _Completions:
    def __init__(self, api: ChatApi) -> None:
        self._api = api

    async def create(self, params: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return await self._api.create_completion(normalize_params(params, **kwargs))


class _Hosted:
    def __init__(self, api: ChatApi) -> None:
        self._api = api

    async def create(self, params: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._api.create_hosted_completion(normalize_params(params, **kwargs))

    async def execute_tool(
        self, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return await self._api.execute_hosted_tool(normalize_params(params, **kwargs))

    executeTool = execute_tool


class _Runs:
    def __init__(self, api: ChatApi) -> None:
        self._api = api

    async def create(self, params: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._api.create_chat_run(normalize_params(params, **kwargs))

    async def get(self, run_id: str) -> dict[str, Any]:
        return await self._api.get_chat_run(run_id)

    async def cancel(self, run_id: str, reason: str | None = None) -> dict[str, Any]:
        return await self._api.cancel_chat_run(run_id, reason)

    async def confirm_cost(
        self, run_id: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return await self._api.confirm_chat_run_cost(run_id, normalize_params(params, **kwargs))

    confirmCost = confirm_cost

    def stream_events(
        self,
        run_id: str,
        *,
        last_event_id: int | str | None = None,
        lastEventId: int | str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        resume_id = last_event_id if last_event_id is not None else lastEventId
        return self._api.stream_chat_run_events(run_id, resume_id)

    streamEvents = stream_events


class ChatApi(EventEmitter):
    def __init__(self, client: ApiClient, projects: ProjectsApi) -> None:
        super().__init__()
        self.client = client
        self.projects = projects
        self.tools = ChatToolsApi(projects)
        self.completions = _Completions(self)
        self.hosted = _Hosted(self)
        self.runs = _Runs(self)
        self._active_streams: dict[str, ChatStream] = {}
        self._models: dict[str, dict[str, Any]] = {}
        socket = client.socket
        socket.on("jobTokens", self._handle_tokens)
        socket.on("llmJobResult", self._handle_result)
        socket.on("llmJobError", self._handle_error)
        socket.on("jobState", self._handle_state)
        socket.on("swarmLLMModels", self._handle_models)

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._models.items()}

    async def wait_for_models(self, timeout: float = 10) -> dict[str, dict[str, Any]]:
        if self._models:
            return self.models
        future: asyncio.Future[dict[str, dict[str, Any]]] = (
            asyncio.get_running_loop().create_future()
        )

        def ready(models: dict[str, dict[str, Any]]) -> None:
            if models and not future.done():
                future.set_result(models)

        remove = self.on("modelsUpdated", ready)
        try:
            return await asyncio.wait_for(future, timeout)
        except TimeoutError as error:
            raise TimeoutError("Timeout waiting for LLM models") from error
        finally:
            remove()

    waitForModels = wait_for_models

    async def estimate_cost(
        self, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        data = normalize_params(params, **kwargs)
        messages = await normalize_vision_messages(data["messages"])
        stripped = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                content = [
                    {"type": "image_url", "image_url": {"url": "[image]"}}
                    if isinstance(part, dict) and part.get("type") == "image_url"
                    else part
                    for part in content
                ]
            stripped.append({**message, "content": content})
        serialized = json.dumps(stripped, separators=(",", ":"), ensure_ascii=False)
        # JavaScript's string length counts UTF-16 code units. Counting the
        # encoded units keeps the server estimate identical for non-BMP text.
        javascript_length = len(serialized.encode("utf-16-le")) // 2
        input_tokens = (javascript_length + 3) // 4
        max_output = data.get("maxTokens") or data.get("max_tokens")
        if max_output is None:
            info = self._models.get(data["model"], {})
            defaults = info.get("maxOutputTokens", {})
            complex_thinking = data.get("think") is True and data.get("taskProfile") in {
                "coding",
                "reasoning",
            }
            max_output = defaults.get("thinkingComplexDefault") if complex_thinking else None
            max_output = max_output or defaults.get("default") or 4096
        path = "/".join(
            quote(str(item), safe="")
            for item in (
                data.get("tokenType", "sogni"),
                data["model"],
                input_tokens,
                max_output,
            )
        )
        response = await self.client.socket.get(f"/api/v1/job-llm/estimate/{path}")
        quote_data = response["quote"]
        return {
            "costInUSD": quote_data["costInUSD"],
            "costInSogni": quote_data["costInSogni"],
            "costInSpark": quote_data["costInSpark"],
            "costInToken": quote_data["costInToken"],
            "inputTokens": quote_data["inputTokens"],
            "outputTokens": quote_data["outputTokens"],
        }

    estimateCost = estimate_cost

    async def create_completion(self, params: dict[str, Any]) -> ChatStream | dict[str, Any]:
        if params.get("autoExecuteTools"):
            if params.get("stream"):
                raise ValueError("autoExecuteTools is not supported with stream=True")
            return await self._completion_with_tools(params)
        job_id = new_id()
        messages = await normalize_vision_messages(params["messages"])
        request: dict[str, Any] = {
            "jobID": job_id,
            "type": "llm",
            "model": params["model"],
            "messages": messages,
            "appSource": params.get("appSource") or self.client.app_source,
            "max_tokens": params.get("maxTokens", params.get("max_tokens")),
            "temperature": params.get("temperature"),
            "top_p": params.get("topP", params.get("top_p")),
            "top_k": params.get("topK", params.get("top_k")),
            "min_p": params.get("minP", params.get("min_p")),
            "stream": params.get("stream"),
            "repetition_penalty": params.get("repetitionPenalty", params.get("repetition_penalty")),
            "frequency_penalty": params.get("frequencyPenalty", params.get("frequency_penalty")),
            "presence_penalty": params.get("presencePenalty", params.get("presence_penalty")),
            "stop": params.get("stop"),
            "tokenType": params.get("tokenType"),
            "tools": params.get("tools"),
            "tool_choice": params.get("toolChoice", params.get("tool_choice")),
            "sogni_tools": "creative-tools"
            if str(params.get("sogniTools", params.get("sogni_tools", ""))).lower() == "rich"
            else params.get("sogniTools", params.get("sogni_tools")),
            "sogni_tool_execution": params.get(
                "sogniToolExecution", params.get("sogni_tool_execution")
            ),
            "taskProfile": params.get("taskProfile"),
            "response_format": params.get("responseFormat", params.get("response_format")),
        }
        if params.get("safeContentFilter") is not None:
            request["safeContentFilter"] = params["safeContentFilter"]
        if params.get("billingMode"):
            request["billingMode"] = params["billingMode"]
        if "think" in params:
            request["chat_template_kwargs"] = {"enable_thinking": bool(params["think"])}
        request = {key: value for key, value in request.items() if value is not None}
        stream = ChatStream(job_id)
        self._active_streams[job_id] = stream
        await self.client.socket.send("llmJobRequest", request)
        if params.get("stream"):
            return stream
        try:
            return await stream.wait(300)
        except TimeoutError as error:
            self._active_streams.pop(job_id, None)
            raise TimeoutError(f"Chat completion timed out after 300s (jobID: {job_id})") from error

    async def _completion_with_tools(self, params: dict[str, Any]) -> dict[str, Any]:
        messages = list(params["messages"])
        history = []
        for round_index in range(int(params.get("maxToolRounds", 5))):
            result = await self.create_completion(
                {**params, "messages": messages, "stream": False, "autoExecuteTools": False}
            )
            if result.get("finishReason") != "tool_calls" or not result.get("tool_calls"):
                if history:
                    result["toolHistory"] = history
                return result
            results = await self.tools.execute_all(
                result["tool_calls"],
                tokenType=params.get("tokenType"),
                onToolCall=params.get("onToolCall"),
                onToolProgress=params.get("onToolProgress"),
            )
            history.append(
                {"round": round_index, "toolCalls": result["tool_calls"], "toolResults": results}
            )
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": result.get("content"),
                        "tool_calls": result["tool_calls"],
                    },
                    *[
                        {
                            "role": "tool",
                            "content": tool_result["content"],
                            "tool_call_id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                        }
                        for tool_call, tool_result in zip(
                            result["tool_calls"], results, strict=True
                        )
                    ],
                ]
            )
        raise RuntimeError(f"Max tool calling rounds ({params.get('maxToolRounds', 5)}) exceeded")

    async def create_hosted_completion(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("stream"):
            raise ValueError("chat.hosted.create currently supports non-streaming requests only.")
        messages = await normalize_vision_messages(params["messages"])
        template_kwargs = params.get("chatTemplateKwargs", params.get("chat_template_kwargs"))
        if template_kwargs is None and "think" in params:
            template_kwargs = {"enable_thinking": bool(params["think"])}
        body = {
            "model": params["model"],
            "messages": messages,
            "app_source": params.get("appSource", params.get("app_source"))
            or self.client.app_source,
            "max_tokens": params.get("maxTokens", params.get("max_tokens")),
            "temperature": params.get("temperature"),
            "top_p": params.get("topP", params.get("top_p")),
            "top_k": params.get("topK", params.get("top_k")),
            "min_p": params.get("minP", params.get("min_p")),
            "repetition_penalty": params.get("repetitionPenalty", params.get("repetition_penalty")),
            "frequency_penalty": params.get("frequencyPenalty", params.get("frequency_penalty")),
            "presence_penalty": params.get("presencePenalty", params.get("presence_penalty")),
            "stop": params.get("stop"),
            "token_type": params.get("tokenType", params.get("token_type")),
            "billingMode": params.get("billingMode"),
            "tools": params.get("tools"),
            "tool_choice": params.get("toolChoice", params.get("tool_choice")),
            "sogni_tools": params.get("sogniTools", params.get("sogni_tools")),
            "sogni_tool_execution": params.get(
                "sogniToolExecution", params.get("sogni_tool_execution")
            ),
            "task_profile": params.get("taskProfile", params.get("task_profile")),
            "media_references": params.get("mediaReferences", params.get("media_references")),
            "api_media_references": params.get(
                "apiMediaReferences", params.get("api_media_references")
            ),
            "safe_content_filter": params.get(
                "safeContentFilter", params.get("safe_content_filter")
            ),
            "chat_template_kwargs": template_kwargs,
            "response_format": params.get("responseFormat", params.get("response_format")),
        }
        if str(body.get("sogni_tools", "")).lower() == "rich":
            body["sogni_tools"] = "creative-tools"
        try:
            return await self.client.rest.post("/v1/chat/completions", body, timeout=300)
        except ApiError as error:
            extracted = extract_chat_job_error_fields(error.payload)
            if extracted:
                raise ChatJobError(
                    str(error),
                    status=error.status,
                    payload=error.payload,
                    **{key: value for key, value in extracted.items() if key != "message"},
                ) from error
            raise

    async def execute_hosted_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self.client.rest.post(
            "/v1/creative-agent/tools/execute",
            {
                "tool": params["tool"],
                "arguments": params.get("arguments", {}),
                "app_source": params.get("appSource", params.get("app_source"))
                or self.client.app_source,
                "token_type": params.get("tokenType", params.get("token_type")),
                "safe_content_filter": params.get(
                    "safeContentFilter", params.get("safe_content_filter")
                ),
            },
            timeout=300,
        )

    async def _run_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        try:
            return await self.client.rest.request(method, path, json_body=body, headers=headers)
        except ApiError as error:
            extracted = extract_chat_job_error_fields(error.payload)
            if extracted:
                raise ChatJobError(
                    str(error),
                    status=error.status,
                    payload=error.payload,
                    **{key: value for key, value in extracted.items() if key != "message"},
                ) from error
            raise

    async def create_chat_run(self, params: dict[str, Any]) -> dict[str, Any]:
        assert_chat_run_external_media(params)
        body = {
            "messages": params["messages"],
            "tools": params.get("tools"),
            "tool_choice": params.get("toolChoice"),
            "model": params.get("model"),
            "sampling": params.get("sampling"),
            "media_references": params.get("mediaReferences"),
            "media_context": params.get("mediaContext"),
            "max_estimated_capacity_units": params.get("maxEstimatedCapacityUnits"),
            "confirm_cost": params.get("confirmCost"),
            "session_id": params.get("sessionId"),
            "client_message_id": params.get("clientMessageId"),
            "token_type": params.get("tokenType"),
            "billing_mode": params.get("billingMode"),
            "app_source": params.get("appSource") or self.client.app_source,
            "runtime_config": params.get("runtimeConfig"),
        }
        headers = (
            {"Idempotency-Key": params["idempotencyKey"]} if params.get("idempotencyKey") else None
        )
        response = await self._run_request("POST", "/v1/chat/runs", body, headers)
        return response["data"]["run"]

    async def get_chat_run(self, run_id: str) -> dict[str, Any]:
        response = await self._run_request("GET", f"/v1/chat/runs/{quote(run_id, safe='')}")
        return response["data"]["run"]

    async def cancel_chat_run(self, run_id: str, reason: str | None = None) -> dict[str, Any]:
        response = await self._run_request(
            "POST",
            f"/v1/chat/runs/{quote(run_id, safe='')}/cancel",
            {"reason": reason} if reason else {},
        )
        return response["data"]["run"]

    async def confirm_chat_run_cost(self, run_id: str, params: dict[str, Any]) -> dict[str, Any]:
        response = await self._run_request(
            "POST",
            f"/v1/chat/runs/{quote(run_id, safe='')}/confirm-cost",
            {
                "tool_call_id": params["toolCallId"],
                "decision": params["decision"],
                "overrides": params.get("overrides"),
                "reason": params.get("reason"),
            },
        )
        return response["data"]["run"]

    async def stream_chat_run_events(
        self, run_id: str, last_event_id: int | str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        headers = {"Accept": "text/event-stream"}
        if last_event_id is not None:
            headers["Last-Event-ID"] = str(last_event_id)
        buffer: list[str] = []
        async for line in self.client.rest.stream_lines(
            f"/v1/chat/runs/{quote(run_id, safe='')}/events/stream",
            headers=headers,
            timeout=None,
        ):
            if line:
                buffer.append(line)
                continue
            for frame in parse_sse_chunk("\n".join(buffer)):
                buffer.clear()
                if frame["event"] == "run_status" or frame["data"] is None:
                    continue
                if isinstance(frame["data"], dict):
                    yield frame["data"]
        if buffer:
            for frame in parse_sse_chunk("\n".join(buffer)):
                if frame["event"] != "run_status" and isinstance(frame["data"], dict):
                    yield frame["data"]

    def _handle_models(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        self._models = {
            model_id: ({"workers": value} if isinstance(value, (int, float)) else value)
            for model_id, value in data.items()
        }
        self.emit("modelsUpdated", self.models)

    def _handle_tokens(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        stream = self._active_streams.get(data.get("jobID"))
        if not stream:
            return
        chunk = {
            "jobID": data["jobID"],
            "content": data.get("content", ""),
            "role": data.get("role"),
            "finishReason": data.get("finishReason"),
            "usage": data.get("usage"),
            "tool_calls": data.get("tool_calls"),
        }
        stream._push_chunk(chunk)
        self.emit("token", chunk)

    def _handle_result(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        stream = self._active_streams.pop(data.get("jobID"), None)
        if not stream:
            return
        stream._complete(
            data.get("timeTaken", 0),
            data.get("usage"),
            worker_name=data.get("workerName"),
            cost=data.get("cost"),
        )
        if stream.final_result:
            self.emit("completed", stream.final_result)

    def _handle_error(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        stream = self._active_streams.pop(data.get("jobID"), None)
        if not stream:
            return
        error = ChatJobError(
            data.get("error_message") or str(data.get("error")),
            code=data.get("error_code"),
            error_type=str(data.get("error")) if data.get("error") is not None else None,
            job_id=data.get("jobID"),
            payload=data,
            subscription_limit=data.get("subscriptionLimit"),
            required_plans=data.get("requiredPlans"),
            feature=data.get("feature"),
            limitation=data.get("limitation"),
        )
        stream._fail(error)
        self.emit(
            "error",
            {
                "jobID": data.get("jobID"),
                "error": str(data.get("error")),
                "errorCode": data.get("error_code"),
                "message": str(error),
                "workerName": data.get("workerName"),
            },
        )

    def _handle_state(self, data: Any) -> None:
        if not isinstance(data, dict) or data.get("jobID") not in self._active_streams:
            return
        stream = self._active_streams[data["jobID"]]
        if data.get("workerName"):
            stream._worker_name = data["workerName"]
        self.emit(
            "jobState",
            {
                "jobID": data["jobID"],
                "type": data.get("type"),
                "workerName": data.get("workerName"),
                "queuePosition": data.get("queuePosition"),
                "modelId": data.get("modelId"),
                "estimatedCost": data.get("estimatedCost"),
            },
        )
