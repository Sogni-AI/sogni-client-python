# Sogni Client for Python

An async Python SDK for image, video, audio, and LLM inference on the Sogni
Supernet. It follows the public surface and wire protocol of the TypeScript
`sogni-client`, while using Python naming conventions and async iterators.

> The Python port is currently beta. Keep credentials in environment variables
> or your system keychain; never commit them to source control.

## Install

From the standalone repository:

```bash
python -m pip install -e .
```

Python 3.10 or newer is required.

## Create an image

```python
import asyncio
import os

from sogni_client import SogniClient


async def main() -> None:
    async with await SogniClient.create(api_key=os.environ["SOGNI_API_KEY"]) as sogni:
        project = await sogni.projects.create(
            type="image",
            model_id="z_image_turbo_bf16",
            positive_prompt="A tiny observatory above a sea of clouds",
            negative_prompt="text, watermark",
            number_of_media=1,
            width=1024,
            height=1024,
            steps=8,
        )
        print(await project.wait_for_completion())


asyncio.run(main())
```

`SogniClient.create()` generates a unique application ID when one is not
provided. Pass `app_id="..."` when you deliberately need a stable socket
identity.

## Chat

Socket-backed completion:

```python
result = await sogni.chat.completions.create(
    model="qwen3.6-35b-a3b-gguf-iq4xs",
    messages=[{"role": "user", "content": "Give me three visual concepts."}],
)
print(result["content"])
```

Hosted OpenAI-compatible completion:

```python
result = await sogni.chat.hosted.create(
    model="qwen3.6-35b-a3b-gguf-iq4xs",
    messages=[{"role": "user", "content": "Describe a surreal album cover."}],
)
```

For streaming socket chat, pass `stream=True` and iterate over the returned
`ChatStream` with `async for`.

## Durable workflows

```python
workflow = await sogni.workflows.start(
    input={"prompt": "Create a four-panel character turnaround"},
    idempotency_key="turnaround-001",
)

async for event in sogni.workflows.stream_events(workflow["id"]):
    print(event["event"], event["data"])
```

The client also exposes:

- `sogni.account` for authentication, balances, rewards, transactions, and subscriptions
- `sogni.projects` for generation, uploads, model discovery, and estimates
- `sogni.chat` for socket, hosted, tool, and durable-run APIs
- `sogni.workflows` and `sogni.workflows.templates`
- `sogni.replay` and `sogni.stats`

Python `snake_case` arguments are preferred. Common JavaScript-style aliases
remain accepted to simplify migration.

## Compatibility

This release tracks the current TypeScript source at `5.1.0-alpha.24`. The
REST, WebSocket, and SSE contracts are covered by credential-free protocol
tests, including authentication refresh, uploads, project state recovery,
streaming chat, workflows, templates, replay, and the canonical 24 hosted-tool
schemas.

The Python API is async-first; `AsyncSogniClient` is an alias of
`SogniClient`, not a synchronous wrapper. Browser-only cookie coordination and
multi-tab behavior have no Python equivalent. Local image references are
uploaded with their detected MIME type, but the TypeScript client's optional
browser-side image resizing is not reproduced. All 24 canonical tool schemas
are exposed; the local project-backed executor handles the six direct media
generation tools, while the remaining tools run through the hosted or durable
chat APIs. Live, credentialed smoke tests are intentionally separate from the
default test suite.

## Token authentication

```python
sogni = await SogniClient.create(auth_type="token")
await sogni.set_tokens(token=access_token, refresh_token=refresh_token)
```

Username/password login and signing are available through `sogni.account.login`.
API-key use does not require storing a wallet password.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
ruff check sogni_client tests
ruff format --check sogni_client tests
python -m build
```

Live integration tests require explicit credentials and are not run by default.
