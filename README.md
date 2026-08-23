# Sogni Client for Python

An async Python SDK for image, video, audio, and LLM inference on the Sogni
Supernet. It follows the public surface and wire protocol of the TypeScript
`sogni-client`, while using Python naming conventions and async iterators.

> The Python port is currently beta. Keep credentials in environment variables
> or your system keychain; never commit them to source control.

[Official quickstart](https://docs.sogni.ai/sogni-sdk/python/) ·
[Examples](https://github.com/Sogni-AI/sogni-client-python/tree/main/examples) ·
[Sogni API reference](https://docs.sogni.ai/api-reference/)

## Install

Install the latest beta directly from the official GitHub repository:

```bash
python -m pip install "sogni-client @ git+https://github.com/Sogni-AI/sogni-client-python.git@main"
```

For an editable source checkout:

```bash
git clone https://github.com/Sogni-AI/sogni-client-python.git
cd sogni-client-python
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

## Edit an image with Krea 2 Identity Edit

Pass one or two local reference images through `context_images`. For two-image
edits, place the base scene first and the identity or detail reference second.

```python
project = await sogni.projects.create(
    type="image",
    model_id="krea2_identity_edit_v1_2",
    positive_prompt=(
        "Change only the jacket to vivid sapphire blue. Preserve the exact "
        "facial identity, expression, framing, background, and lighting."
    ),
    number_of_media=1,
    width=1024,
    height=1024,
    steps=10,
    guidance=1,
    token_type="spark",
    context_images=["reference.png"],
)
print(await project.wait_for_completion(timeout=900))
```

The runnable example accepts one or two image paths and can also create a batch:

```bash
python examples/krea_identity_edit.py reference.png \
  --prompt "Change only the jacket to vivid sapphire blue; preserve identity."

python examples/krea_identity_edit.py scene.png identity.png \
  --prompt "Use the first image as the base scene and the second for identity." \
  --count 4
```

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

## Documentation

- [Python SDK quickstart](https://docs.sogni.ai/sogni-sdk/python/)
- [Sogni SDK overview](https://docs.sogni.ai/sogni-sdk/)
- [REST API reference](https://docs.sogni.ai/api-reference/)
