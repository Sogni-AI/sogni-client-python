"""Stream an LLM response over the Sogni socket transport."""

import asyncio
import os

from sogni_client import SogniClient


async def main() -> None:
    api_key = os.environ["SOGNI_API_KEY"]
    async with await SogniClient.create(
        api_key=api_key,
        app_id="sogni-python-chat-example",
        app_source="sogni-python-examples",
    ) as sogni:
        stream = await sogni.chat.completions.create(
            model="qwen3.6-35b-a3b-gguf-iq4xs",
            messages=[{"role": "user", "content": "Pitch three surreal album covers."}],
            stream=True,
        )
        async for chunk in stream:
            print(chunk.get("content", ""), end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
