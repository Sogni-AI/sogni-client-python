"""Generate one image with an API key from the environment."""

import asyncio
import os

from sogni_client import SogniClient


async def main() -> None:
    api_key = os.environ["SOGNI_API_KEY"]
    async with await SogniClient.create(api_key=api_key) as sogni:
        project = await sogni.projects.create(
            type="image",
            model_id="z_image_turbo_bf16",
            positive_prompt="A glass greenhouse drifting above Singapore at dawn",
            negative_prompt="text, watermark",
            number_of_media=1,
            width=1024,
            height=1024,
            steps=8,
        )
        for url in await project.wait_for_completion():
            print(url)


if __name__ == "__main__":
    asyncio.run(main())
