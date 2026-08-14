"""Edit one or two local reference images with Krea 2 Identity Edit v1.2."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from sogni_client import SogniClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference_images",
        nargs="+",
        type=Path,
        help="One or two local reference images; place the base scene first.",
    )
    parser.add_argument("--prompt", required=True, help="The edit instruction.")
    parser.add_argument("--count", type=int, default=1, help="Number of outputs (default: 1).")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= len(args.reference_images) <= 2:
        parser.error("Krea 2 Identity Edit requires one or two reference images")
    if args.count < 1:
        parser.error("--count must be at least 1")
    for reference in args.reference_images:
        if not reference.is_file():
            parser.error(f"reference image does not exist: {reference}")
    return args


async def main(args: argparse.Namespace) -> None:
    async with await SogniClient.create(api_key=os.environ["SOGNI_API_KEY"]) as sogni:
        project = await sogni.projects.create(
            type="image",
            model_id="krea2_identity_edit_v1_2",
            positive_prompt=args.prompt,
            number_of_media=args.count,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance=1,
            token_type="spark",
            context_images=args.reference_images,
        )
        for url in await project.wait_for_completion(timeout=900):
            print(url)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
