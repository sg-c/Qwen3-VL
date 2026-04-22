#!/usr/bin/env python3
"""Try out local Qwen3-VL for object recognition in a local image."""

import argparse
import base64
import os
import sys

from openai import OpenAI


def encode_image(image_path: str) -> str:
    """Base64-encode a local image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def recognize_objects(image_path: str, base_url: str, model: str, prompt: str | None = None) -> str:
    """Send a local image to the local Qwen3-VL server and return the response."""
    client = OpenAI(
        api_key="not-needed",  # local serving usually ignores this
        base_url=base_url,
    )

    b64_image = encode_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}",
                    },
                },
                {
                    "type": "text",
                    "text": prompt or "List all objects you can recognize in this image.",
                },
            ],
        }
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
    )
    return completion.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize objects in an image using local Qwen3-VL")
    parser.add_argument("image", help="Path to the local image file")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN_BASE_URL", "http://0.0.0.0:7860/v1"),
        help="Base URL of the local OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", "\"./Qwen3-VL-8B-Instruct\""),
        help="Model name to use",
    )
    parser.add_argument(
        "--prompt",
        default="List all objects you can recognize in this image, be concise.",
        help="Prompt to send along with the image",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Sending {args.image} to {args.base_url} (model: {args.model}) ...\n")
    result = recognize_objects(args.image, args.base_url, args.model, args.prompt)
    print(result)


if __name__ == "__main__":
    main()
