#!/usr/bin/env python3
"""Simple object recognition using local Qwen3-VL."""

import argparse
import os
import sys

from tryout_utils import default_base_url, default_model, vlm_chat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recognize objects in an image using local Qwen3-VL"
    )
    parser.add_argument("image", help="Path to the local image file")
    parser.add_argument(
        "--base-url",
        default=default_base_url(),
        help="Base URL of the local OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--model",
        default=default_model(),
        help="Model name to use",
    )
    parser.add_argument(
        "--prompt",
        default="Describe only what you see in the image.",
        help="Prompt to send along with the image",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Sending {args.image} to {args.base_url} (model: {args.model}) ...\n")
    result = vlm_chat(args.image, args.base_url, args.model, args.prompt)
    print(result)


if __name__ == "__main__":
    main()
