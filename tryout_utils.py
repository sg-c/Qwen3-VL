#!/usr/bin/env python3
"""Shared utilities for Qwen3-VL tryout scripts."""

import base64
import os

from openai import OpenAI


def encode_image(image_path: str) -> str:
    """Base64-encode a local image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def vlm_chat(
    image_path: str,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str = "not-needed",
) -> str:
    """Send a local image to a Qwen3-VL server and return the text response."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    b64_image = encode_image(image_path)

    # Quick Validation: ensure <image> token is present for Qwen3-VL
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
    )
    return completion.choices[0].message.content or ""


def default_base_url() -> str:
    """Return the default VLM base URL from env or fallback."""
    return os.environ.get("QWEN_BASE_URL", "http://0.0.0.0:7860/v1")


def default_model() -> str:
    """Return the default model name from env or fallback."""
    return os.environ.get("QWEN_MODEL", './Qwen3-VL-8B-Instruct')
