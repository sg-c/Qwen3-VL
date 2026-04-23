#!/usr/bin/env python3
"""2D object detection and visualization using local Qwen3-VL."""

import argparse
import json
import os
import sys

from PIL import Image, ImageColor, ImageDraw, ImageFont

from tryout_utils import default_base_url, default_model, vlm_chat

COLORS = [
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "pink",
    "purple",
    "brown",
    "gray",
    "beige",
    "turquoise",
    "cyan",
    "magenta",
    "lime",
    "navy",
    "maroon",
    "teal",
    "olive",
    "coral",
    "lavender",
    "violet",
    "gold",
    "silver",
] + [name for (name, _) in ImageColor.colormap.items()]


def _extract_json(text: str) -> str:
    """Strip markdown fences from model output."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            text = "\n".join(lines[i + 1 :])
            text = text.split("```")[0]
            break
    return text.strip()


def _load_font(size: int = 14):
    """Try to load a usable font."""
    for font_name in ("NotoSansCJK-Regular.ttc", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def plot_bounding_boxes(
    image: Image.Image,
    model_response: str,
    output_path: str | None = None,
) -> Image.Image:
    """Parse JSON bbox output and draw boxes on the image.

    Coordinates are expected in Qwen3-VL relative format (0-1000).
    """
    img = image.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font = _load_font()

    raw_json = _extract_json(model_response)

    try:
        boxes = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"Failed to parse model response as JSON: {e}", file=sys.stderr)
        print("Raw response:", raw_json, file=sys.stderr)
        return img

    if not isinstance(boxes, list):
        boxes = [boxes]

    for i, box in enumerate(boxes):
        color = COLORS[i % len(COLORS)]
        bbox = box.get("bbox_2d", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        abs_x1 = int(x1 / 1000 * width)
        abs_y1 = int(y1 / 1000 * height)
        abs_x2 = int(x2 / 1000 * width)
        abs_y2 = int(y2 / 1000 * height)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)),
            outline=color,
            width=3,
        )

        label = box.get("label", "")
        if label:
            draw.text((abs_x1 + 8, abs_y1 + 6), label, fill=color, font=font)

    if output_path:
        img.save(output_path)
        print(f"Saved annotated image to {output_path}")

    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect objects in an image using local Qwen3-VL and visualize bounding boxes."
    )
    parser.add_argument("image", help="Path to the local image file")
    parser.add_argument(
        "--categories",
        required=True,
        help='Comma-separated list of object categories, e.g. "car,person,bicycle"',
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save the annotated image (if omitted, image is displayed)",
    )
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
    args = parser.parse_args()
    
    if not os.path.isfile(args.image):
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    categories_str = ", ".join(categories)

    prompt = (
        f'Locate every instance that belongs to the following categories: "{categories_str}". '
        'Report bbox coordinates in JSON format like this: '
        '[{"bbox_2d": [x1, y1, x2, y2], "label": "category_name"}, ...]'
    )

    print(f"Sending {args.image} to {args.base_url} (model: {args.model}) ...")
    print(f"Prompt: {prompt}\n")

    response = vlm_chat(args.image, args.base_url, args.model, prompt)
    print("Model response:")
    print(response)
    print()

    image = Image.open(args.image)
    plot_bounding_boxes(image, response, args.output)

    if not args.output:
        image.show()


if __name__ == "__main__":
    main()
