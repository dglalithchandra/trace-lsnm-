"""Create one 384-value TRACE A1 research embedding from a fundus image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_models import TraceA1Encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="Optional .npy destination for the embedding")
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if args.output and args.output.suffix.lower() != ".npy":
        raise ValueError("--output must end in .npy")

    with Image.open(args.image) as opened:
        image = np.asarray(opened.convert("RGB"))
    encoder = TraceA1Encoder(WORKSPACE, checkpoint=args.checkpoint)
    result = encoder.embed_image(image, image_id=str(args.image.resolve()))
    embedding = result.pop("embedding")
    if embedding is not None and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, embedding)
        result["embedding_file"] = str(args.output.resolve())
    elif embedding is not None:
        result["embedding"] = embedding.tolist()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
