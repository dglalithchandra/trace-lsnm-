"""Score one colour-fundus image with the research-only RFMiD baseline."""

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

from trace_models import TraceRFMiDScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest scores to print (default: 10; use 45 for all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not 1 <= args.top_k <= 45:
        raise ValueError("--top-k must be between 1 and 45")
    with Image.open(args.image) as opened:
        image = np.asarray(opened.convert("RGB"))
    scorer = TraceRFMiDScorer(WORKSPACE)
    output = scorer.score_image(image, image_id=str(args.image.resolve()))
    output["scores"] = output["scores"][: args.top_k]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
