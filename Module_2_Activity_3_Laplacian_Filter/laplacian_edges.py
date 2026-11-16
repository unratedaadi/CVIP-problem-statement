"""Module 2, Activity 3: highlight image edges with a Laplacian filter.

Install with: python3 -m pip install opencv-python
Run the built-in demo: python3 laplacian_edges.py
Use your own image: python3 laplacian_edges.py photo.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, help="Input image path")
    parser.add_argument("-o", "--output", type=Path, default=Path("laplacian_edges.png"))
    args = parser.parse_args()

    if args.image is None:
        # A flat background and simple shapes make the edge response easy to see.
        gray = np.full((240, 320), 35, dtype=np.uint8)
        cv2.rectangle(gray, (35, 65), (140, 175), 200, thickness=cv2.FILLED)
        cv2.circle(gray, (228, 120), 55, 240, thickness=cv2.FILLED)
    else:
        gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            parser.error(f"Could not read image: {args.image}")
        if args.image.resolve() == args.output.resolve():
            parser.error("Choose an output path different from the input image.")

    # Smooth small intensity variations before taking a second derivative.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    response = cv2.Laplacian(blurred, cv2.CV_64F, ksize=1)

    # Keep the magnitude of both positive and negative responses for display.
    edges = cv2.convertScaleAbs(response)
    try:
        saved = cv2.imwrite(str(args.output), edges)
    except cv2.error as error:
        parser.error(f"Could not save the edge map: {error}")
    if not saved:
        parser.error(f"Could not save the edge map: {args.output}")

    print(f"Saved Laplacian edge map to {args.output}")


if __name__ == "__main__":
    main()
