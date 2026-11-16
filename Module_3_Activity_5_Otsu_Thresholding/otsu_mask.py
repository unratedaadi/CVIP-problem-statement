import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Generate a binary mask with Otsu's thresholding.")
    parser.add_argument("image", type=Path, help="Input image")
    parser.add_argument("-o", "--output", type=Path, default=Path("binary_mask.png"))
    parser.add_argument("--invert", action="store_true", help="Use white for dark objects")
    args = parser.parse_args()

    gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        parser.error(f"Could not read image: {args.image}")
    if args.output.suffix.lower() != ".png":
        parser.error("Use a .png output to preserve exact mask values.")
    if args.output.resolve() == args.image.resolve():
        parser.error("Choose a different output path to preserve the input image.")

    polarity = cv2.THRESH_BINARY_INV if args.invert else cv2.THRESH_BINARY
    threshold, mask = cv2.threshold(gray, 0, 255, polarity | cv2.THRESH_OTSU)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), mask):
        parser.exit(1, f"Could not save mask: {args.output}\n")

    foreground_pixels = cv2.countNonZero(mask)
    print(f"Otsu threshold: {threshold:.0f}")
    print(f"Foreground pixels: {foreground_pixels:,} ({foreground_pixels / mask.size:.1%})")
    print(f"Mask saved to {args.output}")


if __name__ == "__main__":
    main()
