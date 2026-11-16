import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Extract and display RGB channels.")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("sample.png"),
        help="input image (default: included sample.png)",
    )
    parser.add_argument("--no-show", action="store_true", help="save without opening a window")
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"could not open image: {args.image}")

    blue, green, red = cv2.split(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Original Image")

    for ax, name, channel in zip(axes[1:], ("Red", "Green", "Blue"), (red, green, blue)):
        channel_path = output_dir / f"{name.lower()}_channel.png"
        if not cv2.imwrite(str(channel_path), channel):
            raise OSError(f"Could not save {channel_path}")
        ax.imshow(channel, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{name} Channel")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "rgb_comparison.png", dpi=150)
    print(f"Saved three channel images and the comparison to {output_dir.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
