"""Save the input image and all post-ReLU feature maps as one figure."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from cnn import forward

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "assets" / "sample_digit.png"
WEIGHTS = ROOT / "assets" / "weights.npz"


def load_image(path, invert):
    with Image.open(path) as source:
        gray = ImageOps.exif_transpose(source).convert("L")
        if invert:
            gray = ImageOps.invert(gray)
        fitted = ImageOps.pad(gray, (16, 16), method=Image.Resampling.NEAREST, color=0)
    return np.asarray(fitted, dtype=np.float32) / 255


def plot_feature_layer(fig, grid, features, layer):
    height, width, count = features.shape
    bound = max(float(features.max()), 1e-8)
    first_row = 2 if layer == 1 else 5
    title = fig.add_subplot(grid[first_row - 1, :4])
    title.set_axis_off()
    title.text(
        0, 0.43, f"Conv {layer} + ReLU  ·  {count} maps  ·  {height} × {width} pixels",
        fontsize=12, weight="bold", color="#1d324a",
    )
    for channel in range(count):
        row, column = divmod(channel, 4)
        ax = fig.add_subplot(grid[first_row + row, column])
        plotted = ax.imshow(
            features[:, :, channel], cmap="magma", vmin=0, vmax=bound,
            interpolation="nearest",
        )
        ax.set_title(f"Channel {channel + 1:02d}", fontsize=8, color="#283e55", pad=5)
        ax.set_axis_off()

    colorbar_ax = fig.add_subplot(grid[first_row:first_row + count // 4, 4])
    colorbar = fig.colorbar(plotted, cax=colorbar_ax)
    colorbar.set_label("Activation", fontsize=8, color="#283e55")
    colorbar.ax.tick_params(labelsize=7, colors="#283e55")


def save_figure(pixels, layers, prediction, output):
    fig = plt.figure(figsize=(9, 12), layout="constrained", facecolor="#f6f7fa")
    grid = fig.add_gridspec(
        8, 5, height_ratios=[1.30, 0.22, 1, 1, 0.22, 1, 1, 1],
        width_ratios=[1, 1, 1, 1, 0.075],
    )
    fig.suptitle("CNN Feature Maps", fontsize=20, weight="bold", color="#142a42")

    sample = fig.add_subplot(grid[0, 0])
    sample.imshow(pixels, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    sample.set_axis_off()

    info = fig.add_subplot(grid[0, 1:4])
    info.set_axis_off()
    info.text(0.02, 0.73, f"Predicted digit  {prediction}", fontsize=16, weight="bold")
    info.text(
        0.02, 0.52, "Input image  ·  16 × 16 grayscale\n"
                    "Brighter pixels indicate stronger responses.\n"
                    "Channels share one scale within each layer.",
        fontsize=10, va="top", linespacing=1.65, color="#283e55",
    )

    for index, features in enumerate(layers, start=1):
        plot_feature_layer(fig, grid, features[0], index)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Path to a digit image")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "feature_maps.png")
    parser.add_argument("--invert", action="store_true", help="Use for dark ink on a light background")
    args = parser.parse_args()

    try:
        pixels = load_image(args.image, args.invert)
    except OSError as exc:
        parser.error(f"Could not open {args.image}: {exc}")

    with np.load(WEIGHTS, allow_pickle=False) as saved:
        params = dict(saved)
    logits, layers, _ = forward(pixels[None, :, :, None], params)
    prediction = int(logits[0].argmax())
    save_figure(pixels, layers, prediction, args.output)
    print(f"Predicted digit: {prediction}")
    for index, features in enumerate(layers, start=1):
        height, width, count = features.shape[1:]
        active = np.count_nonzero(features)
        print(
            f"Conv {index} + ReLU: {count} maps, {height} x {width} pixels, "
            f"{active}/{features.size} positive activations"
        )
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
