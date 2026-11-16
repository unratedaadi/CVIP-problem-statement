"""Rebuild the CNN weights and select a reproducible held-out demo image."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from cnn import backward, cross_entropy, forward, initialize

ROOT = Path(__file__).resolve().parent
SEED = 42
SAMPLE_DIGIT = 7


def train(epochs, batch_size, learning_rate):
    digits = load_digits()
    train_ids, test_ids = train_test_split(
        np.arange(digits.target.size),
        test_size=0.20,
        random_state=SEED,
        stratify=digits.target,
    )
    images = (digits.images.astype(np.float32) / 16).repeat(2, axis=1).repeat(2, axis=2)
    images = images[..., None]
    rng = np.random.default_rng(SEED)
    params = initialize(rng)
    history = []
    for epoch in range(1, epochs + 1):
        ids = rng.permutation(train_ids)
        total_loss = 0
        for start in range(0, ids.size, batch_size):
            batch = ids[start:start + batch_size]
            logits, features, cache = forward(images[batch], params)
            loss, delta = cross_entropy(logits, digits.target[batch])
            total_loss += loss * batch.size
            for name, gradient in backward(delta, features, cache, params).items():
                params[name] -= learning_rate * gradient
        history.append(total_loss / ids.size)
        print(f"Epoch {epoch:02d}/{epochs:02d}  training loss {history[-1]:.4f}")

    logits, _, _ = forward(images[test_ids], params)
    correct = int(np.count_nonzero(logits.argmax(axis=1) == digits.target[test_ids]))
    sample_id = int(test_ids[np.flatnonzero(digits.target[test_ids] == SAMPLE_DIGIT)[0]])

    asset_dir = ROOT / "assets"
    asset_dir.mkdir(exist_ok=True)
    np.savez_compressed(asset_dir / "weights.npz", **params)
    pixels = np.rint(images[sample_id, :, :, 0] * 255).astype(np.uint8)
    Image.fromarray(pixels).save(asset_dir / "sample_digit.png")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary = {
        "seed": SEED,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_images": int(train_ids.size),
        "test_images": int(test_ids.size),
        "test_correct": correct,
        "test_accuracy": correct / test_ids.size,
        "training_loss": [round(value, 6) for value in history],
        "sample_dataset_index": sample_id,
        "sample_label": SAMPLE_DIGIT,
    }
    (results_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Held-out accuracy {correct}/{test_ids.size} ({summary['test_accuracy']:.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not (0 < args.learning_rate < 1):
        parser.error("Epochs and batch size must be positive, and 0 < learning rate < 1.")
    train(args.epochs, args.batch_size, args.learning_rate)
