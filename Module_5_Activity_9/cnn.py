"""A two-layer CNN for 16 × 16 grayscale digit images, implemented in NumPy."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def initialize(rng):
    def weights(shape, fan_in):
        values = rng.standard_normal(shape, dtype=np.float32)
        return values * np.float32(np.sqrt(2 / fan_in))

    return {
        "kernel1": weights((3, 3, 1, 8), 9),
        "bias1": np.zeros(8, dtype=np.float32),
        "kernel2": weights((3, 3, 8, 12), 72),
        "bias2": np.zeros(12, dtype=np.float32),
        "head": weights((4 * 4 * 12, 10), 4 * 4 * 12),
        "head_bias": np.zeros(10, dtype=np.float32),
    }


def convolve(images, kernel, bias):
    # Pad by one pixel so a 3 × 3 kernel preserves spatial dimensions.
    padded = np.pad(images, ((0, 0), (1, 1), (1, 1), (0, 0)))
    windows = sliding_window_view(padded, (3, 3), axis=(1, 2))
    rows = windows.transpose(0, 1, 2, 4, 5, 3).reshape(-1, 9 * images.shape[-1])
    output = np.einsum("ij,jk->ik", rows, kernel.reshape(rows.shape[1], -1)) + bias
    return output.reshape(*images.shape[:3], bias.size), rows


def average_pool(features):
    batch, height, width, channels = features.shape
    blocks = features.reshape(batch, height // 2, 2, width // 2, 2, channels)
    return blocks.mean(axis=(2, 4))


def forward(images, params):
    conv1, rows1 = convolve(images, params["kernel1"], params["bias1"])
    relu1 = np.maximum(conv1, 0)
    pool1 = average_pool(relu1)
    conv2, rows2 = convolve(pool1, params["kernel2"], params["bias2"])
    relu2 = np.maximum(conv2, 0)
    pool2 = average_pool(relu2)
    flat = pool2.reshape(images.shape[0], -1)
    logits = flat @ params["head"] + params["head_bias"]
    return logits, (relu1, relu2), (rows1, rows2, pool1, pool2, flat)


def unpool(gradient):
    return gradient.repeat(2, axis=1).repeat(2, axis=2) * 0.25


def cross_entropy(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    totals = exponentials.sum(axis=1, keepdims=True)
    samples = np.arange(labels.size)
    loss = (np.log(totals[:, 0]) - shifted[samples, labels]).mean()
    gradient = exponentials / totals
    gradient[samples, labels] -= 1
    return float(loss), gradient / labels.size


def backward(logit_gradient, features, cache, params):
    relu1, relu2 = features
    rows1, rows2, pool1, pool2, flat = cache

    pool2_gradient = (logit_gradient @ params["head"].T).reshape(pool2.shape)
    conv2_gradient = unpool(pool2_gradient) * (relu2 > 0)
    delta2 = conv2_gradient.reshape(-1, relu2.shape[-1])

    # Each entry in a convolution window contributes to a padded input pixel.
    kernel2 = params["kernel2"]
    window_gradient = (delta2 @ kernel2.reshape(-1, relu2.shape[-1]).T).reshape(
        *conv2_gradient.shape[:3], *kernel2.shape[:3]
    )
    batch, height, width, channels = pool1.shape
    padded_gradient = np.zeros((batch, height + 2, width + 2, channels), dtype=np.float32)
    for row in range(3):
        for col in range(3):
            padded_gradient[:, row:row + height, col:col + width] += window_gradient[
                :, :, :, row, col, :
            ]

    pool1_gradient = padded_gradient[:, 1:-1, 1:-1, :]
    conv1_gradient = unpool(pool1_gradient) * (relu1 > 0)
    delta1 = conv1_gradient.reshape(-1, relu1.shape[-1])
    return {
        "kernel1": (rows1.T @ delta1).reshape(params["kernel1"].shape),
        "bias1": delta1.sum(axis=0),
        "kernel2": (rows2.T @ delta2).reshape(kernel2.shape),
        "bias2": delta2.sum(axis=0),
        "head": flat.T @ logit_gradient,
        "head_bias": logit_gradient.sum(axis=0),
    }
