# Module 2 · Activity 3 — Laplacian Edge Detection

## Aim

Highlight image edges using OpenCV's Laplacian filter.

## Run the sample

Open a terminal in this folder and run the following commands. On Windows,
use `py` if `python3` is unavailable.

```bash
python3 -m pip install -r requirements.txt
python3 laplacian_edges.py sample_input.png
```

The script writes `laplacian_edges.png`. Compare it with the included
`sample_edges.png`. Run the script without an input to use its built-in demo,
or pass your own image path to filter a different image.

## Method and result

1. Load the input image in grayscale.
2. Apply a 3 × 3 Gaussian blur to reduce noise before differentiation.
3. Use the Laplacian to sum the horizontal and vertical second derivatives.
4. Convert the signed response to an 8-bit absolute magnitude and save it.

The sample contains a rectangle and a circle. Their boundaries appear brighter
in the edge map, while their flat interiors and the background appear black.
