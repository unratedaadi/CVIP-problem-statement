# Module 3 · Image Segmentation
## Activity 5 · Generate a Binary Mask Using Otsu’s Thresholding

### Aim

Automatically separate objects from the background in an image and save the result as a binary mask.

### Principle

Otsu’s method uses the grayscale histogram to choose a single threshold. For each candidate threshold, it splits the pixels into two classes and measures how far apart their mean intensities are, weighted by the relative size of each class. It picks a threshold that maximizes this between-class variance. The method works best when the two intensity groups are well separated.

### Procedure

1. Load the input as an 8-bit grayscale image.
2. Call `cv2.threshold` with `cv2.THRESH_BINARY | cv2.THRESH_OTSU`. The threshold argument is set to zero because OpenCV computes the actual threshold.
3. Set pixels above the selected threshold to 255 (white foreground). Pixels at or below it become 0 (black background).
4. Save the mask as a PNG and report the threshold and the number of white pixels.

### Run

From this folder, use Python 3 with OpenCV installed.

```bash
python3 -m pip install -r requirements.txt
python3 otsu_mask.py sample/input.png -o binary_mask.png
```

If your object is darker than its background, add `--invert` to make it white in the mask. The archive includes a generated input and its corresponding output in `sample/`.

### Observation and result

| Measure | Sample result |
| --- | --- |
| Image size (width × height) | 320 × 256 pixels |
| Selected threshold | 80 |
| Foreground pixels | 18,743 (22.9%) |
| Background pixels | 63,177 (77.1%) |
| Mask intensity values | 0 and 255 |

The circle and rectangle both appear white against a black background. On this synthetic sample, the mask matches the known shape boundaries pixel for pixel. The threshold can change with different inputs. Uneven lighting or similar object and background intensities can make a single global threshold less effective.
