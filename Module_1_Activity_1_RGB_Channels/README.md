# Module 1 · Activity 1 — Extract and Display Individual RGB Channels

**Aim.** Extract and display the red, green, and blue components of a color image using OpenCV and Matplotlib.

## Run the activity

Extract the ZIP, open a terminal in the extracted folder, then run:

```sh
python -m pip install -r requirements.txt
python rgb_channels.py
```

On macOS, use `python3` if that is your Python command. The script uses the included sample image by default. To use your own image, run `python rgb_channels.py "my_photo.jpg"`. Add `--no-show` to save results without opening a window.

## Method and observation

OpenCV loads a color image in blue, green, red (BGR) order. The program splits the image into three intensity arrays and converts the original to RGB for display with Matplotlib. Each channel is shown in grayscale on the same 0–255 scale. Brighter pixels indicate more of the corresponding color.

The included sample is a synthetic color chart with red, green, and blue tiles on top and cyan, magenta, and yellow tiles below. The tiles fade from full intensity at the top to zero at the bottom. For example, the cyan tile appears bright in the green and blue channel panels.

## Result

The `outputs/` folder contains `red_channel.png`, `green_channel.png`, `blue_channel.png`, and `rgb_comparison.png`. The archive includes the results for the sample. Running the script saves the results for the chosen image to the same filenames.
