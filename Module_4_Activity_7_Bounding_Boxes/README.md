# Module 4 · Problem Solving Activity 7
## Draw Bounding Boxes Around Objects Using OpenCV

**Aim**  
Detect separate objects in an image and draw a bounding box around each one.

### Run the activity

Extract the ZIP, open a terminal in the extracted project folder, and run:

    python -m pip install -r requirements.txt
    python bounding_boxes.py

Use `python3` if your system uses that command for Python 3. The program reads
`objects.jpg` and writes `bounding_boxes.jpg` next to the script.

### Method

1. Convert the input to grayscale and blur it with a 5 × 5 Gaussian kernel.
2. Apply Otsu’s threshold with binary inversion to separate the dark shapes from the light background.
3. Retrieve external contours and discard any with area below 300 square pixels.
4. Compute each axis-aligned box with `cv2.boundingRect()`. Draw and number the boxes from left to right.

### Result

The included synthetic sample contains a circle, a rectangle, and a triangle. The
program detected all three shapes and saved the included annotated image. The
console reported `Objects detected: 3`.

To try a different image, replace `objects.jpg` with a photo of separated dark
objects on a plain light background. Touching objects can produce a single
contour and share one bounding box. The numbered labels refer to detected regions.
