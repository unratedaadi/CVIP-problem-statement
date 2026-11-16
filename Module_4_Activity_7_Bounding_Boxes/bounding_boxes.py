from pathlib import Path

import cv2

folder = Path(__file__).resolve().parent
input_path = folder / "objects.jpg"
output_path = folder / "bounding_boxes.jpg"

image = cv2.imread(str(input_path))
if image is None:
    raise SystemExit(f"Could not open {input_path}. Add a readable image and try again.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, mask = cv2.threshold(
    blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
)
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

min_area = 300  # Square pixels. Lower this value for smaller objects.
object_count = 0
green = (0, 255, 0)

for contour in sorted(contours, key=lambda region: cv2.boundingRect(region)[0]):
    if cv2.contourArea(contour) < min_area:
        continue

    x, y, width, height = cv2.boundingRect(contour)
    object_count += 1

    cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), green, 2)
    cv2.putText(
        image,
        f"Object {object_count}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        green,
        2,
        cv2.LINE_AA,
    )

if not cv2.imwrite(str(output_path), image):
    raise SystemExit(f"Could not save {output_path}.")

print(f"Objects detected: {object_count}")
print(f"Output saved to {output_path}")
