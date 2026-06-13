from ultralytics import YOLO
import cv2
import math

# ---------------- SETTINGS ----------------

MODEL_PATH = "runs/segment/train4/weights/best.pt"
IMAGE_PATH = "/home/Hari/Downloads/IMAGE/DJI_0012.JPG"

# Drone info (EDIT THESE)
MIN_HEIGHT = 20    # meters
MAX_HEIGHT = 60    # meters
CAMERA_FOV = 84    # degrees (most drones: 80–90)

CONF = 0.25

# -----------------------------------------


# Load model
model = YOLO(MODEL_PATH)

# Read image
img = cv2.imread(IMAGE_PATH)

img_h, img_w, _ = img.shape

# Predict
results = model.predict(source=IMAGE_PATH, conf=CONF)

r = results[0]

count = 0

for box in r.boxes:

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    px_width = x2 - x1
    px_height = y2 - y1


    # ----------- Calculate meters/pixel -----------

    # Ground width at min height
    ground_min = 2 * MIN_HEIGHT * math.tan(math.radians(CAMERA_FOV/2))

    # Ground width at max height
    ground_max = 2 * MAX_HEIGHT * math.tan(math.radians(CAMERA_FOV/2))

    m_per_px_min = ground_min / img_w
    m_per_px_max = ground_max / img_w


    # ----------- Object size range -----------

    width_min = px_width * m_per_px_min
    width_max = px_width * m_per_px_max

    height_min = px_height * m_per_px_min
    height_max = px_height * m_per_px_max


    # ----------- Draw ------------

    cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)

    label = (
        f"W: {width_min:.2f}-{width_max:.2f}m "
        f"H: {height_min:.2f}-{height_max:.2f}m"
    )

    cv2.putText(
        img,
        label,
        (x1, y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        2
    )

    print(f"Object {count}")
    print(f" Pixel: {px_width}px x {px_height}px")
    print(f" Width: {width_min:.2f}m to {width_max:.2f}m")
    print(f" Height: {height_min:.2f}m to {height_max:.2f}m")
    print("--------------")

    count += 1


# Save output
cv2.imwrite("output_range_meter1.jpg", img)

print("✅ Done. Saved: output_range_meter.jpg")
