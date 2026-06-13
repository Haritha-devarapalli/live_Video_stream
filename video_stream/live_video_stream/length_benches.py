from flask import Flask, Response
from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import subprocess
import threading
import time
import os
import requests
import numpy as np
from datetime import datetime

app = Flask(__name__)

SIM_INTERFACE = "usb0"

# ------------------- IMAGE CAPTURE SETTINGS -------------------
CAPTURE_INTERVAL = 15  # seconds
VM_API_URL = "http://104.154.141.198:3001/upload"

latest_frame = None
frame_lock = threading.Lock()

# ------------------- SIM CHECK & CONNECT -------------------
def has_sim_ip():
    result = subprocess.run(["ip", "addr", "show", SIM_INTERFACE],
                            capture_output=True, text=True)
    return "inet " in result.stdout

def bring_up_sim():
    print("[*] Bringing up LTE data connection...")
    subprocess.run(["sudo", "qmi-network", SIM_INTERFACE, "start"])
    time.sleep(6)

def add_default_route():
    print("[*] Setting mobile network as default route...")
    subprocess.run(["sudo", "ip", "route", "del", "default"], stderr=subprocess.DEVNULL)
    subprocess.run(["sudo", "ip", "route", "add", "default", "dev", SIM_INTERFACE, "metric", "100"])

def ping_test():
    result = subprocess.run(["ping", "-c", "2", "8.8.8.8"], capture_output=True, text=True)
    return "0% packet loss" in result.stdout

print("\n" + "="*50)
print("      Checking LTE SIM Connection     ")
print("="*50)

if not has_sim_ip():
    bring_up_sim()

if has_sim_ip():
    add_default_route()

if ping_test():
    print("[+] SIM Internet Active & Stable")
else:
    print("[!] SIM Ping Failed. Check Signal or APN")

print("LTE Setup Complete\n")

# ------------------- Camera Start -------------------
print("Starting Raspberry Pi Camera Module 3...")
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 30}
)
picam2.configure(config)
picam2.start()
time.sleep(2)
print("Camera started successfully!")

# ------------------- YOLO Models -------------------
print("Loading YOLO models...")
model_custom = YOLO("runs/segment/train4/weights/best.pt")  # your custom model
model_coco = YOLO("yolov8n.pt")  # COCO model
print("Models loaded!")

# ------------------- FFmpeg RTMP -------------------
rtmp_url = "rtmp://104.154.141.198/live/stream1"
ffmpeg_cmd = [
    "ffmpeg", "-y", "-f", "rawvideo",
    "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
    "-s", "640x480", "-r", "25", "-i", "-",
    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
    "-b:v", "1200k", "-maxrate", "1200k", "-bufsize", "2400k",
    "-g", "50", "-f", "flv", rtmp_url
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# ------------------- HELPER FUNCTIONS -------------------
def get_object_metrics(box_or_mask):
    """
    box_or_mask: either [x1,y1,x2,y2] box or binary mask (numpy)
    Returns: length (pixels), angle (degrees)
    """
    if isinstance(box_or_mask, np.ndarray):
        contours, _ = cv2.findContours(box_or_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0, 0
        rect = cv2.minAreaRect(contours[0])
        (w, h) = rect[1]
        angle = rect[2]
        length = max(w, h)
        return int(length), round(angle, 1)
    else:
        x1, y1, x2, y2 = box_or_mask
        w = x2 - x1
        h = y2 - y1
        length = int(max(w, h))
        angle = 0
        return length, angle

def overlay_metrics(frame, boxes, labels, lengths, angles):
    """
    Draw bounding boxes, labels, and metrics on the frame
    """
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = labels[i]
        length = lengths[i]
        angle = angles[i]

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display label above box
        text = f"{label} L:{length}px A:{angle}°"
        cv2.putText(frame, text, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# ------------------- FRAME UPLOAD THREAD -------------------
def periodic_image_upload():
    while True:
        time.sleep(CAPTURE_INTERVAL)
        with frame_lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()

        _, jpeg = cv2.imencode('.jpg', frame_copy)
        files = {'image': ('frame.jpg', jpeg.tobytes(), 'image/jpeg')}

        try:
            response = requests.post(VM_API_URL, files=files, timeout=10)
            if response.status_code == 200:
                print("[✓] Image uploaded successfully (plain image)")
            else:
                print(f"[!] Upload failed: {response.status_code}")
        except Exception as e:
            print(f"[!] Upload exception: {e}")

# ------------------- FRAME GENERATOR -------------------
def gen_frames():
    global latest_frame
    while True:
        frame = picam2.capture_array()  # RGB frame
        plain_frame = frame.copy()

        # YOLO detections
        results_custom = model_custom(frame)[0]
        results_coco = model_coco(frame)[0]

        # Prepare lists for overlay
        boxes = []
        labels = []
        lengths = []
        angles = []

        for i, box in enumerate(results_custom.boxes.xyxy.cpu().numpy()):
            boxes.append(box)
            # Use YOLO labels if available, else generic
            label = results_custom.names[int(results_custom.boxes.cls[i])]
            labels.append(label)

            length, angle = get_object_metrics(box)
            lengths.append(length)
            angles.append(angle)

        # Overlay bounding boxes, labels, length & angle
        frame = overlay_metrics(frame, boxes, labels, lengths, angles)

        # Draw COCO YOLO results normally
        frame = results_coco.plot()  # optional, can overlay custom + coco together if needed

        # Save clean frame for upload
        with frame_lock:
            latest_frame = plain_frame

        # FFmpeg streaming
        try:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ffmpeg_proc.stdin.write(frame_bgr.tobytes())
        except:
            print("[!] FFmpeg stream issue, retrying...")

        # Flask MJPEG stream
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# ------------------- FLASK ROUTES -------------------
@app.route('/')
def index():
    return f'''
    <h1>Raspberry Pi Camera - YOLO Live Stream</h1>
    <img src="/stream" width="720">
    <p>RTMP: {rtmp_url}</p>
    '''

@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------- SERVER START -------------------
if __name__ == '__main__':
    print("\nServer starting on port 5001...\n")

    threading.Thread(
        target=periodic_image_upload,
        daemon=True
    ).start()

    threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=5001, threaded=True),
        daemon=True
    ).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping system...")
        picam2.stop()
        ffmpeg_proc.terminate()
