#!/usr/bin/env python3
# sim_global.py - EC200 + trained YOLO + bounding boxes + periodic image upload
from flask import Flask, Response
from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import subprocess
import threading
import time
import math
import requests

app = Flask(__name__)

SIM_INTERFACE = "usb0"   # EC200 LTE interface
CAPTURE_INTERVAL = 2     # seconds
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

# ------------------- YOLO Model -------------------
print("Loading trained YOLO model...")
model_custom = YOLO("runs/segment/train4/weights/best.pt")  # Trained model
print("Model loaded!")

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

# ------------------- PERIODIC IMAGE UPLOAD -------------------
def periodic_image_upload():
    global latest_frame
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
                print("[✓] Image uploaded successfully")
            else:
                print(f"[!] Upload failed: {response.status_code}")
        except Exception as e:
            print(f"[!] Upload exception: {e}")

# ------------------- FRAME GENERATOR -------------------
def gen_frames():
    global latest_frame
    cam_center_x = 640 // 2
    cam_center_y = 480 // 2

    while True:
        frame = picam2.capture_array()
        plain_frame = frame.copy()  # save clean frame before YOLO

        # --- Trained YOLO detection ---
        results_custom = model_custom(frame)[0]

        y_offset = 20
        for box, cls_id in zip(results_custom.boxes.xyxy, results_custom.boxes.cls):
            x1, y1, x2, y2 = box.cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            obj_center_x = x1 + width / 2
            obj_center_y = y1 + height / 2
            dx = obj_center_x - cam_center_x
            dy = obj_center_y - cam_center_y
            angle = math.degrees(math.atan2(dy, dx))
            label = results_custom.names[int(cls_id)]

            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            # Show metrics + label
            text = f"{label} | W:{width:.1f} H:{height:.1f} A:{angle:.1f}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
            y_offset += 20

        # --- Update latest_frame for periodic upload ---
        with frame_lock:
            latest_frame = plain_frame

        # --- Send to FFmpeg ---
        try:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ffmpeg_proc.stdin.write(frame_bgr.tobytes())
        except:
            print("[!] FFmpeg stream issue, retrying...")

        # --- Yield for Flask ---
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# ------------------- Flask Routes -------------------
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

# ------------------- Server Start -------------------
if __name__ == '__main__':
    print("\nServer starting on port 5001...\n")

    threading.Thread(target=periodic_image_upload, daemon=True).start()
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001, threaded=True), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping system...")
        picam2.stop()
        ffmpeg_proc.terminate()
