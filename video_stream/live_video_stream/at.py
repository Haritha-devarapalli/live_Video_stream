# sim_global.py - Raspberry Pi + EC200-CN + YOLO + RTMP + LTE Auto Connect
from flask import Flask, Response
from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import subprocess
import threading
import time
import os

app = Flask(__name__)

SIM_INTERFACE = "usb0"   # change to wwan0 if needed

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

print("\n======================================")
print("      Checking LTE SIM Connection     ")
print("======================================")

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
model_custom = YOLO("runs/segment/train4/weights/best.pt")
model_coco = YOLO("yolov8n.pt")
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

# ------------------- Frame Generator -------------------
def gen_frames():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = model_custom(frame)[0].plot()
        frame = model_coco(frame)[0].plot()

        try:
            ffmpeg_proc.stdin.write(frame.tobytes())
        except:
            print("[!] FFmpeg stream issue, retrying...")

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
    print("Server starting on port 5001...")
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001, threaded=True), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping system...")
        picam2.stop()
        ffmpeg_proc.terminate()
