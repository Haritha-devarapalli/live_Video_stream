#!/usr/bin/env python3

from flask import Flask, Response
from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import subprocess
import threading
import time
import math
import requests
from datetime import datetime

# ------------------- APP -------------------
app = Flask(__name__)

# ------------------- SETTINGS -------------------

SIM_INTERFACE = "usb0"

CAPTURE_INTERVAL = 2
VM_API_URL = "http://104.154.141.198:3001/upload"
RTMP_URL = "rtmp://104.154.141.198/live/stream1"

# New folder every boot
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

print("SESSION:", SESSION_ID)

# ------------------- GLOBAL FRAME -------------------

latest_clean = None
frame_lock = threading.Lock()


# ------------------- SIM SETUP -------------------

def has_sim_ip():
    return "inet " in subprocess.run(
        ["ip","addr","show",SIM_INTERFACE],
        capture_output=True,text=True
    ).stdout


def bring_up_sim():
    subprocess.run(["sudo","qmi-network",SIM_INTERFACE,"start"])
    time.sleep(6)


def add_route():
    subprocess.run(["sudo","ip","route","del","default"],
                   stderr=subprocess.DEVNULL)

    subprocess.run(["sudo","ip","route","add","default","dev",SIM_INTERFACE])


def ping_test():
    return "0% packet loss" in subprocess.run(
        ["ping","-c","2","8.8.8.8"],
        capture_output=True,text=True
    ).stdout


print("Checking SIM...")

if not has_sim_ip():
    bring_up_sim()

if has_sim_ip():
    add_route()

print("SIM OK" if ping_test() else "SIM FAIL")


# ------------------- CAMERA -------------------

print("Starting camera...")

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size":(640,480),"format":"RGB888"},
    controls={"FrameRate":30}
)

picam2.configure(config)
picam2.start()

time.sleep(2)

print("Camera ready")


# ------------------- YOLO -------------------

print("Loading YOLO...")

model = YOLO("runs/segment/train4/weights/best.pt")

print("YOLO loaded")


# ------------------- FFMPEG -------------------

print("Starting FFmpeg...")

ffmpeg = subprocess.Popen([

    "ffmpeg","-y",

    "-f","rawvideo",
    "-pix_fmt","bgr24",
    "-s","640x480",
    "-r","25",
    "-i","-",

    "-c:v","libx264",
    "-preset","ultrafast",
    "-tune","zerolatency",

    "-b:v","1200k",
    "-maxrate","1200k",
    "-bufsize","2400k",

    "-f","flv",
    RTMP_URL

], stdin=subprocess.PIPE)


# ------------------- IMAGE UPLOAD THREAD -------------------

def upload_loop():

    global latest_clean

    last = 0

    while True:

        if time.time() - last < CAPTURE_INTERVAL:
            time.sleep(0.1)
            continue

        with frame_lock:

            if latest_clean is None:
                continue

            frame = latest_clean.copy()


        _, jpg = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        )

        files = {
            "image": ("img.jpg", jpg.tobytes(), "image/jpeg")
        }

        data = {
            "session_id": SESSION_ID
        }

        try:

            r = requests.post(
                VM_API_URL,
                files=files,
                data=data,
                timeout=5
            )

            if r.status_code == 200:
                print("[✓] Uploaded")

        except:
            print("[!] Upload failed")

        last = time.time()


# ------------------- FRAME GENERATOR -------------------

def gen_frames():

    global latest_clean

    cx = 320
    cy = 240


    while True:

        # Capture
        frame = picam2.capture_array()

        clean = frame.copy()


        # YOLO detect
        results = model(frame)[0]


        for box, cls in zip(results.boxes.xyxy,
                            results.boxes.cls):

            x1,y1,x2,y2 = box.cpu().numpy()

            w = x2-x1
            h = y2-y1

            mx = x1 + w/2
            my = y1 + h/2

            angle = math.degrees(
                math.atan2(my-cy,mx-cx)
            )

            label = results.names[int(cls)]

            text = f"{label} W:{w:.1f} H:{h:.1f} A:{angle:.1f}"


            # Draw rectangle
            cv2.rectangle(
                frame,
                (int(x1),int(y1)),
                (int(x2),int(y2)),
                (0,0,255),2
            )


            # Text position near box
            tx = int(x1)
            ty = int(y1) - 8

            if ty < 15:
                ty = int(y1) + 15


            # Draw RED text (no background)
            cv2.putText(
                frame,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,255),   # RED
                1,
                cv2.LINE_AA
            )


        # Save clean frame for upload
        with frame_lock:
            latest_clean = clean


        # Send to RTMP
        try:

            ffmpeg.stdin.write(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                ).tobytes()
            )

        except:
            print("[!] FFmpeg issue")


        # Send to browser
        _, jpg = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )

        yield (

            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg.tobytes() + b"\r\n"

        )


# ------------------- FLASK -------------------

@app.route("/")
def index():
    return "<img src='/stream'>"


@app.route("/stream")
def stream():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ------------------- START -------------------

if __name__=="__main__":

    print("Starting upload thread...")

    threading.Thread(
        target=upload_loop,
        daemon=True
    ).start()


    print("Starting server...")

    app.run(
        host="0.0.0.0",
        port=5001,
        threaded=True
    )
