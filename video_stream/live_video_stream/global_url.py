from flask import Flask, Response
import cv2
from ultralytics import YOLO
import subprocess
import threading
import time

app = Flask(__name__)

# -------------------
# Camera & YOLO Setup
# -------------------
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 25)

# Load YOLO models
model_custom = YOLO("runs/segment/train4/weights/best.pt")
model_coco = YOLO("yolov8n.pt")

# -------------------
# FFmpeg RTMP Setup
# -------------------
rtmp_url = "rtmp://104.154.141.198/live/stream1"

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", "640x480",
    "-r", "25",
    "-i", "-",                  # input from stdin
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-b:v", "1200k",
    "-maxrate", "1200k",
    "-bufsize", "2500k",
    "-g", "30",
    "-keyint_min", "30",
    "-pix_fmt", "yuv420p",
    "-f", "flv",
    rtmp_url
]

# Start FFmpeg process
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# -------------------
# Frame Generator for MJPEG
# -------------------
def gen_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        # YOLO inference
        frame = model_custom(frame)[0].plot()
        frame = model_coco(frame)[0].plot()

        # Send frame to FFmpeg (for RTMP)
        try:
            ffmpeg_proc.stdin.write(frame.tobytes())
        except:
            pass

        # Encode for MJPEG
        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# -------------------
# Flask Routes
# -------------------
@app.route('/')
def index():
    return "<h1>Raspberry Pi YOLO Live Stream</h1><video width=640 controls autoplay><source src='/stream' type='multipart/x-mixed-replace; boundary=frame'></video>"

@app.route('/stream')
def stream():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------
# Run Flask in a Thread
# -------------------
def run_flask():
    app.run(host="0.0.0.0", port=5001, threaded=True)

if __name__ == "__main__":
    print("Starting Flask MJPEG server on port 5001...")
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    print("Streaming to RTMP:", rtmp_url)
