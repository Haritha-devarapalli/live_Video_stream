import cv2
from ultralytics import YOLO
from flask import Flask, Response

app = Flask(__name__)

# Load models
model_custom = YOLO("/home/Hari/project/project1/runs/segment/train4/weights/best.pt")  # your custom model
model_coco = YOLO("yolov8n.pt")  # default COCO classes

# Video output path
output_file = "/home/Hari/project/project1/output.mp4"

def generate_frames():
    cap = cv2.VideoCapture("/dev/video0")  # use the correct camera
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open camera /dev/video1")

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20  # default 20 if FPS not available

    # Define VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Custom model
        result_custom = model_custom(frame)
        frame_custom = result_custom[0].plot()

        # COCO model
        result_coco = model_coco(frame_custom)
        try:
            frame_final = result_coco[0].plot()
        except ValueError:
            frame_final = frame_custom  # fallback if COCO throws array error

        # Write frame to video
        out.write(frame_final)

        # Encode for streaming
        ret, buffer = cv2.imencode('.jpg', frame_final)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()

@app.route('/stream')
def stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
