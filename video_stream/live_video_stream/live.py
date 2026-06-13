import cv2
from ultralytics import YOLO
from flask import Flask, Response

app = Flask(__name__)

# Load models
model_custom = YOLO("/home/Hari/project/project1/runs/segment/train4/weights/best.pt")   # segmentation for benches etc
model_coco = YOLO("yolov8n.pt")  # default coco classes

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Custom model
        result_custom = model_custom(frame)
        frame = result_custom[0].plot()

        # COCO model
        result_coco = model_coco(frame)
        frame = result_coco[0].plot()

        # Encode and yield as streaming bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/stream')
def stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
