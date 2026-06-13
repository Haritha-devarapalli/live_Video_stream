from ultralytics import YOLO
import cv2

# Load your trained segmentation model
model = YOLO("runs/segment/train4/weights/best.pt")

# Path to input image
image_path = "/home/Hari/Downloads/IMAGE/33.jpg"

# Run prediction
results = model.predict(source=image_path, conf=0.25, save=False)  # don't save in default folder

# Get the annotated image
annotated_image = results[0].plot()  # returns numpy array with masks drawn

# Save it to desired location
cv2.imwrite("output4.jpg", annotated_image)

print("✅ Annotated image saved as output.jpg in current folder")
