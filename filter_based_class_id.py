from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")  # Load your model
person_class_id = model.names.index('person')  # Assuming 'person' is the class name for people

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame)
    boxes = results.boxes.xyxy[results.boxes.cls == person_class_id].cpu().tolist()  # Filter detections to only 'person'

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        crop_img = frame[y1:y2, x1:x2]
        cv2.imwrite(f'cropped_person_{idx}.jpg', crop_img)
