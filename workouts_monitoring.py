import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture("path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

gym_object = solutions.AIGym(
    line_thickness=2,
    view_img=True,
    pose_type="pushup",
    kpts_to_check=[6, 8, 10],
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    results = model.track(im0, verbose=False)  # Tracking recommended
    # results = model.predict(im0)  # Prediction also supported
    im0 = gym_object.start_counting(im0, results)

cv2.destroyAllWindows()
