from ultralytics import YOLO

# Load the trained pose estimation model
model = YOLO('yolov8n-pose.pt')

# Run predictions
results = model('path/to/your/image.jpg')


"""
labels = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
"""

# Access keypoints for each detected person
for person in results.keypoints:
        wrist_coords = person.xy[person.names.index('wrist')]  # Assuming 'wrist' is part of your model's keypoints
            print(f'Wrist coordinates: {wrist_coords}')
