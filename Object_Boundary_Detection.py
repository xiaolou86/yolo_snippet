import cv2

# Assuming 'box' is the bounding box of a detected object [x1, y1, x2, y2]
# Define your ROI
roi = (100, 100, 400, 400)  # [x1, y1, x2, y2] of the ROI

# Check if the box is inside the ROI
if box[0] >= roi[0] and box[1] >= roi[1] and box[2] <= roi[2] and box[3] <= roi[3]:
    print("Object within boundary detected.")
