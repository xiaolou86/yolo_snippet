# Perform object detection and annotate frame
results = model(im, conf=0.4)

#old version
annotated_image = results.render()[0]  # This will render the detections directly onto the image

#new version
annotated_image = results.plot()[0]  # This will render the detections directly onto the image



video_writer.write(annotated_image)  # Save frame to video
cv2.imshow("YOLO Object Detection", annotated_image)
