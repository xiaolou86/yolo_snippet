import cv2
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO

# Setup GPIO
GPIO.setmode(GPIO.BCM)  # Use BCM GPIO numbering
GPIO_pin = 18  # Replace with your GPIO pin number
GPIO.setup(GPIO_pin, GPIO.OUT)

# Initialize PiCamera
picam2 = Picamera2()
# Camera setup code here...

# Initialize YOLO model
model = YOLO('best (1).pt') 

record = False
video_writer = None
frame_size = (800, 600)  # Frame size from your camera setup

while True:
    # Capture frame from PiCamera
    im = picam2.capture_array()
    
    # Perform object detection with YOLO
    results = model(im, conf=0.4)
    
    # Check for your specific class detection
    if 'person' in results.names:  # Replace 'person' with your interested class
        GPIO.output(GPIO_pin, GPIO.HIGH)  # Set GPIO pin high
        
        # Start recording if not already
        if not record:
            video_writer = cv2.VideoWriter('detected_event.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, frame_size)
            record = True
            start_time = cv2.getTickCount()
    
    # Record video if activated
    if record:
        video_writer.write(im)
        if (cv2.getTickCount() - start_time)/cv2.getTickFrequency() > 10:  # Record for 10 seconds
            GPIO.output(GPIO_pin, GPIO.LOW)  # Reset GPIO pin
            video_writer.release()  # Stop recording
            record = False
    
    # Display image & exit condition here...

# Cleanup
GPIO.cleanup()
