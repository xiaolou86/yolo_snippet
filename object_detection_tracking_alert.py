import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

# Email setup
password = "your_app_password"
from_email = "your_email @yeongnamtan.com"
to_email = "receiver_email@gmail.com"

server = smtplib.SMTP("smtp.gmail.com: 587")
server.starttls()
server.login(from_email, password)

def send_email(to_email, from_email, object_detected=1):
    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = "Security Alert"
    message_body = f"ALERT - {object_detected} objects have been detected!!"
    message.attach(MIMEText(message_body, "plain"))
    server.sendmail(from_email, to_email, message.as_string())

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.email_sent = False
        self.model = YOLO("yolov8n.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.idle_objects = {}

    def predict(self, im0):
        results = self.model(im0)
        return results

    def plot_bboxes(self, results, im0):
        class_ids = []
        annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return im0, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            # Update idle time for detected objects
            current_time = time.time()
            for cls in class_ids:
                if cls not in self.idle_objects:
                    self.idle_objects[cls] = current_time
                elif current_time - self.idle_objects[cls] > 60:  # 60 seconds threshold
                    if not self.email_sent:
                        send_email(to_email, from_email, len(class_ids))
                        self.email_sent = True
            else:
                self.email_sent = False

            cv2.imshow("YOLOv8 Detection", im0)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        server.quit()

detector = ObjectDetection(capture_index=0)
detector()
