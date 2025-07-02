from ultralytics import YOLO
import cv2
import numpy as np

class YOLO:
       def __init__(self, model_path):
           self.model = YOLO(model_path)

       def detect(self, frame):
           if isinstance(frame, str):  # If frame is base64 data
               frame = cv2.imdecode(np.frombuffer(base64.b64decode(frame.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
           results = self.model(frame)[0]
           return results.boxes.xyxy if results.boxes else []