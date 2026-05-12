from ultralytics import YOLO
from app.config import YOLO_WEIGHTS
from app.constants import YOLO_TARGETS

yolo_model = YOLO(YOLO_WEIGHTS)
print("YOLO loaded successfully")

def run_yolo(frame):
    detections = []
    results = yolo_model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in YOLO_TARGETS:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((YOLO_TARGETS[cls_id], conf, (x1, y1, x2, y2)))

    return detections