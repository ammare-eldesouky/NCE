CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
WINDOW_NAME = "NCE - Smart Exam Monitor"
TARGET_MEAN_Y = 130

YAW_THRESHOLD = 25
PITCH_THRESHOLD = 20
NO_FACE_TIMEOUT = 5.0
MULTI_FACE_TIMEOUT = 5.0
HEAD_TURN_TIMEOUT = 5.0
OBJECT_TIMEOUT = 5.0
MAX_WARNINGS = 3

USE_YOLO = True
YOLO_WEIGHTS = "yolov8n.pt"

CSV_FILE = "nce_violations.csv"

# local testing defaults
STUDENT_ID = "STU_001"
SESSION_ID = "sess_local_001"
EXAM_ID = "EXAM_001"
COURSE_ID = "COURSE_001"

# backend callback endpoints
BACKEND_URL = "http://127.0.0.1:3000"
LIVE_VIOLATION_ENDPOINT = "/violations/live"
FINAL_REPORT_ENDPOINT = "/reports/final"