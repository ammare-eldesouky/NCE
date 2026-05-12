import numpy as np

YOLO_TARGETS = {
    67: "cell phone",
    73: "book",
}

MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,   -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

CHEATING_DESCRIPTIONS = {
    "NO_FACE": "Student hid face or moved away from camera",
    "MULTIPLE_FACES": "More than one face appeared in front of the camera",
    "HEAD_TURN_LEFT": "Student looked far left attempting to cheat",
    "HEAD_TURN_RIGHT": "Student looked far right attempting to cheat",
    "HEAD_TURN_DOWN": "Student looked down attempting to cheat",
    "HEAD_TURN_UP": "Student looked up attempting to cheat",
    "PHONE_DETECTED": "Student is holding a phone to cheat",
    "BOOK_DETECTED": "Student is holding a book to cheat",
    "PERSON_DETECTED": "Another person appeared in the camera assisting with cheating",
}

WARNING_REASON_MAP = {
    "NO_FACE": "Face not visible!",
    "MULTIPLE_FACES": "Multiple faces detected!",
    "HEAD_TURN_LEFT": "Head turned away!",
    "HEAD_TURN_RIGHT": "Head turned away!",
    "HEAD_TURN_DOWN": "Head turned away!",
    "HEAD_TURN_UP": "Head turned away!",
    "PHONE_DETECTED": "Phone detected!",
    "BOOK_DETECTED": "Book detected!",
}