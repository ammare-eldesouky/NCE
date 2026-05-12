import cv2
from app.config import MAX_WARNINGS

def draw_label(frame, text, pos, color=(0, 255, 0), scale=0.6, thickness=2):
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness
    )

def draw_warning_overlay(frame, warning_num, reason):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    cv2.putText(
        frame,
        f"WARNING {warning_num}/{MAX_WARNINGS}",
        (120, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 255),
        3
    )
    cv2.putText(
        frame,
        reason,
        (60, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

def draw_terminated_overlay(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(
        frame,
        "EXAM TERMINATED",
        (90, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 255),
        3
    )
    cv2.putText(
        frame,
        "Maximum warnings exceeded",
        (130, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2
    )