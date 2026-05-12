from __future__ import annotations

import cv2
import time

from app.config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    YAW_THRESHOLD,
    PITCH_THRESHOLD,
    NO_FACE_TIMEOUT,
    MULTI_FACE_TIMEOUT,
    HEAD_TURN_TIMEOUT,
    OBJECT_TIMEOUT,
    USE_YOLO,
    CSV_FILE,
)
from app.constants import WARNING_REASON_MAP
from app.logger_csv import log_violation
from app.vision.lighting import enhance_lighting
from app.vision.face_detection import detect_faces
from app.vision.head_pose import get_mesh_results, get_head_pose
from app.vision.object_detection import run_yolo
from app.drawing.overlays import (
    draw_label,
    draw_warning_overlay,
    draw_terminated_overlay,
)


def _log_callback(student_id, session_id, violation_type):
    log_violation(CSV_FILE, student_id, session_id, violation_type)


def analyze_frame(frame, state):
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    enhanced, mean_y = enhance_lighting(frame)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    h, w, _ = enhanced.shape
    now = time.time()

    if state.terminated:
        draw_terminated_overlay(enhanced)
        return {
            "frame": enhanced,
            "warnings": state.warnings,
            "terminated": state.terminated,
            "violation": state.last_warning_type,
            "message": state.last_message,
            "face_count": 0,
            "mean_y": mean_y,
            "new_warning_added": False,
        }

    det_results = detect_faces(rgb)
    face_count = len(det_results.detections) if det_results.detections else 0
    current_violation = None
    new_warning_added = False

    if det_results.detections:
        for detection in det_results.detections:
            bbox = detection.location_data.relative_bounding_box
            bx = int(bbox.xmin * w)
            by = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            box_color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
            cv2.rectangle(enhanced, (bx, by), (bx + bw, by + bh), box_color, 2)

    if face_count == 0:
        if state.no_face_start is None:
            state.no_face_start = now
        elapsed = now - state.no_face_start
        remaining = NO_FACE_TIMEOUT - elapsed
        draw_label(enhanced, f"NO FACE  [{remaining:.1f}s]", (10, 40), (0, 0, 255))
        if elapsed >= NO_FACE_TIMEOUT:
            state.no_face_start = None
            current_violation = "NO_FACE"
            _, new_warning_added = state.add_warning(current_violation, _log_callback)
    else:
        state.no_face_start = None

    if face_count > 1:
        if state.multi_face_start is None:
            state.multi_face_start = now
        elapsed = now - state.multi_face_start
        remaining = MULTI_FACE_TIMEOUT - elapsed
        draw_label(
            enhanced,
            f"MULTIPLE FACES ({face_count})  [{remaining:.1f}s]",
            (10, 70),
            (0, 0, 255),
        )
        if elapsed >= MULTI_FACE_TIMEOUT:
            state.multi_face_start = None
            current_violation = "MULTIPLE_FACES"
            _, new_warning_added = state.add_warning(current_violation, _log_callback)
    else:
        state.multi_face_start = None

    if face_count == 1:
        draw_label(enhanced, "Face: OK", (10, 40), (0, 255, 0))

    mesh_results = get_mesh_results(rgb)
    head_violation = False

    if mesh_results.multi_face_landmarks:
        landmarks = mesh_results.multi_face_landmarks[0].landmark
        pitch, yaw, roll = get_head_pose(landmarks, w, h)

        if pitch is not None:
            direction = "FORWARD"
            color = (0, 255, 0)
            head_vtype = None

            if yaw < -YAW_THRESHOLD:
                direction, color, head_violation = "LEFT", (0, 165, 255), True
                head_vtype = "HEAD_TURN_LEFT"
            elif yaw > YAW_THRESHOLD:
                direction, color, head_violation = "RIGHT", (0, 165, 255), True
                head_vtype = "HEAD_TURN_RIGHT"
            elif pitch < -PITCH_THRESHOLD:
                direction, color, head_violation = "DOWN", (0, 165, 255), True
                head_vtype = "HEAD_TURN_DOWN"
            elif pitch > PITCH_THRESHOLD:
                direction, color, head_violation = "UP", (0, 165, 255), True
                head_vtype = "HEAD_TURN_UP"

            draw_label(enhanced, f"Head: {direction}", (10, 100), color)
            draw_label(
                enhanced,
                f"Y:{yaw:+.0f} P:{pitch:+.0f} R:{roll:+.0f}",
                (10, 125),
                (180, 180, 180),
                scale=0.5,
            )

            if head_violation:
                if state.head_turn_start is None:
                    state.head_turn_start = now
                elapsed = now - state.head_turn_start
                remaining = HEAD_TURN_TIMEOUT - elapsed
                draw_label(
                    enhanced,
                    f"Head turned [{remaining:.1f}s]",
                    (10, 150),
                    (0, 165, 255),
                )
                if elapsed >= HEAD_TURN_TIMEOUT:
                    state.head_turn_start = None
                    current_violation = head_vtype
                    _, new_warning_added = state.add_warning(current_violation, _log_callback)
            else:
                state.head_turn_start = None

    if USE_YOLO:
        yolo_dets = run_yolo(enhanced)
        det_labels = [label for label, _, _ in yolo_dets]

        for label, conf, (x1, y1, x2, y2) in yolo_dets:
            cv2.rectangle(enhanced, (x1, y1), (x2, y2), (0, 0, 255), 2)
            draw_label(
                enhanced,
                f"{label} {conf:.0%}",
                (x1, y1 - 8),
                (0, 0, 255),
                scale=0.55,
            )

        if "cell phone" in det_labels:
            if state.phone_start is None:
                state.phone_start = now
            elapsed = now - state.phone_start
            remaining = OBJECT_TIMEOUT - elapsed
            draw_label(enhanced, f"Phone detected [{remaining:.1f}s]", (10, 175), (0, 0, 255))
            if elapsed >= OBJECT_TIMEOUT:
                state.phone_start = None
                current_violation = "PHONE_DETECTED"
                _, new_warning_added = state.add_warning(current_violation, _log_callback)
        else:
            state.phone_start = None

        if "book" in det_labels:
            if state.book_start is None:
                state.book_start = now
            elapsed = now - state.book_start
            remaining = OBJECT_TIMEOUT - elapsed
            draw_label(enhanced, f"Book detected [{remaining:.1f}s]", (10, 200), (0, 0, 255))
            if elapsed >= OBJECT_TIMEOUT:
                state.book_start = None
                current_violation = "BOOK_DETECTED"
                _, new_warning_added = state.add_warning(current_violation, _log_callback)
        else:
            state.book_start = None

    if now < state.warning_display_until:
        reason = WARNING_REASON_MAP.get(state.last_warning_type, "Violation!")
        draw_warning_overlay(enhanced, state.warnings, reason)

    bar_color = (
        (0, 200, 0) if state.warnings == 0
        else (0, 165, 255) if state.warnings == 1
        else (0, 0, 255)
    )

    cv2.rectangle(enhanced, (0, h - 35), (w, h), (30, 30, 30), -1)
    draw_label(
        enhanced,
        f"Warnings: {state.warnings}/3   Light: {mean_y:.0f}   Faces: {face_count}",
        (10, h - 10),
        bar_color,
        scale=0.55,
    )

    if state.warnings == 0:
        state.last_message = "No warnings"

    return {
        "frame": enhanced,
        "warnings": state.warnings,
        "terminated": state.terminated,
        "violation": current_violation,
        "message": state.last_message,
        "face_count": face_count,
        "mean_y": mean_y,
        "new_warning_added": new_warning_added,
    }