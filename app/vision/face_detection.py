"""
Face detection using mediapipe Tasks API (0.10.30+).
Wraps the new API to stay compatible with the old mp.solutions interface
so monitor.py doesn't need any changes.
"""
from __future__ import annotations

import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ── Model download ────────────────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_detector.tflite")


def _ensure_model() -> None:
    if not os.path.exists(_MODEL_PATH):
        print("[NCE AI] Downloading face_detector.tflite …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[NCE AI] face_detector.tflite ready.")


_ensure_model()

# ── Detector instance ─────────────────────────────────────────────────────────
_detector = mp_vision.FaceDetector.create_from_options(
    mp_vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
    )
)


# ── Compatibility wrappers ────────────────────────────────────────────────────

class _RelBBox:
    """Mirrors mp.solutions relative_bounding_box (normalised 0-1)."""
    def __init__(self, xmin: float, ymin: float, width: float, height: float):
        self.xmin   = xmin
        self.ymin   = ymin
        self.width  = width
        self.height = height


class _Detection:
    def __init__(self, task_detection, img_w: int, img_h: int):
        bb = task_detection.bounding_box          # pixel-absolute in Tasks API
        self.location_data = type("LD", (), {
            "relative_bounding_box": _RelBBox(
                xmin   = bb.origin_x / img_w,
                ymin   = bb.origin_y / img_h,
                width  = bb.width    / img_w,
                height = bb.height   / img_h,
            )
        })()


class _DetectionResult:
    def __init__(self, detections: list):
        self.detections = detections   # empty list → same as old API returning None


# ── Public API (same signature as old code) ───────────────────────────────────

class _FaceDetectorCompat:
    """Dummy object so api.py can call face_detector.close()."""
    def close(self): pass


face_detector = _FaceDetectorCompat()


def detect_faces(rgb_frame) -> _DetectionResult:
    """Drop-in replacement for the old mediapipe solutions detect call."""
    h, w = rgb_frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result   = _detector.detect(mp_image)
    compat   = [_Detection(d, w, h) for d in (result.detections or [])]
    return _DetectionResult(compat)