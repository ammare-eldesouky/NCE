"""
Head-pose estimation using mediapipe Tasks API (0.10.30+).
Wraps the new FaceLandmarker to stay compatible with the old
mp.solutions.face_mesh interface used in monitor.py.
"""
from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from app.constants import MODEL_POINTS, LANDMARK_IDS

# ── Model download ────────────────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def _ensure_model() -> None:
    if not os.path.exists(_MODEL_PATH):
        print("[NCE AI] Downloading face_landmarker.task …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[NCE AI] face_landmarker.task ready.")


_ensure_model()

# ── Landmarker instance ───────────────────────────────────────────────────────
_landmarker = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=4,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
)


# ── Compatibility wrappers ────────────────────────────────────────────────────

class _Landmark:
    __slots__ = ("x", "y", "z")
    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z


class _FaceLandmarks:
    def __init__(self, raw_landmarks):
        self.landmark = [_Landmark(lm.x, lm.y, lm.z) for lm in raw_landmarks]


class _MeshResult:
    """Mirrors the old FaceMesh result with .multi_face_landmarks."""
    def __init__(self, task_result):
        if task_result.face_landmarks:
            self.multi_face_landmarks = [
                _FaceLandmarks(face_lms)
                for face_lms in task_result.face_landmarks
            ]
        else:
            self.multi_face_landmarks = []


# ── Public API ────────────────────────────────────────────────────────────────

class _FaceMeshCompat:
    """Dummy so api.py can call face_mesh.close()."""
    def close(self): pass


face_mesh = _FaceMeshCompat()


def get_mesh_results(rgb_frame) -> _MeshResult:
    """Drop-in replacement for the old face_mesh.process() call."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result   = _landmarker.detect(mp_image)
    return _MeshResult(result)


def get_head_pose(landmarks, frame_w: int, frame_h: int):
    """Unchanged — works with the same landmark format."""
    image_points = np.array([
        (landmarks[i].x * frame_w, landmarks[i].y * frame_h)
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = frame_w
    center       = (frame_w / 2, frame_h / 2)

    cam_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1        ],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points,
        cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None, None, None

    rmat, _   = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2]