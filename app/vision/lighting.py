import cv2
import numpy as np
from app.config import TARGET_MEAN_Y

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance_lighting(frame_bgr):
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    mean_y = float(np.mean(y))
    diff = TARGET_MEAN_Y - mean_y
    alpha = np.clip(1.0 + diff / 255.0, 0.7, 1.8)
    beta = np.clip(diff * 0.8, -60, 60)

    y = cv2.convertScaleAbs(y, alpha=alpha, beta=beta)
    y = CLAHE.apply(y)

    out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    return out, mean_y