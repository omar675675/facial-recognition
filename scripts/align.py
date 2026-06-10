#!/usr/bin/env python3
"""
5-point face alignment for ArcFace (w600k_r50).

ArcFace embeddings are only comparable when every face is warped to the same
canonical 5-point template (eyes / nose / mouth-corners) via a similarity
transform. The *exact same* alignment must be applied at enrollment and at
recognition time — otherwise stored and live embeddings live in different
spaces and matching degrades.

The detector exposes no keypoints, so we derive the 5 points from MediaPipe
FaceLandmarker (already a dependency) and warp to InsightFace's standard
112x112 template.
"""

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_LANDMARKER_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def load_landmarker() -> mp_vision.FaceLandmarker:
    """Create a single-face FaceLandmarker in IMAGE mode.

    Not thread-safe — give each worker thread its own instance.
    """
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_LANDMARKER_MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)

# InsightFace / ArcFace canonical 5-point template for a 112x112 crop.
# Order: left-eye, right-eye, nose, left-mouth, right-mouth — where
# "left" / "right" mean the LEFT / RIGHT side of the image.
_ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],   # left eye  (image-left)
    [73.5318, 51.5014],   # right eye (image-right)
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner  (image-left)
    [70.7299, 92.2041],   # right mouth corner (image-right)
], dtype=np.float32)

# MediaPipe FaceLandmarker indices reduced to the 5 ArcFace points.
# Eye centres are the midpoint of (outer, inner) corners. Image-left points
# correspond to the subject's right-side features (frontal convention).
_MP_5PT = (
    (33, 133),    # left eye  (image-left)  = subject's right eye
    (362, 263),   # right eye (image-right) = subject's left eye
    (1,),         # nose tip
    (61,),        # left mouth corner  (image-left)
    (291,),       # right mouth corner (image-right)
)


def landmarks_5pt(crop_bgr: np.ndarray, landmarker) -> np.ndarray | None:
    """Return a 5x2 float32 array of (x, y) pixel coords, or None if no face."""
    h, w = crop_bgr.shape[:2]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    pts = np.empty((5, 2), dtype=np.float32)
    for i, idxs in enumerate(_MP_5PT):
        pts[i, 0] = np.mean([lm[j].x for j in idxs]) * w
        pts[i, 1] = np.mean([lm[j].y for j in idxs]) * h
    return pts


def align_from_5pt(crop_bgr: np.ndarray, pts5: np.ndarray) -> np.ndarray | None:
    """Warp a face crop to the 112x112 ArcFace template from its 5 points."""
    M, _ = cv2.estimateAffinePartial2D(pts5, _ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(crop_bgr, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderValue=0)


def align_face(crop_bgr: np.ndarray, landmarker) -> np.ndarray | None:
    """Detect 5 landmarks and return the aligned 112x112 crop, or None."""
    pts = landmarks_5pt(crop_bgr, landmarker)
    if pts is None:
        return None
    return align_from_5pt(crop_bgr, pts)
