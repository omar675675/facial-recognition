#!/usr/bin/env python3
"""
5-point face alignment for ArcFace (w600k_r50).

ArcFace embeddings are only comparable when every face is warped to the same
canonical 5-point template (eyes / nose / mouth-corners) via a similarity
transform. The *exact same* alignment must be applied at enrollment and at
recognition time — otherwise stored and live embeddings live in different
spaces and matching degrades.

Keypoints come from SCRFD (InsightFace `det_10g`): it emits 5 points in the
exact order ArcFace was trained on, and its onnxruntime session is thread-safe
so one shared instance serves the whole worker pool.

It runs on **CPU on purpose**. On GPU it's faster per-call (~2ms) but it then
fights the dGPU that's already saturated with decode + detect + embed — at ~10
streams that collapses alignment to 40–250ms AND collides with the detector's
CUDA-graph capture (CUDA error 906). On CPU (~5–9ms/face, ~600/s across the
pool) it stays off the dGPU entirely: no contention, no stream conflict.
"""

from pathlib import Path

import cv2
import numpy as np

# InsightFace / ArcFace canonical 5-point template for a 112x112 crop.
# Order matches SCRFD's kps output: left-eye, right-eye, nose, left-mouth,
# right-mouth (where "left"/"right" mean the LEFT/RIGHT side of the image).
_ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],   # left eye  (image-left)
    [73.5318, 51.5014],   # right eye (image-right)
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner  (image-left)
    [70.7299, 92.2041],   # right mouth corner (image-right)
], dtype=np.float32)

_SCRFD_MODEL = Path.home() / ".insightface" / "models" / "buffalo_l" / "det_10g.onnx"
_SCRFD_INPUT = 160   # det input size; face crops letterbox into this. 160 ≈ 2ms, 128 ≈ 1.7ms.


def load_scrfd(input_size: int = _SCRFD_INPUT):
    """Load SCRFD det_10g on CPU (one shared, thread-safe instance).

    intra_op_num_threads=1 so each concurrent detect() uses a single core —
    parallelism comes from the worker pool, which avoids oversubscribing the
    CPU when many faces align at once. We pre-warm once so SCRFD's per-(stride)
    anchor cache is populated; after that concurrent detect() calls only read
    it, so there's no write race.
    """
    if not _SCRFD_MODEL.exists():
        raise FileNotFoundError(
            f"{_SCRFD_MODEL} not found. Fetch it once with:\n"
            "  python -c \"import insightface; insightface.app.FaceAnalysis(name='buffalo_l')\""
        )
    import onnxruntime as ort
    ort.set_default_logger_severity(3)   # silence SCRFD's 640-baked anchor-size warnings
    from insightface.model_zoo.scrfd import SCRFD
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(_SCRFD_MODEL), sess_options=so,
                                providers=["CPUExecutionProvider"])
    det = SCRFD(str(_SCRFD_MODEL), session=sess)
    det.prepare(ctx_id=-1, input_size=(input_size, input_size))   # ctx_id<0 → CPU
    det.detect(np.zeros((input_size, input_size, 3), np.uint8))    # warm the anchor cache
    return det


def align_from_5pt(crop_bgr: np.ndarray, pts5: np.ndarray) -> np.ndarray | None:
    """Warp a face crop to the 112x112 ArcFace template from its 5 points."""
    M, _ = cv2.estimateAffinePartial2D(pts5, _ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(crop_bgr, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderValue=0)


def align_face(crop_bgr: np.ndarray, det) -> np.ndarray | None:
    """Detect 5 SCRFD keypoints in the crop and return the aligned 112x112 face.

    Returns None if no face is found (tiny / extreme-angle / blurred) — callers
    must not feed an unaligned crop to the aligned DB.
    """
    _, kpss = det.detect(crop_bgr)
    if kpss is None or len(kpss) == 0:
        return None
    return align_from_5pt(crop_bgr, kpss[0].astype(np.float32))
