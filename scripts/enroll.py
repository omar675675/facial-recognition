#!/usr/bin/env python3
"""
Enrollment: video → face crops (rtsp_trt) → angle buckets (MediaPipe) → ArcFace embeddings → PostgreSQL + Qdrant

Usage:
    python scripts/enroll.py    # enrolls everyone in enrollment_videos/
"""

import sys
import tempfile
import os
import contextlib
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import onnxruntime as ort
import psycopg2
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

# ── paths ──────────────────────────────────────────────────────────────────────

REPO_DIR       = Path(__file__).resolve().parent.parent
RTSP_TRT_DIR   = REPO_DIR.parent / "rtsp_trt"
ENROLLMENT_DIR = REPO_DIR / "enrollment_videos"
MODEL_PATH     = REPO_DIR / "models" / "w600k_r50.onnx"
RTSP_TRT_CFG   = RTSP_TRT_DIR / "config.yaml"
ENROLL_CFG     = REPO_DIR / "config.yaml"

sys.path.insert(0, str(RTSP_TRT_DIR.parent))
from rtsp_trt import Pipeline, DetectionEvent  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import align_face, load_scrfd  # noqa: E402

# ── settings ───────────────────────────────────────────────────────────────────

PG_DSN            = "postgresql://fr:fr@localhost:5432/fr"
QDRANT_URL        = "http://localhost:6333"
QDRANT_COLLECTION = "faces"
EMBEDDING_DIM     = 512

# ── stdout suppressor (silences C-extension engine logs) ──────────────────────

@contextlib.contextmanager
def _suppress_fd1():
    fd = sys.stdout.fileno()
    saved = os.dup(fd)
    null  = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, fd)
    os.close(null)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, fd)
        os.close(saved)

# ── head-pose (MediaPipe FaceLandmarker + solvePnP) ───────────────────────────

_LANDMARKER_MODEL = REPO_DIR / "models" / "face_landmarker.task"
_LANDMARKER_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

BUCKETS = ["front", "left", "right", "up", "down", "up_right", "up_left", "down_right", "down_left"]

_BUCKET_SCORE = {
    "front":      lambda y, p: -(abs(y) + abs(p)),
    "left":       lambda y, p: -y,
    "right":      lambda y, p:  y,
    "up":         lambda y, p: -p,
    "down":       lambda y, p:  p,
    "up_right":   lambda y, p:  y - p,
    "up_left":    lambda y, p: -y - p,
    "down_right": lambda y, p:  y + p,
    "down_left":  lambda y, p: -y + p,
}


def load_pose_estimator() -> mp_vision.FaceLandmarker:
    if not _LANDMARKER_MODEL.exists():
        print(f"[mediapipe] downloading face_landmarker.task …")
        urllib.request.urlretrieve(_LANDMARKER_URL, str(_LANDMARKER_MODEL))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_LANDMARKER_MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def head_pose(crop_bgr: np.ndarray, landmarker: mp_vision.FaceLandmarker) -> tuple[float | None, float | None]:
    """Return (yaw_deg, pitch_deg) or (None, None) if no face detected.
    Uses MediaPipe's own facial transformation matrix — no custom solvePnP."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks or not res.facial_transformation_matrixes:
        return None, None
    R = np.array(res.facial_transformation_matrixes[0])[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
    return float(yaw), float(pitch)


def angle_bucket(yaw: float, pitch: float, yaw_thresh: float, pitch_thresh: float) -> str | None:
    """Map (yaw, pitch) to one of the 8 bucket names, or None if unclassified."""
    is_left  = yaw   < -yaw_thresh
    is_right = yaw   >  yaw_thresh
    is_up    = pitch < -pitch_thresh
    is_down  = pitch >  pitch_thresh

    if is_right and is_up:   return "up_right"
    if is_left  and is_up:   return "up_left"
    if is_right and is_down: return "down_right"
    if is_left  and is_down: return "down_left"
    if is_right:             return "right"
    if is_left:              return "left"
    if is_down:              return "down"
    if is_up:                return "up"
    return "front"

# ── ArcFace recognizer ─────────────────────────────────────────────────────────

def load_recognizer() -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(MODEL_PATH), providers=providers)


def extract_embedding(sess: ort.InferenceSession, crop_bgr: np.ndarray) -> np.ndarray:
    """112×112 BGR → 512-d L2-normalised float32 embedding."""
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 127.5
    img = img.transpose(2, 0, 1)[np.newaxis]
    emb = sess.run(None, {sess.get_inputs()[0].name: img})[0][0]
    return emb / np.linalg.norm(emb)


def augment_crops(crop: np.ndarray, aug_cfg: dict) -> list[tuple[str, np.ndarray]]:
    """Return (name, image) pairs: original + one variant per augmentation."""
    variants = [("original", crop)]

    if "blur" in aug_cfg:
        img = cv2.GaussianBlur(crop.copy(), (0, 0), float(aug_cfg["blur"]["sigma"]))
        variants.append(("blur", img))

    if "jpeg" in aug_cfg:
        q = int(aug_cfg["jpeg"]["quality"])
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, q])
        variants.append(("jpeg", cv2.imdecode(buf, cv2.IMREAD_COLOR)))

    if "brightness" in aug_cfg:
        alpha = float(aug_cfg["brightness"]["contrast"])
        beta  = float(aug_cfg["brightness"]["shift"])
        img = np.clip(alpha * crop.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        variants.append(("brightness", img))

    if "gamma" in aug_cfg:
        gamma = float(aug_cfg["gamma"]["gamma"])
        lut = (np.arange(256, dtype=np.float32) / 255) ** gamma * 255
        variants.append(("gamma", lut.astype(np.uint8)[crop]))

    if "rotation" in aug_cfg:
        angle = float(aug_cfg["rotation"]["max_angle"])
        M = cv2.getRotationMatrix2D((56, 56), angle, 1.0)
        img = cv2.warpAffine(crop.copy(), M, (112, 112), borderMode=cv2.BORDER_REFLECT)
        variants.append(("rotation", img))

    if "noise" in aug_cfg:
        sigma = float(aug_cfg["noise"]["sigma"])
        noise = np.random.normal(0, sigma, crop.shape).astype(np.float32)
        img = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        variants.append(("noise", img))

    if "perspective" in aug_cfg:
        dx = float(aug_cfg["perspective"]["strength"]) * 112
        src = np.float32([[0, 0], [112, 0], [112, 112], [0, 112]])
        dst = np.float32([[dx, 0], [112 - dx, 0], [112, 112], [0, 112]])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(crop.copy(), M, (112, 112), borderMode=cv2.BORDER_REFLECT)
        variants.append(("perspective", img))

    if "flip" in aug_cfg:
        variants.append(("flip", cv2.flip(crop, 1)))

    if "grayscale" in aug_cfg:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variants.append(("grayscale", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

    if "hue_sat" in aug_cfg:
        hue_shift = float(aug_cfg["hue_sat"]["hue_shift"])
        sat_scale = float(aug_cfg["hue_sat"]["saturation_scale"])
        img = cv2.cvtColor(crop.copy(), cv2.COLOR_BGR2HSV).astype(np.float32)
        img[:, :, 0] = (img[:, :, 0] + hue_shift) % 180
        img[:, :, 1] = np.clip(img[:, :, 1] * (1 + sat_scale), 0, 255)
        variants.append(("hue_sat", cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2BGR)))

    if "cutout" in aug_cfg:
        h_cut = int(aug_cfg["cutout"]["height"])
        img = crop.copy()
        img[112 - h_cut:, :] = 0
        variants.append(("cutout", img))

    if "downscale" in aug_cfg:
        scale = float(aug_cfg["downscale"]["scale"])
        small = max(1, int(112 * scale))
        img = cv2.resize(cv2.resize(crop, (small, small)), (112, 112))
        variants.append(("downscale", img))

    if "motion_blur" in aug_cfg:
        k = int(aug_cfg["motion_blur"]["kernel_size"])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        variants.append(("motion_blur", cv2.filter2D(crop.copy(), -1, kernel)))

    if "histogram_eq" in aug_cfg:
        img = cv2.cvtColor(crop.copy(), cv2.COLOR_BGR2YUV)
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        variants.append(("histogram_eq", cv2.cvtColor(img, cv2.COLOR_YUV2BGR)))

    return variants

# ── database ───────────────────────────────────────────────────────────────────

def ensure_pg_schema(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id          SERIAL PRIMARY KEY,
                name        TEXT NOT NULL UNIQUE,
                enrolled_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
    conn.commit()


def upsert_person(conn, name: str) -> int:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO persons (name)
            VALUES (%s)
            ON CONFLICT (name) DO UPDATE SET enrolled_at = NOW()
            RETURNING id
        """, (name,))
        pid = cur.fetchone()[0]
    conn.commit()
    return pid


def ensure_qdrant_collection(client: QdrantClient):
    names = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in names:
        client.create_collection(
            QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def replace_person_points(client: QdrantClient, person_id: int,
                          person_name: str, embeddings: list):
    """Store one searchable point per template (payload = name only).

    Qdrant indexes and searches the vectors itself, so recognition never has
    to ship embedding matrices back over the wire or rerank in Python.
    """
    # Drop this person's previous templates, then insert the fresh set.
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(must=[
            FieldCondition(key="person_id", match=MatchValue(value=person_id))
        ]),
    )
    points = [
        PointStruct(
            id=person_id * 100_000 + i,
            vector=emb.tolist(),
            payload={"person_id": person_id, "person_name": person_name},
        )
        for i, emb in enumerate(embeddings)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)

# ── enrollment logic ───────────────────────────────────────────────────────────

def collect_crops(video_path: Path, yaw_thresh: float, pitch_thresh: float,
                  landmarker: mp_vision.FaceLandmarker, det,
                  engine_conf: float, engine_nms: float,
                  crop_padding: float = 0.0,
                  blur_thresh: float = 0.0) -> dict:
    """Run rtsp_trt on the video; return one 112×112 crop per angle bucket.
    Keeps the most extreme (best-representative) frame per bucket."""
    candidates: dict[str, tuple[float, np.ndarray]] = {}

    def on_detection(d: DetectionEvent):
        if d.conf == 0:
            return
        x, y, w, h = int(d.x), int(d.y), int(d.w), int(d.h)
        fh, fw = d.frame.shape[:2]

        pose_crop = d.frame[y:y+h, x:x+w]
        if min(pose_crop.shape[:2]) < 4:
            return
        pose_112 = cv2.resize(pose_crop, (112, 112))
        gray = cv2.cvtColor(pose_112, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_thresh:
            return
        yaw, pitch = head_pose(pose_112, landmarker)
        if yaw is None:
            return

        bucket = angle_bucket(yaw, pitch, yaw_thresh, pitch_thresh)
        print(f"  [debug] yaw={yaw:+.1f}  pitch={pitch:+.1f}  → {bucket or 'none'}", file=sys.stderr)
        if not bucket:
            return

        pad_x = int(w * crop_padding)
        pad_y = int(h * crop_padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(fw, x + w + pad_x)
        y2 = min(fh, y + h + pad_y)
        # Keep the full-res box crop; alignment (which needs sharp landmarks)
        # runs once per winning bucket after the pass, not per frame.
        crop = d.frame[y1:y2, x1:x2].copy()

        score = _BUCKET_SCORE[bucket](yaw, pitch)
        if bucket not in candidates or score > candidates[bucket][0]:
            candidates[bucket] = (score, crop)

    cfg = yaml.safe_load(RTSP_TRT_CFG.read_text())
    cfg["streams"] = [str(video_path)]
    cfg["display"] = False
    cfg.setdefault("detection", {})
    cfg["detection"]["conf_threshold"] = engine_conf
    cfg["detection"]["nms_threshold"]  = engine_nms

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp = f.name

    try:
        with _suppress_fd1():
            p = Pipeline(tmp)
            p.set_on_detection(on_detection)
            p.run()
            del p
    finally:
        os.unlink(tmp)

    aligned = {}
    for b, (_, box_crop) in candidates.items():
        crop = align_face(box_crop, det)
        if crop is None:
            print(f"  [warn] alignment failed for bucket {b}; skipping", file=sys.stderr)
            continue
        aligned[b] = crop
    return aligned


def enroll_video(video_path: Path, sess: ort.InferenceSession,
                 pg_conn, qdrant: QdrantClient,
                 yaw_thresh: float, pitch_thresh: float,
                 landmarker: mp_vision.FaceLandmarker, det,
                 engine_conf: float, engine_nms: float,
                 aug_cfg: dict | None = None,
                 crop_padding: float = 0.0,
                 blur_thresh: float = 0.0):
    aug_cfg  = aug_cfg or {}
    name     = video_path.stem.replace("_", " ").title()
    buckets  = collect_crops(video_path, yaw_thresh, pitch_thresh, landmarker, det,
                             engine_conf, engine_nms, crop_padding, blur_thresh)

    captured    = list(buckets.keys())
    missing     = [b for b in BUCKETS if b not in buckets]
    missing_str = "  missing: " + ", ".join(missing) if missing else ""
    print(f"{name:<20} captured: {', '.join(captured) or 'none'}{missing_str}")

    if not buckets:
        return

    embeddings = []
    for bucket, crop in buckets.items():
        for _, aug_crop in augment_crops(crop, aug_cfg):
            embeddings.append(extract_embedding(sess, aug_crop))

    person_id = upsert_person(pg_conn, name)
    replace_person_points(qdrant, person_id, name, embeddings)

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    cfg          = yaml.safe_load(ENROLL_CFG.read_text())
    enroll       = cfg["enrollment"]
    yaw_thresh   = float(enroll["yaw_threshold"])
    pitch_thresh = float(enroll["pitch_threshold"])
    engine_conf  = float(enroll["engine"]["detection"]["conf_threshold"])
    engine_nms   = float(enroll["engine"]["detection"]["nms_threshold"])
    crop_padding = float(enroll.get("crop_padding", 0.0))
    blur_thresh  = float(enroll.get("blur_threshold", 0.0))
    aug_cfg      = enroll["augmentation"]

    exts  = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI", "*.mkv")
    paths = [p for ext in exts for p in sorted(ENROLLMENT_DIR.glob(ext))]

    if not paths:
        print("No videos found.")
        return

    sess       = load_recognizer()
    landmarker = load_pose_estimator()   # MediaPipe — pose bucketing only (offline, speed irrelevant)
    det        = load_scrfd()            # SCRFD — alignment, identical to recognition
    pg_conn    = psycopg2.connect(PG_DSN)
    qdrant     = QdrantClient(url=QDRANT_URL, prefer_grpc=True)

    ensure_pg_schema(pg_conn)
    ensure_qdrant_collection(qdrant)

    for p in paths:
        enroll_video(p, sess, pg_conn, qdrant, yaw_thresh, pitch_thresh,
                     landmarker, det, engine_conf, engine_nms, aug_cfg, crop_padding, blur_thresh)

    pg_conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
