#!/usr/bin/env python3
"""
Enrollment: video → face crops (rtsp_trt) → angle buckets (MediaPipe) → ArcFace embeddings → PostgreSQL + Qdrant

Usage:
    python scripts/enroll.py    # enrolls everyone in enrollment_videos/
"""

import sys
import tempfile
import os
from pathlib import Path

import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import onnxruntime as ort
import psycopg2
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── paths ──────────────────────────────────────────────────────────────────────

REPO_DIR       = Path(__file__).resolve().parent.parent
RTSP_TRT_DIR   = REPO_DIR.parent / "rtsp_trt"
ENROLLMENT_DIR = REPO_DIR / "enrollment_videos"
EMBEDDINGS_DIR = REPO_DIR / "embeddings"
MODEL_PATH     = REPO_DIR / "models" / "w600k_r50.onnx"
RTSP_TRT_CFG   = RTSP_TRT_DIR / "config.yaml"
ENROLL_CFG     = REPO_DIR / "config.yaml"

sys.path.insert(0, str(RTSP_TRT_DIR.parent))
from rtsp_trt import Pipeline, DetectionEvent  # noqa: E402

# ── settings ───────────────────────────────────────────────────────────────────

PG_DSN            = "postgresql://fr:fr@localhost:5432/fr"
QDRANT_URL        = "http://localhost:6333"
QDRANT_COLLECTION = "faces"
EMBEDDING_DIM     = 512

# ── head-pose (MediaPipe Tasks + solvePnP) ────────────────────────────────────

# Generic 3-D face model points (mm) matched to _LANDMARK_IDS
_FACE_3D = np.array([
    [  0.0,    0.0,    0.0],   # nose tip        (1)
    [  0.0, -330.0,  -65.0],   # chin            (152)
    [-225.0,  170.0, -135.0],  # left eye outer  (263)
    [ 225.0,  170.0, -135.0],  # right eye outer (33)
    [-150.0, -150.0, -125.0],  # left mouth      (287)
    [ 150.0, -150.0, -125.0],  # right mouth     (57)
], dtype=np.float64)

_LANDMARK_IDS  = [1, 152, 263, 33, 287, 57]
_LANDMARKER_MODEL = REPO_DIR / "models" / "face_landmarker.task"
_LANDMARKER_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _load_face_landmarker(min_face_presence_score: float = 0.5):
    if not _LANDMARKER_MODEL.exists():
        print(f"[mediapipe] downloading face_landmarker.task …")
        urllib.request.urlretrieve(_LANDMARKER_URL, str(_LANDMARKER_MODEL))
    try:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_LANDMARKER_MODEL)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_presence_score=min_face_presence_score,
        )
    except TypeError:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_LANDMARKER_MODEL)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
    return mp_vision.FaceLandmarker.create_from_options(opts)


BUCKETS = ["front", "left", "right", "down", "up_right", "up_left", "down_right", "down_left"]


def head_pose(crop_bgr: np.ndarray, landmarker: mp_vision.FaceLandmarker):
    """Return (yaw_deg, pitch_deg) or (None, None) if no face detected."""
    h, w = crop_bgr.shape[:2]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if not result.face_landmarks:
        return None, None

    lm = result.face_landmarks[0]
    pts2d = np.array([[lm[i].x * w, lm[i].y * h] for i in _LANDMARK_IDS], dtype=np.float64)

    f = float(w)
    cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_FACE_3D, pts2d, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None

    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
    return yaw, pitch


def angle_bucket(yaw: float, pitch: float, yaw_thresh: float, pitch_thresh: float):
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
    if not is_up:            return "front"
    return None  # pure up — no bucket defined

# ── ArcFace recognizer ─────────────────────────────────────────────────────────

def load_recognizer() -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(MODEL_PATH), providers=providers)
    print(f"[recognizer] loaded {MODEL_PATH.name} ({sess.get_providers()[0]})")
    return sess


def extract_embedding(sess: ort.InferenceSession, crop_bgr: np.ndarray) -> np.ndarray:
    """112×112 BGR uint8 → 512-d L2-normalised float32 embedding."""
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


def replace_qdrant_point(client: QdrantClient, person_id: int,
                          person_name: str, embeddings: list,
                          anchor_vec: np.ndarray):
    matrix = np.stack(embeddings)
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=person_id,
                vector=anchor_vec.tolist(),
                payload={
                    "person_id": person_id,
                    "person_name": person_name,
                    "embeddings": matrix.tolist(),
                },
            )
        ],
    )

# ── enrollment logic ───────────────────────────────────────────────────────────

def collect_crops(video_path: Path, yaw_thresh: float, pitch_thresh: float,
                  landmarker: mp_vision.FaceLandmarker,
                  engine_conf: float, engine_nms: float) -> dict:
    """Run rtsp_trt on the video; return one 112×112 crop per angle bucket."""
    buckets: dict[str, np.ndarray] = {}

    def on_detection(d: DetectionEvent):
        if len(buckets) == len(BUCKETS):
            return
        x, y, w, h = int(d.x), int(d.y), int(d.w), int(d.h)
        crop = d.frame[y:y+h, x:x+w]
        if min(crop.shape[:2]) < 4:
            return
        crop = cv2.resize(crop, (112, 112))

        yaw, pitch = head_pose(crop, landmarker)
        if yaw is None:
            return

        bucket = angle_bucket(yaw, pitch, yaw_thresh, pitch_thresh)
        if bucket and bucket not in buckets:
            buckets[bucket] = crop

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
        p = Pipeline(tmp)
        p.set_on_detection(on_detection)
        p.run()
    finally:
        os.unlink(tmp)

    return buckets


def enroll_video(video_path: Path, sess: ort.InferenceSession,
                 pg_conn, qdrant: QdrantClient,
                 yaw_thresh: float, pitch_thresh: float,
                 landmarker: mp_vision.FaceLandmarker,
                 engine_conf: float, engine_nms: float,
                 aug_cfg: dict | None = None):
    aug_cfg = aug_cfg or {}

    name = video_path.stem.replace("_", " ").title()
    print(f"\n── {name}  ({video_path.name})")

    print("  [1/3] collecting crops by angle...")
    buckets = collect_crops(video_path, yaw_thresh, pitch_thresh, landmarker,
                            engine_conf, engine_nms)
    captured = list(buckets.keys())
    missing  = [b for b in BUCKETS if b not in buckets]
    print(f"        captured: {captured}")
    if missing:
        print(f"        missing:  {missing}")

    if not buckets:
        print("        no faces detected — skipping")
        return

    crop_dir = EMBEDDINGS_DIR / video_path.stem
    crop_dir.mkdir(parents=True, exist_ok=True)

    print("  [2/3] extracting embeddings...")
    embeddings = []
    total_saved = 0
    for bucket, crop in buckets.items():
        variants = augment_crops(crop, aug_cfg)
        for aug_name, aug_crop in variants:
            fname = f"{bucket}.jpg" if aug_name == "original" else f"{bucket}_{aug_name}.jpg"
            cv2.imwrite(str(crop_dir / fname), aug_crop)
            total_saved += 1
            embeddings.append(extract_embedding(sess, aug_crop))
    aug_per_crop = len(embeddings) // len(buckets) if buckets else 0
    print(f"        saved {total_saved} crops → {crop_dir}")
    print(f"        {len(embeddings)} embeddings ({len(buckets)} crops × {aug_per_crop} variants)")

    print("  [3/3] storing...")
    bucket_names  = list(buckets.keys())
    aug_per_crop  = len(embeddings) // len(buckets)
    front_idx     = bucket_names.index("front") if "front" in bucket_names else 0
    anchor_vec    = embeddings[front_idx * aug_per_crop]

    person_id = upsert_person(pg_conn, name)
    replace_qdrant_point(qdrant, person_id, name, embeddings, anchor_vec)
    print(f"        PostgreSQL id={person_id}  |  1 point ({len(embeddings)} embeddings) → Qdrant  ✓")

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    cfg          = yaml.safe_load(ENROLL_CFG.read_text())
    enroll       = cfg["enrollment"]
    yaw_thresh   = float(enroll["yaw_threshold"])
    pitch_thresh = float(enroll["pitch_threshold"])
    engine_conf  = float(enroll["engine"]["detection"]["conf_threshold"])
    engine_nms   = float(enroll["engine"]["detection"]["nms_threshold"])
    mp_presence  = float(enroll["mediapipe"]["min_face_presence_score"])
    aug_cfg      = enroll["augmentation"]
    print(f"[config] yaw={yaw_thresh}°  pitch={pitch_thresh}°  "
          f"engine_conf={engine_conf}  engine_nms={engine_nms}  "
          f"mp_presence={mp_presence}  augmentations={list(aug_cfg.keys())}")

    exts  = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI", "*.mkv")
    paths = [p for ext in exts for p in sorted(ENROLLMENT_DIR.glob(ext))]

    if not paths:
        print("No videos found.")
        return

    sess       = load_recognizer()
    landmarker = _load_face_landmarker(mp_presence)
    pg_conn    = psycopg2.connect(PG_DSN)
    qdrant     = QdrantClient(url=QDRANT_URL)

    ensure_pg_schema(pg_conn)
    ensure_qdrant_collection(qdrant)

    for p in paths:
        enroll_video(p, sess, pg_conn, qdrant, yaw_thresh, pitch_thresh,
                     landmarker, engine_conf, engine_nms, aug_cfg)

    pg_conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
