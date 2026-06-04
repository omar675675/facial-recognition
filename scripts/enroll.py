#!/usr/bin/env python3
"""
Enrollment: video → face crops (rtsp_trt) → ArcFace embeddings → PostgreSQL + Qdrant

Usage:
    python scripts/enroll.py    # enrolls everyone in enrollment_videos/
"""

import sys
import tempfile
import os
from pathlib import Path

import cv2
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
MODEL_PATH     = REPO_DIR / "models" / "w600k_r50.onnx"
RTSP_TRT_CFG   = RTSP_TRT_DIR / "config.yaml"

sys.path.insert(0, str(RTSP_TRT_DIR.parent))
from rtsp_trt import Pipeline, DetectionEvent  # noqa: E402

# ── settings ───────────────────────────────────────────────────────────────────

PG_DSN            = "postgresql://fr:fr@localhost:5432/fr"
QDRANT_URL        = "http://localhost:6333"
QDRANT_COLLECTION = "faces"
EMBEDDING_DIM     = 512

# ── ArcFace recognizer ─────────────────────────────────────────────────────────

def load_recognizer() -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(MODEL_PATH), providers=providers)
    print(f"[recognizer] loaded {MODEL_PATH.name} ({sess.get_providers()[0]})")
    return sess


def extract_embedding(sess: ort.InferenceSession, crop_bgr: np.ndarray) -> np.ndarray:
    """112×112 BGR uint8 → 512-d L2-normalised float32 embedding."""
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 127.5                 # [-1, 1]
    img = img.transpose(2, 0, 1)[np.newaxis]     # (1, 3, 112, 112)
    emb = sess.run(None, {sess.get_inputs()[0].name: img})[0][0]
    return emb / np.linalg.norm(emb)

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
                          person_name: str, embeddings: list):
    matrix = np.stack(embeddings)                    # (N, 512)
    centroid = matrix.mean(axis=0)
    centroid /= np.linalg.norm(centroid)             # re-normalise

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=person_id,
                vector=centroid.tolist(),
                payload={
                    "person_id": person_id,
                    "person_name": person_name,
                    "embeddings": matrix.tolist(),
                },
            )
        ],
    )

# ── enrollment logic ───────────────────────────────────────────────────────────

def collect_crops(video_path: Path) -> list:
    """Run rtsp_trt on the video and return all face crops."""
    crops = []

    def on_detection(d: DetectionEvent):
        x, y, w, h = int(d.x), int(d.y), int(d.w), int(d.h)
        crop = d.frame[y:y+h, x:x+w]
        if min(crop.shape[:2]) < 4:
            return
        crops.append(cv2.resize(crop, (112, 112)))

    cfg = yaml.safe_load(RTSP_TRT_CFG.read_text())
    cfg["streams"] = [str(video_path)]
    cfg["display"] = False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp = f.name

    try:
        p = Pipeline(tmp)
        p.set_on_detection(on_detection)
        p.run()
    finally:
        os.unlink(tmp)

    return crops


def enroll_video(video_path: Path, sess: ort.InferenceSession,
                 pg_conn, qdrant: QdrantClient):
    name = video_path.stem.replace("_", " ").title()
    print(f"\n── {name}  ({video_path.name})")

    print("  [1/3] collecting crops...")
    crops = collect_crops(video_path)
    print(f"        {len(crops)} crops from video")

    if not crops:
        print("        no faces detected — skipping")
        return

    print("  [2/3] extracting embeddings...")
    embeddings = [extract_embedding(sess, c) for c in crops]
    print(f"        {len(embeddings)} embeddings")

    print("  [3/3] storing...")
    person_id = upsert_person(pg_conn, name)
    replace_qdrant_point(qdrant, person_id, name, embeddings)
    print(f"        PostgreSQL id={person_id}  |  1 point ({len(embeddings)} embeddings) → Qdrant  ✓")

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    exts  = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI", "*.mkv")
    paths = [p for ext in exts for p in sorted(ENROLLMENT_DIR.glob(ext))]

    if not paths:
        print("No videos found.")
        return

    sess    = load_recognizer()
    pg_conn = psycopg2.connect(PG_DSN)
    qdrant  = QdrantClient(url=QDRANT_URL)

    ensure_pg_schema(pg_conn)
    ensure_qdrant_collection(qdrant)

    for p in paths:
        enroll_video(p, sess, pg_conn, qdrant)

    pg_conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
