#!/usr/bin/env python3
"""
Recognition: live face detection + identity matching with cv2 display.

Usage:
    python scripts/recognize.py              # uses rtsp_trt/config.yaml
    python scripts/recognize.py my.yaml
"""

import os
import sys
import time
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 — initialises the CUDA context
import yaml
from qdrant_client import QdrantClient

# ── paths ──────────────────────────────────────────────────────────────────────

REPO_DIR     = Path(__file__).resolve().parent.parent
RTSP_TRT_DIR = REPO_DIR.parent / "rtsp_trt"
MODEL_PATH   = REPO_DIR / "models" / "w600k_r50.engine"

sys.path.insert(0, str(RTSP_TRT_DIR.parent))
from rtsp_trt import Pipeline, StatsEvent, DetectionEvent  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import align_face, load_landmarker  # noqa: E402

# FaceLandmarker isn't thread-safe → one instance per worker thread.
_tls = threading.local()

def _get_landmarker():
    lm = getattr(_tls, "landmarker", None)
    if lm is None:
        lm = load_landmarker()
        _tls.landmarker = lm
    return lm

# ── settings ───────────────────────────────────────────────────────────────────

QDRANT_URL        = "http://localhost:6333"
QDRANT_COLLECTION = "faces"
ENROLL_CFG        = REPO_DIR / "config.yaml"

# ── recognizer ─────────────────────────────────────────────────────────────────

_IN_NAME  = "input.1"
_OUT_NAME = "683"

class Recognizer:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self._engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self._ctx    = self._engine.create_execution_context()
        self._ctx.set_input_shape(_IN_NAME, (1, 3, 112, 112))
        self._h_in   = cuda.pagelocked_empty((1, 3, 112, 112), np.float32)
        self._h_out  = cuda.pagelocked_empty((1, 512),         np.float32)
        self._d_in   = cuda.mem_alloc(self._h_in.nbytes)
        self._d_out  = cuda.mem_alloc(self._h_out.nbytes)
        self._stream = cuda.Stream()
        self._ctx.set_tensor_address(_IN_NAME,  int(self._d_in))
        self._ctx.set_tensor_address(_OUT_NAME, int(self._d_out))
        self._lock   = threading.Lock()
        print(f"[recognizer] {Path(engine_path).name}  TensorRT {trt.__version__}")

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5
        with self._lock:
            np.copyto(self._h_in[0], img.transpose(2, 0, 1))
            cuda.memcpy_htod_async(self._d_in, self._h_in, self._stream)
            self._ctx.execute_async_v3(self._stream.handle)
            cuda.memcpy_dtoh_async(self._h_out, self._d_out, self._stream)
            self._stream.synchronize()
            emb = self._h_out[0].copy()
        return emb / np.linalg.norm(emb)


def load_recognizer() -> Recognizer:
    return Recognizer(str(MODEL_PATH))


def identify(qdrant: QdrantClient, embedding: np.ndarray,
             match_threshold: float) -> tuple[str, float]:
    # Qdrant searches the indexed templates natively and returns the nearest
    # one with its cosine score — payload is just the name (no matrices).
    hits = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=embedding.tolist(),
        with_payload=True,
        limit=1,
    ).points
    if not hits:
        return "Unknown", 0.0

    best = hits[0]
    score = float(best.score)
    if score < match_threshold:
        return "Unknown", score
    return best.payload["person_name"], score

# ── drawing ────────────────────────────────────────────────────────────────────

def draw_detection(frame: np.ndarray, x: int, y: int, w: int, h: int,
                   name: str, score: float) -> None:
    known = name != "Unknown"
    color = (0, 210, 0) if known else (0, 50, 200)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"{name}  {score * 100:.0f}%" if known else "Unknown"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color, -1)
    cv2.putText(frame, label, (x + 4, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS {fps:.1f}", (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 210, 0), 2, cv2.LINE_AA)

# ── shared state ───────────────────────────────────────────────────────────────

_accum:         dict = {}   # detection accumulator for the current frame
_latest_frame:  dict = {}   # newest raw frame per stream (always live)
_labels:        dict = {}   # last-known recognition labels per stream
_track_labels:  dict = {}   # (sid, track_id) -> (name, score)  — locked labels per track
_fps:           dict = {}
_pending:       set  = set()  # streams with a recognition job already in-flight
_lock = threading.Lock()


def _recognize_frame(sid, frame, dets, fps, rec, qdrant, match_threshold, crop_padding, track_lock_threshold):
    try:
        labels = []
        fh, fw = frame.shape[:2]
        for x, y, w, h, _, track_id in dets:
            pad_x = int(w * crop_padding)
            pad_y = int(h * crop_padding)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(fw, x + w + pad_x)
            y2 = min(fh, y + h + pad_y)
            crop = frame[y1:y2, x1:x2]
            if min(crop.shape[:2]) < 4:
                continue

            key = (sid, track_id)

            # Align to the ArcFace template — must match enrollment exactly.
            t0 = time.perf_counter()
            aligned = align_face(crop, _get_landmarker())
            t_align = time.perf_counter() - t0
            if aligned is None:
                # No landmarks (tiny / extreme-angle face). Feeding an unaligned
                # crop to an aligned DB would pollute matching, so skip and keep
                # this track's locked label if we have one.
                if track_id > 0:
                    with _lock:
                        locked = _track_labels.get(key)
                    if locked:
                        labels.append((x, y, w, h, locked[0], locked[1]))
                continue

            t1 = time.perf_counter()
            emb = rec.embed(aligned)
            t_embed = time.perf_counter() - t1
            t2 = time.perf_counter()
            name, score = identify(qdrant, emb, match_threshold)
            t_query = time.perf_counter() - t2

            if track_id > 0:
                if name != "Unknown" and score >= track_lock_threshold:
                    with _lock:
                        _track_labels[key] = (name, score)
                elif name == "Unknown":
                    with _lock:
                        locked = _track_labels.get(key)
                    if locked:
                        name, score = locked

            labels.append((x, y, w, h, name, score))
            tag = f"{name}  {score*100:.0f}%" if name != "Unknown" else "Unknown"
            print(f"[detect] stream={sid}  track={track_id}  {tag}  box=({x},{y},{w},{h})  "
                  f"align={t_align*1e3:.1f}ms embed={t_embed*1e3:.1f}ms query={t_query*1e3:.1f}ms", flush=True)

        with _lock:
            _labels[sid] = {"labels": labels, "fps": fps}
    except Exception as e:
        print(f"[render] {e}", flush=True)
    finally:
        with _lock:
            _pending.discard(sid)


def _flush(sid, rec, qdrant, executor, match_threshold, crop_padding, track_lock_threshold):
    state = _accum.pop(sid)
    if sid in _pending:
        # recognition still running for this stream — drop frame to avoid queue buildup
        return
    _pending.add(sid)
    executor.submit(_recognize_frame, sid, state["frame"],
                    state["dets"], _fps.get(sid, 0.0), rec, qdrant, match_threshold, crop_padding, track_lock_threshold)


def make_on_detection(rec, qdrant, executor, match_threshold, crop_padding, track_lock_threshold):
    def on_detection(d: DetectionEvent):
        sid, fnum = d.stream_id, d.frame_num
        with _lock:
            _latest_frame[sid] = np.array(d.frame)  # always keep newest frame for display
            if sid in _accum and _accum[sid]["fnum"] != fnum:
                _flush(sid, rec, qdrant, executor, match_threshold, crop_padding, track_lock_threshold)
            if sid not in _accum:
                _accum[sid] = {"fnum": fnum, "frame": d.frame, "dets": []}
            if d.conf > 0:
                _accum[sid]["dets"].append(
                    (int(d.x), int(d.y), int(d.w), int(d.h), d.conf, d.track_id))
    return on_detection


def make_on_stats():
    def on_stats(s: StatsEvent):
        with _lock:
            _fps[s.stream_id] = s.fps
    return on_stats

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    enroll_cfg      = yaml.safe_load(ENROLL_CFG.read_text())
    recog_cfg       = enroll_cfg["recognition"]
    match_threshold   = float(recog_cfg["similarity_threshold"])
    engine_conf       = float(recog_cfg["engine"]["detection"]["conf_threshold"])
    crop_padding        = float(recog_cfg.get("crop_padding", 0.0))
    track_lock_threshold = float(recog_cfg.get("track_lock_threshold", match_threshold))
    print(f"[config] similarity_threshold={match_threshold}  track_lock_threshold={track_lock_threshold}  engine_conf={engine_conf}  crop_padding={crop_padding}")

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(RTSP_TRT_DIR / "config.yaml")
    cfg      = yaml.safe_load(Path(cfg_path).read_text())
    cfg.setdefault("detection", {})
    cfg["detection"]["conf_threshold"] = engine_conf
    n        = len(cfg.get("streams", []))

    cell_w = cfg.get("tiler", {}).get("cell_width",  640)
    cell_h = cfg.get("tiler", {}).get("cell_height", 360)
    cols   = min(n, cfg.get("tiler", {}).get("columns", 2))
    rows   = (n + cols - 1) // cols
    win_w, win_h = cols * cell_w, rows * cell_h

    rec     = load_recognizer()
    qdrant   = QdrantClient(url=QDRANT_URL)
    executor = ThreadPoolExecutor(max_workers=max(4, n * 2))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        run_cfg = f.name

    p = Pipeline(run_cfg)
    p.set_on_detection(make_on_detection(rec, qdrant, executor, match_threshold, crop_padding, track_lock_threshold))
    p.set_on_stats(make_on_stats())
    p.start()

    # Wait for the pipeline thread to actually set running=True
    deadline = time.monotonic() + 10.0
    while not p.running and time.monotonic() < deadline:
        time.sleep(0.05)
    if not p.running:
        print("Pipeline failed to start")
        return

    print("Running — press Esc to quit")
    try:
        while p.running:
            with _lock:
                live   = dict(_latest_frame)
                labels = dict(_labels)
                fps_snap = dict(_fps)

            if live:
                canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                for sid, frame in live.items():
                    r, c         = divmod(sid, cols)
                    cell         = cv2.resize(frame, (cell_w, cell_h))
                    src_h, src_w = frame.shape[:2]
                    sx, sy       = cell_w / src_w, cell_h / src_h
                    for x, y, w, h, name, score in labels.get(sid, {}).get("labels", []):
                        draw_detection(cell,
                                       int(x*sx), int(y*sy),
                                       int(w*sx), int(h*sy),
                                       name, score)
                    draw_fps(cell, fps_snap.get(sid, 0.0))
                    canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = cell
                cv2.imshow("Recognition", canvas)

            if cv2.waitKey(30) == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        p.stop()
        executor.shutdown(wait=False)
        cv2.destroyAllWindows()
        os.unlink(run_cfg)


if __name__ == "__main__":
    main()
