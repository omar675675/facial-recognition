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

# ── settings ───────────────────────────────────────────────────────────────────

QDRANT_URL        = "http://localhost:6333"
QDRANT_COLLECTION = "faces"
MATCH_THRESHOLD   = 0.35

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


def identify(qdrant: QdrantClient, embedding: np.ndarray) -> tuple[str, float]:
    hits = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=embedding.tolist(),
        limit=1,
    ).points
    if not hits or hits[0].score < MATCH_THRESHOLD:
        return "Unknown", hits[0].score if hits else 0.0
    return hits[0].payload["person_name"], hits[0].score

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

_accum:        dict = {}   # detection accumulator for the current frame
_latest_frame: dict = {}   # newest raw frame per stream (always live)
_labels:       dict = {}   # last-known recognition labels per stream
_fps:          dict = {}
_pending:      set  = set()  # streams with a recognition job already in-flight
_lock = threading.Lock()


def _recognize_frame(sid, frame, dets, fps, rec, qdrant):
    try:
        labels = []
        for x, y, w, h, _ in dets:
            crop = frame[y:y+h, x:x+w]
            if min(crop.shape[:2]) < 4:
                continue
            name, score = identify(qdrant, rec.embed(cv2.resize(crop, (112, 112))))
            labels.append((x, y, w, h, name, score))
            tag = f"{name}  {score*100:.0f}%" if name != "Unknown" else "Unknown"
            print(f"[detect] stream={sid}  {tag}  box=({x},{y},{w},{h})", flush=True)
        with _lock:
            _labels[sid] = {"labels": labels, "fps": fps}
    except Exception as e:
        print(f"[render] {e}", flush=True)
    finally:
        with _lock:
            _pending.discard(sid)


def _flush(sid, rec, qdrant, executor):
    state = _accum.pop(sid)
    if sid in _pending:
        # recognition still running for this stream — drop frame to avoid queue buildup
        return
    _pending.add(sid)
    executor.submit(_recognize_frame, sid, state["frame"],
                    state["dets"], _fps.get(sid, 0.0), rec, qdrant)


def make_on_detection(rec, qdrant, executor):
    def on_detection(d: DetectionEvent):
        sid, fnum = d.stream_id, d.frame_num
        with _lock:
            _latest_frame[sid] = np.array(d.frame)  # always keep newest frame for display
            if sid in _accum and _accum[sid]["fnum"] != fnum:
                _flush(sid, rec, qdrant, executor)
            if sid not in _accum:
                _accum[sid] = {"fnum": fnum, "frame": d.frame, "dets": []}
            if d.conf > 0:
                _accum[sid]["dets"].append(
                    (int(d.x), int(d.y), int(d.w), int(d.h), d.conf))
    return on_detection


def make_on_stats():
    def on_stats(s: StatsEvent):
        with _lock:
            _fps[s.stream_id] = s.fps
    return on_stats

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(RTSP_TRT_DIR / "config.yaml")
    cfg      = yaml.safe_load(Path(cfg_path).read_text())
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
    p.set_on_detection(make_on_detection(rec, qdrant, executor))
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
