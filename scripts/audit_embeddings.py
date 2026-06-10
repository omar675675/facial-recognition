#!/usr/bin/env python3
"""
Cross-person similarity audit.
Loads crops from the embeddings folder, extracts embeddings, and reports any
pair from different persons whose cosine similarity exceeds the threshold.

Usage:
    python scripts/audit_embeddings.py
    python scripts/audit_embeddings.py --threshold 0.40
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO_DIR       = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = REPO_DIR / "embeddings"
MODEL_PATH     = REPO_DIR / "models" / "w600k_r50.onnx"


def load_model() -> ort.InferenceSession:
    sess = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"[model] {MODEL_PATH.name}  ({sess.get_providers()[0]})\n")
    return sess


def extract_embedding(sess: ort.InferenceSession, img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 127.5
    img = img.transpose(2, 0, 1)[np.newaxis]
    emb = sess.run(None, {sess.get_inputs()[0].name: img})[0][0]
    return emb / np.linalg.norm(emb)


def load_person(sess: ort.InferenceSession, person_dir: Path) -> list[tuple[str, np.ndarray]]:
    results = []
    for img_path in sorted(person_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [warn] could not read {img_path.name}")
            continue
        results.append((img_path.name, extract_embedding(sess, img)))
    return results


OUT_DIR = REPO_DIR / "audit"

LABEL_H  = 28   # px height of the text bar under each crop
PAD      = 6    # px gap between the two crops
BG_COLOR = (30, 30, 30)


def make_pair_image(
    a_img: np.ndarray, a_label: str,
    b_img: np.ndarray, b_label: str,
    sim: float,
) -> np.ndarray:
    """Two 112×112 crops side-by-side with labels and a similarity header."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    crop_w     = 112
    total_w    = crop_w * 2 + PAD
    header_h   = LABEL_H
    total_h    = header_h + crop_w + LABEL_H

    canvas = np.full((total_h, total_w, 3), BG_COLOR, dtype=np.uint8)

    # header
    header = f"{sim * 100:.1f}% similarity"
    (tw, _), _ = cv2.getTextSize(header, font, 0.55, 1)
    cv2.putText(canvas, header, ((total_w - tw) // 2, header_h - 6),
                font, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

    # crops
    canvas[header_h:header_h + crop_w,          0:crop_w]            = a_img
    canvas[header_h:header_h + crop_w, crop_w + PAD:crop_w * 2 + PAD] = b_img

    # per-crop labels (truncate if too wide)
    for label, x_off in [(a_label, 0), (b_label, crop_w + PAD)]:
        y = header_h + crop_w + LABEL_H - 6
        cv2.putText(canvas, label, (x_off + 2, y),
                    font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="cosine similarity threshold (default: 0.40)")
    parser.add_argument("--out", type=Path, default=OUT_DIR,
                        help="folder to save pair images")
    args = parser.parse_args()

    person_dirs = sorted(d for d in EMBEDDINGS_DIR.iterdir() if d.is_dir())
    if len(person_dirs) < 2:
        print("Need at least 2 persons enrolled in the embeddings folder.")
        return

    sess = load_model()

    persons: dict[str, list[tuple[str, np.ndarray]]] = {}
    raw_crops: dict[str, dict[str, np.ndarray]] = {}   # person → filename → image
    for d in person_dirs:
        embs = load_person(sess, d)
        if embs:
            persons[d.name] = embs
            raw_crops[d.name] = {
                fname: cv2.imread(str(d / fname))
                for fname, _ in embs
            }
            print(f"  {d.name}: {len(embs)} crops")

    print(f"\nAuditing cross-person pairs  (threshold ≥ {args.threshold * 100:.0f}%) …\n")

    args.out.mkdir(parents=True, exist_ok=True)

    found = 0
    names = list(persons.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            hits = []
            for a_file, a_emb in persons[a_name]:
                for b_file, b_emb in persons[b_name]:
                    sim = float(np.dot(a_emb, b_emb))
                    if sim >= args.threshold:
                        hits.append((sim, a_file, b_file))

            if hits:
                hits.sort(reverse=True)
                print(f"── {a_name}  ↔  {b_name}  ({len(hits)} pair(s))")
                for sim, a_file, b_file in hits:
                    print(f"   {sim * 100:.1f}%   {a_file}  ↔  {b_file}")
                    a_img = raw_crops[a_name][a_file]
                    b_img = raw_crops[b_name][b_file]
                    pair  = make_pair_image(
                        a_img, f"{a_name}/{a_file}",
                        b_img, f"{b_name}/{b_file}",
                        sim,
                    )
                    stem = (
                        f"{sim * 100:.0f}pct"
                        f"__{a_name}_{a_file.replace('.jpg', '')}"
                        f"__{b_name}_{b_file.replace('.jpg', '')}"
                    )
                    cv2.imwrite(str(args.out / f"{stem}.jpg"), pair)
                print()
                found += len(hits)

    if found == 0:
        print(f"No cross-person pairs at or above {args.threshold * 100:.0f}%.")
    else:
        print(f"{found} problematic pair(s) found → {args.out}")


if __name__ == "__main__":
    main()
