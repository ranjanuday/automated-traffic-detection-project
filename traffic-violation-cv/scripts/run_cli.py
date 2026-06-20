"""CLI batch runner: process a folder of images -> annotated outputs + DB rows.

Usage:
    python -m scripts.run_cli data/samples
    python -m scripts.run_cli /path/to/images --out data/annotated
"""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import cv2

from backend import config, db
from backend.pipeline import engine
from backend.pipeline.annotate import annotate

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch traffic-violation analysis")
    ap.add_argument("folder", help="folder of input images")
    ap.add_argument("--out", default=str(config.ANNOTATED_DIR))
    ap.add_argument("--stop-line", type=float, default=0.55)
    args = ap.parse_args()

    db.init_db()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in Path(args.folder).iterdir() if p.suffix.lower() in _EXTS]
    if not images:
        print("No images found.")
        return

    total_v = 0
    for p in sorted(images):
        img = cv2.imread(str(p))
        if img is None:
            print(f"skip (unreadable): {p}")
            continue
        frame, metrics = engine.process(img, meta={"stop_line_y": args.stop_line})
        uid = uuid.uuid4().hex[:12]
        anno_name = f"{uid}_annotated.jpg"
        cv2.imwrite(str(out_dir / anno_name), annotate(frame))
        # store original under uploads for traceability
        orig_name = f"{uid}{p.suffix.lower()}"
        cv2.imwrite(str(config.UPLOAD_DIR / orig_name), img)
        eid = db.save_result(orig_name, anno_name, frame, metrics)
        total_v += len(frame.violations)
        print(f"{p.name}: {len(frame.violations)} violations "
              f"({metrics['detector_backend']}, {metrics['timings_ms']['total']}ms) -> evidence #{eid}")

    print(f"\nDone. {len(images)} images, {total_v} violations. "
          f"Open the dashboard to review.")


if __name__ == "__main__":
    main()
