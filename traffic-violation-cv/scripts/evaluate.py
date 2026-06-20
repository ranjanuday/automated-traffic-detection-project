"""Performance evaluation harness.

Computes the metrics the brief asks for:
  * Per violation-class Precision / Recall / F1 + image-level Accuracy
    (from a presence-based ground truth).
  * Object-detection mAP@0.5 (true Average Precision over IoU-matched boxes)
    when a bbox-level ground truth is supplied.

Violation ground-truth (data/ground_truth.json):
{
  "image1.jpg": {"violations": ["no_helmet", "triple_riding"]},
  "image2.jpg": {"violations": []}
}

Detection ground-truth (for mAP), pass with --detection-gt:
{
  "image1.jpg": {"boxes": [{"cls": "car", "bbox": [x1,y1,x2,y2]}, ...]}
}

Usage:
    python -m scripts.evaluate data/samples data/ground_truth.json
    python -m scripts.evaluate data/samples data/gt.json --detection-gt data/boxes.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from backend import config
from backend.pipeline import engine

VTYPES = config.VIOLATION_TYPES


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def _iou(a, b) -> float:
    """IoU of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _average_precision(matches: list[tuple[float, int]], n_gt: int) -> float:
    """All-point AP from (confidence, is_true_positive) pairs for one class."""
    if n_gt == 0:
        return 0.0
    matches.sort(key=lambda m: m[0], reverse=True)
    tp = fp = 0
    recalls, precisions = [], []
    for _conf, is_tp in matches:
        tp += is_tp
        fp += 1 - is_tp
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))
    # Monotone-decreasing precision envelope, then integrate over recall.
    ap, prev_r = 0.0, 0.0
    for i in range(len(recalls)):
        p_max = max(precisions[i:]) if precisions[i:] else 0.0
        ap += (recalls[i] - prev_r) * p_max
        prev_r = recalls[i]
    return ap


def detection_map(folder: str, det_gt: dict, iou_thr: float = 0.5) -> None:
    """Compute and print object-detection mAP@iou_thr over the GT classes."""
    from collections import defaultdict
    per_cls_matches: dict[str, list] = defaultdict(list)
    per_cls_ngt: dict[str, int] = defaultdict(int)
    for name, truth in det_gt.items():
        img = cv2.imread(str(Path(folder) / name))
        if img is None:
            continue
        frame, _ = engine.process(img)
        gts = truth.get("boxes", [])
        for g in gts:
            per_cls_ngt[g["cls"]] += 1
        used = [False] * len(gts)
        preds = sorted(frame.detections, key=lambda d: d.confidence, reverse=True)
        for d in preds:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if g["cls"] != d.cls_name or used[j]:
                    continue
                i = _iou(d.bbox, g["bbox"])
                if i > best_iou:
                    best_iou, best_j = i, j
            is_tp = 1 if best_iou >= iou_thr and best_j >= 0 else 0
            if is_tp:
                used[best_j] = True
            per_cls_matches[d.cls_name].append((d.confidence, is_tp))
    print(f"\nObject-detection mAP@{iou_thr}")
    print(f"{'class':16}{'AP':>8}{'nGT':>6}")
    aps = []
    for cls in sorted(per_cls_ngt):
        ap = _average_precision(list(per_cls_matches[cls]), per_cls_ngt[cls])
        aps.append(ap)
        print(f"{cls:16}{ap:8.3f}{per_cls_ngt[cls]:6}")
    if aps:
        print(f"\nmAP@{iou_thr} = {sum(aps)/len(aps):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate violation detection")
    ap.add_argument("folder")
    ap.add_argument("ground_truth")
    ap.add_argument("--detection-gt", default="",
                    help="optional bbox ground-truth JSON -> object mAP@0.5")
    args = ap.parse_args()

    gt = json.loads(Path(args.ground_truth).read_text())
    counts = {v: {"tp": 0, "fp": 0, "fn": 0} for v in VTYPES}
    img_correct = 0

    for name, truth in gt.items():
        path = Path(args.folder) / name
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip {name}: unreadable")
            continue
        frame, _ = engine.process(img)
        pred = {v.vtype for v in frame.violations}
        true = set(truth.get("violations", []))

        if pred == true:
            img_correct += 1
        for v in VTYPES:
            if v in pred and v in true:
                counts[v]["tp"] += 1
            elif v in pred and v not in true:
                counts[v]["fp"] += 1
            elif v not in pred and v in true:
                counts[v]["fn"] += 1

    print(f"\nImage-level exact-match accuracy: {img_correct}/{len(gt)} "
          f"= {img_correct / max(1, len(gt)):.2%}\n")
    print(f"{'class':16}{'P':>8}{'R':>8}{'F1':>8}{'TP':>5}{'FP':>5}{'FN':>5}")
    macro = defaultdict(float)
    used = 0
    for v in VTYPES:
        c = counts[v]
        if c["tp"] + c["fp"] + c["fn"] == 0:
            continue
        p, r, f1 = _prf(c["tp"], c["fp"], c["fn"])
        macro["p"] += p; macro["r"] += r; macro["f1"] += f1; used += 1
        print(f"{v:16}{p:8.2f}{r:8.2f}{f1:8.2f}{c['tp']:5}{c['fp']:5}{c['fn']:5}")
    if used:
        print(f"\nMacro avg: P={macro['p']/used:.2f} "
              f"R={macro['r']/used:.2f} F1={macro['f1']/used:.2f}")

    if args.detection_gt:
        det_gt = json.loads(Path(args.detection_gt).read_text())
        detection_map(args.folder, det_gt)


if __name__ == "__main__":
    main()
