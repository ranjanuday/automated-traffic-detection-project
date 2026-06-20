"""Pipeline orchestrator.

Wires the stages together:
    preprocess -> detect -> plate OCR -> violation detectors -> annotate
Returns a fully-populated Frame plus timing/metrics. Pure function-ish: takes
an image, returns results; persistence is the caller's job (separation of
concerns).
"""
from __future__ import annotations

import time

import numpy as np

from backend import config
from backend.pipeline import detect, plate, preprocess
from backend.pipeline.annotate import annotate
from backend.pipeline.types import Frame, Violation, iou
from backend.pipeline.violations import all_detectors


def _restore_status() -> str:
    """Report deep-restoration availability without forcing a model load."""
    try:
        from backend.pipeline import restore
        return restore.status()
    except Exception:  # noqa: BLE001
        return "unavailable"


def _dedupe(violations: list[Violation]) -> list[Violation]:
    """Drop near-duplicate violations of the same type (IoU > 0.9)."""
    kept: list[Violation] = []
    for v in sorted(violations, key=lambda x: -x.confidence):
        if any(v.vtype == k.vtype and iou(v.bbox, k.bbox) > 0.9 for k in kept):
            continue
        kept.append(v)
    return kept


def _attach_plates(frame: Frame) -> None:
    """Best-effort: label each violation with the nearest vehicle's plate."""
    vehicle_idx = {i: d for i, d in enumerate(frame.detections)
                   if d.cls_name in config.PLATE_BEARING}
    for v in frame.violations:
        best_i, best_iou = None, 0.0
        for i, d in vehicle_idx.items():
            score = iou(v.bbox, d.bbox)
            if score > best_iou:
                best_i, best_iou = i, score
        if best_i is not None and best_i in frame.plates:
            v.plate = frame.plates[best_i]


def process(image: np.ndarray, meta: dict | None = None) -> tuple[Frame, dict]:
    """Run the full pipeline on a BGR image. Returns (frame, metrics)."""
    t0 = time.perf_counter()
    meta = meta or {}
    enhanced, pre_report = preprocess.enhance(
        image,
        deep_derain=bool(meta.get("deep_derain")),
        deep_deblur=bool(meta.get("deep_deblur")),
    )

    frame = Frame(image=enhanced, original=image, meta=meta)
    frame.meta["preprocess"] = pre_report

    t_pre = time.perf_counter()
    frame.detections = detect.detect(enhanced)
    t_det = time.perf_counter()

    plate.annotate_plates(frame)
    t_ocr = time.perf_counter()

    for detector in all_detectors():
        try:
            frame.violations.extend(detector.run(frame))
        except Exception as exc:  # one bad detector shouldn't sink the frame
            print(f"[engine] detector {detector.vtype} failed: {exc}")
    frame.violations = _dedupe(frame.violations)
    _attach_plates(frame)
    t_vio = time.perf_counter()

    metrics = {
        "detector_backend": detect.model_status(),
        "ocr_backend": plate.status(),
        "restore_backend": _restore_status(),
        "n_objects": len(frame.detections),
        "n_violations": len(frame.violations),
        "timings_ms": {
            "preprocess": round((t_pre - t0) * 1000, 1),
            "detect": round((t_det - t_pre) * 1000, 1),
            "ocr": round((t_ocr - t_det) * 1000, 1),
            "violations": round((t_vio - t_ocr) * 1000, 1),
            "total": round((t_vio - t0) * 1000, 1),
        },
    }
    return frame, metrics
