"""License-plate detection + OCR.

Two-stage ANPR:
  1. DETECT  - a YOLOv5 license-plate detector (config.PLATE_WEIGHTS) tight-
     crops the plate region inside each vehicle. Big OCR accuracy win.
  2. OCR     - EasyOCR reads the (tight) crop and we keep plate-shaped strings.
Both stages degrade gracefully: no detector -> OCR the whole vehicle crop;
no EasyOCR -> return None so the pipeline still completes.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from backend import config
from backend.pipeline.types import Detection, Frame

_reader = None
_status = "uninitialised"
_plate_det = None
_plate_status = "uninitialised"

# Generic plate pattern: 4-10 chars, letters+digits, allows spaces/dashes.
_PLATE_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-]{3,9}[A-Z0-9]$")


def status() -> str:
    _ensure()
    _ensure_detector()
    det = "+yolov5_detector" if _plate_det is not None else ""
    return _status + det


def _ensure():
    global _reader, _status
    if _reader is not None or _status == "unavailable":
        return
    try:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        _status = "easyocr"
    except Exception as exc:  # noqa: BLE001
        _reader = None
        _status = "unavailable"
        print(f"[plate] EasyOCR unavailable ({exc}); plate OCR disabled.")


def _ensure_detector():
    """Lazily load the optional YOLOv5 plate detector (trusted local file)."""
    global _plate_det, _plate_status
    if _plate_det is not None or _plate_status == "unavailable":
        return
    if not config.PLATE_WEIGHTS.exists():
        _plate_status = "unavailable"
        return
    try:
        import torch
        import yolov5
        # torch>=2.6 defaults weights_only=True; the YOLOv5 checkpoint stores the
        # model class, so it needs weights_only=False. ONLY use trusted weights
        # from the documented sources -- a malicious .pt could execute code on
        # load. The patch is scoped via try/finally so torch.load is always
        # restored (never left globally weakened, even on failure).
        _orig = torch.load
        try:
            torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
            _plate_det = yolov5.load(str(config.PLATE_WEIGHTS.resolve()), device="cpu")
        finally:
            torch.load = _orig
        _plate_det.conf = 0.25
        _plate_status = "loaded"
    except Exception as exc:  # noqa: BLE001
        _plate_det = None
        _plate_status = "unavailable"
        print(f"[plate] YOLOv5 detector unavailable ({exc}); OCR on full crop.")


def _localize_plate(vehicle_crop: np.ndarray) -> np.ndarray:
    """Return a tight plate crop if the detector finds one, else the input."""
    _ensure_detector()
    if _plate_det is None or vehicle_crop.size == 0:
        return vehicle_crop
    try:
        res = _plate_det(vehicle_crop[:, :, ::-1])  # detector expects RGB
        dets = res.pred[0].tolist()
        if not dets:
            return vehicle_crop
        x1, y1, x2, y2, _conf, _cls = max(dets, key=lambda d: d[4])
        h, w = vehicle_crop.shape[:2]
        px, py = int((x2 - x1) * 0.08), int((y2 - y1) * 0.15)  # small padding
        x1, y1 = max(0, int(x1) - px), max(0, int(y1) - py)
        x2, y2 = min(w, int(x2) + px), min(h, int(y2) + py)
        tight = vehicle_crop[y1:y2, x1:x2]
        return tight if tight.size else vehicle_crop
    except Exception:  # noqa: BLE001
        return vehicle_crop


def _clean(text: str) -> str | None:
    t = re.sub(r"[^A-Z0-9 \-]", "", text.upper()).strip()
    return t if _PLATE_RE.match(t) and any(c.isdigit() for c in t) else None


def read_plate(frame: Frame, vehicle: Detection) -> str | None:
    """Best-effort plate string for one vehicle (detect -> tight-crop -> OCR)."""
    _ensure()
    if _reader is None:
        return None
    x1, y1, x2, y2 = vehicle.bbox
    crop = frame.image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = _localize_plate(crop)  # tighten to the plate when detector available
    # Upscale small crops to help OCR.
    if crop.shape[0] < 120:
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    best, best_conf = None, 0.0
    for (_box, text, conf) in _reader.readtext(gray):
        cleaned = _clean(text)
        if cleaned and conf > best_conf:
            best, best_conf = cleaned, conf
    return best


def annotate_plates(frame: Frame) -> None:
    """Populate frame.plates {detection_index: plate_string} for vehicles."""
    for idx, det in enumerate(frame.detections):
        if det.cls_name in config.PLATE_BEARING:
            plate = read_plate(frame, det)
            if plate:
                frame.plates[idx] = plate
