"""Vehicle & road-user detection (COCO + Indian model fusion).

Single COCO YOLOv8 is great at people/cars/etc. but has *no* auto-rickshaw
class, so three-wheelers get called 'car'. A South-Asian YOLOv8 knows
auto-rickshaw (CNG) / rickshaw but is weaker at general objects (people).

So we FUSE:
  * COCO (config.YOLO_WEIGHTS)  -> the workhorse: person, car, motorcycle,
    bus, truck, bicycle, traffic_light.
  * Indian (config.INDIAN_WEIGHTS, ONNX) -> specialist: contributes
    auto_rickshaw / rickshaw, and *re-labels* any COCO vehicle it overlaps
    with high IoU (fixing 'car' -> 'auto_rickshaw').

If the ML stack is missing entirely we fall back to a deterministic mock so the
whole app stays demoable.
"""
from __future__ import annotations

import hashlib

import numpy as np

from backend import config
from backend.pipeline.types import Detection, iou

_coco = None
_indian = None
_indian_labels: list[str] = []
_status = "uninitialised"
_loaded = False

# Indian categories COCO cannot produce -> always merged in.
_INDIAN_ONLY = {"auto_rickshaw", "rickshaw"}
# COCO vehicle labels an overlapping auto-rickshaw should override.
_OVERRIDABLE = {"car", "motorcycle", "truck"}


def model_status() -> str:
    """Human-readable backend state for the UI banner."""
    _ensure()
    return _status


def _ensure():
    global _coco, _indian, _indian_labels, _status, _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from ultralytics import YOLO  # noqa: WPS433 (lazy import on purpose)
        _coco = YOLO(config.YOLO_WEIGHTS)
        parts = ["coco"]
        if config.PREFER_INDIAN and config.INDIAN_WEIGHTS.exists():
            _indian = YOLO(str(config.INDIAN_WEIGHTS), task="detect")
            if config.INDIAN_LABELS.exists():
                with open(config.INDIAN_LABELS) as fh:
                    _indian_labels = [ln.strip() for ln in fh if ln.strip()]
            parts.append("indian")
        _status = "yolo-fusion:" + "+".join(parts)
    except Exception as exc:  # ML stack absent or weights unavailable
        _coco = None
        _status = "mock"
        print(f"[detect] detector unavailable ({exc}); using mock detector.")


def detect(image: np.ndarray) -> list[Detection]:
    """Detect road users in a BGR image -> fused list[Detection]."""
    _ensure()
    if _coco is None:
        return _mock_detect(image)
    base = _coco_detect(image)
    if _indian is None:
        return base
    return _fuse(base, _indian_detect(image))


def _coco_detect(image: np.ndarray) -> list[Detection]:
    res = _coco.predict(image, conf=config.DETECT_CONF, verbose=False)
    names = _coco.names
    out: list[Detection] = []
    for r in res:
        for box in r.boxes:
            raw = names[int(box.cls)]
            norm = _normalise_coco(raw)
            if norm is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            out.append(Detection(cls_name=norm, confidence=float(box.conf),
                                  bbox=(x1, y1, x2, y2), raw_label=raw))
    return out


def _indian_detect(image: np.ndarray) -> list[Detection]:
    res = _indian.predict(image, conf=config.DETECT_CONF, verbose=False)
    out: list[Detection] = []
    for r in res:
        for box in r.boxes:
            cid = int(box.cls)
            raw = _indian_labels[cid] if cid < len(_indian_labels) \
                else _indian.names.get(cid, str(cid))
            norm = config.INDIAN_CLASS_MAP.get(raw)
            if norm is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            out.append(Detection(cls_name=norm, confidence=float(box.conf),
                                  bbox=(x1, y1, x2, y2), raw_label=raw))
    return out


def _fuse(base: list[Detection], indian: list[Detection]) -> list[Detection]:
    """COCO base + Indian auto-rickshaws (re-labelling overlapped vehicles)."""
    fused = list(base)
    for d in indian:
        if d.cls_name not in _INDIAN_ONLY:
            continue  # trust COCO for person/car/etc.; only borrow rickshaws
        # Does this rickshaw overlap a COCO vehicle that was mislabelled?
        overlapped = None
        for b in fused:
            if b.cls_name in _OVERRIDABLE and iou(d.bbox, b.bbox) >= 0.45:
                overlapped = b
                break
        if overlapped is not None:
            # Re-label the COCO box (keep its tighter geometry/confidence).
            fused[fused.index(overlapped)] = Detection(
                cls_name=d.cls_name, confidence=max(d.confidence,
                                                    overlapped.confidence),
                bbox=overlapped.bbox, raw_label=d.raw_label)
        elif not any(iou(d.bbox, b.bbox) >= 0.55 for b in fused):
            fused.append(d)  # genuinely new object COCO missed
    return fused


def _normalise_coco(raw: str) -> str | None:
    if raw in config.VEHICLE_CLASSES:
        return config.VEHICLE_CLASSES[raw]
    if raw == config.PERSON_CLASS:
        return "person"
    if raw == config.TRAFFIC_LIGHT_CLASS:
        return "traffic_light"
    return None


def _mock_detect(image: np.ndarray) -> list[Detection]:
    """Deterministic pseudo-detections seeded by image bytes (demo fallback)."""
    h, w = image.shape[:2]
    seed = int(hashlib.md5(image.tobytes()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    def box(cx, cy, bw, bh):
        return (max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)),
                min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2)))

    dets: list[Detection] = []
    mcx, mcy = w * 0.4, h * 0.7
    dets.append(Detection("motorcycle", 0.88,
                          box(mcx, mcy, w * 0.18, h * 0.30), "motorcycle"))
    n_riders = 3 if rng.random() > 0.4 else 2
    for i in range(n_riders):
        rx = mcx - w * 0.05 + i * w * 0.05
        ry = mcy - h * 0.18
        dets.append(Detection("person", 0.8 - i * 0.05,
                              box(rx, ry, w * 0.07, h * 0.22), "person"))
    dets.append(Detection("car", 0.91,
                          box(w * 0.78, h * 0.6, w * 0.25, h * 0.22), "car"))
    dets.append(Detection("traffic_light", 0.7,
                          box(w * 0.9, h * 0.15, w * 0.04, h * 0.12),
                          "traffic light"))
    return dets
