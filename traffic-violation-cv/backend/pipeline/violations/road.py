"""Zone/context violations: illegal parking, wrong-side, seatbelt.

These three are inherently *context-dependent* — a single still frame can't
know traffic-flow direction or see inside a windscreen reliably. We implement
them honestly: geometry/zone-driven where defensible, model-pluggable where a
specialised model is the only credible path, and we never fabricate confidence.

Per-camera config travels in frame.meta:
    no_parking_zones : list[list[(x,y)]]   polygons in pixel coords
    wrong_side_zone  : {"y": <px>, "dir": "left"|"right"}
"""
from __future__ import annotations

import cv2
import numpy as np

from backend import config
from backend.pipeline.types import Detection, Frame, Violation
from backend.pipeline.violations.base import ViolationDetector, register


@register
class IllegalParkingDetector(ViolationDetector):
    """Vehicle whose centre sits inside a configured no-parking polygon."""

    vtype = "illegal_parking"

    def run(self, frame: Frame) -> list[Violation]:
        zones = frame.meta.get("no_parking_zones") or []
        if not zones:
            return []
        polys = [np.array(z, dtype=np.int32) for z in zones]
        out: list[Violation] = []
        for v in self.vehicles(frame):
            cx, cy = v.center
            if any(cv2.pointPolygonTest(p, (cx, cy), False) >= 0
                   for p in polys):
                out.append(Violation(
                    vtype=self.vtype, confidence=0.8, bbox=v.bbox,
                    note=f"{v.cls_name} stopped inside no-parking zone",
                ))
        return out


@register
class WrongSideDetector(ViolationDetector):
    """Vehicle present on the wrong carriageway half.

    Honest caveat: true wrong-side needs motion/orientation. From a still we
    only flag vehicles whose centre is on the configured 'wrong' side of a
    divider line, at reduced confidence.
    """

    vtype = "wrong_side"

    def run(self, frame: Frame) -> list[Violation]:
        zone = frame.meta.get("wrong_side_zone")
        if not zone:
            return []
        divider_x = int(zone.get("x", frame.image.shape[1] // 2))
        wrong = zone.get("dir", "left")
        out: list[Violation] = []
        for v in self.vehicles(frame):
            cx, _ = v.center
            on_left = cx < divider_x
            if (wrong == "left" and on_left) or (wrong == "right" and not on_left):
                out.append(Violation(
                    vtype=self.vtype, confidence=0.5, bbox=v.bbox,
                    note=f"{v.cls_name} on wrong carriageway half (advisory)",
                ))
        return out


@register
class NoSeatbeltDetector(ViolationDetector):
    """Driver/front-passenger seatbelt non-compliance.

    Runs a seatbelt model (config.SEATBELT_WEIGHTS, classes no_seatbelt/
    seat_belt) on the windscreen region of each detected car. Falls back to a
    pluggable callable in frame.meta['seatbelt_model'] (crop -> (belted, conf)),
    and abstains entirely if neither is available (no fabricated confidence).
    """

    vtype = "no_seatbelt"

    def __init__(self) -> None:
        self._model = None
        self._tried = False

    def _load(self):
        if self._tried:
            return
        self._tried = True
        if config.SEATBELT_WEIGHTS.exists():
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(config.SEATBELT_WEIGHTS))
            except Exception as exc:  # noqa: BLE001
                print(f"[no_seatbelt] model load failed: {exc}")

    def _classify(self, crop: np.ndarray) -> tuple[bool, float]:
        """Return (belted, confidence) from the seatbelt model.

        Supports both a classifier (probs over no_seatbelt/seat_belt) and a
        detector (boxes). Abstains (belted=True, conf 0) when uncertain.
        """
        res = self._model.predict(crop, verbose=False, conf=0.25)
        r = res[0]
        # Classification model: single label over the crop.
        if getattr(r, "probs", None) is not None:
            name = self._model.names[r.probs.top1].lower()
            conf = float(r.probs.top1conf)
            belted = not ("no" in name and "belt" in name)
            return belted, conf
        # Detection model: look for belt / no-belt boxes.
        belt_conf, nobelt_conf = 0.0, 0.0
        for b in (r.boxes or []):
            name = self._model.names[int(b.cls)].lower()
            conf = float(b.conf)
            if "no" in name and "belt" in name:
                nobelt_conf = max(nobelt_conf, conf)
            elif "belt" in name:
                belt_conf = max(belt_conf, conf)
        if nobelt_conf > belt_conf and nobelt_conf > 0:
            return False, nobelt_conf
        if belt_conf > 0:
            return True, belt_conf
        return True, 0.0  # nothing seen -> don't flag

    def run(self, frame: Frame) -> list[Violation]:
        self._load()
        model_cb = frame.meta.get("seatbelt_model")
        # Seatbelt is only credible from a windscreen/front-facing view. On
        # exterior surveillance frames we'd over-flag every car, so this is
        # opt-in: enable via meta['seatbelt_check']=True (camera is windscreen
        # facing) or supply a custom callable. Otherwise abstain.
        if model_cb is None and not (
                self._model is not None and frame.meta.get("seatbelt_check")):
            return []
        out: list[Violation] = []
        cars = self.of(frame, "car")
        # When the whole frame IS the windscreen view (no car box), classify it.
        regions = cars if cars else [None]
        for v in regions:
            if v is None:
                crop, bbox = frame.image, (0, 0, frame.image.shape[1],
                                           frame.image.shape[0])
            else:
                x1, y1, x2, y2 = v.bbox
                wh = int((y2 - y1) * 0.6)  # windscreen = upper part of car
                crop, bbox = frame.image[y1:y1 + wh, x1:x2], v.bbox
            if crop.size == 0:
                continue
            belted, conf = (model_cb(crop) if model_cb is not None
                            else self._classify(crop))
            if not belted:
                out.append(Violation(
                    vtype=self.vtype, confidence=round(conf, 2), bbox=bbox,
                    note="seatbelt model: not fastened",
                ))
        return out
