"""Rider-related violations: triple riding & helmet non-compliance.

Both hinge on associating `person` detections with a two-wheeler, so they
share that logic via `_riders_on`.
"""
from __future__ import annotations

from backend import config
from backend.pipeline.types import (
    Detection, Frame, Violation, overlap_ratio, union_bbox,
)
from backend.pipeline.violations.base import ViolationDetector, register


def _riders_on(bike: Detection, persons: list[Detection]) -> list[Detection]:
    """Persons sufficiently overlapping a two-wheeler box == its riders."""
    return [p for p in persons if overlap_ratio(p.bbox, bike.bbox) > 0.25]


@register
class TripleRidingDetector(ViolationDetector):
    """3+ persons on a single two-wheeler."""

    vtype = "triple_riding"

    def run(self, frame: Frame) -> list[Violation]:
        out: list[Violation] = []
        persons = self.persons(frame)
        for bike in self.of(frame, "motorcycle") + self.of(frame, "bicycle"):
            riders = _riders_on(bike, persons)
            if len(riders) >= 3:
                boxes = [bike.bbox] + [r.bbox for r in riders]
                conf = min(0.99, 0.6 + 0.1 * len(riders))
                out.append(Violation(
                    vtype=self.vtype,
                    confidence=round(conf, 2),
                    bbox=union_bbox(boxes),
                    note=f"{len(riders)} riders detected on one two-wheeler",
                ))
        return out


@register
class NoHelmetDetector(ViolationDetector):
    """Two-wheeler riders not wearing a helmet.

    Robust mode needs a helmet-specialised model at config.HELMET_WEIGHTS.
    Without it we apply a colour/region heuristic on the head zone and clearly
    lower the confidence so reviewers know it's advisory, not definitive.
    """

    vtype = "no_helmet"

    def __init__(self) -> None:
        self._helmet_model = None
        self._tried = False

    def _load(self):
        if self._tried:
            return
        self._tried = True
        if config.HELMET_WEIGHTS.exists():
            try:
                from ultralytics import YOLO
                self._helmet_model = YOLO(str(config.HELMET_WEIGHTS))
            except Exception as exc:  # noqa: BLE001
                print(f"[no_helmet] helmet model load failed: {exc}")

    def run(self, frame: Frame) -> list[Violation]:
        self._load()
        persons = self.persons(frame)
        bikes = self.of(frame, "motorcycle")
        out: list[Violation] = []
        for bike in bikes:
            for rider in _riders_on(bike, persons):
                helmeted, conf = self._is_helmeted(frame, rider)
                if not helmeted:
                    out.append(Violation(
                        vtype=self.vtype,
                        confidence=round(conf, 2),
                        bbox=rider.bbox,
                        note=("helmet model: no helmet"
                              if self._helmet_model
                              else "heuristic (advisory) — no specialised "
                                   "helmet model loaded"),
                    ))
        return out

    def _is_helmeted(self, frame: Frame, rider: Detection) -> tuple[bool, float]:
        x1, y1, x2, y2 = rider.bbox
        # Run the helmet model on the upper body (more context than head-only);
        # fall back to a head-only texture heuristic when no model is loaded.
        upper_h = int((y2 - y1) * 0.5)
        upper = frame.image[y1:y1 + upper_h, x1:x2]
        if upper.size == 0:
            return True, 0.0  # can't tell -> don't flag

        if self._helmet_model is not None:
            res = self._helmet_model.predict(upper, verbose=False, conf=0.25)
            helmet_conf, nohelmet_conf = 0.0, 0.0
            for r in res:
                for b in r.boxes:
                    name = self._helmet_model.names[int(b.cls)].lower()
                    conf = float(b.conf)
                    positive = (any(k in name for k in ("helmet", "hardhat"))
                                and not name.startswith(("no", "non"))
                                and "without" not in name)
                    # 'head' (bare) is the canonical no-helmet signal in the
                    # helmet/head/person scheme; also handle explicit negatives.
                    negative = (name in ("head", "nohelmet", "no-helmet")
                                or name.startswith(("no", "non"))
                                or "without" in name)
                    if positive:
                        helmet_conf = max(helmet_conf, conf)
                    elif negative:
                        nohelmet_conf = max(nohelmet_conf, conf)
            if helmet_conf >= nohelmet_conf and helmet_conf > 0:
                return True, helmet_conf
            if nohelmet_conf > 0:
                return False, nohelmet_conf
            return True, 0.0  # model saw neither -> don't flag (avoid FPs)

        # Heuristic fallback: helmets are smooth; bare head/hair = more texture.
        import cv2
        head = frame.image[y1:y1 + int((y2 - y1) * 0.3), x1:x2]
        gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
        density = cv2.Canny(gray, 80, 160).mean()
        return (density < 18.0), 0.45
